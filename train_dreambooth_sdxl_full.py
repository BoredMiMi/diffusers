#!/usr/bin/env python
# coding=utf-8
"""Minimal DreamBooth training script for Stable Diffusion XL.
Fine-tunes the entire SDXL model on a small set of images describing a custom
concept. Optimised for a single GPU and small datasets.
"""

import argparse
import itertools
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler


class DreamBoothDataset(Dataset):
    def __init__(self, instance_data_root, instance_prompt, size=1024, repeats=1, center_crop=False):
        self.size = size
        self.center_crop = center_crop
        self.instance_prompt = instance_prompt

        image_paths = sorted(Path(instance_data_root).iterdir())
        self.instance_images = list(itertools.chain.from_iterable(itertools.repeat(p, repeats) for p in image_paths))

        self.resize = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)
        self.cropper = transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size)
        self.to_tensor = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        )

    def __len__(self):
        return len(self.instance_images)

    def __getitem__(self, index):
        image = Image.open(self.instance_images[index])
        image = exif_transpose(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        original_size = (image.height, image.width)
        image = self.resize(image)
        if self.center_crop:
            y1 = max(0, int(round((image.height - self.size) / 2.0)))
            x1 = max(0, int(round((image.width - self.size) / 2.0)))
            image = self.cropper(image)
        else:
            y1, x1, h, w = self.cropper.get_params(image, (self.size, self.size))
            image = crop(image, y1, x1, h, w)
        crop_top_left = (y1, x1)
        pixel_values = self.to_tensor(image)

        return {
            "pixel_values": pixel_values,
            "prompt": self.instance_prompt,
            "original_size": original_size,
            "crop_top_left": crop_top_left,
        }


def tokenize_prompt(tokenizer, prompt):
    text_inputs = tokenizer(
        prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    return text_inputs.input_ids


def encode_prompt(text_encoders, tokenizers, prompt):
    prompt_embeds_list = []
    for i, text_encoder in enumerate(text_encoders):
        tokens = tokenize_prompt(tokenizers[i], prompt)
        embeds = text_encoder(tokens.to(text_encoder.device), output_hidden_states=True, return_dict=False)
        pooled = embeds[0]
        hidden = embeds[-1][-2]
        bs_embed, seq_len, _ = hidden.shape
        hidden = hidden.view(bs_embed, seq_len, -1)
        prompt_embeds_list.append(hidden)
    prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)
    pooled_prompt_embeds = pooled.view(bs_embed, -1)
    return prompt_embeds, pooled_prompt_embeds


def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str = None, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")


def compute_time_ids(original_size, crop_top_left, target_size):
    add_time_ids = list(original_size + crop_top_left + target_size)
    return torch.tensor([add_time_ids])


def parse_args():
    parser = argparse.ArgumentParser(description="DreamBooth SDXL full fine-tuning")
    parser.add_argument("--pretrained_model_name_or_path", required=True)
    parser.add_argument("--instance_data_dir", required=True)
    parser.add_argument("--instance_prompt", required=True)
    parser.add_argument("--output_dir", default="sdxl-dreambooth")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16"], default="no")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_text_encoder", action="store_true")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
        project_config=ProjectConfiguration(project_dir=args.output_dir),
    )
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    tokenizer_one = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer_2", use_fast=False
    )
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
    )
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        variant=args.variant,
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder_2",
        revision=args.revision,
        variant=args.variant,
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        if args.train_text_encoder:
            text_encoder_one.gradient_checkpointing_enable()
            text_encoder_two.gradient_checkpointing_enable()

    train_dataset = DreamBoothDataset(args.instance_data_dir, args.instance_prompt, args.resolution)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=1)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    unet.to(accelerator.device, dtype=weight_dtype)
    text_encoder_one.to(accelerator.device, dtype=weight_dtype)
    text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    optimizer = torch.optim.AdamW(
        itertools.chain(
            unet.parameters(),
            text_encoder_one.parameters() if args.train_text_encoder else [],
            text_encoder_two.parameters() if args.train_text_encoder else [],
        ),
        lr=args.learning_rate,
    )
    lr_scheduler = get_scheduler(
        "constant", optimizer=optimizer, num_warmup_steps=0, num_training_steps=args.max_train_steps
    )

    unet, text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, text_encoder_one, text_encoder_two, optimizer, train_dataloader, lr_scheduler
    )

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                if isinstance(model, type(accelerator.unwrap_model(unet))):
                    sub_dir = "unet"
                elif isinstance(model, type(accelerator.unwrap_model(text_encoder_one))):
                    sub_dir = "text_encoder"
                elif isinstance(model, type(accelerator.unwrap_model(text_encoder_two))):
                    sub_dir = "text_encoder_2"
                else:
                    continue
                model.save_pretrained(os.path.join(output_dir, sub_dir))
                if weights:
                    weights.pop()

    def load_model_hook(models, input_dir):
        while len(models) > 0:
            model = models.pop()
            if isinstance(model, type(accelerator.unwrap_model(text_encoder_one))):
                load_model = text_encoder_cls_one.from_pretrained(input_dir, subfolder="text_encoder")
                model.config = load_model.config
            elif isinstance(model, type(accelerator.unwrap_model(text_encoder_two))):
                load_model = text_encoder_cls_two.from_pretrained(input_dir, subfolder="text_encoder_2")
                model.config = load_model.config
            else:
                load_model = UNet2DConditionModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)
            model.load_state_dict(load_model.state_dict())
            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    global_step = 0
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)

    for epoch in range(num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                latents = vae.encode(
                    batch["pixel_values"].to(device=accelerator.device, dtype=weight_dtype)
                ).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                add_time_ids = torch.cat(
                    [
                        compute_time_ids(s, c, (args.resolution, args.resolution))
                        for s, c in zip(batch["original_size"], batch["crop_top_left"])
                    ]
                ).to(accelerator.device, dtype=weight_dtype)

                prompt_embeds, pooled_prompt_embeds = encode_prompt(
                    [text_encoder_one, text_encoder_two], [tokenizer_one, tokenizer_two], batch["prompt"]
                )
                prompt_embeds = prompt_embeds.to(accelerator.device, dtype=weight_dtype)
                pooled_prompt_embeds = pooled_prompt_embeds.to(accelerator.device, dtype=weight_dtype)

                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    prompt_embeds,
                    added_cond_kwargs={"time_ids": add_time_ids, "text_embeds": pooled_prompt_embeds},
                )[0]

                target = noise
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    global_step += 1
                    if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                if global_step >= args.max_train_steps:
                    break
        if global_step >= args.max_train_steps:
            break

    if accelerator.is_main_process:
        accelerator.save_state(args.output_dir)
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            unet=accelerator.unwrap_model(unet),
            text_encoder=text_encoder_one,
            text_encoder_2=text_encoder_two,
            vae=vae,
            torch_dtype=weight_dtype,
        )
        pipeline.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torchvision import transforms
from diffusers import UNet2DConditionModel, DDPMScheduler
from typing import Dict, Optional


# ─── CONFIGURATION ──────────────────────────────────────────────────────────
# Shared config file for model architecture & scheduler:
CONFIG_YAML_PATH = "configuration_hyper.yaml"

# Where all outputs go; each run gets a subfolder under here:
OUTPUT_DIR = "./generatedSGDLinear"

# How many images per class (same for every model):
NUM_IMAGES_PER_CLASS = 20

# Random seed base (each run will offset by model index):
BASE_SEED = 42

# Device to use:
DEVICE_STR = "cuda"    # or "cpu"


# Per class guidance to ensure individual handling
PER_CLASS_GUIDANCE: Dict[int, float] = {
    0: 5.0,
    1: 3.0,
    2: 3.5,  
    3: 4.0,
    4: 7.0,
}

# List your models here. Each entry needs:
#  • name          — a short label for the folder
#  • checkpoint    — path to .pth with model_state_dict & label_emb_state_dict
#  • guidance_scale— CFG scale to use for that run
MODEL_RUNS = [
    {
        "name": "ModelSGD1",
        "checkpoint": "TraniOutputs/diffusion_model-2025-08-28-19-47/ckpt_epoch_999.pt",
        "guidance_scale": 0,
    },
        {
        "name": "ModelSGD2",
        "checkpoint": "TraniOutputs/diffusion_model-2025-08-28-19-47/ckpt_epoch_949.pt",
        "guidance_scale": 0,
    },
            {
        "name": "ModelSGD3",
        "checkpoint": "TraniOutputs/diffusion_model-2025-08-28-19-47/ckpt_epoch_899.pt",
        "guidance_scale": 0,
    },
                {
        "name": "ModelSGD4",
        "checkpoint": "TraniOutputs/diffusion_model-2025-08-28-19-47/ckpt_epoch_799.pt",
        "guidance_scale": 0,
    },
    # {
    #     "name": "ModelWeighted",
    #     "checkpoint": "./TraniOutputs/diffusion_model-2025-08-04-08-37/ckpt_epoch_999.pt",
    #     "guidance_scale": 3,
    # },
    # add more entries here as needed
]
# ─────────────────────────────────────────────────────────────────────────────

def _get_uncond_emb_like(emb: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(emb, device=emb.device)

@torch.no_grad()
def generate_for_run(
    model: nn.Module,
    label_emb: nn.Embedding,
    image_size: int,
    noise_scheduler: DDPMScheduler,
    out_folder: str,
    device: torch.device,
    num_classes: int,
    num_images_per_class: int,
    guidance_scale: float,
    seed_offset: int,
    per_class_guidance: Optional[Dict[int, float]] = None,
):
    model.eval()
    label_emb.eval()

    to_pil = transforms.ToPILImage()
    uncond_emb = None
    
    per_class_guidance = per_class_guidance or {}

    for class_idx in range(num_classes):
        class_guidance = per_class_guidance.get(class_idx, guidance_scale)
        class_dir = os.path.join(out_folder, f"class_{class_idx}")
        os.makedirs(class_dir, exist_ok=True)

        cond_lbl = torch.tensor([class_idx], device=device)
        cond_emb = label_emb(cond_lbl).unsqueeze(1)

        if uncond_emb is None:
            uncond_emb = _get_uncond_emb_like(cond_emb)

        for img_idx in range(num_images_per_class):
            seed = seed_offset + class_idx * num_images_per_class + img_idx
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)

            x = torch.randn((1, 3, image_size, image_size), device=device)
               
            noise_scheduler.set_timesteps(noise_scheduler.config.num_train_timesteps, device=device)
            for t in noise_scheduler.timesteps:
                t_tensor = t  
                with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                    noise_c   = model(x, t_tensor, encoder_hidden_states=cond_emb).sample
                    noise_u = model(x, t_tensor, encoder_hidden_states=uncond_emb).sample
                    noise_pred = noise_u + class_guidance * (noise_c - noise_u)
                x = noise_scheduler.step(noise_pred, t, x).prev_sample

            img = (x.clamp(-1, 1) + 1) / 2.0
            pil = to_pil(img.squeeze(0).cpu())
            pil.save(os.path.join(class_dir, f"seed_{seed:04d}.png"))

def main():
    # 1) Read shared YAML
    with open(CONFIG_YAML_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    image_size = cfg.get("general", {})["image_size"]

    device = torch.device(DEVICE_STR)

    for run_idx, run_cfg in enumerate(MODEL_RUNS):
        name           = run_cfg["name"]
        ckpt_path      = run_cfg["checkpoint"]
        guidance_scale = run_cfg["guidance_scale"]

        print(f"\n=== Running '{name}' (guidance_scale={guidance_scale}) ===")

        # 2) Build fresh model & scheduler
        model = UNet2DConditionModel(
            sample_size         = model_cfg["sample_size"],
            in_channels         = model_cfg["in_channels"],
            out_channels        = model_cfg["out_channels"],
            layers_per_block    = model_cfg["layers_per_block"],
            block_out_channels  = model_cfg["block_out_channels"],
            down_block_types    = model_cfg["down_block_types"],
            up_block_types      = model_cfg["up_block_types"],
            norm_num_groups     = model_cfg["norm_num_groups"],
            dropout             = model_cfg["dropout_scale"],
            cross_attention_dim = model_cfg["cross_attention_dim"],
        ).to(device)

        noise_scheduler = DDPMScheduler(
            num_train_timesteps = train_cfg["diff_time_step"],
            beta_schedule       = train_cfg["beta_scheduler"],
        )

        # 3) Load checkpoint
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

        w = ckpt["label_emb_state_dict"]["weight"]
        num_classes, emb_dim = w.shape
        label_emb = nn.Embedding(num_classes, emb_dim).to(device)
        label_emb.load_state_dict(ckpt["label_emb_state_dict"])

        # 4) Generate images
        out_folder = os.path.join(OUTPUT_DIR, name)
        os.makedirs(out_folder, exist_ok=True)

        seed_offset = BASE_SEED + run_idx * (num_classes * NUM_IMAGES_PER_CLASS)
        generate_for_run(
            model=model,
            label_emb=label_emb,
            image_size=image_size,
            noise_scheduler=noise_scheduler,
            out_folder=out_folder,
            device=device,
            num_classes=num_classes,
            num_images_per_class=NUM_IMAGES_PER_CLASS,
            guidance_scale=guidance_scale,
            seed_offset=seed_offset,
            per_class_guidance=PER_CLASS_GUIDANCE
        )

        print(f"Saved run '{name}' into {out_folder}")

if __name__ == "__main__":
    main()

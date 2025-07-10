"""Classifier‑Free Guidance (CFG) DDPM
-------------------------------------------------
# For simplicity of transfering and running the code it is in the same file
# main cause been working on remote server and therefore easy handling localy (crtl + c/ctrl + v)
"""

import os, random
import shutil
import glob
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import time
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import UNet2DConditionModel, DDPMScheduler
from tqdm import tqdm
from torchvision.utils import make_grid
from pytorch_fid import fid_score
import datetime
import yaml
from accelerate import Accelerator
from accelerate.utils import set_seed 

def seed_everything(seed: int = 42):

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)           

    # cuDNN / convolution determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    set_seed(seed, device_specific=True)       
    
# -----------------------------------------------------------------------------
# Helper – returns 0‑vector embedding for the unconditional path
# -----------------------------------------------------------------------------

def _get_uncond_emb_like(emb: torch.Tensor) -> torch.Tensor:
    """Return an embedding of zeros with the same shape as *emb* (no gradient)."""
    return torch.zeros_like(emb, device=emb.device)


def main():
    torch.manual_seed(420)
    seed = 420
    seed_everything(seed)   
    
    accelerator = Accelerator()

    # ------------------------------------------------------------
    # Load configuration (with sensible fallbacks for the new keys)
    # ------------------------------------------------------------
    with open("configuration_hyper.yaml", "r") as f:
        config = yaml.safe_load(f)

    model_config = config["model"]
    training_config = config["training"]
    general_config = config["general"]

    # probability of dropping condition during training
    drop_cond_prob = training_config.get("drop_cond_prob", 0.1)
    # guidance scale used at inference
    guidance_scale = general_config.get("guidance_scale", 5.0)

    # ------------------------------------------------------------
    # Dataset & label paths 
    # ------------------------------------------------------------
    train_dir = "./DDR/small_train"
    labels_file = "./DDR/small_train.txt"

    # ------------------------------------------------------------
    # Output bookkeeping
    # ------------------------------------------------------------
    torch.backends.cudnn.benchmark = True
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%d-%m-%H-%M")

    folder_path = f"./diffusion_model-{formatted_time}"
    dst_folder = "./dataresized"  # real images for FID

    save_model_path = f"{folder_path}/diffusion_model_last.pth"
    save_model_path_best = f"{folder_path}/diffusion_model_best.pth"

    yaml_file_src = "configuration_hyper.yaml"
    yaml_file_dst = os.path.join(folder_path, "current_configuration_hyper.yaml")

    # ------------------------------------------------------------
    # Hyper‑parameters
    # ------------------------------------------------------------
    image_size = general_config["image_size"]
    batch_size = training_config["batch_size"]
    num_epochs = 1
    learning_rate = training_config["learning_rate"]

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        shutil.copy2(yaml_file_src, yaml_file_dst)

    # ------------------------------------------------------------
    # Parse labels file
    # ------------------------------------------------------------
    label_dict = {}
    unique_labels = set()
    with open(labels_file, "r") as lf:
        for line in lf:
            line = line.strip()
            if not line:
                continue
            filename, lbl = line.split()
            lbl = int(lbl)
            label_dict[filename] = lbl
            unique_labels.add(lbl)
    num_classes = len(unique_labels)

    # ------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------
    class ImageFolderWithLabels(Dataset):
        def __init__(self, folder, label_dict, image_size=256):
            exts = ["*.jpg", "*.jpeg", "*.png"]
            self.paths = []
            for ext in exts:
                self.paths.extend(glob.glob(os.path.join(folder, ext)))
            self.label_dict = label_dict
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        def __len__(self):
            return len(self.paths)
        def __getitem__(self, idx):
            path = self.paths[idx]
            filename = os.path.basename(path)
            img = Image.open(path).convert("RGB")
            img = self.transform(img)
            if filename not in self.label_dict:
                raise ValueError(f"Label not found for {filename}. Check your labels.txt.")
            label = torch.tensor(self.label_dict[filename], dtype=torch.long)
            return img, label

    train_dataset = ImageFolderWithLabels(train_dir, label_dict, image_size=image_size)
    if len(train_dataset) == 0:
        raise ValueError(f"No images found in {train_dir}. Please place at least one image in the directory.")

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=True)

    # ------------------------------------------------------------
    # Model & noise scheduler
    # ------------------------------------------------------------
    model = UNet2DConditionModel(
        sample_size=model_config["sample_size"],
        in_channels=model_config["in_channels"],
        out_channels=model_config["out_channels"],
        layers_per_block=model_config["layers_per_block"],
        block_out_channels=model_config["block_out_channels"],
        down_block_types=model_config["down_block_types"],
        up_block_types=model_config["up_block_types"],
        norm_num_groups=model_config["norm_num_groups"],
        dropout=model_config["dropout_scale"],
        cross_attention_dim=model_config["cross_attention_dim"]
    )

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=training_config["diff_time_step"],
        beta_schedule=training_config["beta_scheduler"],
    )

    # ------------------------------------------------------------
    # Class embedding (conditioning)
    # ------------------------------------------------------------
    emb_dim = model_config["cross_attention_dim"]
    label_emb = nn.Embedding(num_classes, emb_dim)

    # ------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(label_emb.parameters()),
        lr=learning_rate
    )
    total_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in label_emb.parameters())
    print(f"Total parameters (model + label_emb): {total_params}")

    with open(f"{folder_path}/epoch_times.txt", "w") as f:
        f.write(f"Trainable Param: {total_params}\n")

    # Prepare with accelerator
    model, label_emb, optimizer, train_loader = accelerator.prepare(
        model, label_emb, optimizer, train_loader
    )

    # ------------------------------------------------------------
    # Training loop 
    # ------------------------------------------------------------
    epoch_times, epoch_loss, fid_scores = [], [], []
    best_FID = 1e8

    for epoch in range(num_epochs):
        epoch_start = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=not accelerator.is_local_main_process)
        model.train()

        for batch_imgs, batch_lbls in pbar:
            # Sample random timesteps for each image
            timesteps = torch.randint(0,
                                      noise_scheduler.config.num_train_timesteps,
                                      (batch_imgs.size(0),),
                                      device=batch_imgs.device,
                                      dtype=torch.long)
            noise = torch.randn_like(batch_imgs)
            noisy_imgs = noise_scheduler.add_noise(batch_imgs, noise, timesteps)

            # --------------------------------------------------
            # Classifier‑free guidance trick during *training*
            # Randomly drop conditioning with probability drop_cond_prob
            # --------------------------------------------------
            keep_mask = (torch.rand(batch_lbls.size(0), device=batch_lbls.device) > drop_cond_prob)
            emb_cond = label_emb(batch_lbls)
            emb_uncond = _get_uncond_emb_like(emb_cond)
            # Choose conditional or unconditional embedding per sample
            emb = torch.where(keep_mask.unsqueeze(-1), emb_cond, emb_uncond).unsqueeze(1)  # (B,1,D)

            optimizer.zero_grad(set_to_none=True)
            with accelerator.autocast():
                noise_pred = model(noisy_imgs, timesteps, encoder_hidden_states=emb).sample
                loss = nn.SmoothL1Loss()(noise_pred, noise)
            accelerator.backward(loss)
            optimizer.step()

            pbar.set_postfix({"loss": loss.item()})

        epoch_times.append(time.time() - epoch_start)
        epoch_loss.append(loss.item())

        # --------------------------------------------------
        # Periodic image generation + FID every 100 epochs
        # --------------------------------------------------
        if (epoch + 1) % 100 == 0:
            epoch_folder = generate_image(
                model, label_emb, image_size, noise_scheduler,
                folder_path, epoch, accelerator.device,
                num_classes=num_classes,
                guidance_scale=guidance_scale
            )
            fid_value = calculateFID(epoch_folder, dst_folder, device=batch_imgs.device)
            fid_scores.append(fid_value)

            # Save checkpoints
            if accelerator.is_main_process:
                torch.save({
                    'model_state_dict': accelerator.unwrap_model(model).state_dict(),
                    'label_emb_state_dict': accelerator.unwrap_model(label_emb).state_dict()
                }, save_model_path)
                # Best FID
                if fid_value < best_FID:
                    best_FID = fid_value
                    torch.save({
                        'model_state_dict': accelerator.unwrap_model(model).state_dict(),
                        'label_emb_state_dict': accelerator.unwrap_model(label_emb).state_dict()
                    }, save_model_path_best)
        else:
            fid_scores.append(0)

        # Logging
        if accelerator.is_main_process:
            with open(f"{folder_path}/epoch_times.txt", "a") as f:
                f.write(f"Epoch {epoch+1}: {epoch_times[-1]:.2f} sec, {epoch_loss[-1]:.6f} loss, {fid_scores[-1]:.2f} FID\n")

    # Final save
    if accelerator.is_main_process:
        torch.save({
            'model_state_dict': accelerator.unwrap_model(model).state_dict(),
            'label_emb_state_dict': accelerator.unwrap_model(label_emb).state_dict()
        }, save_model_path)

# -----------------------------------------------------------------------------
# GENERATION with classifier‑free guidance at inference time
# -----------------------------------------------------------------------------
@torch.no_grad()
def generate_image(model,
                   label_emb,
                   image_size,
                   noise_scheduler,
                   folder_path,
                   epoch,
                   device,
                   num_classes=5,
                   seeds_per_class=None,
                   guidance_scale: float = 5.0):
    """Generate *numOfImages* per class using CFG with *guidance_scale*."""
    numOfImages = 2  # images per class
    model.eval()
    epoch_folder = os.path.join(folder_path, f"epoch_{epoch+1}")
    os.makedirs(epoch_folder, exist_ok=True)

    if seeds_per_class is None:
        seeds_per_class = [[c*numOfImages + i for i in range(numOfImages)] for c in range(num_classes)]

    emb_dim = label_emb.embedding_dim
    uncond_emb_template = torch.zeros((1, emb_dim), device=device)

    for class_idx in range(num_classes):
        for img_idx, seed in enumerate(seeds_per_class[class_idx]):
            torch.manual_seed(seed)
            if device.type == 'cuda':
                torch.cuda.manual_seed_all(seed)

            x = torch.randn((1, 3, image_size, image_size), device=device)
            cond_labels = torch.tensor([class_idx], device=device, dtype=torch.long)

            cond_emb = label_emb(cond_labels)           # (1, D)
            uncond_emb = uncond_emb_template.clone()    # (1, D)
            cond_emb = cond_emb.unsqueeze(1)            # (1,1,D)
            uncond_emb = uncond_emb.unsqueeze(1)        # (1,1,D)

            # DDPM reverse process
            for t in reversed(range(noise_scheduler.config.num_train_timesteps)):
                t_tensor = torch.tensor([t], device=device, dtype=torch.long)
                with torch.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                    noise_cond = model(x, t_tensor, encoder_hidden_states=cond_emb).sample
                    noise_uncond = model(x, t_tensor, encoder_hidden_states=uncond_emb).sample
                    # CFG equation
                    noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
                x = noise_scheduler.step(noise_pred, t, x).prev_sample

            x = (x.clamp(-1, 1) + 1) / 2.0  # to [0,1]
            img_pil = transforms.ToPILImage()(x.squeeze(0).cpu())
            img_name = f"gen_class_{class_idx}_img_{img_idx}_seed_{seed}_epoch_{epoch+1}.png"
            img_pil.save(os.path.join(epoch_folder, img_name))

    return epoch_folder

# -----------------------------------------------------------------------------
# FID utility (unchanged)
# -----------------------------------------------------------------------------
def calculateFID(generated_images_folder, dataset_images_folder, device, batch_size=2):
    return fid_score.calculate_fid_given_paths(
        [generated_images_folder, dataset_images_folder],
        batch_size=batch_size,
        device=device,
        dims=2048,
    )


if __name__ == "__main__":
    main()

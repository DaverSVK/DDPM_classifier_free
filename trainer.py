

import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm

from generator import generate_image
from fid_calculator import calculateFID

def _get_uncond_emb_like(emb: torch.Tensor) -> torch.Tensor:
    """Return an embedding of zeros with the same shape as *emb* (no gradient)."""
    return torch.zeros_like(emb, device=emb.device)

def train_model(
    model,
    label_emb,
    train_loader,
    optimizer,
    noise_scheduler,
    accelerator,
    num_epochs,
    drop_cond_prob,
    folder_path,
    image_size,
    num_classes,
    guidance_scale
):
    epoch_times, epoch_loss, fid_scores = [], [], []
    best_FID = 1e8

    for epoch in range(num_epochs):
        epoch_start = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=not accelerator.is_local_main_process)
        model.train()

        for batch_imgs, batch_lbls in pbar:
            timesteps = torch.randint(0,
                                      noise_scheduler.config.num_train_timesteps,
                                      (batch_imgs.size(0),),
                                      device=batch_imgs.device,
                                      dtype=torch.long)
            noise = torch.randn_like(batch_imgs)
            noisy_imgs = noise_scheduler.add_noise(batch_imgs, noise, timesteps)

            keep_mask = (torch.rand(batch_lbls.size(0), device=batch_lbls.device) > drop_cond_prob)
            emb_cond = label_emb(batch_lbls)
            emb_uncond = _get_uncond_emb_like(emb_cond)
            emb = torch.where(keep_mask.unsqueeze(-1), emb_cond, emb_uncond).unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)
            with accelerator.autocast():
                noise_pred = model(noisy_imgs, timesteps, encoder_hidden_states=emb).sample
                loss = nn.SmoothL1Loss()(noise_pred, noise)
            accelerator.backward(loss)
            optimizer.step()

            pbar.set_postfix({"loss": loss.item()})

        epoch_times.append(time.time() - epoch_start)
        epoch_loss.append(loss.item())

        if (epoch + 1) % 100 == 0:
            save_model_path_per = f"{folder_path}/ckpt_epoch_{epoch:03d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": accelerator.unwrap_model(model).state_dict(),
                    "label_emb_state_dict": accelerator.unwrap_model(label_emb).state_dict(),
                },
                save_model_path_per,
            )
            epoch_folder = generate_image(
                model, label_emb, image_size, noise_scheduler,
                folder_path, epoch, accelerator.device,
                num_classes=num_classes,
                guidance_scale=guidance_scale
            )
            fid_value = calculateFID(epoch_folder, "./dataresized", device=accelerator.device)
            fid_scores.append(fid_value)

            if accelerator.is_main_process:
                if fid_value < best_FID:
                    best_FID = fid_value
                    torch.save({
                        'model_state_dict': accelerator.unwrap_model(model).state_dict(),
                        'label_emb_state_dict': accelerator.unwrap_model(label_emb).state_dict()
                    }, f"{folder_path}/diffusion_model_best.pth")
        else:
            fid_scores.append(0)

        if accelerator.is_main_process:
            with open(f"{folder_path}/epoch_times.txt", "a") as f:
                f.write(f"Epoch {epoch+1}: {epoch_times[-1]:.2f} sec, {epoch_loss[-1]:.6f} loss, {fid_scores[-1]:.2f} FID\n")

    if accelerator.is_main_process:
        torch.save({
            'model_state_dict': accelerator.unwrap_model(model).state_dict(),
            'label_emb_state_dict': accelerator.unwrap_model(label_emb).state_dict()
        }, f"{folder_path}/diffusion_model_last.pth")


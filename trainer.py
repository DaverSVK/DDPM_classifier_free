import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from generator import generate_image
from fid_calculator import calculateFID

def _get_uncond_emb_like(emb: torch.Tensor) -> torch.Tensor:
    """Return an embedding of zeros with the same shape as *emb* (no gradient)."""
    return torch.zeros_like(emb, device=emb.device)

def train_model(
    model,
    train_loader,
    optimizer,
    noise_scheduler,
    accelerator,
    num_epochs,
    drop_cond_prob,
    folder_path,
    image_size,
    num_classes,
    guidance_scale,
    lr_scheduler=None,
    start_epoch=0,
    plateau_metric="train_loss",
    use_null_class=True,  # set True if your model was built with num_class_embeds = num_classes + 1
):
    device = accelerator.device
    null_id = num_classes if use_null_class else None

    epoch_times, epoch_loss, fid_scores = [], [], []
    best_FID = float("inf")
    global_step = 0

    # define loss once (slightly faster than instantiating every step)
    loss_fn = nn.SmoothL1Loss()
    # reduce UI overhead: update tqdm every N steps
    log_every = 20

    for epoch in range(start_epoch, num_epochs):
        # timing start
        epoch_start = time.time()

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{num_epochs}",
            disable=not accelerator.is_local_main_process
        )
        model.train()

        running_loss, num_batches = 0.0, 0

        for step, (batch_imgs, batch_lbls) in enumerate(pbar):
            # ---- Fast host→device path + better conv layout ----
            batch_imgs = (
                batch_imgs
                .contiguous(memory_format=torch.channels_last)
                .to(device, non_blocking=True)
            )
            labels = batch_lbls.to(device, non_blocking=True, dtype=torch.long)

            # ---- Work that doesn't need autograd ----
            with torch.no_grad():
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps,
                    (batch_imgs.size(0),), device=device, dtype=torch.long
                )
                noise = torch.randn_like(batch_imgs)  # GPU RNG
                noisy_imgs = noise_scheduler.add_noise(batch_imgs, noise, timesteps)

                if use_null_class and drop_cond_prob > 0.0 and null_id is not None:
                    drop_mask = torch.rand(labels.size(0), device=device) < drop_cond_prob
                    # vectorized replacement of some labels with null_id
                    labels = labels.masked_fill(drop_mask, null_id)

            optimizer.zero_grad(set_to_none=True)

            # ---- Forward + loss under mixed precision ----
            with accelerator.autocast():
                noise_pred = model(noisy_imgs, timesteps, class_labels=labels).sample
                loss = loss_fn(noise_pred, noise)

            accelerator.backward(loss)
            optimizer.step()

            # ---- Logging (lightweight) ----
            running_loss += loss.detach().float().item()
            num_batches += 1
            if (step + 1) % log_every == 0:
                pbar.set_postfix({"loss": f"{running_loss / num_batches:.6f}"})

            global_step += 1

        # ensure all CUDA work is done before timing
        accelerator.wait_for_everyone()
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)

        epoch_time = time.time() - epoch_start
        epoch_avg_loss = running_loss / max(1, num_batches)
        epoch_times.append(epoch_time)
        epoch_loss.append(epoch_avg_loss)

        # optional LR scheduler on val/train metric
        if lr_scheduler is not None:
            lr_scheduler.step(epoch_avg_loss)

        current_lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else float("nan")

        # periodic checkpointing / (optional) FID
        if (epoch + 1) % 50 == 0:
            save_model_path_per = f"{folder_path}/ckpt_epoch_{epoch:03d}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": accelerator.unwrap_model(model).state_dict(),
                },
                save_model_path_per,
            )
            # If/when you re-enable FID, keep it outside autocast and eval() the generator.
            # epoch_folder = generate_image(
            #     model, label_emb, image_size, noise_scheduler,
            #     folder_path, epoch, accelerator.device,
            #     num_classes=num_classes,
            #     guidance_scale=guidance_scale
            # )
            # fid_value = calculateFID(epoch_folder, "./dataresized", device=accelerator.device)
            # fid_scores.append(fid_value)
            # if accelerator.is_main_process and fid_value < best_FID:
            #     best_FID = fid_value
            #     torch.save(
            #         {
            #             'model_state_dict': accelerator.unwrap_model(model).state_dict(),
            #             'label_emb_state_dict': accelerator.unwrap_model(label_emb).state_dict()
            #         },
            #         f"{folder_path}/diffusion_model_best.pth"
            #     )
        else:
            fid_scores.append(0)

        if accelerator.is_main_process:
            with open(f"{folder_path}/epoch_times.txt", "a") as f:
                f.write(
                    f"Epoch {epoch+1}: {epoch_times[-1]:.2f} sec, "
                    f"{epoch_loss[-1]:.6f} loss, {fid_scores[-1]:.2f} FID, {current_lr:.6f} LR\n"
                )

    if accelerator.is_main_process:
        torch.save(
            {"model_state_dict": accelerator.unwrap_model(model).state_dict()},
            f"{folder_path}/diffusion_model_last.pth"
        )


import os
import torch
from torchvision import transforms
from PIL import Image

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

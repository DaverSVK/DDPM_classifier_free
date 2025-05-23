import os
import shutil
import glob
from PIL import Image
import torch
import torch.nn as nn
import time
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from diffusers import UNet2DModel, DDPMScheduler
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from pytorch_fid import fid_score
import datetime
import torchvision.models as models
import yaml
from accelerate import Accelerator

def main():
    accelerator = Accelerator()  # Initialize Accelerator
    # Load configuration here you can change the hyperparameters
    with open("configuration_hyper.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    model_config = config["model"]
    training_config = config["training"]
    general_config = config["general"]
    
    # Training dataset to be chosen
    train_dir = "./IDRID_resize/train"
    # train_dir = "./DDR/test"
    
    ############################################
    # Saving path
    ############################################ 
    
    torch.backends.cudnn.benchmark = False
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%d-%m-%H-%M")
    
    # Output folder
    folder_path = f"./diffusion_model-{formatted_time}"
    dst_folder = "./dataresized"
    
    # Model paths
    save_model_path = f"{folder_path}/diffusion_model_last.pth"
    save_model_path_best = f"{folder_path}/diffusion_model_best.pth"
    
    # Copy the configuration file for history
    yaml_file_src = "configuration_hyper.yaml"
    yaml_file_dst = os.path.join(folder_path, "current_configuration_hyper.yaml")
    
    
    ############################################
    # Configuration
    ############################################
    
    image_size = general_config["image_size"]
    batch_size = training_config["batch_size"]
    num_epochs = training_config["epochs"]
    learning_rate = training_config["learning_rate"]
    

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        shutil.copy2(yaml_file_src, yaml_file_dst)

    ############################################
    # Dataset
    ############################################
    class ImageFolderDataset(Dataset):
        def __init__(self, folder, image_size=256):
            exts = ["*.jpg", "*.jpeg", "*.png"]
            self.paths = []
            for ext in exts:
                self.paths.extend(glob.glob(os.path.join(folder, ext)))
            
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        
        def __len__(self):
            return len(self.paths)
        
        def __getitem__(self, idx):
            path = self.paths[idx]
            img = Image.open(path)
            img = self.transform(img)
            return img

    train_dataset = ImageFolderDataset(train_dir, image_size=image_size)
    if len(train_dataset) == 0:
        raise ValueError(f"No images found in {train_dir}. Please place at least one image in the directory.")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    ############################################
    # Model & Scheduler
    ############################################
    model = UNet2DModel(
        sample_size=model_config["sample_size"],
        in_channels=model_config["in_channels"],
        out_channels=model_config["out_channels"],
        layers_per_block=model_config["layers_per_block"],
        block_out_channels=model_config["block_out_channels"],
        down_block_types=model_config["down_block_types"],
        # mid_block_type=model_config["mid_block_type"], # Hardcoded in library
        up_block_types=model_config["up_block_types"],
        norm_num_groups=model_config["norm_num_groups"],
        dropout=model_config["dropout_scale"],
    )

    if os.path.exists("./diffusion_model-2025-03-05-23-15/diffusion_model_last.pth"):
        print("Loading model from diffusion_model_best.pth...")
        model.load_state_dict(torch.load("./diffusion_model-2025-03-05-23-15/diffusion_model_best.pth", map_location="cpu"))
        
    
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=training_config["diff_time_step"],
        beta_schedule="squaredcos_cap_v2"
    )

    ############################################
    # Optimizer & AMP
    ############################################
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
        
    with open(f"{folder_path}/epoch_times.txt", "w") as f:
        f.write(f"Trainable Param: {total_params}\n")

    # Prepare model, optimizer, and dataloader with Accelerator
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    
    ############################################
    # Training Loop
    ############################################
    epoch_times = []
    epoch_loss = []
    fid_scores = []
    best_FID = 10000

    for epoch in range(num_epochs):
        
        try:
            running_loss = 0.0                    
            num_batches  = 0 
            epoch_start_time = time.time() 
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}") 
            model.train()
            for batch in pbar:
                
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch.shape[0],), device=batch.device, dtype=torch.long)
                noise = torch.randn_like(batch)
                noisy_images = noise_scheduler.add_noise(batch, noise, timesteps)

                optimizer.zero_grad(set_to_none=True)
                
                with accelerator.autocast():
                    noise_pred = model(noisy_images, timesteps).sample
                    # loss = nn.L1Loss()(noise_pred, noise)
                    # loss = nn.MSELoss()(noise_pred, noise)
                    loss = nn.SmoothL1Loss()(noise_pred, noise)


                accelerator.backward(loss)
                optimizer.step()
                loss_scalar = accelerator.gather(loss.detach()).mean().item()
                running_loss += loss_scalar
                num_batches  += 1
                # optimizer.zero_grad(set_to_none=True)
                pbar.set_postfix({"loss": loss.item()})

            torch.cuda.empty_cache()          
            accelerator.wait_for_everyone()  
            avg_loss = running_loss / num_batches        
            epoch_loss.append(avg_loss)    
            epoch_end_time = time.time()
            epoch_duration = epoch_end_time - epoch_start_time
            epoch_times.append(epoch_duration)

            if epoch % 50 == 0:
                epoch_folder = generate_image(model, image_size, noise_scheduler, folder_path, epoch, accelerator)
                fid_value = calculateFID(epoch_folder, dst_folder, device=batch.device)
                fid_scores.append(fid_value)
                # Save the model 
                if accelerator.is_main_process:
                    torch.save(accelerator.unwrap_model(model).state_dict(), save_model_path)
                if fid_value < best_FID:
                    if accelerator.is_main_process:
                        torch.save(accelerator.unwrap_model(model).state_dict(), save_model_path_best)
                    best_FID = fid_value
            else:
                fid_scores.append(0)
            if accelerator.is_main_process:
                with open(f"{folder_path}/epoch_times.txt", "a") as f:
                    f.write(f"Epoch {epoch}: {epoch_duration:.2f} seconds, {epoch_loss[epoch]} loss, {fid_scores[epoch]} FID\n")
        except:
            print("Wooopsie")
    if accelerator.is_main_process:
        torch.save(accelerator.unwrap_model(model).state_dict(), save_model_path)

############################################
# Generation (Inference)
############################################
def generate_image(model, image_size, noise_scheduler, folder_path, epoch, accelerator):
    model.eval()
    epoch_folder = os.path.join(folder_path, f"epoch_{epoch+1}")
    os.makedirs(epoch_folder, exist_ok=True)
    
    @torch.no_grad()
    def generate_images(num_images=10):
        x = torch.randn((num_images, 3, image_size, image_size), device=accelerator.device)
        for t in reversed(range(noise_scheduler.config.num_train_timesteps)):
            t_tensor = torch.tensor([t] * num_images, device=accelerator.device, dtype=torch.long)
            with accelerator.autocast():
                noise_pred = model(x, t_tensor).sample
            x = noise_scheduler.step(noise_pred, t, x).prev_sample
        x = (x.clamp(-1, 1) + 1) / 2
        return x

    generated = generate_images(num_images=10)
    for i, img_tensor in enumerate(generated):
        img_pil = transforms.ToPILImage()(img_tensor.cpu())
        img_pil.save(os.path.join(epoch_folder, f"generated_{i}_epoch_{epoch}.png"))
    return epoch_folder

############################################
# FID calculation
############################################           
def calculateFID(generated_images_folder, dataset_images_folder, device, batch_size=2):
    fid_value = fid_score.calculate_fid_given_paths(
        [generated_images_folder, dataset_images_folder],
        batch_size=batch_size,
        device=device,
        dims=2048
    )
    return fid_value
    
if __name__ == '__main__':
    main()
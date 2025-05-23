# For simplicity of transfering and running the code it is in the same file
# main cause been working on remote server and therefore easy handling localy (crtl + c/ctrl + v)

import os
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
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from pytorch_fid import fid_score
import datetime
import yaml
from accelerate import Accelerator

def main():
    torch.manual_seed(420)
    accelerator = Accelerator()  # Initialize Accelerator for later use
    
    # Load configuration
    with open("configuration_hyper.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    model_config = config["model"]
    training_config = config["training"]
    general_config = config["general"]
    
    # Dataset selection
    # train_dir = "./IDRID_resize/train"
    train_dir = "./DDR/filtered_procesed_train"
    
    # Path to labels.txt
    labels_file = "./DDR/filtered_train.txt"
    
    ############################################
    # Saving path
    ############################################
    torch.backends.cudnn.benchmark = True
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
    # Parse labels file
    ############################################
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

    
    ############################################
    # Dataset loading
    ############################################
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
            
            label = self.label_dict[filename]
            label = torch.tensor(label, dtype=torch.long)
            
            return img, label

    train_dataset = ImageFolderWithLabels(train_dir, label_dict, image_size=image_size)
    if len(train_dataset) == 0:
        raise ValueError(f"No images found in {train_dir}. Please place at least one image in the directory.")
    
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=True)
    
    ############################################
    # Data load check
    ############################################
    
    # def show_images_from_loader(loader, num_batches=10):
    #     for i, (images, labels) in enumerate(loader):
    #         if i >= num_batches:
    #             break
    #         images = images * 0.5 + 0.5  
    #         # img_array = np.array(images)

    #         # # Split the channels
    #         # r_channel = img_array[:,:,0]
    #         # g_channel = img_array[:,:,1]
    #         # b_channel = img_array[:,:,2]

    #         # # Print the maximum value for each channel
    #         # print("Max R:", r_channel.max())
    #         # print("Max G:", g_channel.max())
    #         # print("Max B:", b_channel.max())
            
    #         # print("Mix R:", r_channel.min())
    #         # print("Max G:", g_channel.min())
    #         # print("Max B:", b_channel.min())

    #         grid_img = make_grid(images, nrow=4)
    #         plt.figure(figsize=(12, 6))
    #         plt.imshow(grid_img.permute(1, 2, 0))
    #         plt.title(f"Batch {i+1} - Labels: {labels.tolist()}")
    #         plt.axis('off')
    #         plt.show()

    # show_images_from_loader(train_loader, num_batches=10)
    
    
    ############################################
    # Model & Scheduler
    ############################################
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

    ############################################
    # Class Embedding for Conditioning
    ############################################
    label_emb = nn.Embedding(num_classes, model_config["cross_attention_dim"])

    ############################################
    # Optimizer & AMP
    ############################################
    optimizer = torch.optim.Adam(list(model.parameters()) + list(label_emb.parameters()), 
                                 lr=learning_rate)
    total_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in label_emb.parameters())
    print(f"Total parameters (model + label_emb): {total_params}")
        
    with open(f"{folder_path}/epoch_times.txt", "w") as f:
        f.write(f"Trainable Param: {total_params}\n")

    model, label_emb, optimizer, train_loader = accelerator.prepare(
        model, label_emb, optimizer, train_loader
    )
    ############################################
    # Loading previously trained model
    ############################################
    
    # checkpoint_path = "./diffusion_model-2025-14-04-14-10/diffusion_model_last.pth"
    # checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # model.load_state_dict(checkpoint['model_state_dict'])
    # label_emb.load_state_dict(checkpoint['label_emb_state_dict'])

    # model.eval()  # Put model in eval mode
    # label_emb.eval()
    
    # epoch_folder = generate_image(
    #     model, label_emb, image_size, noise_scheduler,
    #     folder_path, epoch, accelerator.device,
    #     num_classes
    #     )
    
    
    ############################################
    # Training Loop
    ############################################
    epoch_times = []
    epoch_loss = []
    fid_scores = []
    best_FID = 1e8  # large initial value

    for epoch in range(num_epochs):
        epoch_start_time = time.time() 
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", disable=not accelerator.is_local_main_process)
        
        model.train()
        for batch in pbar:
            batch_imgs, batch_lbls = batch

            # Sample random timesteps
            timesteps = torch.randint(0,
                                      noise_scheduler.config.num_train_timesteps,
                                      (batch_imgs.shape[0],),
                                      device=batch_imgs.device,
                                      dtype=torch.long)
            noise = torch.randn_like(batch_imgs)
            noisy_images = noise_scheduler.add_noise(batch_imgs, noise, timesteps)

            # Get label embeddings (for cross-attention)
            with accelerator.autocast():
                emb = label_emb(batch_lbls)           
                emb = emb.unsqueeze(1)                

            optimizer.zero_grad(set_to_none=True)
            
            with accelerator.autocast():
                noise_pred = model(noisy_images, timesteps, encoder_hidden_states=emb).sample
                loss = nn.SmoothL1Loss()(noise_pred, noise)

            accelerator.backward(loss)
            optimizer.step()

            pbar.set_postfix({"loss": loss.item()})
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_duration)
        epoch_loss.append(loss.item())
        
        if (epoch+1) % 100 == 0:
            epoch_folder = generate_image(
                model, label_emb, image_size, noise_scheduler,
                folder_path, epoch, accelerator.device,
                num_classes
            )
            # FID
            fid_value = calculateFID(epoch_folder, dst_folder, device=batch_imgs.device)
            fid_scores.append(fid_value)

            if accelerator.is_main_process:
                torch.save({
                    'model_state_dict': accelerator.unwrap_model(model).state_dict(),
                    'label_emb_state_dict': accelerator.unwrap_model(label_emb).state_dict()
                }, save_model_path)

            if fid_value < best_FID:
                best_FID = fid_value
                if accelerator.is_main_process:
                    torch.save({
                        'model_state_dict': accelerator.unwrap_model(model).state_dict(),
                        'label_emb_state_dict': accelerator.unwrap_model(label_emb).state_dict()
                    }, save_model_path_best)
        else:
            fid_scores.append(0)
        
        if accelerator.is_main_process:
            with open(f"{folder_path}/epoch_times.txt", "a") as f:
                f.write(f"Epoch {epoch+1}: {epoch_duration:.2f} sec, {epoch_loss[-1]:.6f} loss, {fid_scores[-1]:.2f} FID\n")

    # Final save of the model
    if accelerator.is_main_process:
        torch.save({
            'model_state_dict': accelerator.unwrap_model(model).state_dict(),
            'label_emb_state_dict': accelerator.unwrap_model(label_emb).state_dict()
        }, save_model_path)


############################################
# Generation (Inference)
############################################

@torch.no_grad()
def generate_image(
    model,
    label_emb,
    image_size,
    noise_scheduler,
    folder_path,
    epoch,
    device,
    num_classes=5,
    seeds_per_class=None
):

    numOfImages = 2 #  number of images per class
    model.eval()
    epoch_folder = os.path.join(folder_path, f"epoch_{epoch+1}")
    os.makedirs(epoch_folder, exist_ok=True)

    if seeds_per_class is None:
        seeds_per_class = [
            list(range(c*numOfImages, c*numOfImages+numOfImages)) for c in range(num_classes)
        ]

    if isinstance(seeds_per_class[0], int) and len(seeds_per_class) == num_classes * numOfImages:
        seeds_chunked = [
            seeds_per_class[c*numOfImages:(c+1)*numOfImages] for c in range(num_classes)
        ]
        seeds_per_class = seeds_chunked
    
    
    # -------------------------------------------------------------------------
    # Generating images
    # -------------------------------------------------------------------------
    all_generated = []
    all_labels = []

    for class_idx in range(num_classes):
        for i, seed in enumerate(seeds_per_class[class_idx]):
            torch.manual_seed(seed)
            if device.type == 'cuda':
                torch.cuda.manual_seed_all(seed)

            x = torch.randn((1, 3, image_size, image_size), device=device)

            random_labels = torch.tensor([class_idx], device=device, dtype=torch.long)
            emb = label_emb(random_labels)  #
            emb = emb.unsqueeze(1)         

            for t in reversed(range(noise_scheduler.config.num_train_timesteps)):
                t_tensor = torch.tensor([t], device=device, dtype=torch.long)
                with torch.autocast(device_type='cuda', enabled=(device.type=='cuda')):
                    noise_pred = model(x, t_tensor, encoder_hidden_states=emb).sample
                x = noise_scheduler.step(noise_pred, t, x).prev_sample
                
            x = (x.clamp(-1, 1) + 1) / 2.0

            # Convert tensor to PIL Image
            img_pil = transforms.ToPILImage()(x.squeeze(0).cpu())
    
            img_name = f"gen_class_{class_idx}_img_{i}_seed_{seed}_epoch_{epoch+1}.png"
            img_pil.save(os.path.join(epoch_folder, img_name))

            all_generated.append(x)
            all_labels.append(class_idx)

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

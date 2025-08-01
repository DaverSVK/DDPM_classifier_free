import os
import shutil
import torch
import torch.nn as nn
import datetime
import yaml
from accelerate import Accelerator
from diffusers import UNet2DConditionModel, DDPMScheduler

from data_loader import get_data_loader
from trainer import train_model

def main():
    torch.manual_seed(420)
    accelerator = Accelerator()

    with open("configuration_hyper.yaml", "r") as f:
        config = yaml.safe_load(f)

    model_config = config["model"]
    training_config = config["training"]
    general_config = config["general"]

    drop_cond_prob = training_config.get("drop_cond_prob", 0.7)
    guidance_scale = general_config.get("guidance_scale", 5.0)

    train_dir = "./TestSmall/train"
    labels_file = "./TestSmall/labels.txt"

    torch.backends.cudnn.benchmark = True
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d-%H-%M")

    folder_path = f"./diffusion_model-{formatted_time}"
    
    image_size = general_config["image_size"]
    batch_size = training_config["batch_size"]
    num_epochs = training_config["epochs"]
    learning_rate = training_config["learning_rate"]

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        shutil.copy2("configuration_hyper.yaml", os.path.join(folder_path, "current_configuration_hyper.yaml"))

    train_loader, num_classes = get_data_loader(train_dir, labels_file, image_size, batch_size)

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

    emb_dim = model_config["cross_attention_dim"]
    label_emb = nn.Embedding(num_classes, emb_dim)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(label_emb.parameters()),
        lr=learning_rate
    )
    total_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in label_emb.parameters())
    print(f"Total parameters (model + label_emb): {total_params}")

    with open(f"{folder_path}/epoch_times.txt", "w") as f:
        f.write(f"Trainable Param: {total_params}\n")

    model, label_emb, optimizer, train_loader = accelerator.prepare(
        model, label_emb, optimizer, train_loader
    )
    
    train_model(
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
    )

if __name__ == "__main__":
    main()
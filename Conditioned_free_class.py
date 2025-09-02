import os
import shutil
import torch
import torch.nn as nn
import datetime
import yaml
from accelerate import Accelerator
from diffusers import UNet2DConditionModel, DDPMScheduler

from data_loader import get_data_loader
from collections import OrderedDict
from trainer import train_model
from torch.optim.lr_scheduler import LambdaLR
import math

def _strip_module_prefix(sd):
    # Robust in case anything ever gets saved with "module." keys
    if not any(k.startswith("module.") for k in sd.keys()):
        return sd
    return OrderedDict((k[len("module."):], v) for k, v in sd.items())

def load_simple_checkpoint(checkpoint_path, model, label_emb, strict=True):
    """
    Loads weights saved as:
        { "epoch": int, "model_state_dict": ..., "label_emb_state_dict": ... }
    Returns the next epoch to run (epoch+1) if present, else 0.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # Handle expected keys (and be tolerant if someone used different names)
    model_sd = ckpt.get("model_state_dict") or ckpt.get("model") or ckpt["model_state_dict"]
    label_sd = ckpt.get("label_emb_state_dict") or ckpt.get("label_emb") or ckpt["label_emb_state_dict"]

    model.load_state_dict(_strip_module_prefix(model_sd), strict=strict)
    label_emb.load_state_dict(_strip_module_prefix(label_sd), strict=strict)

    start_epoch = int(ckpt.get("epoch", -1)) + 1
    if start_epoch < 0:
        start_epoch = 0
    return start_epoch
def main():
    torch.manual_seed(420)
    accelerator = Accelerator()

    with open("configuration_hyper.yaml", "r") as f:
        config = yaml.safe_load(f)

    model_config = config["model"]
    training_config = config["training"]
    general_config = config["general"]

    drop_cond_prob = training_config.get("drop_cond_prob", 0.2)
    guidance_scale = general_config.get("guidance_scale", 3.0)
    
    lr_start = float(training_config.get("lr_start", 0.02))
    lr_end = float(training_config.get("lr_end", 0.0001))

    train_dir = "./DDR/filtered_procesed_train"
    labels_file = "./DDR/filtered_train.txt"

    torch.backends.cudnn.benchmark = True
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d-%H-%M")

    folder_path = f"./TraniOutputs/diffusion_model-{formatted_time}"
    
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
    
    start_epoch = 0
    resume_path = 0 #"./TraniOutputs/diffusion_model-2025-08-04-08-37/ckpt_epoch_999.pt"
    if resume_path and os.path.isfile(resume_path):
        start_epoch = load_simple_checkpoint(resume_path, model, label_emb, strict=True)
        print(f"Loaded weights from {resume_path}. Resuming at epoch {start_epoch}.")
    else:
        print("No resume_path provided or file not found; starting fresh.")

    # optimizer = torch.optim.Adam(
    #     list(model.parameters()) + list(label_emb.parameters()),
    #     lr=learning_rate
    # )
    
    optimizer = torch.optim.SGD(
    list(model.parameters()) + list(label_emb.parameters()),
    lr=lr_start,
    momentum=0.9
)
    total_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in label_emb.parameters())
    print(f"Total parameters (model + label_emb): {total_params}")

    with open(f"{folder_path}/epoch_times.txt", "w") as f:
        f.write(f"Trainable Param: {total_params}\n")

    model, label_emb, optimizer, train_loader = accelerator.prepare(
        model, label_emb, optimizer, train_loader
    )
    
    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0:
        raise RuntimeError("DataLoader has zero length; cannot build LR schedule.")
    total_steps = num_epochs * steps_per_epoch
    start_step = start_epoch * steps_per_epoch

    if lr_start <= 0:
        raise ValueError("training.lr_start must be > 0 for LambdaLR to work correctly.")

    def _linear_lr_lambda(current_step: int):
      
        s = min(current_step, total_steps)
        t = s / float(total_steps) if total_steps > 0 else 1.0
        lr_now = lr_start + (lr_end - lr_start) * t

        return lr_now / lr_start

    

    scheduler = LambdaLR(
        optimizer,
        lr_lambda=_linear_lr_lambda,
        last_epoch=start_step - 1  # so that step= start_step gives the right lr
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
        guidance_scale,
        scheduler=scheduler,
        steps_per_epoch=steps_per_epoch
    )

if __name__ == "__main__":
    main()
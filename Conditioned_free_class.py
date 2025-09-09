import os
import shutil
import torch
import torch.nn as nn
import datetime
import yaml
from accelerate import Accelerator
from diffusers import UNet2DModel, DDPMScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import diffusers


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
    accelerator = Accelerator(mixed_precision="fp16") 
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(420)
    accelerator = Accelerator()

    with open("configuration_hyper.yaml", "r") as f:
        config = yaml.safe_load(f)

    model_config = config["model"]
    training_config = config["training"]
    general_config = config["general"]

    drop_cond_prob = training_config.get("drop_cond_prob", 0.2)
    guidance_scale = general_config.get("guidance_scale", 3.0)
    
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
    use_null_class = True             
    num_class_embeds = num_classes + (1 if use_null_class else 0)
    
    model = UNet2DModel(
        sample_size=model_config["sample_size"],      
        in_channels=model_config["in_channels"],      
        out_channels=model_config["out_channels"],    
        layers_per_block=model_config["layers_per_block"],       
        block_out_channels=model_config["block_out_channels"],   
        down_block_types=model_config["down_block_types"],       
        up_block_types=model_config["up_block_types"],           
        norm_num_groups=model_config["norm_num_groups"],         
        dropout=model_config["dropout_scale"],                    
        time_embedding_dim=model_config.get("time_embedding_dim", 512),
        # class_embed_type=model_config.get("class_embed_type", "simple"),
        num_class_embeds=num_class_embeds,
        attention_head_dim=model_config.get("attention_head_dim", 8),
    )
    print("diffusers version:", diffusers.__version__)
    print("num_class_embeds:", getattr(model.config, "num_class_embeds", None))
    print("has class_embedding:", hasattr(model, "class_embedding"))
    assert getattr(model, "class_embedding", None) is not None, "class_embedding was not created"
    model = model.to(memory_format=torch.channels_last)
    
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=training_config["diff_time_step"],   # 1000
        beta_schedule=training_config["beta_scheduler"],         # 'linear'
        beta_start=training_config.get("beta_start", 1e-4),
        beta_end=training_config.get("beta_end", 2e-2),
        prediction_type="epsilon",
    )

    label_emb = None
    start_epoch = 0
    resume_path = 0 #"./TraniOutputs/diffusion_model-2025-08-04-08-37/ckpt_epoch_999.pt"
    if resume_path and os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location="cpu")
        model_sd = ckpt.get("model_state_dict") or ckpt.get("model") or ckpt["model_state_dict"]
        model.load_state_dict(_strip_module_prefix(model_sd), strict=True)
        start_epoch = int(ckpt.get("epoch", -1)) + 1
    else:
        print("No resume_path provided or file not found; starting fresh.")
    
    try:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
            amsgrad=False,
            fused=True,         
            foreach=False        
        )
        print("AdamW fused requested: True")
    except TypeError:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
            amsgrad=False,
            foreach=True
    )
    print("AdamW fused not available; using foreach=True")

    total_params = sum(p.numel() for p in model.parameters())  # <- fixed
    print(f"Total parameters (model): {total_params}")

    with open(f"{folder_path}/epoch_times.txt", "w") as f:
        f.write(f"Trainable Param: {total_params}\n")

    objects = [model, optimizer, train_loader]
    model, optimizer, train_loader = accelerator.prepare(*objects)
    print("=== Global settings Info ===")
    print("Mixed precision:", getattr(accelerator, "mixed_precision", "no"))

    print("AdamW fused:", getattr(optimizer, "fused", False))
    print("Device:", accelerator.device)
    print("Batch size:", batch_size)
    print("Dataloader workers:", train_loader.num_workers)
    print("Pin memory:", getattr(train_loader, "pin_memory", None))
    print("Channels-last:", next(model.parameters()).is_contiguous(memory_format=torch.channels_last))
    
    train_model(
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
        start_epoch=0,
        use_null_class=True,  
)

if __name__ == "__main__":
    main()
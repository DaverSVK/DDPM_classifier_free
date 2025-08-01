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
from torch_fidelity import calculate_metrics

def calculateKID(gen_folder, real_folder, device="cuda", batch_size=50, subset_size=200):
    metrics = calculate_metrics(
        input1=gen_folder,
        input2=real_folder,
        cuda=(device=="cpu"),
        # disable everything except KID
        isc=False, frechet=False, kid=True,
        kid_subset_size=subset_size,
        batch_size=batch_size,
    )
    # torch-fidelity returns a dict with 'KID_Full^2' and 'KID_Subset^2'
    # we usually report the mean subset MMD (not squared)
    # kid_mean = metrics["KID_Subset"]
    # kid_std  = metrics["KID_Subset_std"]
    print("Available metric keys:", metrics.keys())
    return 0

if __name__ == "__main__":
    gen = "./epoch_1/images"
    real = "./DDRResized/images"
    kid_mean, kid_std = calculateKID(gen, real, device="cuda")
    print(f"KID: {kid_mean:.6f} ± {kid_std:.6f}")
Kernel Inception Distance: 0.07466536521911621 ± 0.002420825902367168
# Available metric keys: dict_keys(['kernel_inception_distance_mean', 'kernel_inception_distance_std'])
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
    # Define the folder path containing original images
    dst_folder = "./DDRResized/images"
    # Define the folder path containing generated images
    epoch_folder = "./epoch_1/images"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fid_value = calculateFID(epoch_folder, dst_folder, device=device)
    print("FID value: "+ str(fid_value))     

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
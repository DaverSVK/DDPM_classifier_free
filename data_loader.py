
import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from collections import Counter
from torch.utils.data import WeightedRandomSampler


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
        label = torch.tensor(self.label_dict[filename], dtype=torch.long)
        return img, label

def get_data_loader(train_dir, labels_file, image_size, batch_size):
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

    train_dataset = ImageFolderWithLabels(train_dir, label_dict, image_size=image_size)
    if len(train_dataset) == 0:
        raise ValueError(f"No images found in {train_dir}. Please place at least one image in the directory.")

    # train_loader = DataLoader(train_dataset,
    #                           batch_size=batch_size,
    #                           shuffle=True,
    #                           num_workers=0,
    #                           pin_memory=True)

    label_list   = [label_dict[os.path.basename(p)] for p in train_dataset.paths]
    label_counts = Counter(label_list)                       

    sample_weights = [1.0 / label_counts[lbl] for lbl in label_list]

    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True                    
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,                    
        num_workers=0,
        pin_memory=True
    )

    return train_loader, num_classes

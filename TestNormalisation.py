import os
from pathlib import Path

from PIL import Image
import torch
from torchvision import transforms
from tqdm import tqdm

# ==== STATIC PARAMETERS ====
INPUT_DIR = Path("./DDR/filtered_procesed_train")   
OUTPUT_DIR = Path("./sanity_out_normalisation")       
IMAGE_SIZE = 256                        
RECURSIVE = True                        

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Pillow resample constant (compat across versions)
RESAMPLE = getattr(Image, "BICUBIC", Image.Resampling.BICUBIC)

to_tensor_and_norm = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=RESAMPLE),
    transforms.ToTensor(),  # [0,1]
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # -> [-1,1]
])
to_pil = transforms.ToPILImage()


def denormalize_like_yours(x: torch.Tensor) -> torch.Tensor:
    # x in [-1,1] -> [0,1]
    return (x.clamp(-1, 1) + 1) / 2


def find_images(root: Path, recursive: bool):
    if recursive:
        yield from (p for p in root.rglob("*") if p.suffix.lower() in ALLOWED_EXTS)
    else:
        yield from (p for p in root.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_EXTS)


def process_image(in_path: Path, out_path: Path):
    img = Image.open(in_path).convert("RGB")
    x = to_tensor_and_norm(img)       # (C,H,W) in [-1,1]
    x_denorm = denormalize_like_yours(x)  # back to [0,1]
    out_img = to_pil(x_denorm)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(out_path)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    images = list(find_images(INPUT_DIR, RECURSIVE))
    if not images:
        print(f"No images with {sorted(ALLOWED_EXTS)} found in {INPUT_DIR}")
        return

    for p in tqdm(images, desc="Normalize → De-normalize → Save"):
        rel = p.relative_to(INPUT_DIR) if RECURSIVE else Path(p.name)
        out_path = OUTPUT_DIR / rel  # keep same extension/structure
        process_image(p, out_path)

    print(f"Done. Wrote {len(images)} images to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()

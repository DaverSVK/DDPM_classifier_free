import os
import cv2
import json
import math
import numpy as np
import matplotlib.pyplot as plt
import csv
from collections import Counter, defaultdict

# =========================
# CONFIGURATION
# =========================
RESIZED_BASE = "resized_and_sorted"  # expects subfolders: 0..4
GENERATED_BASE = "GeneratedOutputs/generated5/ckpt_epoch_10005"  # expects subfolders: class_0..class_4
CLASSES = [0, 1, 2, 3, 4]
BINS = 256
BAR_ALPHA = 1.0  # 1.0 is solid
SAVE_FIG = "histogram_tower_resized_vs_generated.png"
SAVE_CSV = "histogram_diagnostics.csv"
VALID_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')

# =========================
# UTILITIES
# =========================
def ensure_rgb_uint8(img):
    """
    Convert input image (BGR/GRAY/BGRA, uint8/uint16/float) to RGB uint8.
    """
    if img is None:
        return None

    original_dtype = str(img.dtype)

    # If grayscale, make 3 channels first
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Drop alpha if present
    if img.ndim == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    # Convert to float32 for scaling if needed
    if img.dtype == np.uint16:
        img = (img.astype(np.float32) / 65535.0) * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
    elif img.dtype == np.float32 or img.dtype == np.float64:
        # assume 0..1 or 0..255; if max <= 1.0, scale to 0..255
        maxv = float(np.nanmax(img)) if img.size > 0 else 1.0
        img = img * (255.0 if maxv <= 1.0 else 1.0)
        img = np.clip(img, 0, 255).astype(np.uint8)
    elif img.dtype != np.uint8:
        # Fallback cast
        img = np.clip(img.astype(np.float32), 0, 255).astype(np.uint8)

    # Now convert BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, original_dtype

def compute_rgb_histogram_uint8(img_rgb_uint8, bins=BINS):
    """Return per-channel histogram with shape (3, bins)."""
    chans = cv2.split(img_rgb_uint8)  # R, G, B
    hist = [cv2.calcHist([ch], [0], None, [bins], [0, 256]).flatten() for ch in chans]
    return np.array(hist, dtype=np.float64)

def unique_levels_per_channel(img_rgb_uint8):
    """Return tuple of counts of unique intensity levels per channel (R,G,B)."""
    return tuple(int(np.unique(img_rgb_uint8[..., i]).size) for i in range(3))

def iter_images(folder):
    if not os.path.isdir(folder):
        return
    for fn in os.listdir(folder):
        if fn.lower().endswith(VALID_EXTS):
            yield os.path.join(folder, fn)

def to_prob_per_channel(hist_3xbins, eps=1e-12):
    h = np.maximum(hist_3xbins, 0)
    denom = h.sum(axis=1, keepdims=True) + eps
    return h / denom

def process_folder(folder):
    """
    Process all images in folder:
      - Convert to RGB uint8
      - Accumulate mean histogram (per channel)
      - Track diagnostics: dtype counts, nonzero bins, unique levels
    Returns:
      {
        'mean_hist_p': (3, bins) probabilities,
        'sum_hist': (3, bins) raw summed histogram (for nonzero bins),
        'images': int,
        'dtype_counts': dict,
        'avg_unique_levels': (R,G,B) averages over images,
        'nonzero_bins': (R,G,B) from summed histogram
      }
    or None if folder missing/empty.
    """
    dtype_counter = Counter()
    unique_levels_accum = np.zeros(3, dtype=np.float64)
    sum_hist = np.zeros((3, BINS), dtype=np.float64)
    n = 0

    for path in iter_images(folder):
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            continue
        img, odt = ensure_rgb_uint8(raw)
        dtype_counter[odt] += 1

        # unique levels (after normalization to uint8, because we compare 8-bit hist)
        u = unique_levels_per_channel(img)
        unique_levels_accum += np.array(u, dtype=np.float64)

        # histogram
        h = compute_rgb_histogram_uint8(img, BINS)
        sum_hist += h
        n += 1

    if n == 0:
        return None

    mean_hist = sum_hist / float(n)
    mean_hist_p = to_prob_per_channel(mean_hist)
    nonzero_bins = tuple(int(np.count_nonzero(sum_hist[i])) for i in range(3))
    avg_unique_levels = tuple(float(unique_levels_accum[i]) / n for i in range(3))

    return {
        'mean_hist_p': mean_hist_p,
        'sum_hist': sum_hist,
        'images': n,
        'dtype_counts': dict(dtype_counter),
        'avg_unique_levels': avg_unique_levels,
        'nonzero_bins': nonzero_bins
    }

def compare_class(cls_id):
    resized_folder = os.path.join(RESIZED_BASE, str(cls_id))
    generated_folder = os.path.join(GENERATED_BASE, f"class_{cls_id}")

    resized = process_folder(resized_folder)
    generated = process_folder(generated_folder)

    return resized, generated

# =========================
# PLOTTING (TOWER GRAPHS)
# =========================
def plot_tower(resized_dicts, generated_dicts):
    """
    resized_dicts, generated_dicts: dict[class_id] -> result dict from process_folder (or None)
    Layout:
      For each class:
        Row 1: Resized (R,G,B columns)
        Row 2: Generated (R,G,B columns)
    """
    n_classes = len(resized_dicts)
    fig, axes = plt.subplots(n_classes * 2, 3, figsize=(15, max(3, n_classes * 3)), sharex=True, sharey=True)
    fig.suptitle("Resized (top) vs Generated (bottom) — RGB Histograms as Bar Charts (normalized)", fontsize=16)

    channel_labels = ("R", "G", "B")
    channel_colors = ("r", "g", "b")
    x = np.arange(BINS)

    for r, cls in enumerate(sorted(resized_dicts.keys())):
        r_info = resized_dicts[cls]
        g_info = generated_dicts[cls]

        for c in range(3):
            # Top: Resized
            ax_top = axes[r*2, c]
            if r_info is not None:
                ax_top.bar(x, r_info['mean_hist_p'][c], color=channel_colors[c], alpha=BAR_ALPHA, width=1.0)
            else:
                ax_top.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax_top.transAxes)
            if c == 0:
                ax_top.set_ylabel(f"class_{cls}\nResized", fontsize=10)
            if r == 0:
                ax_top.set_title(channel_labels[c], fontsize=12)

            # Bottom: Generated
            ax_bottom = axes[r*2 + 1, c]
            if g_info is not None:
                ax_bottom.bar(x, g_info['mean_hist_p'][c], color=channel_colors[c], alpha=BAR_ALPHA, width=1.0)
            else:
                ax_bottom.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax_bottom.transAxes)
            if c == 0:
                ax_bottom.set_ylabel("Generated", fontsize=10)

    for ax in np.ravel(axes):
        ax.set_xlim(-10, BINS + 9)
        
    if r_info is not None:
        total_prob_r = np.sum(r_info['mean_hist_p'][c])
        total_count_r = np.sum(r_info['sum_hist'][c])
        print(f"class_{cls} resized {channel_labels[c]}: prob_sum={total_prob_r:.6f}, raw_sum={total_count_r}")

    if g_info is not None:
        total_prob_g = np.sum(g_info['mean_hist_p'][c])
        total_count_g = np.sum(g_info['sum_hist'][c])
        print(f"class_{cls} generated {channel_labels[c]}: prob_sum={total_prob_g:.6f}, raw_sum={total_count_g}")


    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(SAVE_FIG, dpi=200)
    print(f"Saved figure to: {SAVE_FIG}")

# =========================
# CSV DIAGNOSTICS
# =========================
def save_diagnostics_csv(resized_dicts, generated_dicts):
    """
    Write one row per (class, set) with:
      class, set, images, nonzero_bins_R/G/B, avg_unique_R/G/B, dtype_json
    """
    rows = []
    for cls in sorted(resized_dicts.keys()):
        for set_name, info in (("resized", resized_dicts[cls]), ("generated", generated_dicts[cls])):
            if info is None:
                rows.append({
                    "class": cls,
                    "set": set_name,
                    "images": 0,
                    "nonzero_bins_R": 0, "nonzero_bins_G": 0, "nonzero_bins_B": 0,
                    "avg_unique_R": 0.0, "avg_unique_G": 0.0, "avg_unique_B": 0.0,
                    "dtype_counts": "{}"
                })
            else:
                rows.append({
                    "class": cls,
                    "set": set_name,
                    "images": info['images'],
                    "nonzero_bins_R": info['nonzero_bins'][0],
                    "nonzero_bins_G": info['nonzero_bins'][1],
                    "nonzero_bins_B": info['nonzero_bins'][2],
                    "avg_unique_R": round(info['avg_unique_levels'][0], 3),
                    "avg_unique_G": round(info['avg_unique_levels'][1], 3),
                    "avg_unique_B": round(info['avg_unique_levels'][2], 3),
                    "dtype_counts": json.dumps(info['dtype_counts'])
                })

    fieldnames = [
        "class", "set", "images",
        "nonzero_bins_R", "nonzero_bins_G", "nonzero_bins_B",
        "avg_unique_R", "avg_unique_G", "avg_unique_B",
        "dtype_counts"
    ]

    with open(SAVE_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Saved diagnostics to: {SAVE_CSV}")

# =========================
# MAIN
# =========================
def main():
    resized_dicts = {}
    generated_dicts = {}

    for cls in CLASSES:
        r_info, g_info = compare_class(cls)
        resized_dicts[cls] = r_info
        generated_dicts[cls] = g_info

    # Print a compact console summary
    print("\n=== Folder diagnostics (higher nonzero bins & unique levels => richer distribution) ===")
    for cls in CLASSES:
        for tag, info in (("resized", resized_dicts[cls]), ("generated", generated_dicts[cls])):
            if info is None:
                print(f"class_{cls:<2} [{tag:9}]  images=0  nonzero_bins=(0,0,0)  avg_unique=(0,0,0)  dtypes={{}}")
            else:
                nz = info['nonzero_bins']
                au = tuple(round(x, 1) for x in info['avg_unique_levels'])
                print(f"class_{cls:<2} [{tag:9}]  images={info['images']:<4}  nonzero_bins={nz}  avg_unique={au}  dtypes={info['dtype_counts']}")

    # Save CSV + plot
    save_diagnostics_csv(resized_dicts, generated_dicts)
    plot_tower(resized_dicts, generated_dicts)

if __name__ == "__main__":
    main()

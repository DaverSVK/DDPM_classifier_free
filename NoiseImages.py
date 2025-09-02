import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Set your folder path
folder_path = "generated5/ckpt_epoch_10005/class_2"   # <-- Change to your folder path

# Create output directory
output_dir = "output_graphs"
os.makedirs(output_dir, exist_ok=True)

# Prepare lists
indices = []
values_R = []
values_G = []
values_B = []
colors = []

# Loop through all images
for idx, filename in enumerate(sorted(os.listdir(folder_path))):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path).convert('RGB')

        arr = np.array(img)

        # Get 3x3 block from bottom-right corner
        block = arr[-3:, -3:, :]

        # Compute mean for each channel
        avg_R = block[:, :, 0].mean()
        avg_G = block[:, :, 1].mean()
        avg_B = block[:, :, 2].mean()

        # Append to lists
        indices.append(idx)
        values_R.append(avg_R)
        values_G.append(avg_G)
        values_B.append(avg_B)

        # Save color (normalized to 0-1 for matplotlib)
        colors.append((avg_R/255, avg_G/255, avg_B/255))

# Create figure with 4 rows (R, G, B + Color strip)
fig, axs = plt.subplots(4, 1, figsize=(12, 14), gridspec_kw={'height_ratios': [1, 1, 1, 0.3]}, sharex=True)

# Red channel
axs[0].plot(indices, values_R, marker='o', color='red')
axs[0].set_title("3x3 Bottom-Right Avg Red Values")
axs[0].set_ylabel("Red Value")
axs[0].grid(True)

# Green channel
axs[1].plot(indices, values_G, marker='o', color='green')
axs[1].set_title("3x3 Bottom-Right Avg Green Values")
axs[1].set_ylabel("Green Value")
axs[1].grid(True)

# Blue channel
axs[2].plot(indices, values_B, marker='o', color='blue')
axs[2].set_title("3x3 Bottom-Right Avg Blue Values")
axs[2].set_ylabel("Blue Value")
axs[2].grid(True)

luminances = [0.299*r + 0.587*g + 0.114*b for r, g, b in colors]

# Pair indices, colors, luminances together
color_data = list(zip(indices, colors, luminances))

# Sort by luminance ascending
color_data.sort(key=lambda x: x[2])

# Draw sorted colors
for new_idx, (_, color, _) in enumerate(color_data):
    axs[3].add_patch(plt.Rectangle((new_idx, 0), 1, 1, color=color))

axs[3].set_xlim(0, len(color_data))
axs[3].set_ylim(0, 1)
axs[3].axis('off')
axs[3].set_title("Mean Color Strip Sorted by Brightness")
plt.tight_layout()

# Save figure
output_path = os.path.join(output_dir, "rgb_channels_with_colors.png")
plt.savefig(output_path)
plt.close()

print(f"Combined RGB + Color Strip figure saved as: {output_path}")

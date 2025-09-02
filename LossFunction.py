import matplotlib.pyplot as plt
import re
from scipy.signal import medfilt

log_folder = "./TraniOutputs/diffusion_model-2025-08-28-19-47/"
log_file = log_folder+"epoch_times.txt"

epochs = []
losses = []
learningRate = []

# pattern = re.compile(r"Epoch (\d+): .*?, ([0-9.]+) loss")
# Epoch 1001: 267.25 sec, 0.001363 loss, 0.00 FID, 0.000100 LR
pattern = re.compile(r"Epoch (\d+): .*?, ([0-9.]+) loss, .*?, ([0-9.]+) LR")

with open(log_file, "r") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            epochs.append(int(match.group(1)))
            losses.append(float(match.group(2)))
            learningRate.append(float(match.group(3)))

# Apply median filter with window size 5
smoothed_losses = medfilt(losses, kernel_size=5)

# Plot
plt.figure(figsize=(8, 5))
# plt.plot(epochs, losses, marker='o', linestyle='-', label="Original Loss")
plt.plot(epochs, smoothed_losses, linestyle='-', label="Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss per Epoch")
plt.legend()
plt.grid(True)

# Save to file
plt.savefig(log_folder+"loss_plot_filtered.png", dpi=300)
plt.close()


print("Plot saved as loss_plot_filtered.png")


# Plot
plt.figure(figsize=(8, 5))
# plt.plot(epochs, losses, marker='o', linestyle='-', label="Original Loss")
plt.plot(epochs, learningRate, linestyle='-', label="Loss")

plt.xlabel("Epoch")
plt.ylabel("LR")
plt.title("Training Learning rate per Epoch")
plt.legend()
plt.grid(True)

# Save to file
plt.savefig(log_folder+"LR_plot_linear.png", dpi=300)
plt.close()


print("Plot saved as LR_plot_linear.png")

import os
import numpy as np
import random
import matplotlib.pyplot as plt

# Set the data folder path
data_dir = r"C:\Users\j6g\Desktop\Worm\Vis_test\MergedCubes32"

# Get all .npy files
npy_files = [f for f in os.listdir(data_dir) if f.endswith(".npy")]

# Ensure the folder contains files
if not npy_files:
    print("Error: No .npy files found in the folder! Please check the path.")
    exit()

# Randomly select one worm
random_worm = random.choice(npy_files)
worm_path = os.path.join(data_dir, random_worm)

# Load the .npy data (expected shape: 558, 1, 32, 32, 32)
worm_data = np.load(worm_path)

# Ensure the data shape is correct
if worm_data.shape != (558, 1, 32, 32, 32):
    print(f"Error: File {random_worm} has unexpected shape {worm_data.shape}!")
    exit()

# Randomly select 3 cells
random_cells = random.sample(range(558), 3)

# Set colormap
cmap = "inferno"  # Can also be 'gray', 'magma', 'viridis', etc.

# Create visualization
fig, axes = plt.subplots(3, 3, figsize=(10, 10))

for i, cell_id in enumerate(random_cells):
    cell_data = worm_data[cell_id, 0, :, :, :]  # Shape: (32, 32, 32)

    # Extract three slices
    xy_slice = cell_data[:, :, 16]  # Middle XY plane
    xz_slice = cell_data[:, 16, :]  # Middle XZ plane
    yz_slice = cell_data[16, :, :]  # Middle YZ plane

    # Plot the images
    axes[i, 0].imshow(xy_slice, cmap='viridis')
    axes[i, 0].set_title(f"Worm: {random_worm}\nCell {cell_id} - XY (Z=12)")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(xz_slice, cmap='viridis')
    axes[i, 1].set_title(f"Cell {cell_id} - XZ (Y=12)")
    axes[i, 1].axis("off")

    axes[i, 2].imshow(yz_slice, cmap='viridis')
    axes[i, 2].set_title(f"Cell {cell_id} - YZ (X=12)")
    axes[i, 2].axis("off")

plt.tight_layout()
plt.show()

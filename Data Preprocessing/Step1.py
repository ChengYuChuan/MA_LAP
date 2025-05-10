import os
import numpy as np
from scipy.ndimage import distance_transform_edt

def apply_soft_mask_outside(voxel, alpha=10.0):
    """Apply soft mask outside the selected region"""
    original_voxel = voxel[0]  # Shape: (32, 32, 32)
    mask = voxel[1]            # Shape: (32, 32, 32)

    distance = distance_transform_edt(1 - mask)
    soft_mask = 1 / (1 + np.exp(alpha * (distance - 1)))

    masked_voxel = original_voxel.astype(np.float64) * soft_mask
    voxel = np.concatenate([voxel, masked_voxel[np.newaxis, :, :, :]], axis=0)

    return voxel

# Set input and output directories
input_dir = "C:/Users/j6g/Desktop/Vis_test/2ChannelMaskedCube32"
output_dir = "C:/Users/j6g/Desktop/Vis_test/3ChannelMaskedCube32"

os.makedirs(output_dir, exist_ok=True)

# Read all .npy files
npy_files = [f for f in os.listdir(input_dir) if f.endswith(".npy")]

# Keep track of processed and skipped files
processed_files = []
skipped_files = []

for file in npy_files:
    # Load numpy file, expected shape: (2, 32, 32, 32)
    data = np.load(os.path.join(input_dir, file))

    if data.shape != (2, 32, 32, 32):
        print(f"Skipped {file} because shape is not (2, 32, 32, 32)")
        skipped_files.append(file)
        continue

    # Get the two channels
    channel_0 = data[0]  # Original data
    channel_1 = data[1]  # Label data

    # Step 1: Check the center 3x3x3 region for labels
    center = (16, 16, 16)
    half_size = 2
    center_region = channel_1[
        center[0] - half_size: center[0] + half_size + 1,
        center[1] - half_size: center[1] + half_size + 1,
        center[2] - half_size: center[2] + half_size + 1
    ]

    nonzero_labels = center_region[center_region > 0]

    # Step 2: If no labels in the center region, choose the most frequent label in the entire cube
    if nonzero_labels.size == 0:
        print(f"{file} has no label in the {half_size}x{half_size}x{half_size} center region, selecting most frequent non-zero label from entire cube")

        # Get all non-zero labels in the cube
        nonzero_labels_cube = channel_1[channel_1 > 0]

        if nonzero_labels_cube.size == 0:
            print(f"Skipped {file} because there are no valid labels in the entire cube")
            skipped_files.append(file)
            continue

        # Determine the most common label
        unique_labels, counts = np.unique(nonzero_labels_cube, return_counts=True)
        most_common_label = unique_labels[np.argmax(counts)]
    else:
        # Step 3: Normally, select the most common label in the center region
        unique_labels, counts = np.unique(nonzero_labels, return_counts=True)
        most_common_label = unique_labels[np.argmax(counts)]

    # Step 4: Create a binary mask for the most common label
    mask = (channel_1 == most_common_label)

    # Step 5: Construct a 2-channel voxel (original + binary mask)
    voxel = np.stack([channel_0, mask], axis=0)

    # Step 6: Apply soft mask outside the selected region
    voxel_3channel = apply_soft_mask_outside(voxel, alpha=0.6)

    # Step 7: Save the processed 3-channel voxel cube
    output_path = os.path.join(output_dir, file)
    np.save(output_path, voxel_3channel)

    processed_files.append(file)

    # Debug info for each file (optional)
    # print(f"File: {file}")
    # print(f"  Selected label: {most_common_label}")
    # print(f"  Saved to: {output_path}\n")

# Save the list of skipped files
with open("skipped_files.txt", "w") as f:
    for file in skipped_files:
        f.write(file + "\n")

# Save the list of successfully processed files
with open("processed_files.txt", "w") as f:
    for file in processed_files:
        f.write(file + "\n")

print(f"\n=== Step1 Completed ===")
print(f"Total input files: {len(npy_files)}")
print(f"Successfully processed: {len(processed_files)}")
print(f"Skipped: {len(skipped_files)} (recorded in skipped_files.txt)")

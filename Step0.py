import torch
from torch.utils.data import Dataset, DataLoader
import os

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
from v3dpy.loaders import Raw as V3DREADER
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path

def read_all_coordinates(txt_path):
    """Read all coordinates and aligned numbers from txt file"""
    try:
        df = pd.read_csv(txt_path, delimiter='\t')
        coordinates = []
        for _, row in df.iterrows():
            # Round the coordinates to nearest whole number
            x = round(float(row['x']))
            y = round(float(row['y']))
            z = round(float(row['z']))
            aligned_no = int(row['Aligned_No'])
            coordinates.append((x, y, z, aligned_no))
        return coordinates
    except Exception as e:
        print(f"Error reading {txt_path}: {e}")
        return None


def save_cube_as_npy(cube_data, save_path):
    """Save cube data as numpy array"""
    np.save(save_path, cube_data)


def save_cube_as_raw(cube_data, save_path, dtype=np.uint8):
    """Save cube data as raw format, ensuring it matches V3DREADER() output."""
    # Ensure data type matches the original raw format
    cube_data = cube_data.astype(dtype)

    # Save as raw binary file
    with open(save_path, 'wb') as f:
        f.write(cube_data.tobytes())

    # print(f"Saved cube as RAW: {save_path}")


def extract_cube(data, center, size):
    """Extract a cubic region from 4D data (C, D, H, W) with zero-padding if necessary."""
    C, D, H, W = data.shape  # Get full dimensions
    z, y, x = center
    half = size // 2

    # Compute slicing bounds ensuring they stay within the data range
    z_min, z_max = max(z - half, 0), min(z + half, D)
    y_min, y_max = max(y - half, 0), min(y + half, H)
    x_min, x_max = max(x - half, 0), min(x + half, W)

    # Extract the cube
    cube = data[:, z_min:z_max, y_min:y_max, x_min:x_max]

    # Compute necessary padding for each axis (before, after)
    pad_z = ((max(0, half - z), max(0, (z + half) - D)))
    pad_y = ((max(0, half - y), max(0, (y + half) - H)))
    pad_x = ((max(0, half - x), max(0, (x + half) - W)))

    # Pad the cube to ensure it is always (C, size, size, size)
    cube_padded = np.pad(cube, ((0, 0), pad_z, pad_y, pad_x), mode='constant', constant_values=0)

    return cube_padded


raw_dir = "C:/Users/j6g/Desktop/Vis_test/Crop Raw"
raw_masked_dir = "C:/Users/j6g/Desktop/Golden Sample/Masked"
txt_dir = "C:/Users/j6g/Desktop/Golden Sample/Processed for GNN"
output_dir = "C:/Users/j6g/Desktop/Vis_test/2ChannelMaskedCube32"

cube_size = 32
save_as_npy = True  # 切換 True 儲存為 .npy，False 儲存為 .raw

# original_data = V3DREADER().load("C:/Users/j6g/Desktop/Worm/Vis_test/Masked test/worm_001.raw")
# print(original_data.shape, original_data.dtype)

# cube = np.fromfile("example_cube.raw", dtype=original_data.dtype)
# print(cube.shape)  # Should match (C, D, H, W)



# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process each file
for raw_file in os.listdir(raw_dir):
    if not raw_file.endswith('.raw'):  # Adjust file extension if needed
        continue

    print(f"Processing {raw_file}...")

    # Find corresponding txt file
    txt_file = Path(raw_file).stem + '.txt'
    txt_path = os.path.join(txt_dir, txt_file)

    if not os.path.exists(txt_path):
        print(f"Warning: No txt file found for {raw_file}")
        continue

    # Read all coordinates for this file
    coordinates = read_all_coordinates(txt_path)
    if coordinates is None:
        continue

    # Load raw data (only once per file)
    data = V3DREADER().load(os.path.join(raw_dir, raw_file))
    third_channel_data = data[2:3, :, :, :]  # 這樣就保持 4D 格式
    data_np = third_channel_data.numpy() if torch.is_tensor(third_channel_data) else third_channel_data
    print(f"data_np shape {data_np.shape}")

    # Find corresponding masked raw file
    masked_file_path = os.path.join(raw_masked_dir, raw_file)
    if not os.path.exists(masked_file_path):
        print(f"Warning: No masked file found for {raw_file}")
        continue

    # Load masked raw data
    masked_data = V3DREADER().load(masked_file_path)
    masked_np = masked_data.numpy() if torch.is_tensor(masked_data) else masked_data
    print(f"Masked_np shape {masked_np.shape}")


    # 確保拼接時的 shape 一致
    print(f"data_np shape: {data_np.shape}, masked_np shape: {masked_np.shape}")
    assert data_np.shape[1:] == masked_np.shape[1:], "Shape mismatch between data_np and masked_np"

    # Concatenate third channel data with masked data
    combined_data = np.concatenate((data_np, masked_np), axis=0)  # Concatenate along channel axis

    # Process each coordinate set
    for x, y, z, aligned_no in coordinates:
        center_point = (z, y, x)  # Note the order: (z,y,x)

        # Extract cube
        cube_data = extract_cube(combined_data, center_point, cube_size)

        # Save cube
        save_path = os.path.join(output_dir, f'{Path(raw_file).stem}_cube_{aligned_no:03d}')
        if save_as_npy:
            save_cube_as_npy(cube_data, save_path + ".npy")
        else:
            save_cube_as_raw(cube_data, save_path + ".raw")

        print(f"Saved cube_{aligned_no:03d} - Coordinates (x,y,z): ({x}, {y}, {z})")

    print(f"Completed processing {raw_file} - {len(coordinates)} cubes extracted")
    print("-" * 50)

print("Processing complete!")

# Verify the results
saved_cubes = os.listdir(output_dir)
print(f"\nTotal cubes saved: {len(saved_cubes)}")


cube_sizes = []
for cube_file in saved_cubes[:5]:  # Check first 5 cubes
    cube = np.load(os.path.join(output_dir, cube_file), allow_pickle=True)
    cube_sizes.append(cube.shape)
print("\nSample cube shapes:", cube_sizes)

# Optional: Print distribution of cube sizes
# cube_sizes = []
# for cube_file in saved_cubes[:5]:  # Check first 5 cubes
#     if save_as_npy:
#         cube = np.load(os.path.join(output_dir, cube_file))
#     else:
#         with open(os.path.join(output_dir, cube_file), 'rb') as f:
#             cube = np.frombuffer(f.read(), dtype=np.uint8)  # Adjust dtype based on original data
#     cube_sizes.append(cube.shape)
# print("\nSample cube shapes:", cube_sizes)

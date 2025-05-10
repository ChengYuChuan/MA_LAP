import os
import numpy as np

# Set input and output directories
input_dir = "C:/Users/j6g/Desktop/Vis_test/3ChannelMaskedCube32"
output_dir = "C:/Users/j6g/Desktop/Vis_test/MaskedCube32"
os.makedirs(output_dir, exist_ok=True)

# Get all .npy files
npy_files = [f for f in os.listdir(input_dir) if f.endswith(".npy")]

for file in npy_files:
    # Load the 3-channel numpy file, expected shape: (3, 32, 32, 32)
    voxel_3channel = np.load(os.path.join(input_dir, file))

    # Check if the file contains a third channel
    if voxel_3channel.shape[0] < 3:
        print(f"Skipped {file} because it does not contain a third channel")
        continue

    # Extract the third channel (index 2), shape: (32, 32, 32)
    third_channel = voxel_3channel[2]

    # Save the third channel to the output directory
    output_path = os.path.join(output_dir, file)
    np.save(output_path, third_channel)

    print(f"Third channel of file: {file} has been saved to {output_path}")

print("All third channels have been saved!")

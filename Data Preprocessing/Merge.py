import os
import numpy as np
from tqdm import tqdm

# Set the directory containing the .npy files
data_dir = r"C:\Users\j6g\Desktop\Vis_test\Cubes32"

# Read all .npy files in the directory
npy_files = [f for f in os.listdir(data_dir) if f.endswith(".npy")]

# Create a structure to store data for 196 worms
worms_data = {}

# Record the cell_id occurrences for each worm, and maintain order
worm_cells = {}

for file in tqdm(npy_files, desc="Processing npy files"):
    # Parse filename, e.g., worm_001_cube_001.npy
    parts = file.replace(".npy", "").split("_")
    worm_id = int(parts[1])  # Worm number
    cell_id = int(parts[3])  # Cell number (should be ordered accordingly)

    # Ensure cell_id is within valid range
    if not (1 <= cell_id <= 558):
        print(f"Warning: cell_id ({cell_id}) in {file} is out of range (1~558), skipped!")
        continue

    # Initialize data structure for the worm
    if worm_id not in worms_data:
        worms_data[worm_id] = np.zeros((558, 1, 32, 32, 32), dtype=np.float32)
        worm_cells[worm_id] = []

    # Record cell_id and filename to maintain order
    worm_cells[worm_id].append((cell_id, file))

# Ensure cell_id order is correct and insert data
for worm_id in tqdm(worms_data.keys(), desc="Sorting and verifying order"):
    # Sort by cell_id (ensure correct cube_xxx order)
    worm_cells[worm_id].sort()

    # Check for missing cell_ids
    sorted_cell_ids = [c[0] for c in worm_cells[worm_id]]
    if sorted_cell_ids != list(range(1, 559)):
        missing_cells = set(range(1, 559)) - set(sorted_cell_ids)
        print(f"Warning: worm_{worm_id} is missing {len(missing_cells)} cells, filled with zeros!")
        print(f"Missing cell_ids: {sorted(missing_cells)}")

    # Load and store the data
    for idx, (cell_id, file) in enumerate(worm_cells[worm_id]):
        file_path = os.path.join(data_dir, file)
        data = np.load(file_path)  # Shape: (24, 24, 24)
        worms_data[worm_id][idx, 0, :, :, :] = data  # Ensure cells are filled in sorted order

# Specify the output directory
output_dir = r"C:\Users\j6g\Desktop\Worm\Vis_test\MergedCubes32"
os.makedirs(output_dir, exist_ok=True)

# Save the .npy files for the 196 worms
for worm_id, array in tqdm(worms_data.items(), desc="Saving merged files"):
    save_path = os.path.join(output_dir, f"worm_{worm_id:03d}.npy")
    np.save(save_path, array)

print("Processing complete! 196 .npy files have been saved.")

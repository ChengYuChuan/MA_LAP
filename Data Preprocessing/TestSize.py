import os
import numpy as np

folder = "C:/Users/j6g/Desktop/Vis_test/MergedCubes32"
bad_files = []

for f in os.listdir(folder):
    if f.endswith('.npy'):
        try:
            data = np.load(os.path.join(folder, f))
            if data.shape != (558, 1, 32, 32, 32):
                print(f"⚠️ {f} shape: {data.shape}")
                bad_files.append(f)
        except Exception as e:
            print(f" Error loading {f}: {e}")
            bad_files.append(f)

print(f"Found {len(bad_files)} invalid files.")
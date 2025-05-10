import os
import numpy as np
from tqdm import tqdm

# 設定 npy 檔案的目錄
data_dir = r"C:\Users\j6g\Desktop\Vis_test\Cubes32"

# 讀取所有 npy 檔案
npy_files = [f for f in os.listdir(data_dir) if f.endswith(".npy")]

# 建立結構來儲存 196 隻蟲的數據
worms_data = {}

# 記錄每隻蟲的 cell_id 出現次數，並儲存順序
worm_cells = {}

for file in tqdm(npy_files, desc="Processing npy files"):
    # 解析檔名，例如 worm_001_cube_001.npy
    parts = file.replace(".npy", "").split("_")
    worm_id = int(parts[1])  # 第幾隻蟲
    cell_id = int(parts[3])  # 第幾個細胞 (應按照此排序)

    # 確保 cell_id 在合法範圍內
    if not (1 <= cell_id <= 558):
        print(f"警告：{file} 的 cell_id ({cell_id}) 超出範圍 (1~558)，已跳過！")
        continue

    # 初始化該蟲的數據結構
    if worm_id not in worms_data:
        worms_data[worm_id] = np.zeros((558, 1, 32, 32, 32), dtype=np.float32)
        worm_cells[worm_id] = []

    # 記錄 cell_id，確保順序
    worm_cells[worm_id].append((cell_id, file))

# 確保 cell_id 排序正確，並填入數據
for worm_id in tqdm(worms_data.keys(), desc="Sorting and verifying order"):
    # 按照 `cell_id` 排序 (確保 cube_xxx 順序正確)
    worm_cells[worm_id].sort()

    # 檢查是否有錯誤順序
    sorted_cell_ids = [c[0] for c in worm_cells[worm_id]]
    if sorted_cell_ids != list(range(1, 559)):
        missing_cells = set(range(1, 559)) - set(sorted_cell_ids)
        print(f"警告：worm_{worm_id} 缺少 {len(missing_cells)} 個細胞，已填補 0！")
        print(f"缺失 cell_id: {sorted(missing_cells)}")

    # 讀取並存入數據
    for idx, (cell_id, file) in enumerate(worm_cells[worm_id]):
        file_path = os.path.join(data_dir, file)
        data = np.load(file_path)  # Shape: (24, 24, 24)
        worms_data[worm_id][idx, 0, :, :, :] = data  # 確保 cell 按照排序填入

# 指定輸出目錄
output_dir = r"C:\Users\j6g\Desktop\Worm\Vis_test\MergedCubes32"
os.makedirs(output_dir, exist_ok=True)

# 儲存 196 隻蟲的 npy 檔案
for worm_id, array in tqdm(worms_data.items(), desc="Saving merged files"):
    save_path = os.path.join(output_dir, f"worm_{worm_id:03d}.npy")
    np.save(save_path, array)

print("處理完成！已儲存 196 個 npy 檔案。")

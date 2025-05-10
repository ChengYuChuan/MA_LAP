import os
import numpy as np
import random
import matplotlib.pyplot as plt

# 設定資料夾路徑
data_dir = r"C:\Users\j6g\Desktop\Worm\Vis_test\MergedCubes32"

# 獲取所有 npy 檔案
npy_files = [f for f in os.listdir(data_dir) if f.endswith(".npy")]

# 確保資料夾內有檔案
if not npy_files:
    print("錯誤：資料夾內沒有任何 npy 檔案！請檢查路徑。")
    exit()

# 隨機選擇一隻蟲
random_worm = random.choice(npy_files)
worm_path = os.path.join(data_dir, random_worm)

# 讀取 npy 數據 (shape: 558,1,24,24,24)
worm_data = np.load(worm_path)  # Shape: (558, 1, 24, 24, 24)

# 確保數據形狀正確
if worm_data.shape != (558, 1, 32, 32, 32):
    print(f"錯誤：檔案 {random_worm} 的形狀不符合預期 {worm_data.shape}！")
    exit()

# 隨機選擇 3 個細胞
random_cells = random.sample(range(558), 3)

# 設定顏色映射
cmap = "inferno"  # 可改為 'gray'、'magma'、'viridis' 等

# 創建視覺化
fig, axes = plt.subplots(3, 3, figsize=(10, 10))

for i, cell_id in enumerate(random_cells):
    cell_data = worm_data[cell_id, 0, :, :, :]  # Shape: (24, 24, 24)

    # 取得三個剖面
    xy_slice = cell_data[:, :, 12]  # 中間的 XY 平面
    xz_slice = cell_data[:, 12, :]  # 中間的 XZ 平面
    yz_slice = cell_data[12, :, :]  # 中間的 YZ 平面

    # 繪製圖像
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

import os
import numpy as np
from scipy.ndimage import distance_transform_edt

def apply_soft_mask_outside(voxel, alpha=10.0):
    """ 計算 mask 外部的距離，並應用 soft mask """
    original_voxel = voxel[0]  # (24,24,24)
    mask = voxel[1]            # (24,24,24)

    distance = distance_transform_edt(1 - mask)
    soft_mask = 1 / (1 + np.exp(alpha * (distance - 1)))

    masked_voxel = original_voxel.astype(np.float64) * soft_mask
    voxel = np.concatenate([voxel, masked_voxel[np.newaxis, :, :, :]], axis=0)

    return voxel

# 設定輸入與輸出資料夾
input_dir = "C:/Users/j6g/Desktop/Vis_test/2ChannelMaskedCube32"
output_dir = "C:/Users/j6g/Desktop/Vis_test/3ChannelMaskedCube32"

os.makedirs(output_dir, exist_ok=True)

# 讀取所有 .npy 檔案
npy_files = [f for f in os.listdir(input_dir) if f.endswith(".npy")]

# 記錄處理過和被跳過的檔案
processed_files = []
skipped_files = []

for file in npy_files:
    # 讀取 numpy 檔案，形狀預期為 (2, 24, 24, 24)
    data = np.load(os.path.join(input_dir, file))

    if data.shape != (2, 32, 32, 32):
        print(f"跳過 {file}，因為 shape 不符合 (2, 32, 32, 32)")
        skipped_files.append(file)
        continue

    # 取得兩個通道
    channel_0 = data[0]  # 原始資料
    channel_1 = data[1]  # 標籤數據

    # **Step 1: 檢查中心 3x3x3 區域的標籤**
    center = (16, 16, 16)
    half_size = 2
    center_region = channel_1[
        center[0] - half_size: center[0] + half_size + 1,
        center[1] - half_size: center[1] + half_size + 1,
        center[2] - half_size: center[2] + half_size + 1
    ]

    nonzero_labels = center_region[center_region > 0]

    # **Step 2: 如果中心 3x3x3 內無標籤，則選擇整個 cube 內最多的標籤**
    if nonzero_labels.size == 0:
        print(f"{file} 的中心 {half_size}x{half_size}x{half_size} 內無標籤，改為選擇整個 cube 內最多的非0標籤")

        # 計算整個 24x24x24 內非 0 的標籤
        nonzero_labels_cube = channel_1[channel_1 > 0]

        if nonzero_labels_cube.size == 0:
            print(f"跳過 {file}，因為整個 cube 內也沒有有效標籤")
            skipped_files.append(file)
            continue

        # 計算最大佔比標籤
        unique_labels, counts = np.unique(nonzero_labels_cube, return_counts=True)
        most_common_label = unique_labels[np.argmax(counts)]
    else:
        # **Step 3: 正常情況下，選擇中心 3x3x3 區域內最多的標籤**
        unique_labels, counts = np.unique(nonzero_labels, return_counts=True)
        most_common_label = unique_labels[np.argmax(counts)]

    # **Step 4: 建立 mask**
    mask = (channel_1 == most_common_label)

    # **Step 5: 建立三通道的 voxel**
    voxel = np.stack([channel_0, mask], axis=0)

    # **Step 6: 應用 soft mask**
    voxel_3channel = apply_soft_mask_outside(voxel, alpha=0.6)

    # **Step 7: 儲存處理後的 voxel cube**
    output_path = os.path.join(output_dir, file)
    np.save(output_path, voxel_3channel)

    processed_files.append(file)

    # **輸出該 cube 的處理結果**
    # print(f"檔案: {file}")
    # print(f"  選擇的標籤: {most_common_label}")
    # print(f"  已處理並儲存至: {output_path}\n")

# **儲存被跳過的檔案**
with open("skipped_files.txt", "w") as f:
    for file in skipped_files:
        f.write(file + "\n")

# **儲存成功處理的檔案**
with open("processed_files.txt", "w") as f:
    for file in processed_files:
        f.write(file + "\n")

print(f"\n=== Step1 完成 ===")
print(f"總共輸入檔案數: {len(npy_files)}")
print(f"成功處理: {len(processed_files)} 個")
print(f"跳過: {len(skipped_files)} 個 (已記錄在 skipped_files.txt)")

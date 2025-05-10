import os
import numpy as np

# 設定輸入與輸出資料夾
input_dir = "C:/Users/j6g/Desktop/Vis_test/3ChannelMaskedCube32"
output_dir = "C:/Users/j6g/Desktop/Vis_test/MaskedCube32"
os.makedirs(output_dir, exist_ok=True)

# 獲取所有 .npy 檔案
npy_files = [f for f in os.listdir(input_dir) if f.endswith(".npy")]

for file in npy_files:
    # 讀取 3-channel 的 numpy 檔案，預期形狀為 (3, 24, 24, 24)
    voxel_3channel = np.load(os.path.join(input_dir, file))

    # 確認檔案是否含有第三個通道
    if voxel_3channel.shape[0] < 3:
        print(f"跳過 {file}，因為不含第三個通道")
        continue

    # 提取第三個通道 (索引為 2)，其形狀為 (24,24,24)
    third_channel = voxel_3channel[2]

    # 儲存第三個通道至輸出資料夾
    output_path = os.path.join(output_dir, file)
    np.save(output_path, third_channel)

    print(f"檔案: {file} 的第三個通道已儲存至 {output_path}")

print("所有檔案的第三個通道均已儲存！")

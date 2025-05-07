import torch
import torch.nn as nn
import numpy as np
import sys
# --- Dataset ---
from CubeDataset import CubeDataset, get_train_loaders
#---Transform---
import torchvision.transforms as transforms
import utils
# from torch_geometric.contrib.nn.models.rbcd_attack import LOSS_TYPE
# --- internal function ---
from utils import get_logger, load_checkpoint, create_optimizer, save_checkpoint, RunningAverage
from utils import _split_and_move_to_gpu, TensorboardFormatter
from transform import Standardize, RandomFlip, RandomRotate90, RandomRotate, ToTensor
from buildingblocks import SingleConv, DoubleConv, ResBlockPNI, ResNetBlock, Encoder, Decoder
from buildingblocks import create_encoders, create_decoders
from loss import get_loss_criterion
# --- Scheduler ---
import torch.optim.lr_scheduler as lr_scheduler
# --- LAPNetTrainer ---
from LAPNetTrainer import LAPNetTrainer

LossType = sys.argv[1] # "SSIMLoss" or "MSELoss"
Cubesets = sys.argv[2] # "Cubes" or "MaskedCubes"
CubeSize = sys.argv[3] # "24" or "32"
PoolType = sys.argv[4] # 'avg' or 'max'
Learning_Rate = float(sys.argv[5]) # 0.0001
window_size = sys.argv[6] # cube24 should be 5 or 3, cube32 should 7 or 11
alpha = float(sys.argv[7])


window_size = int(window_size)
# Learning_Rate = 0.001
# alpha = 0.8

logger = get_logger('Trainer')

random_state = np.random.RandomState(66)  # 這樣才是正確的隨機狀態
transform_pipeline = transforms.Compose([
    Standardize(mean=0, std=1, min_max=True), #TODO 這樣的標準化請問是全部558個細胞一請算標準化還是？
    # RandomFlip(random_state),        # 預設隨機沿 (2,3,4) 翻轉
    # RandomRotate90(random_state),      # 隨機以 90 度倍數旋轉
    # RandomRotate(random_state, axes=[(2, 1)], angle_spectrum=45, mode='reflect'),
    ToTensor(expand_dims=True)        # 若資料為 (24,24,24) 則轉換成 (1,1,24,24,24)
])
loaders = get_train_loaders(transform=transform_pipeline,num_workers=2, batch_size= 2) # training setting


folder_path = "/home/hd/hd_hd/hd_uu312/MergedCubes32"
train_dataset = CubeDataset(folder_path, transform=transform_pipeline, split="train")
val_dataset = CubeDataset(folder_path, transform=transform_pipeline, split="val")

# 打印各个 split 的数据集大小
print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

# Define Autoencoder model, if you change the basic module, you should change layer order ALSO!!!!!
# DoubleConv gcr or ResNetBlock cge
# [32,64,128,256]
# [16,32,64,128] class LatentEncoder(nn.Module) 的def forward(self, x, return_layer=-2):
#  ResBlockPNI gce,    ResNetBlock cge,    DoubleConv gcr
class LatentEncoder(nn.Module):
  def __init__(self, in_channels=1, f_maps=[32,64,128], layer_order='gce', num_groups=8, pool_type='max'):
      super(LatentEncoder, self).__init__()

      self.latent_channels = f_maps[-1]
      # Create Encoders
      self.encoders = create_encoders(
          in_channels=in_channels,
          pool_type = 'max',
          f_maps=f_maps,
          basic_module=ResBlockPNI,
          conv_kernel_size=3,
          conv_padding=1,
          layer_order=layer_order,
          num_groups=num_groups,
          pool_kernel_size=2,
          downsample_mode='conv'
      )
  def forward(self, x, return_layers=None):
    batch_size, num_cubes, c, d, h, w = x.shape
    x = x.view(batch_size * num_cubes, c, d, h, w)

    feats = []
    for encoder in self.encoders:
        x = encoder(x)
        feats.append(x)

    # 預設回傳最後一層
    if return_layers is None:
        return_layers = [-1]

    out_feats = []
    total_layers = len(feats)

    for i in return_layers:
        # 支援負數 index
        encoder_index = i if i >= 0 else total_layers + i

        if encoder_index < 0 or encoder_index >= total_layers:
            raise IndexError(f"Invalid return_layer index: {i} (resolved as {encoder_index}). Total layers: {total_layers}")

        feat = feats[encoder_index]

        # 根據 encoder 層數推導壓縮倍率（每層 downsampling ×2）
        downscale_factor = 2 ** encoder_index
        D_out = d // downscale_factor
        H_out = h // downscale_factor
        W_out = w // downscale_factor

        # Debug log（可選）
        # print(f"[DEBUG] Layer {i} → resolved as encoder[{encoder_index}] | Shape: {feat.shape} → reshape to: {(batch_size, num_cubes, feat.shape[1], D_out, H_out, W_out)}")

        # reshape 回 batch 格式
        feat = feat.view(batch_size, num_cubes, feat.shape[1], int(D_out), int(H_out), int(W_out))
        out_feats.append(feat)

    return out_feats  # list of (B, N, C, D, H, W)


  # def forward(self, x, return_layer=-2):
  #     batch_size, num_cubes, c, d, h, w = x.shape  # (2, 558, 1, 24, 24, 24)

  #     # 先 reshape 成 (2*558, 1, 24, 24, 24)，然後送進 Encoder
  #     x = x.view(batch_size * num_cubes, c, d, h, w)

  #     feats = []
  #     for encoder in self.encoders:
  #         x = encoder(x)
  #         feats.append(x)

  #     # 從指定層選擇特徵，然後 reshape 回 batch 格式
  #     x = feats[return_layer]  # e.g. return_layer = -1 (最深層), -2 (次深層)

  #     # 根據 encoder 層數反推壓縮倍率（每層 /2）
  #     downscale_factor = 2 ** (len(self.encoders)-1)

  #     D_out = d // downscale_factor
  #     H_out = h // downscale_factor
  #     W_out = w // downscale_factor

  #     # 重新 reshape 回 (2, 558, 256, 3, 3, 3)
  #     x = x.view(batch_size, num_cubes, self.latent_channels, D_out, H_out, W_out) #TODO reshape的形狀需要調整

  #     #TODO 想測試到底要用幾層的特徵來做loss計算，因為我覺得Unet的結構如果取最底層的結構，很可能會過度壓縮
  #     return x  # 最終輸出 (2, 558, 256, 3, 3, 3)

    # 還沒確定要用哪一個
    # def forward(self, x):
    #   batch_size, num_cubes, c, d, h, w = x.shape
    #   x = x.view(batch_size * num_cubes, c, d, h, w)

    #   feats = []
    #   for encoder in self.encoders:
    #       x = encoder(x)
    #       feats.append(x)

    #   return [f.view(batch_size, num_cubes, *f.shape[1:]) for f in feats]  # 回傳多層 list


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params


# Create an instance of your Autoencoder model
model = LatentEncoder()

# 載入預訓練 autoencoder 的 state_dict
full_state = torch.load('/home/hd/hd_hd/hd_uu312/LAP path/CheckPoint_BS2_RBPNI_32_3Layers_CD_Cube32_L1/best_checkpoint.pytorch', map_location='cpu')
ae_model_state = full_state['model_state_dict']

# 過濾出 encoder 部分
encoder_state = {
    k.replace("encoder.", ""): v
    for k, v in ae_model_state.items()
    if k.startswith("encoder.")
}

# 載入到你新的 encoder 模型中
model.load_state_dict(encoder_state, strict=False)
print("✅ 成功只載入 encoder 權重！")

# Move the model to the appropriate device ('cuda:0' or 'cpu') before training
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

num_params = count_parameters(model)
print(f"Total number of trainable parameters: {num_params}")

# --- Loss ---
# loss_criterion = get_loss_criterion(name="DifferentiableHungarianLoss").to(device)
loss_criterion = get_loss_criterion(name='MultiLayerHungarianLoss',
                                    layer_weights=[0.2, 0.8],
                                    penalty_weight=0.35,
                                    penalty_mode="global" ).to(device)
# --- Evaluation ---
# eval_criterion = get_loss_criterion(name="DifferentiableHungarianLoss").to(device)
eval_criterion = get_loss_criterion(name='MultiLayerHungarianLoss',
                                    layer_weights=[0.2, 0.8],
                                    penalty_weight=0.35,
                                    penalty_mode="global" ).to(device)

# --- Optimizer ---
optimizer = create_optimizer('Adam', model, learning_rate=Learning_Rate, weight_decay=0.00001)
# optimizer = create_optimizer('AdamW', model, learning_rate=0.0001, weight_decay=0.00001)
# optimizer = create_optimizer('SGD', model, learning_rate=0.0001, weight_decay=0.00001)

# --- Scheduler ---
lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1, factor=0.5, min_lr=0.00001) # 每15個epoch衰減學習率
# lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0.000005)
lr_scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,       # 每次降一半
    patience=3,       # 容忍 5 次沒有進步
    min_lr=1e-6,      # 最低不小於這個
    verbose=True
)


tensorboard_formatter = TensorboardFormatter(log_channelwise=True)

# Batch = 2
# trainer_config = {
#   "checkpoint_dir" : "/content/drive/MyDrive/Masterarbeit Code/LAP CheckPoint 15Lambda035penalty",
#   "validate_after_iters" : 22, # usually it's half of one epoch iterations
#   "log_after_iters" : 11, # usually it's half of validate_after_iters
#   "max_num_epochs" : 10,
#   "max_num_iterations" : 880 # training data: 196, 196/batch size * max epoch= max iteration
#   }
import math

# 假設你總共有 196 組資料，其中 90% 用於訓練
total_data = 196
train_ratio = 0.9
batch_size = 2
num_epochs = 15

# 動態計算 iterations
train_data = int(total_data * train_ratio)  # 176
iters_per_epoch = math.ceil(train_data / batch_size)  # 88
max_num_iterations = iters_per_epoch * num_epochs  # 880

trainer_config = {
    "checkpoint_dir": "/content/drive/MyDrive/Masterarbeit Code/LAP CheckPoint 15Lambda035penalty",
    "validate_after_iters": iters_per_epoch // 4,  # 每 1/2 個 epoch 驗證一次 → 22
    "log_after_iters": iters_per_epoch // 8,       # 每 1/4 個 epoch logging → 11
    "max_num_epochs": num_epochs,
    "max_num_iterations": max_num_iterations       # 880
}

trainer = LAPNetTrainer(model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, loss_criterion=loss_criterion,
                       eval_criterion=eval_criterion, loaders=loaders, tensorboard_formatter=tensorboard_formatter,
                       resume=None, pre_trained=None, **trainer_config)

trainer.fit()
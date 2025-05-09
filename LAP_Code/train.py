import torch
import torch.nn as nn
import numpy as np
import sys
import math
# --- Dataset ---
from CubeDataset import CubeDataset, get_train_loaders
#---Transform---
import torchvision.transforms as transforms
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

logger = get_logger('Trainer')

random_state = np.random.RandomState(66)
transform_pipeline = transforms.Compose([
    Standardize(z_score=True, min_max=True),
    ToTensor(expand_dims=True)        # from (32,32,32) reform into (1,1,32,32,32)
])
loaders = get_train_loaders(transform=transform_pipeline,num_workers=2, batch_size= 2) # training setting


folder_path = "/home/hd/hd_hd/hd_uu312/MergedCubes32"
train_dataset = CubeDataset(folder_path, transform=transform_pipeline, split="train")
val_dataset = CubeDataset(folder_path, transform=transform_pipeline, split="val")

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")

# [32,64,128,256]
# [16,32,64,128] class LatentEncoder(nn.Module) 的def forward(self, x, return_layer=-2):
#  ResBlockPNI gce OR ResNetBlock cge OR DoubleConv gcr
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

    # Defaultly return last layer
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

        # according to numbers of encoder layers（each layer downsampling ×2）
        downscale_factor = 2 ** encoder_index
        D_out = d // downscale_factor
        H_out = h // downscale_factor
        W_out = w // downscale_factor

        # Debug log (Option)
        # print(f"[DEBUG] Layer {i} → resolved as encoder[{encoder_index}] | Shape: {feat.shape} → reshape to: {(batch_size, num_cubes, feat.shape[1], D_out, H_out, W_out)}")

        # reshape back to batch format
        feat = feat.view(batch_size, num_cubes, feat.shape[1], int(D_out), int(H_out), int(W_out))
        out_feats.append(feat)

    return out_feats  # list of (B, N, C, D, H, W)

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params


# Create an instance of your Autoencoder model
model = LatentEncoder()

# 載入預訓練 autoencoder 的 state_dict
full_state = torch.load('/home/hd/hd_hd/hd_uu312/LAP path/CheckPoint_BS2_RBPNI_32_3Layers_CD_Cube32_L1/best_checkpoint.pytorch', map_location='cpu')
ae_model_state = full_state['model_state_dict']

# Filtering encoder part
encoder_state = {
    k.replace("encoder.", ""): v
    for k, v in ae_model_state.items()
    if k.startswith("encoder.")
}

# Load it into the encoder we just created
model.load_state_dict(encoder_state, strict=False)
print("Successfully loaded encoder weights！")

# Move the model to the appropriate device ('cuda:0' or 'cpu') before training
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

num_params = count_parameters(model)
print(f"Total number of trainable parameters: {num_params}")

# --- Loss ---
# loss_criterion = get_loss_criterion(name="DifferentiableHungarianLoss").to(device)
loss_criterion = get_loss_criterion(name='MultiLayerHungarianLoss',
                                    layer_weights=[0.3, 0.7],
                                    penalty_weight=0.5,
                                    penalty_scale=10.0,
                                    penalty_mode="global" ).to(device)
# --- Evaluation ---
# eval_criterion = get_loss_criterion(name="DifferentiableHungarianLoss").to(device)
eval_criterion = get_loss_criterion(name='MultiLayerHungarianLoss',
                                    layer_weights=[0.3, 0.7],
                                    penalty_weight=0.5,
                                    penalty_scale=10.0,
                                    penalty_mode="global" ).to(device)

# --- Optimizer ---
optimizer = create_optimizer('Adam', model, learning_rate=Learning_Rate, weight_decay=0.00001)
# optimizer = create_optimizer('AdamW', model, learning_rate=0.0001, weight_decay=0.00001)
# optimizer = create_optimizer('SGD', model, learning_rate=0.0001, weight_decay=0.00001)

# --- Scheduler ---
# lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=1, factor=0.5, min_lr=0.00001) # 每15個epoch衰減學習率
# lr_scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0.000005)
lr_scheduler = lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.7,
    patience=3,
    min_lr=1e-6,
    verbose=True
)


tensorboard_formatter = TensorboardFormatter(log_channelwise=True)


total_data = 196
train_ratio = 0.9
batch_size = 2
num_epochs = 40

# Dynamic iterations
train_data = int(total_data * train_ratio)  # 176
iters_per_epoch = math.ceil(train_data / batch_size)  # 88
max_num_iterations = iters_per_epoch * num_epochs

trainer_config = {
    "checkpoint_dir": "/home/hd/hd_hd/hd_uu312/LAP_CheckPoint/LAP CheckPoint 15Lambda035penalty",
    "validate_after_iters": iters_per_epoch // 4,  # validate per 1/2 epoch → 22
    "log_after_iters": iters_per_epoch // 8,       # log per 1/4 epoch logging → 11
    "max_num_epochs": num_epochs,
    "max_num_iterations": max_num_iterations
}

trainer = LAPNetTrainer(model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, loss_criterion=loss_criterion,
                       eval_criterion=eval_criterion, loaders=loaders, tensorboard_formatter=tensorboard_formatter,
                       resume=None, pre_trained=None, **trainer_config)

trainer.fit()
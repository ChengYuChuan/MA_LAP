import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from LAP_Code.utils import get_logger
import sys

logger = get_logger('Dataset')
Cubesets = sys.argv[2] # "Cubes" or "MaskedCubes"
CubeSize = sys.argv[3] # "24" or "32"
STR_CubesSize = str(CubeSize)
CubeSize = int(CubeSize)

logger = get_logger('Dataset')
def get_train_loaders(transform=None, num_workers=0, batch_size=1, device='GPU'):
    """
    Returns dictionary containing the training and validation loaders (torch.utils.data.DataLoader).
    Args:
        config:  a top level configuration object containing the 'loaders' key
    Returns:
        dict {
            'train': <train_loader>
            'val': <val_loader>
        }
    """

    logger.info('Creating training and validation set loaders...')
    folder_path = "/content/drive/MyDrive/Masterarbeit Code/MergedCubes32"

    train_dataset = CubeDataset(folder_path,transform=transform, split="train")
    val_dataset = CubeDataset(folder_path,transform=transform, split="val")
    logger.info(f'Number of workers for train/val dataloader: {num_workers}')

    if torch.cuda.device_count() > 1 and not device == 'cpu':
        logger.info(
            f'{torch.cuda.device_count()} GPUs available. Using batch_size = {torch.cuda.device_count()} * {batch_size}'
        )
        batch_size = batch_size * torch.cuda.device_count()

    logger.info(f'Batch size for train/val loader: {batch_size}')
    # when training with volumetric data use batch_size of 1 due to GPU memory constraints
    return {
        'train':DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=num_workers),
        # don't shuffle during validation: useful when showing how predictions for a given batch get better over time
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, num_workers=num_workers)
    }


class CubeDataset(Dataset):
    def __init__(self, folder_path, transform=None, split="train", train_ratio=0.9):
        self.transform = transform
        self.files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npy')])
        total = len(self.files)
        split_idx = int(total * train_ratio)

        if split == "train":
            self.files = self.files[:split_idx]
        elif split == "val":
            self.files = self.files[split_idx:]
        else:
            raise ValueError("split must be 'train' or 'val'")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        cubes = np.load(self.files[idx]).astype(np.float32)  # shape: (558, 1, 32, 32, 32)

        # 新增隨機 permutation
        perm = np.random.permutation(cubes.shape[0])  # 打亂細胞的順序
        inv_perm = np.argsort(perm)  # 反向 permutation，可以還原回原本的排序

        # 使用 permutation 打亂 cube 順序
        if self.transform:
            cubes = torch.stack([self.transform(cube) for cube in cubes[perm]], dim=0)
        else:
            cubes = torch.from_numpy(cubes[perm])

        # 把 permutation 一起回傳
        return cubes, self.files[idx], perm, inv_perm

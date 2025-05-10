# MA_LAP

---

## 🔬 Overview of `LAP_Code`

The `LAP_Code` folder implements a deep learning pipeline for solving the **Linear Assignment Problem (LAP)** between worm cell embeddings, using a 3D convolutional neural network with a latent space alignment strategy.

This system is built to perform cell-wise alignment between two worm volumes represented as cube-wise features, leveraging a pre-trained **AutoEncoder's encoder** and fine-tuning it through **Hungarian loss optimization**. The primary objective is to learn a representation that can robustly match corresponding cells across samples.

---

## 📁 Folder Structure Summary
### LAP_Code
* **`train.py`**: Main entry script for training. It sets up data loading, model construction (with encoder architecture), loss functions, optimizer, and training loop using `LAPNetTrainer`.
* **`CubeDataset.py`**: Defines the `CubeDataset` class for loading `.npy` files (each representing worm cell cubes). Includes data shuffling and split between training and validation.
* **`buildingblocks.py`**: Core network components, including `ResBlockPNI`, `DoubleConv`, `Encoder`, and helper functions like `create_encoders`.
* **`LAPNetTrainer.py`**: Encapsulates the training and validation logic. Handles logging, learning rate scheduling, dynamic loss adjustment, early stopping, and visualization through TensorBoard.
* **`loss.py`**: Implements differentiable Hungarian-based losses including:
  * `DifferentiableHungarianLoss` for pairwise assignments
  * `MultiLayerHungarianLoss` for multi-resolution feature alignment

* **`transform.py`**: Data augmentation and normalization routines, including Z-score and Min-Max standardization.
* **`utils.py`**: Logging, checkpoint handling, optimizer creation, and various training utilities.

### Data Info
* **`worm_shapes_and_sizes_CropRaw.xlsx`**: It shows the 500 sample's shape and the size of the data.
* **`worm_shapes_and_sizes_mask.xlsx`**: It shows the 500 sample's shape and the size of the data (only in one Channel).

#### These are the samples I use:
* **`GoldenSample_shapes_and_sizes_CropRaw.xlsx`**: It shows the 200 sample's shape and the size of the data.
* **`GoldenSample_shapes_and_sizes_Masked.xlsx`**: It shows the 200 sample's shape and the size of the data (only in one Channel).
* **`Label dict.xlsx`**: It shows the name of Worm No.
---

## 🧠 Model Architecture (Encoder)

The encoder is constructed with a flexible modular structure:

```python
class LatentEncoder(nn.Module):
  def __init__(self, in_channels=1, f_maps=[32,64,128], layer_order='gce', pool_type='max'):
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
```

* `f_maps=[32,64,128]` specifies the number of feature maps in each convolutional block.
* `basic_module` (e.g. `ResBlockPNI` or `ResNetBlock`) can be changed via checkpoint naming conventions.
* `layer_order` such as `'gce'` configures the order of GroupNorm, Conv3D, and ELU activations.
### Corresponding combination
- `ResBlockPNI` -> `gce`
- `ResNetBlock` -> `cge`
- `DoubleConv` -> `gcr`

### Example
if we have a check point file like `RBPNI_32_4Layers_CD`, which means we use 
- `ResNetBlock`(RBPNI) as `basic_module`
- `cge` as `layer_order` 
- `f_map` start from 32 and it has 4 Block Layers like: `[32,64,128,256]`
- `conv`(CD) as `downsample_mode`

The encoder takes in batches of shape `(B, N, C, D, H, W)` and outputs multiple latent representations across different feature depths.

---

## 🧪 Data Preparation

Worm cell cubes (`MergedCubes32`) should be downloaded from the [Google Drive link](https://drive.google.com/drive/folders/1HSu7vZkxCNFcxWPZKkHtazc0NycYo5mW?usp=sharing).

```python
loaders = get_train_loaders(transform=transform_pipeline, folder_path="/home/hd/hd_hd/hd_uu312/MergedCubes32", num_workers=2, batch_size=2)
```

Each cube represents a 3D volume (typically 32³ voxels) of individual worm cells.

---

## 🎯 Pretrained Encoder

You can load a pretrained encoder from a checkpoint (e.g., from [this checkpoint directory](https://drive.google.com/drive/folders/1BwaG9Z8-Gz_TkIbIvmxYomcB4i7q35Mf?usp=sharing)) using:

```python
model.load_state_dict(encoder_state, strict=False)
```

Checkpoints are named to indicate architecture and parameters, e.g.:

* `RBPNI_32_4Layers_CD`: Uses `ResNetBlock`, `cge` layer order, 4 layers with `[32,64,128,256]`.

---

## 🧮 Loss Function

The training objective uses a differentiable Hungarian loss based on matching latent features between paired worm samples. It includes:

* Weighted contributions from multiple encoder layers.
* A cosine similarity penalty to encourage global alignment consistency.

* `DifferentiableHungarianLoss` for pairwise assignments
* `MultiLayerHungarianLoss` for multi-resolution feature alignment

```python
loss_criterion = get_loss_criterion(name='MultiLayerHungarianLoss', layer_weights=[0.3, 0.7], penalty_weight=0.5, penalty_scale=10.0)
```
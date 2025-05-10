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
---
# Data Preprocessing

This folder provides a full pipeline for preparing, refining, and visualizing **3D voxel cubes of cellular data** extracted from segmented microscopy volumes of *C. elegans* (worms). It supports raw data parsing, mask processing, voxel cube generation, merging, and visualization.

---

## 🔍 Purpose

**WormCube** is designed to:

* Convert raw microscopy data into uniform 3D voxel cubes.
* Apply biologically relevant masks and refine them using soft boundary masking.
* Organize per-worm data into consistent arrays for deep learning models.
* Enable efficient visual inspection of processed cubes.

This pipeline supports **graph neural networks (GNNs)**, voxel classification, and other 3D learning tasks.

---

## 📁 Pipeline Overview

---

### **Step 0: `Step0.py`**

Extracts 3D cubes (e.g., 32×32×32) centered on annotated neuron coordinates. Combines raw and masked voxel channels.

* **Input**: `.raw` volumes + `.txt` annotation files
* **Output**: 2-channel `.npy` cubes
* **Key features**: Robust padding, coordinate alignment, optional `.raw` or `.npy` output
* Key Reference: [Folder:Processed for GNN](https://github.com/ChengYuChuan/MA_LAP?tab=readme-ov-file#-folder-processed-for-gnn)

---

### **Step 1: `Step1.py`**

Adds a **soft boundary mask** (third channel) using a sigmoid-transformed distance from labeled regions.

* **Input**: 2-channel cubes
* **Output**: 3-channel cubes (`[raw, binary_mask, soft_mask]`)
* **Method**: Applies a decaying weight outside label boundaries

---

### **Step 2: `Step2.py`**

Extracts only the soft-masked voxel (`3rd channel`) for direct use in model training.

* **Input**: 3-channel cubes
* **Output**: 1-channel soft-masked `.npy` cubes
* **Purpose**: Simplifies dataset for downstream models

---

### **Merge: `Merge.py`**

Merges all individual cubes per worm into a single 5D array for efficient batched access.

* **Shape**: `(558, 1, 32, 32, 32)` per worm
* **Output**: One `.npy` file per worm
* **Note**: Automatically fills missing cells with zeroed cubes

---

### **Validation: `TestSize.py`**

Scans the merged dataset and identifies any `.npy` files with incorrect shapes or loading issues.

* **Output**: List of problematic files, if any

---

### **Visualization: `VisMergedData.py`**

Visualizes 3 random cells from a random worm, displaying:

* XY, XZ, YZ mid-slices

* Interactive colormap support (`viridis`, `inferno`, etc.)

* **Output**: Interactive matplotlib plot

---

## 📸 Sample Output

| View     | Example                                                  |
| -------- | -------------------------------------------------------- |
| XY slice | ![xy](https://via.placeholder.com/200x200?text=XY+slice) |
| XZ slice | ![xz](https://via.placeholder.com/200x200?text=XZ+slice) |
| YZ slice | ![yz](https://via.placeholder.com/200x200?text=YZ+slice) |

---

## 🛠️ Requirements

```bash
pip install numpy scipy matplotlib tqdm v3dpy torch plotly pandas
```

---

## ✅ Recommended Workflow

```bash
# 1. Extract 2-channel cubes from raw + mask volumes
python Step0.py

# 2. Add soft boundary as 3rd channel
python Step1.py

# 3. Extract soft-masked cube for training
python Step2.py

# 4. Merge all cubes per worm
python Merge.py

# 5. Check for invalid files
python TestSize.py

# 6. Visualize samples
python VisMergedData.py
```

---

## 📦 Output Directory Structure

```
├── 2ChannelMaskedCube32/      # After Step0
├── 3ChannelMaskedCube32/      # After Step1
├── MaskedCube32/              # After Step2
├── MergedCubes32/             # After Merge
├── skipped_files.txt          # From Step1
├── processed_files.txt        # From Step1
```

---

Great — based on the uploaded file `worm_200.txt`, I’ll update the **README** content to include a clear explanation of the **`Processed for GNN/`** folder and the structure of its files.

---

# Some Explanation of other Reference files or folders

## 🧬 `Coordinates of cube.zip`

### 🔹 Folder: `Processed for GNN/`

This directory contains `.txt` files with **annotated neuron coordinates** for each worm, used as inputs in `Step0.py`.

### 📄 File Format: `worm_###.txt`

Each file corresponds to a single worm and contains tab-separated information about its identified neurons. The key columns include:

| Column         | Description                                                |
| -------------- | ---------------------------------------------------------- |
| `label_number` | Unique numeric ID for the neuron                           |
| `label_name`   | Neuron name (e.g., `ADAL`, `PHAR`, `RIGL`, etc.)           |
| `x`, `y`, `z`  | 3D coordinates of the neuron center (float precision)      |
| `Aligned_No`   | A unique sequential index used as the identifier in output |

### 🔍 Example Row:

```
195	ADAL	1242.000000	84.000000	33.000000	1
```

* `ADAL` is the neuron label.
* The position is at `(x=1242, y=84, z=33)`.
* This is the first aligned neuron in the list.

### 🧪 Usage in Step0

In `Step0.py`, these files are loaded via:

```python
txt_path = os.path.join(txt_dir, txt_file)
coordinates = read_all_coordinates(txt_path)
```

Each row provides a target for extracting a **32×32×32 cube** around the neuron center. The extracted cube is named as:

```
<worm_name>_cube_<Aligned_No>.npy
```

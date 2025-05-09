# MA_LAP

---

## 🔬 Overview of `LAP_Code`

The `LAP_Code` folder implements a deep learning pipeline for solving the **Linear Assignment Problem (LAP)** between worm cell embeddings, using a 3D convolutional neural network with a latent space alignment strategy.

This system is built to perform cell-wise alignment between two worm volumes represented as cube-wise features, leveraging a pre-trained **AutoEncoder's encoder** and fine-tuning it through **Hungarian loss optimization**. The primary objective is to learn a representation that can robustly match corresponding cells across samples.

---

## 📁 Folder Structure Summary

* **`train.py`**: Main entry script for training. It sets up data loading, model construction (with encoder architecture), loss functions, optimizer, and training loop using `LAPNetTrainer`.
* **`CubeDataset.py`**: Defines the `CubeDataset` class for loading `.npy` files (each representing worm cell cubes). Includes data shuffling and split between training and validation.
* **`buildingblocks.py`**: Core network components, including `ResBlockPNI`, `DoubleConv`, `Encoder`, and helper functions like `create_encoders`.
* **`LAPNetTrainer.py`**: Encapsulates the training and validation logic. Handles logging, learning rate scheduling, dynamic loss adjustment, early stopping, and visualization through TensorBoard.
* **`loss.py`**: Implements differentiable Hungarian-based losses including:

  * `DifferentiableHungarianLoss` for pairwise assignments
  * `MultiLayerHungarianLoss` for multi-resolution feature alignment
* **`transform.py`**: Data augmentation and normalization routines, including Z-score and Min-Max standardization.
* **`utils.py`**: Logging, checkpoint handling, optimizer creation, and various training utilities.

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

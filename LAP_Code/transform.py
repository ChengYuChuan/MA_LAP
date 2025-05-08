#---Transform---
from scipy.ndimage import rotate, map_coordinates, gaussian_filter, convolve
import torch
import numpy as np


class Standardize:
    """
        A standardization transformer for 3D data, supporting optional Z-score and/or Min-Max normalization.

        This class is designed for preprocessing 3D voxel-based data, such as cell images or medical volumes.
        It supports:
        - Per-sample (per-cube) automatic Z-score normalization
        - Per-sample Min-Max normalization to scale data into [0, 1]
        - Optional fixed mean and std for consistent dataset-level normalization

        Args:
            z_score (bool): Whether to apply Z-score normalization. Default is True.
            min_max (bool): Whether to apply Min-Max normalization. Default is True.
            mean (float or None): If specified, uses this value as the mean. Otherwise, computed from the input.
            std (float or None): If specified, uses this value as the std. Otherwise, computed from the input.
            eps (float): Small value added to avoid division by zero. Default is 1e-10.

        Returns:
            np.ndarray: A standardized numpy array with the same shape as the input.

        Examples:
            >>> transform = Standardize(z_score=True, min_max=False)
            >>> output = transform(input_cube)

            >>> transform = Standardize(z_score=False, min_max=True)
            >>> output = transform(input_cube)

            >>> transform = Standardize(z_score=True, min_max=True, mean=0.5, std=0.1)
            >>> output = transform(input_cube)
        """
    def __init__(self, z_score=True, min_max=True, mean=None, std=None, eps=1e-10, **kwargs):
        self.z_score = z_score
        self.min_max = min_max
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, m):
        if isinstance(m, torch.Tensor):
            m = m.numpy()

        if self.z_score:
            mean = self.mean if self.mean is not None else np.mean(m)
            std = self.std if self.std is not None else np.std(m)
            m = (m - mean) / np.clip(std, a_min=self.eps, a_max=None)

        if self.min_max:
            min_val = m.min()
            max_val = m.max()
            m = (m - min_val) / (max_val - min_val + self.eps)

        return m

class RandomFlip:
    """
    Randomly flips the image across the given axes. Image can be either 3D (DxHxW) or 4D (CxDxHxW).

    When creating make sure that the provided RandomStates are consistent between raw and labeled datasets,
    otherwise the models won't converge.
    """

    def __init__(self, random_state, axis_prob=0.5, **kwargs):
        assert random_state is not None, 'RandomState cannot be None'
        self.random_state = random_state
        self.axes = (0, 1, 2)
        self.axis_prob = axis_prob

    def __call__(self, m):
        assert m.ndim in [3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'

        for axis in self.axes:
            if self.random_state.uniform() > self.axis_prob:
                if m.ndim == 3:
                    m = np.flip(m, axis)
                else:
                    channels = [np.flip(m[c], axis) for c in range(m.shape[0])]
                    m = np.stack(channels, axis=0)

        return m

class RandomRotate90:
    """
    Rotate an array by 90 degrees around a randomly chosen plane. Image can be either 3D (DxHxW) or 4D (CxDxHxW).

    When creating make sure that the provided RandomStates are consistent between raw and labeled datasets,
    otherwise the models won't converge.

    IMPORTANT: assumes DHW axis order (that's why rotation is performed across (1,2) axis)
    """

    def __init__(self, random_state, **kwargs):
        self.random_state = random_state
        # always rotate around z-axis
        self.axis = (1, 2)

    def __call__(self, m):
        assert m.ndim in [3, 4], 'Supports only 3D (DxHxW) or 4D (CxDxHxW) images'

        # pick number of rotations at random
        k = self.random_state.randint(0, 4)
        # rotate k times around a given plane
        if m.ndim == 3:
            m = np.rot90(m, k, self.axis)
        else:
            channels = [np.rot90(m[c], k, self.axis) for c in range(m.shape[0])]
            m = np.stack(channels, axis=0)

        return m

class RandomRotate:
    """
    Rotate an array by a random degrees from taken from (-angle_spectrum, angle_spectrum) interval.
    Rotation axis is picked at random from the list of provided axes.
    """

    def __init__(self, random_state, angle_spectrum=30, axes=None, mode='reflect', order=0, **kwargs):
        if axes is None:
            axes = [(1, 0), (2, 1), (2, 0)]
        else:
            assert isinstance(axes, list) and len(axes) > 0

        self.random_state = random_state
        self.angle_spectrum = angle_spectrum
        self.axes = axes
        self.mode = mode
        self.order = order

    def __call__(self, m):
        axis = self.axes[self.random_state.randint(len(self.axes))]
        angle = self.random_state.randint(-self.angle_spectrum, self.angle_spectrum)

        if m.ndim == 3:
            m = rotate(m, angle, axes=axis, reshape=False, order=self.order, mode=self.mode, cval=-1)
        else:
            channels = [rotate(m[c], angle, axes=axis, reshape=False, order=self.order, mode=self.mode, cval=-1) for c
                        in range(m.shape[0])]
            m = np.stack(channels, axis=0)

        return m

class ToTensor:
    def __init__(self, expand_dims=True, dtype=np.float32, normalize=False, **kwargs):
        self.expand_dims = expand_dims
        self.dtype = dtype
        self.normalize = normalize

    def __call__(self, m):
        if self.expand_dims and m.ndim == 3:
            m = np.expand_dims(m, axis=0)
        return torch.from_numpy(m.astype(dtype=self.dtype))
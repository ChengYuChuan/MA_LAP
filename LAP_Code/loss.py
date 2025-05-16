import torch
from torch.nn import MSELoss, SmoothL1Loss, L1Loss
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from scipy.optimize import linear_sum_assignment
from math import exp
from utils import get_logger
import math
import warnings

logger = get_logger('Loss')

def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)

class HammingLoss(torch.nn.Module):
    def forward(self, suggested, target):
        errors = suggested * (1.0 - target) + (1.0 - suggested) * target
        return errors.mean(dim=0).sum()

class LAPSolver(torch.autograd.Function):

    @staticmethod
    def forward(ctx, unaries: torch.Tensor, params: dict):
        device = unaries.device
        labelling = torch.zeros_like(unaries)
        unaries_np = unaries.cpu().detach().numpy()
        row_ind, col_ind = linear_sum_assignment(unaries_np)
        labelling[row_ind, col_ind] = 1.

        ctx.labels = labelling
        ctx.col_labels = col_ind
        ctx.params = params
        ctx.unaries = unaries
        ctx.device = device
        return labelling.to(device)

    @staticmethod
    def backward(ctx, unary_gradients: torch.Tensor):
        assert ctx.unaries.shape == unary_gradients.shape

        lambda_val = ctx.params.get("lambda", 15)
        epsilon_val = 1e-6

        unaries = ctx.unaries
        device = unary_gradients.device

        # w′ = w + λ ∇L/∇y
        unaries_prime = unaries + lambda_val * unary_gradients
        unaries_prime_np = unaries_prime.detach().cpu().numpy()

        # yλ = Solver(w′)
        bwd_labels = torch.zeros_like(unaries)
        row_ind, col_ind = linear_sum_assignment(unaries_prime_np)
        bwd_labels[row_ind, col_ind] = 1.

        forward_labels = ctx.labels

        # ∇fλ(w) = −(ŷ − yλ) / λ
        unary_grad_bwd = -(forward_labels - bwd_labels) / (lambda_val + epsilon_val)

        return unary_grad_bwd.to(ctx.device), None

def compute_distance_matrix(A_flat, B_flat, distance_type="MSE"):
    if distance_type == "L1":
        # Uses torch.cdist, which is generally more memory-efficient for L1
        return torch.cdist(A_flat, B_flat, p=1)
    elif distance_type == "L2":
        # Uses torch.cdist, which is generally more memory-efficient for L2
         return torch.cdist(A_flat, B_flat, p=2)
    elif distance_type == "MSE":
        # Optimized MSE calculation to avoid large intermediate tensor
        # ||a - b||^2 = ||a||^2 - 2a·b + ||b||^2
        # MSE = ||a - b||^2 / latent_dim
        A_sq = torch.sum(A_flat**2, dim=1, keepdim=True) # Shape: (num_cells, 1)
        B_sq = torch.sum(B_flat**2, dim=1, keepdim=True) # Shape: (num_cells, 1)
        AB = torch.matmul(A_flat, B_flat.transpose(0, 1)) # Shape: (num_cells, num_cells)

        # Expand A_sq and B_sq for broadcasting
        A_sq = A_sq.expand_as(AB) # Shape: (num_cells, num_cells)
        B_sq = B_sq.transpose(0, 1).expand_as(AB) # Shape: (num_cells, num_cells)

        distance_sq = A_sq - 2 * AB + B_sq # Shape: (num_cells, num_cells)
        # Ensure non-negativity due to floating point inaccuracies
        distance_sq = torch.clamp(distance_sq, min=0)

        latent_dim = A_flat.shape[1]
        mse_matrix = distance_sq / latent_dim # Shape: (num_cells, num_cells)
        return mse_matrix

    else:
        raise ValueError(f"Unsupported distance type: {distance_type}")


class DifferentiableHungarianLoss(nn.Module):
    def __init__(self, distance_type="MSE", lambda_val=20):
        super().__init__()
        self.distance_type = distance_type
        self.lambda_val = lambda_val

    def forward(self, latent, inv_perm_A=None, inv_perm_B=None):
        assert latent.shape[0] == 2, "Latent input must be shape (2, N, ...)"
        num_cells = latent.shape[1]
        latent_dim = latent.shape[2:].numel()

        latent_A = latent[0]
        latent_B = latent[1]

        latent_A = latent_A.view(num_cells, latent_dim)
        latent_B = latent_B.view(num_cells, latent_dim)

        cost_matrix = compute_distance_matrix(latent_A, latent_B, self.distance_type)

        params = {"lambda": self.lambda_val}
        predicted_matching = LAPSolver.apply(cost_matrix, params)

        ideal_matching = torch.zeros_like(predicted_matching)
        ideal_matching[inv_perm_A, inv_perm_B] = 1.0

        loss = HammingLoss()(predicted_matching, ideal_matching)

        col_ind = predicted_matching.argmax(dim=1).detach().cpu().numpy()
        row_ind = np.arange(num_cells)

        return loss, (row_ind, col_ind)

class MultiLayerHungarianLoss(nn.Module):
    def __init__(self, layer_weights, distance_type="MSE", lambda_val=20):
        super().__init__()
        self.layer_weights = layer_weights
        self.distance_type = distance_type
        self.lambda_val = lambda_val

    def forward(self, multi_layer_latents, inv_perm_A=None, inv_perm_B=None):
        assert len(multi_layer_latents) == len(self.layer_weights), \
            "The number of latent layers and weights must match"

        num_cells = multi_layer_latents[0].shape[1]
        latent_dim = multi_layer_latents[0].shape[2:].numel()
        device = multi_layer_latents[0].device

        total_loss = 0
        combined_cost_matrix = torch.zeros((num_cells, num_cells), device=device)

        params = {"lambda": self.lambda_val}
        for weight, layer_latent in zip(self.layer_weights, multi_layer_latents):
            # Calculate latent_dim specific to this layer
            current_latent_dim = layer_latent.shape[2:].numel()

            latent_A_layer = layer_latent[0].view(num_cells, current_latent_dim)
            latent_B_layer = layer_latent[1].view(num_cells, current_latent_dim)

            # It will only work when
            cost = compute_distance_matrix(latent_A_layer, latent_B_layer, self.distance_type)
            combined_cost_matrix += weight * cost

            # Per-layer individual loss (still using Hungarian matching)
            predicted_matching = LAPSolver.apply(cost, params)
            ideal_matching = torch.zeros_like(predicted_matching)
            ideal_matching[inv_perm_A, inv_perm_B] = 1.0

            loss = HammingLoss()(predicted_matching, ideal_matching)
            total_loss += weight * loss

        # Final prediction using the combined cost matrix
        final_predicted_matching = LAPSolver.apply(combined_cost_matrix, params)
        col_ind = final_predicted_matching.argmax(dim=1).detach().cpu().numpy()
        row_ind = np.arange(num_cells)

        return total_loss, (row_ind, col_ind)

def get_loss_criterion(name, weight=None, ignore_index=None, skip_last_target=False, pos_weight=None, window_size=7, alpha=0.2, **loss_kwargs):

    """
    Returns the loss function based on provided parameters.
    :param name: (str) Name of the loss function.
    :param weight: (list or tensor, optional) Class weights for the loss.
    :param ignore_index: (int, optional) Index to ignore in loss calculation.
    :param skip_last_target: (bool) Whether to skip the last target channel.
    :param pos_weight: (tensor, optional) Positive class weight (for BCE-based losses).
    :param loss_kwargs: (dict) Additional keyword arguments for loss functions.
    :return: An instance of the loss function.
    """

    logger.info(f"Creating loss function: {name}")

    if weight is not None:
        weight = torch.tensor(weight).float()
        logger.info(f"Using class weights: {weight}")

    if pos_weight is not None:
        pos_weight = torch.tensor(pos_weight)

    loss = _create_loss(
        name, weight, ignore_index, pos_weight,
        alpha=alpha, window_size=window_size,
        **loss_kwargs
    )

    if torch.cuda.is_available():
        loss = loss.cuda()

    return loss

def _create_loss(name, weight, ignore_index, pos_weight, **loss_kwargs):
    logger.info(f"Creating loss function: {name}")

    if name == 'DifferentiableHungarianLoss':
        distance_type = loss_kwargs.get("distance_type", "MSE")
        lambda_val = loss_kwargs.get("lambda_val", 20)
        return DifferentiableHungarianLoss(distance_type=distance_type, lambda_val=lambda_val)

    elif name == 'MultiLayerHungarianLoss':
        layer_weights = loss_kwargs.get("layer_weights", [0.5, 0.5])
        distance_type = loss_kwargs.get("distance_type", "MSE")
        lambda_val = loss_kwargs.get("lambda_val", 20)
        return MultiLayerHungarianLoss(
            layer_weights=layer_weights,
            distance_type=distance_type,
            lambda_val=lambda_val
        )
    else:
        raise RuntimeError(f"Unsupported loss function: '{name}'")
import torch
from torch.nn import MSELoss, SmoothL1Loss, L1Loss
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from scipy.optimize import linear_sum_assignment
from LAP_Code.utils import get_logger

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

def compute_per_channel_dice(input, target, epsilon=1e-6, weight=None):
    """
    Computes DiceCoefficient as defined in https://arxiv.org/abs/1606.04797 given  a multi channel input and target.
    Assumes the input is a normalized probability, e.g. a result of Sigmoid or Softmax function.

    Args:
         input (torch.Tensor): NxCxSpatial input tensor
         target (torch.Tensor): NxCxSpatial target tensor
         epsilon (float): prevents division by zero
         weight (torch.Tensor): Cx1 tensor of weight per channel/class
    """

    # input and target shapes must match
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    input = flatten(input)
    target = flatten(target)
    target = target.float()

    # compute per channel Dice Coefficient
    intersect = (input * target).sum(-1)
    if weight is not None:
        intersect = weight * intersect

    # here we can use standard dice (input + target).sum(-1) or extension (see V-Net) (input^2 + target^2).sum(-1)
    denominator = (input * input).sum(-1) + (target * target).sum(-1)
    return 2 * (intersect / denominator.clamp(min=epsilon))


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


class DifferentiableHungarianLoss(nn.Module):
    def __init__(self, distance_type="L2", lambda_val=20):
        super().__init__()
        self.distance_type = distance_type
        self.lambda_val = lambda_val

    def compute_distance_matrix(self, A_flat, B_flat):
        if self.distance_type == "L1":
            return torch.cdist(A_flat, B_flat, p=1)
        elif self.distance_type == "L2":
            return torch.cdist(A_flat, B_flat, p=2)
        elif self.distance_type == "MSE":
            return ((A_flat.unsqueeze(1) - B_flat.unsqueeze(0)) ** 2).mean(dim=2)

    def forward(self, latent, inv_perm_A=None, inv_perm_B=None):
        assert latent.shape[0] == 2, "Latent input must be shape (2, N, ...)"
        num_cells = latent.shape[1]
        latent_dim = latent.shape[2:].numel()

        latent_A = latent[0]
        latent_B = latent[1]

        # ✅ 還原 permutation
        if inv_perm_A is not None:
            latent_A = latent_A[inv_perm_A]
        if inv_perm_B is not None:
            latent_B = latent_B[inv_perm_B]

        latent_A = latent_A.view(num_cells, latent_dim)
        latent_B = latent_B.view(num_cells, latent_dim)

        cost_matrix = self.compute_distance_matrix(latent_A, latent_B)

        params = {"lambda": self.lambda_val}
        predicted_matching = LAPSolver.apply(cost_matrix, params)

        identity = torch.eye(num_cells, device=latent.device)

        ideal_cost = torch.sum(identity * cost_matrix)
        predicted_cost = torch.sum(predicted_matching * cost_matrix)
        loss = abs(predicted_cost - ideal_cost) / num_cells

        col_ind = predicted_matching.argmax(dim=1).detach().cpu().numpy()
        row_ind = np.arange(num_cells)

        return loss, (row_ind, col_ind)

class MultiLayerHungarianLoss(nn.Module):
    """
    Multi-layer extension of the differentiable Hungarian loss.
    Aggregates weighted losses from multiple layers of latent features and
    optionally applies a cosine similarity penalty between the predicted
    and identity (ground truth) matching matrices.
    """
    def __init__(self, layer_weights, base_loss_fn=None,
                 penalty_weight=0.1, penalty_mode="global",
                 total_loss_weight=1.0):
        super().__init__()
        self.layer_weights = layer_weights
        self.base_loss_fn = base_loss_fn or DifferentiableHungarianLoss()
        self.penalty_weight = penalty_weight
        self.penalty_mode = penalty_mode  # "none", "per_layer", "global"
        self.total_loss_weight = total_loss_weight  # 新增: 控制total_loss的重要性

    def forward(self, multi_layer_latents, inv_perm_A=None, inv_perm_B=None):
        """
        Calculates the weighted sum of Hungarian losses over multiple layers of
        latent features. If enabled, adds cosine similarity penalty between
        predicted matching and identity matrix.
        """
        total_loss = 0
        similarity_penalty = 0
        match_info = None

        for weight, layer_latent in zip(self.layer_weights, multi_layer_latents):
            loss, info = self.base_loss_fn(layer_latent, inv_perm_A=inv_perm_A, inv_perm_B=inv_perm_B)
            total_loss += weight * loss
            match_info = info  # 只記最後一層的配對資訊

            if self.penalty_mode == "per_layer":
                row_ind, col_ind = info
                num_cells = len(row_ind)
                device = layer_latent.device

                predicted_matching = torch.zeros((num_cells, num_cells), device=device)
                predicted_matching[row_ind, col_ind] = 1.0
                identity = torch.eye(num_cells, device=device)

                cosine_sim = F.cosine_similarity(predicted_matching.flatten(), identity.flatten(), dim=0)
                similarity_penalty += (1.0 - cosine_sim)

        if self.penalty_mode == "global" and match_info is not None:
            row_ind, col_ind = match_info
            num_cells = len(row_ind)
            device = multi_layer_latents[0].device

            predicted_matching = torch.zeros((num_cells, num_cells), device=device)
            predicted_matching[row_ind, col_ind] = 1.0
            identity = torch.eye(num_cells, device=device)

            cosine_sim = F.cosine_similarity(predicted_matching.flatten(), identity.flatten(), dim=0)
            similarity_penalty = (1.0 - cosine_sim)

        if self.penalty_mode in ["per_layer", "global"]:
            # 加入 dynamic weighting
            final_loss = self.total_loss_weight * total_loss + self.penalty_weight * similarity_penalty
        else:
            final_loss = self.total_loss_weight * total_loss

        return final_loss, match_info

    # 新增：允許在train loop中動態調整loss weight
    def set_total_loss_weight(self, new_weight):
        """
        Dynamically adjusts the weight of the total loss term (matching).
        """
        self.total_loss_weight = new_weight

    def set_penalty_weight(self, new_weight):
        """
        Dynamically adjusts the weight of the similarity penalty term.
        """
        self.penalty_weight = new_weight



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

    # 移除 loss_kwargs 內的 return_msssim，避免重複
    return_msssim = loss_kwargs.pop("return_msssim", False)

    loss = _create_loss(
        name, weight, ignore_index, pos_weight,
        alpha=alpha, window_size=window_size,
        **loss_kwargs
    )

    if torch.cuda.is_available():
        loss = loss.cuda()

    return loss

def _create_loss(name, weight, ignore_index, pos_weight, alpha=0.2, window_size=5, return_msssim=False, **loss_kwargs):
    logger.info(f"Creating loss function: {name}")

    if name == 'BCEWithLogitsLoss':
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif name == 'DifferentiableHungarianLoss':
        return DifferentiableHungarianLoss()
    elif name == 'MultiLayerHungarianLoss':
        layer_weights = loss_kwargs.get("layer_weights", [0.5, 0.5])
        base_loss = DifferentiableHungarianLoss()
        return MultiLayerHungarianLoss(layer_weights=layer_weights, base_loss_fn=base_loss)
    else:
        raise RuntimeError(f"Unsupported loss function: '{name}'")
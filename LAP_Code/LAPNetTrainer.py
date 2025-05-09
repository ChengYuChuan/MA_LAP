from sys import prefix

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau

from CubeDataset import get_train_loaders

from utils import get_logger, load_checkpoint, create_optimizer, save_checkpoint, RunningAverage
from utils import _split_and_move_to_gpu, TensorboardFormatter
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import os
import numpy as np
import torch

import matplotlib.pyplot as plt
from io import BytesIO
import torchvision.transforms as T
from PIL import Image

from torch.cuda.amp import autocast, GradScaler

logger = get_logger('LAPNetTrainer')

class EarlyStopping:
    """Simple early stopping based on validation score."""
    def __init__(self, patience=10, higher_is_better=True):
        self.patience = patience
        self.higher_is_better = higher_is_better
        self.best_score = None
        self.bad_epochs = 0

    def step(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
            return False

        improvement = (current_score > self.best_score) if self.higher_is_better else (current_score < self.best_score)

        if improvement:
            self.best_score = current_score
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1

        return self.bad_epochs >= self.patience

class LAPNetTrainer:
    """trainer.

    Args:
        model (Unet3D): UNet 3D model to be trained
        optimizer (nn.optim.Optimizer): optimizer used for training
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): learning rate scheduler
            WARN: bear in mind that lr_scheduler.step() is invoked after every validation step
            (i.e. validate_after_iters) not after every epoch. So e.g. if one uses StepLR with step_size=30
            the learning rate will be adjusted after every 30 * validate_after_iters iterations.
        loss_criterion (callable): loss function
        eval_criterion (callable): used to compute training/validation metric (such as Dice, IoU, AP or Rand score)
            saving the best checkpoint is based on the result of this function on the validation set
        loaders (dict): 'train' and 'val' loaders
        checkpoint_dir (string): dir for saving checkpoints and tensorboard logs
        max_num_epochs (int): maximum number of epochs
        max_num_iterations (int): maximum number of iterations
        validate_after_iters (int): validate after that many iterations
        log_after_iters (int): number of iterations before logging to tensorboard
        validate_iters (int): number of validation iterations, if None validate
            on the whole validation set
        eval_score_higher_is_better (bool): if True higher eval scores are considered better
        best_eval_score (float): best validation score so far (higher better)
        num_iterations (int): useful when loading the model from the checkpoint
        num_epoch (int): useful when loading the model from the checkpoint
        tensorboard_formatter (callable): converts a given batch of input/output/target image to a series of images
            that can be displayed in tensorboard
        skip_train_validation (bool): if True eval_criterion is not evaluated on the training set (used when
            evaluation is expensive)
        resume (string): path to the checkpoint to be resumed
        pre_trained (string): path to the pre-trained model
        max_val_images (int): maximum number of images to log during validation
    """

    def __init__(self, model, optimizer, lr_scheduler, loss_criterion, eval_criterion, loaders, checkpoint_dir,
                 max_num_epochs, max_num_iterations, validate_after_iters=200, log_after_iters=100, validate_iters=None,
                 num_iterations=1, num_epoch=0, eval_score_higher_is_better=True, tensorboard_formatter=None,
                 skip_train_validation=False, resume=None, pre_trained=None, max_val_images=10, **kwargs):

        self.model = model
        self.optimizer = optimizer
        self.scheduler = lr_scheduler
        self.loss_criterion = loss_criterion
        self.eval_criterion = eval_criterion
        self.loaders = loaders
        self.checkpoint_dir = checkpoint_dir
        self.max_num_epochs = max_num_epochs
        self.max_num_iterations = max_num_iterations
        self.validate_after_iters = validate_after_iters
        self.log_after_iters = log_after_iters
        self.validate_iters = validate_iters
        self.num_iterations = num_iterations
        self.num_epochs = num_epoch
        self.skip_train_validation = skip_train_validation
        self.tensorboard_formatter = tensorboard_formatter
        self.max_val_images = max_val_images
        self.scaler = GradScaler()
        self.eval_score_higher_is_better = eval_score_higher_is_better

        logger.info(model)
        logger.info(f"eval_score_higher_is_better: {eval_score_higher_is_better}")

        self.writer = SummaryWriter(
            log_dir=os.path.join(
                checkpoint_dir, 'logs',
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            )
        )

        if eval_score_higher_is_better:
            self.best_eval_score = float('-inf')
        else:
            self.best_eval_score = float('inf')

        if resume:
            logger.info(f"Resuming from checkpoint '{resume}'")
            state = load_checkpoint(resume, self.model, self.optimizer)
            self.best_eval_score = state['best_eval_score']
            self.num_iterations = state['num_iterations']
            self.num_epochs = state['num_epochs']
            self.checkpoint_dir = os.path.split(resume)[0]
        elif pre_trained:
            logger.info(f"Loading pre-trained model from '{pre_trained}'")
            load_checkpoint(pre_trained, self.model, None)
            if not self.checkpoint_dir:
                self.checkpoint_dir = os.path.split(pre_trained)[0]

        # Early stopping
        self.early_stopper = EarlyStopping(patience=20, higher_is_better=eval_score_higher_is_better)

        # Dynamic loss scheduling config
        self.total_loss_initial_weight = 1.0
        self.total_loss_final_weight = 0.3

        self.penalty_initial_weight = 0.2
        self.penalty_final_weight = 0.5

        self.penalty_initial_scale = 5.0
        self.penalty_final_scale = 20.0

        self.decay_start_iter = 22
        self.decay_end_iter = int(self.max_num_iterations * 0.8)

    def fit(self):
        for _ in range(self.num_epochs, self.max_num_epochs):
            should_terminate = self.train()
            if should_terminate:
                logger.info('Early stopping triggered or max iterations reached.')
                return
            self.num_epochs += 1

    def train(self):
        train_losses = RunningAverage()
        train_accuracies = RunningAverage()
        self.model.train()

        for batch in self.loaders['train']:
            if self.num_iterations > self.max_num_iterations:
                return True
            logger.info(f'Training iteration [{self.num_iterations}/{self.max_num_iterations}]. '
                        f'Epoch [{self.num_epochs}/{self.max_num_epochs - 1}]')

            self._adjust_loss_weights()  # 動態調整 total_loss_weight / penalty_weight

            (cubes, paths, perms, inv_perms) = batch
            cubes = cubes.to(next(self.model.parameters()).device)
            inv_perm_A, inv_perm_B = inv_perms[0], inv_perms[1]

            with autocast():
                multi_feats = self.model(cubes, return_layers=[-2, -1])
                loss, (row_ind, col_ind) = self.loss_criterion(multi_feats, inv_perm_A=inv_perm_A, inv_perm_B=inv_perm_B)

            accuracy = np.mean((row_ind == col_ind).astype(np.float32))

            if self.num_iterations % 10 == 0:
                worm_A = os.path.basename(paths[0])
                worm_B = os.path.basename(paths[1])
                logger.info(f"Pairing: {worm_A} vs {worm_B} | Loss: {loss.item():.4f} | Accuracy: {accuracy:.4f}")

            self.writer.add_scalar("train/loss", loss.item(), self.num_iterations)
            self.writer.add_scalar("train/accuracy_identity_match", accuracy, self.num_iterations)

            # 混合精度 backward
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            train_losses.update(loss.item(), 1)
            train_accuracies.update(accuracy, 1)

            torch.cuda.empty_cache()

            # === 每validate_after_iters次做一次Validation
            if self.num_iterations % self.validate_after_iters == 0:
                self.model.eval()
                val_loss, val_accuracy = self.validate()
                self.model.train()

                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_loss)  # take loss as learning reference
                elif self.scheduler is not None:
                    self.scheduler.step()

                self._log_lr()

                is_best = self._is_best_eval_score(val_accuracy)
                self._save_checkpoint(is_best)

                if self.early_stopper.step(val_accuracy):  # take accuracy as early stopping threshold
                    logger.info('Early stopping condition met.')
                    return True

            # === 每log_after_iters次做一次 logging
            if self.num_iterations % self.log_after_iters == 0:
                logger.info(f"Training stats. Loss: {train_losses.avg:.4f} | Accuracy: {train_accuracies.avg:.4f}")
                self._log_stats('train', train_losses.avg, train_accuracies.avg)

                # Log matching matrix, cosine similarity, random pair prediction
                perm_A, perm_B = perms[0], perms[1]
                self._log_random_pair_predictions(cubes, row_ind, col_ind, inv_perm_A, inv_perm_B)

                predicted_matching = torch.zeros((len(row_ind), len(col_ind)), device=loss.device)
                predicted_matching[row_ind, col_ind] = 1.0
                self._log_matching_similarity(predicted_matching)
                self._log_matching_heatmap(predicted_matching)

            self.num_iterations += 1

        return False

    def validate(self):
        logger.info('Validating...')
        val_losses = RunningAverage()
        val_accuracies = RunningAverage()

        self.model.eval()
        with torch.no_grad():
            rs = np.random.RandomState(42)
            indices = (
                list(range(len(self.loaders['val'])))
                if len(self.loaders['val']) <= self.max_val_images
                else rs.choice(len(self.loaders['val']), size=self.max_val_images, replace=False)
            )

            for i, (cubes, paths, perms, inv_perms) in enumerate(tqdm(self.loaders['val'])):
                cubes = cubes.to(next(self.model.parameters()).device)
                inv_perm_A, inv_perm_B = inv_perms[0], inv_perms[1]

                with autocast():
                    multi_feats = self.model(cubes, return_layers=[-2, -1])
                    loss, (row_ind, col_ind) = self.loss_criterion(
                        multi_feats, inv_perm_A=inv_perm_A, inv_perm_B=inv_perm_B
                    )

                val_losses.update(loss.item(), 1)
                accuracy = np.mean((row_ind == col_ind).astype(np.float32))
                val_accuracies.update(accuracy, 1)

                if i == 0:
                    predicted_matching = torch.zeros((len(row_ind), len(col_ind)), device=loss.device)
                    predicted_matching[row_ind, col_ind] = 1.0
                    self._log_matching_similarity(predicted_matching, tag_prefix="val/matching_similarity")
                    self._log_matching_heatmap(predicted_matching, title="Validation Matching Heatmap", tag_prefix="val/matching_heatmap")
                    self._log_random_pair_predictions(cubes, row_ind, col_ind, inv_perm_A, inv_perm_B)

                if self.validate_iters is not None and self.validate_iters <= i:
                    break

        logger.info(f"Validation finished. Loss: {val_losses.avg:.4f}. Accuracy: {val_accuracies.avg:.4f}")
        self._log_stats('val', val_losses.avg, val_accuracies.avg)

        return val_losses.avg, val_accuracies.avg

    def _adjust_loss_weights(self):
        progress = 0.0
        if self.num_iterations > self.decay_start_iter:
            progress = (self.num_iterations - self.decay_start_iter) / (self.decay_end_iter - self.decay_start_iter)
            progress = min(max(progress, 0.0), 1.0)

        # Dynamic Linear Adjusting total_loss and penalty weight
        total_loss_weight = self.total_loss_initial_weight * (1 - progress) + self.total_loss_final_weight * progress
        penalty_weight = self.penalty_initial_weight * (1 - progress) + self.penalty_final_weight * progress

        # penalty scale Dynamic Linear Adjustment
        penalty_scale = self.penalty_initial_scale * (1 - progress) + self.penalty_final_scale * progress

        self.loss_criterion.set_total_loss_weight(total_loss_weight)
        self.loss_criterion.set_penalty_weight(penalty_weight)
        self.loss_criterion.set_penalty_scale(penalty_scale)


    def _save_checkpoint(self, is_best):
        last_file_path = os.path.join(self.checkpoint_dir, 'last_checkpoint.pytorch')
        logger.info(f"Saving checkpoint to '{last_file_path}'")
        save_checkpoint({
            'num_epochs': self.num_epochs + 1,
            'num_iterations': self.num_iterations,
            'model_state_dict': self.model.state_dict(),
            'best_eval_score': self.best_eval_score,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, is_best, checkpoint_dir=self.checkpoint_dir)

    def _is_best_eval_score(self, eval_score):
        is_best = eval_score > self.best_eval_score if self.eval_score_higher_is_better else eval_score < self.best_eval_score
        if is_best:
            logger.info(f"Saving new best model. Validation score improved to {eval_score:.4f}")
            self.best_eval_score = eval_score
        return is_best

    def _log_stats(self, phase, loss_avg, accuracy_avg):
        self.writer.add_scalar(f'{phase}_loss_avg', loss_avg, self.num_iterations)
        self.writer.add_scalar(f'{phase}_accuracy_avg', accuracy_avg, self.num_iterations)

    def _log_lr(self):
        lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('learning_rate', lr, self.num_iterations)

    def _log_params(self):
        logger.info('Logging model parameters and gradients')
        for name, value in self.model.named_parameters():
            self.writer.add_histogram(name, value.data.cpu().numpy(), self.num_iterations)
            if value.grad is not None:
                self.writer.add_histogram(name + '/grad', value.grad.data.cpu().numpy(), self.num_iterations)

    def _log_matching_similarity(self, predicted_matching, tag_prefix="train/matching_similarity"):
        device = predicted_matching.device
        num_cells = predicted_matching.shape[0]

        identity = torch.eye(num_cells, device=device)
        cosine_sim = F.cosine_similarity(
            predicted_matching.flatten(), identity.flatten(), dim=0
        ).item()

        self.writer.add_scalar(tag_prefix, cosine_sim, self.num_iterations)

    def _log_matching_heatmap(self, predicted_matching, title="Matching Heatmap", tag_prefix="train/matching_heatmap"):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(predicted_matching.detach().cpu().numpy(), cmap='viridis', interpolation='nearest')
        ax.set_title(title)
        ax.set_xlabel("Cell Index (Worm B)")
        ax.set_ylabel("Cell Index (Worm A)")
        plt.tight_layout()

        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        transform = T.ToTensor()
        image_tensor = transform(image)

        self.writer.add_image(tag_prefix, image_tensor, self.num_iterations)
        plt.close(fig)

    def _log_random_pair_predictions(
            self, cubes, row_ind, col_ind,
            inv_perm_A, inv_perm_B,
            tag_prefix="train/pair_predictions"):

        batch_size, num_cells, _, _, _, _ = cubes.shape
        assert batch_size == 2, "Expect batch size = 2 for worm A and worm B"

        # === Step 1: 還原 A / B 細胞的原始順序 ===
        cubes_A = cubes[0][inv_perm_A]  # shape: (558, 1, D, H, W)
        cubes_B = cubes[1][inv_perm_B]

        # === Step 2: 隨機取 5 組配對進行可視化 ===
        rs = np.random.RandomState(self.num_iterations)
        sample_indices = rs.choice(len(row_ind), size=5, replace=False)

        fig, axes = plt.subplots(5, 2, figsize=(6, 15))
        for ax_row, idx in zip(axes, sample_indices):
            i = row_ind[idx]  # Original A cells index
            j = col_ind[idx]  # Original B cells index

            is_correct = (i == j)  # identity match

            # === 畫出對應的切片影像 ===
            for k, (cube_tensor, cell_idx) in enumerate([(cubes_A, i), (cubes_B, j)]):
                img = cube_tensor[cell_idx, 0]  # NDHW → take channel 0
                middle_slice = img[img.shape[0] // 2, :, :].cpu().numpy()
                ax = ax_row[k]
                ax.imshow(middle_slice, cmap='gray')
                ax.axis('off')
                if k == 0:
                    ax.set_title(f'Worm A Cell {i}')
                else:
                    ax.set_title(f'Worm B Cell {j}\n{"Correct" if is_correct else "Wrong"}')

        plt.tight_layout()

        # === Log to TensorBoard ===
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        transform = T.ToTensor()
        image_tensor = transform(image)

        self.writer.add_image(tag_prefix, image_tensor, self.num_iterations)
        plt.close(fig)

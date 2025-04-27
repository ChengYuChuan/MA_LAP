from sys import prefix

import torch
import torch.nn as nn
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

logger = get_logger('LAPNetTrainer')


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
                 num_iterations=1, num_epoch=0, eval_score_higher_is_better=False, tensorboard_formatter=None,
                 skip_train_validation=False, resume=None, pre_trained=None, max_val_images=10, **kwargs):
        self.scaler = GradScaler()
        self.max_val_images = max_val_images
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
        self.eval_score_higher_is_better = eval_score_higher_is_better

        logger.info(model)
        logger.info(f'eval_score_higher_is_better: {eval_score_higher_is_better}')

        # initialize the best_eval_score
        if eval_score_higher_is_better:
            self.best_eval_score = float('-inf')
        else:
            self.best_eval_score = float('+inf')

        self.writer = SummaryWriter(
            log_dir=os.path.join(
                checkpoint_dir, 'logs',
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            )
        )

        assert tensorboard_formatter is not None, 'TensorboardFormatter must be provided'
        self.tensorboard_formatter = tensorboard_formatter

        self.num_iterations = num_iterations
        self.num_epochs = num_epoch
        self.skip_train_validation = skip_train_validation

        if resume is not None:
            logger.info(f"Loading checkpoint '{resume}'...")
            state = load_checkpoint(resume, self.model, self.optimizer)
            logger.info(
                f"Checkpoint loaded from '{resume}'. Epoch: {state['num_epochs']}.  Iteration: {state['num_iterations']}. "
                f"Best val score: {state['best_eval_score']}."
            )
            self.best_eval_score = state['best_eval_score']
            self.num_iterations = state['num_iterations']
            self.num_epochs = state['num_epochs']
            self.checkpoint_dir = os.path.split(resume)[0]
        elif pre_trained is not None:
            logger.info(f"Logging pre-trained model from '{pre_trained}'...")
            load_checkpoint(pre_trained, self.model, None)
            if not self.checkpoint_dir:
                self.checkpoint_dir = os.path.split(pre_trained)[0]

    def fit(self):
        for _ in range(self.num_epochs, self.max_num_epochs):
            # train for one epoch
            should_terminate = self.train()

            if should_terminate:
                logger.info('Stopping criterion is satisfied. Finishing training')
                return

            self.num_epochs += 1
        logger.info(f"Reached maximum number of epochs: {self.max_num_epochs}. Finishing training...")

    def train(self):
        """Trains the model for 1 epoch using mixed precision and permutation-aware matching."""
        train_losses = RunningAverage()
        train_accuracies = RunningAverage()
        self.model.train()

        for batch in self.loaders['train']:
            if self.num_iterations % 10 == 0:
                logger.info(f'Training iteration [{self.num_iterations}/{self.max_num_iterations}]. '
                            f'Epoch [{self.num_epochs}/{self.max_num_epochs - 1}]')

            (cubes, paths, perms, inv_perms) = batch
            cubes = cubes.to(next(self.model.parameters()).device)

            inv_perm_A, inv_perm_B = inv_perms[0], inv_perms[1]

            # === 混合精度 Forward Pass ===
            with autocast():
                multi_feats = self.model(cubes, return_layers=[-2, -1])
                loss, (row_ind, col_ind) = self.loss_criterion(
                    multi_feats, inv_perm_A=inv_perm_A, inv_perm_B=inv_perm_B
                )

            accuracy = np.mean((row_ind == col_ind).astype(np.float32))
            worm_A = os.path.basename(paths[0])
            worm_B = os.path.basename(paths[1])
            logger.info(f"Pairing: {worm_A} vs {worm_B} | Loss: {loss.item():.4f} | Accuracy: {accuracy:.4f}")
            if self.num_iterations % self.log_after_iters == 0:
                # Log matching heatmap to TensorBoard
                predicted_matching = torch.zeros((len(row_ind), len(col_ind)), device=loss.device)
                predicted_matching[row_ind, col_ind] = 1.0
                self._log_matching_heatmap(predicted_matching,
                                           title=f"Train Matching Heatmap @iter {self.num_iterations}",
                                           tag_prefix="train/matching_heatmap")
                self._log_matching_similarity(predicted_matching, tag_prefix="train/matching_similarity")

            self.writer.add_scalar("train/loss", loss.item(), self.num_iterations)
            self.writer.add_scalar("train/accuracy_identity_match", accuracy, self.num_iterations)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            train_losses.update(loss.item(), 1)
            train_accuracies.update(accuracy, 1)

            torch.cuda.empty_cache()

            if self.num_iterations % self.validate_after_iters == 0:
                self.model.eval()
                val_accuracy = self.validate()
                self.model.train()

                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_accuracy)
                elif self.scheduler is not None:
                    self.scheduler.step()

                self._log_lr()
                is_best = self._is_best_eval_score(val_accuracy)
                self._save_checkpoint(is_best)

            if self.num_iterations % self.log_after_iters == 0:
                logger.info(f'Training stats. Loss: {train_losses.avg:.4f} | Accuracy: {train_accuracies.avg:.4f}')
                self._log_stats('train', train_losses.avg, train_accuracies.avg)

            if self.should_stop():
                return True

            self.num_iterations += 1

        return False

    def should_stop(self):
        """
        Training will terminate if maximum number of iterations is exceeded or the learning rate drops below
        some predefined threshold (1e-6 in our case)
        """
        if self.max_num_iterations < self.num_iterations:
            logger.info(f'Maximum number of iterations {self.max_num_iterations} exceeded.')
            return True

        min_lr = 1e-6
        lr = self.optimizer.param_groups[0]['lr']
        if lr < min_lr:
            logger.info(f'Learning rate below the minimum {min_lr}.')
            return True

        return False

    def validate(self):
        logger.info('Validating...')

        val_losses = RunningAverage()
        val_accuracies = RunningAverage()

        self.model.eval()
        with torch.no_grad():
            rs = np.random.RandomState(42)
            if len(self.loaders['val']) <= self.max_val_images:
                indices = list(range(len(self.loaders['val'])))
            else:
                indices = rs.choice(len(self.loaders['val']), size=self.max_val_images, replace=False)

            for i, (cubes, paths, perms, inv_perms) in enumerate(tqdm(self.loaders['val'])):
                cubes = cubes.to(next(self.model.parameters()).device)

                inv_perm_A, inv_perm_B = inv_perms[0], inv_perms[1]

                # 混合精度推論
                with autocast():
                    multi_feats = self.model(cubes, return_layers=[-2, -1])
                    loss, (row_ind, col_ind) = self.loss_criterion(
                        multi_feats, inv_perm_A=inv_perm_A, inv_perm_B=inv_perm_B
                    )
                if i == 0:
                    predicted_matching = torch.zeros((len(row_ind), len(col_ind)), device=loss.device)
                    predicted_matching[row_ind, col_ind] = 1.0
                    self._log_matching_heatmap(predicted_matching,
                                               title=f"Validation Matching Heatmap @iter {self.num_iterations}",
                                               tag_prefix="val/matching_heatmap")
                    self._log_matching_similarity(predicted_matching, tag_prefix="val/matching_similarity")

                val_losses.update(loss.item(), 1)

                # ✅ 正確地計算 Validation Accuracy
                accuracy = np.mean((row_ind == col_ind).astype(np.float32))
                val_accuracies.update(accuracy, 1)

                if self.validate_iters is not None and self.validate_iters <= i:
                    break

        logger.info(f'Validation finished. Loss: {val_losses.avg:.4f}. Accuracy: {val_accuracies.avg:.4f}')
        self._log_stats('val', val_losses.avg, val_accuracies.avg)

        return val_accuracies.avg  # ✅ Return validation accuracy 作為 eval score

    def _is_best_eval_score(self, eval_score: float) -> bool:
        # 這裡假設 accuracy 越高越好
        is_best = eval_score > self.best_eval_score

        if is_best:
            logger.info(f'Saving new best model. Validation accuracy improved to {eval_score:.4f}')
            self.best_eval_score = eval_score

        return is_best

    def _save_checkpoint(self, is_best: bool):
        # remove `module` prefix from layer names when using `nn.DataParallel`
        # see: https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/20
        if isinstance(self.model, nn.DataParallel):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()

        last_file_path = os.path.join(self.checkpoint_dir, 'last_checkpoint.pytorch')
        logger.info(f"Saving checkpoint to '{last_file_path}'")

        save_checkpoint({
            'num_epochs': self.num_epochs + 1,
            'num_iterations': self.num_iterations,
            'model_state_dict': state_dict,
            'best_eval_score': self.best_eval_score,
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, is_best, checkpoint_dir=self.checkpoint_dir)

    def _log_lr(self):
        lr = self.optimizer.param_groups[0]['lr']
        self.writer.add_scalar('learning_rate', lr, self.num_iterations)

    def _log_stats(self, phase: str, loss_avg: float, accuracy_avg: float):
        tag_value = {
            f'{phase}_loss_avg': loss_avg,
            f'{phase}_accuracy_avg': accuracy_avg
        }

        for tag, value in tag_value.items():
            self.writer.add_scalar(tag, value, self.num_iterations)

    def _log_params(self):
        logger.info('Logging model parameters and gradients')
        for name, value in self.model.named_parameters():
            self.writer.add_histogram(name, value.data.cpu().numpy(), self.num_iterations)
            self.writer.add_histogram(name + '/grad', value.grad.data.cpu().numpy(), self.num_iterations)

    def _log_matching_similarity(self, predicted_matching, tag_prefix="train/matching_similarity"):
        """
        Compute cosine similarity between predicted matching and identity matrix,
        and log it to TensorBoard.
        """
        device = predicted_matching.device
        num_cells = predicted_matching.shape[0]

        identity = torch.eye(num_cells, device=device)
        cosine_sim = F.cosine_similarity(
            predicted_matching.flatten(), identity.flatten(), dim=0
        ).item()

        self.writer.add_scalar(tag_prefix, cosine_sim, self.num_iterations)

    def _log_matching_heatmap(self, predicted_matching, title="Matching Heatmap", tag_prefix="train/matching_heatmap"):
        """
        Logs the permutation matrix (predicted matching) as a heatmap to TensorBoard.

        Args:
            predicted_matching (torch.Tensor): [N, N] matching matrix with 1s at matched indices
            title (str): Title to display on the heatmap
            tag_prefix (str): TensorBoard tag prefix
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(predicted_matching.detach().cpu().numpy(), cmap='viridis', interpolation='nearest')
        ax.set_title(title)
        ax.set_xlabel("Cell Index (Worm B)")
        ax.set_ylabel("Cell Index (Worm A)")
        plt.tight_layout()

        # Convert figure to TensorBoard compatible format
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        transform = T.ToTensor()
        image_tensor = transform(image)

        # Log to TensorBoard
        self.writer.add_image(tag_prefix, image_tensor, self.num_iterations)
        plt.close(fig)

        @staticmethod
        def _batch_size(input: torch.Tensor) -> int:
            if isinstance(input, list) or isinstance(input, tuple):
                return input[0].size(0)
            else:
                return input.size(0)
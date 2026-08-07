"""
ResNet-compatible EWC (Elastic Weight Consolidation) implementation.

Adapted from lop/algos/ewc.py to work with ResNet's forward() interface
instead of DeepFFNN's predict() interface. Handles the CifarDataSet data
format {"image": ..., "label": ...} in the Fisher estimation dataloader.

Key differences from lop/algos/ewc.py:
    - Uses net.forward(x) instead of net.predict(x) for Fisher computation
    - Accepts a loss_fn callable instead of using self.loss_func internally
    - Handles dict-style dataloader output {"image": ..., "label": ...}
    - No dependency on Backprop base class
"""

import torch
import numpy as np
from torch import Tensor


class ResNetEWC:
    """
    Elastic Weight Consolidation for ResNet-based continual learning.

    After each task, call after_task(dataloader, loss_fn) to:
    1. Estimate the Fisher information matrix on the current task's data
    2. Accumulate it into the global Fisher matrix (with decay γ=0.93)
    3. Save a snapshot of the current parameters

    During training, call penalty() to get the EWC regularization loss term
    to add to the standard classification loss.
    """

    def __init__(self, net: torch.nn.Module, ewc_lambda: float = 5000,
                 device: torch.device = torch.device("cpu"),
                 fisher_sample_size: int = 2000,
                 gamma: float = 0.93,
                 fisher_clip_max: float = 0.01,
                 penalty_clip_max: float = 1.0):
        """
        :param net: the ResNet model
        :param ewc_lambda: EWC regularization strength
        :param device: torch device
        :param fisher_sample_size: max samples for Fisher estimation (efficiency)
        :param gamma: decay factor when accumulating Fisher across tasks
        :param fisher_clip_max: max value for individual Fisher entries
        :param penalty_clip_max: max value for per-parameter squared difference
        """
        self.net = net
        self.ewc_lambda = ewc_lambda
        self.device = device
        self.fisher_sample_size = fisher_sample_size
        self.gamma = gamma
        self.fisher_clip_max = fisher_clip_max
        self.penalty_clip_max = penalty_clip_max

        self.fisher_matrix = None       # {param_name: fisher_tensor}
        self.previous_params = None     # {param_name: param_tensor}

    # ------------------------------------------------------------------ #
    def penalty(self) -> Tensor:
        """
        Compute the EWC regularization loss:
            (lambda / 2) * sum_i F_i * (theta_i - theta_i_old)^2

        :return: scalar tensor (0.0 if no Fisher has been accumulated yet)
        """
        if self.fisher_matrix is None or self.previous_params is None:
            return torch.tensor(0.0, device=self.device)

        loss = torch.tensor(0.0, device=self.device)
        for name, param in self.net.named_parameters():
            if name in self.fisher_matrix:
                fisher = self.fisher_matrix[name]
                prev = self.previous_params[name]
                diff_sq = (param - prev).pow(2)
                diff_sq = torch.clamp(diff_sq, max=self.penalty_clip_max)
                loss = loss + (fisher * diff_sq).sum()

        return (self.ewc_lambda / 2.0) * loss

    # ------------------------------------------------------------------ #
    def after_task(self, dataloader, loss_fn):
        """
        Call after finishing a task: estimate Fisher on the task's data,
        accumulate into the global Fisher matrix, and save parameter snapshot.

        :param dataloader: DataLoader yielding {"image": Tensor, "label": Tensor}
        :param loss_fn: callable(predictions, labels) -> scalar loss
        """
        print("  -> Computing Fisher matrix (sample_size={0})...".format(
            self.fisher_sample_size))
        new_fisher = self._estimate_fisher(dataloader, loss_fn)

        if self.fisher_matrix is None:
            self.fisher_matrix = new_fisher
        else:
            for name in self.fisher_matrix:
                self.fisher_matrix[name] = (
                    self.gamma * self.fisher_matrix[name] + new_fisher[name]
                )
                self.fisher_matrix[name] = torch.clamp(
                    self.fisher_matrix[name], max=self.fisher_clip_max
                )

        self.previous_params = {
            name: param.data.clone()
            for name, param in self.net.named_parameters()
        }

        fisher_norm = sum(t.norm().item() for t in self.fisher_matrix.values())
        print("  -> Accumulated Fisher matrix norm: {0:.6f}".format(fisher_norm))

    # ------------------------------------------------------------------ #
    def _estimate_fisher(self, dataloader, loss_fn) -> dict:
        """
        Estimate the diagonal Fisher information matrix on the given dataloader.

        Fisher is approximated as the squared gradient of the loss w.r.t. each
        parameter, averaged over the (sampled) dataset.

        :param dataloader: DataLoader yielding {"image": ..., "label": ...}
        :param loss_fn: callable(predictions, labels) -> scalar loss
        :return: dict {param_name: fisher_tensor}
        """
        self.net.eval()

        # initialize Fisher dict
        fisher = {}
        for name, param in self.net.named_parameters():
            fisher[name] = torch.zeros_like(param.data).to(self.device)

        # sample a subset for efficiency
        total_samples = len(dataloader.dataset)
        sample_indices = np.random.choice(
            total_samples,
            min(self.fisher_sample_size, total_samples),
            replace=False
        )
        sampled_data = torch.utils.data.Subset(dataloader.dataset, sample_indices)
        sample_loader = torch.utils.data.DataLoader(
            sampled_data,
            batch_size=dataloader.batch_size,
            shuffle=False,
            num_workers=0
        )

        n_samples = 0
        for batch in sample_loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            self.net.zero_grad()
            predictions = self.net.forward(images)
            loss = loss_fn(predictions, labels)
            loss.backward()

            for name, param in self.net.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.data.pow(2) * len(images)
            n_samples += len(images)

        # normalize by number of samples
        for name in fisher:
            fisher[name] /= max(n_samples, 1)
            fisher[name] = torch.clamp(fisher[name], max=self.fisher_clip_max)

        self.net.train()
        return fisher

    # ------------------------------------------------------------------ #
    # Checkpoint support
    # ------------------------------------------------------------------ #
    def get_state(self) -> dict:
        """Return serializable state for checkpointing."""
        return {
            "fisher_matrix": self.fisher_matrix,
            "previous_params": self.previous_params,
        }

    def load_state(self, state: dict):
        """Restore state from a checkpoint."""
        self.fisher_matrix = state.get("fisher_matrix", None)
        self.previous_params = state.get("previous_params", None)

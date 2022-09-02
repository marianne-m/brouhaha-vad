from typing import Dict, Optional, Sequence, Text, Tuple, Union

import torch
from torchmetrics import Metric
import torchmetrics.functional as F
from torchmetrics.functional.regression.mae import _mean_absolute_error_compute



class CustomAUROC(Metric):
    higher_is_better: Optional[bool] = True
    full_state_update: Optional[bool] = False
    def __init__(self, output_transform=None):
        super().__init__()
        self.output_transform = output_transform

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape == target.shape

        preds, target = self.output_transform(preds, target)
        self.auroc = F.auroc(preds, target.int())

    def compute(self):
        return self.auroc


class CustomMeanAbsoluteError(Metric):
    is_differentiable: bool = True
    higher_is_better: bool = False
    full_state_update: bool = False
    sum_abs_error: torch.Tensor
    total: torch.Tensor

    def __init__(
        self,
        output_transform = None,
        mask = False
    ) -> None:
        super().__init__()

        self.add_state("sum_abs_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.output_transform = output_transform
        self.mask = mask

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update state with predictions and targets.
        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        if self.mask:
            weight = target[:,:,0].reshape(-1).int()

        if self.output_transform:
            preds, target = self.output_transform(preds, target)
        
        abs_error = torch.abs(preds - target)


        if self.mask:
            abs_error = weight * abs_error

        sum_abs_error = torch.sum(abs_error)
        n_obs = target.numel()

        self.sum_abs_error += sum_abs_error
        self.total += n_obs

    def compute(self) -> torch.Tensor:
        """Computes mean absolute error over state."""
        return _mean_absolute_error_compute(self.sum_abs_error, self.total)
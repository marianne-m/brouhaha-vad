from numbers import Number
from typing import Optional, List

import torch
import torchmetrics.functional as F
from torchmetrics import Metric
from torchmetrics.functional.classification.f_beta import _safe_divide
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
            output_transform=None,
            mask=False
    ) -> None:
        super().__init__()

        self.add_state("sum_abs_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.output_transform = output_transform
        self.mask = mask

    def update(
            self,
            preds: torch.Tensor,
            target: torch.Tensor,
            weights: torch.Tensor = None
    ) -> None:
        """Update state with predictions and targets.
        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        if self.mask:
            weights = target[:, :, 0].reshape(-1).int()

        if self.output_transform:
            preds, target = self.output_transform(preds, target)

        abs_error = torch.abs(preds - target)

        if weights is not None:
            abs_error = weights * abs_error
            n_obs = int(weights.sum())
        else:
            n_obs = target.numel()

        sum_abs_error = torch.sum(abs_error)
        self.sum_abs_error += sum_abs_error
        self.total += n_obs

    def compute(self) -> torch.Tensor:
        """Computes mean absolute error over state."""
        return _mean_absolute_error_compute(self.sum_abs_error, self.total)


class OptimalFScore(Metric):
    """Optiml F score metric

    Parameters
    ----------
    thresholds : torch.Tensor, optional
        Thresholds used to binarize predictions.
        Defaults to torch.linspace(0.0, 1.0, 51)

    Notes
    -----
    While pyannote.audio conventions is to store speaker activations with
    (batch_size, num_frames, num_speakers)-shaped tensors, this torchmetrics metric
    expects them to be shaped as (batch_size, num_speakers, num_frames) tensors.
    """
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(self, threshold: Optional[torch.Tensor] = None, output_transform=None):
        super().__init__()

        threshold = threshold or torch.linspace(0.0, 1.0, 51)
        self.add_state("threshold", default=threshold, dist_reduce_fx="mean")
        (num_thresholds,) = threshold.shape

        self.output_transform = output_transform

        self.add_state(
            "tp",
            default=torch.zeros((num_thresholds,)),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "tn",
            default=torch.zeros((num_thresholds,)),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fp",
            default=torch.zeros((num_thresholds,)),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fn",
            default=torch.zeros((num_thresholds,)),
            dist_reduce_fx="sum",
        )

    def update(
            self,
            preds: torch.Tensor,
            target: torch.Tensor,
    ) -> None:
        """Compute and accumulate components of diarization error rate

        Parameters
        ----------
        preds : torch.Tensor
            (batch_size, num_speakers, num_frames)-shaped continuous predictions.
        target : torch.Tensor
            (batch_size, num_speakers, num_frames)-shaped (0 or 1) targets.

        Returns
        -------
        false_alarm : torch.Tensor
        missed_detection : torch.Tensor
        speaker_confusion : torch.Tensor
        speech_total : torch.Tensor
            Diarization error rate components accumulated over the whole batch.
        """
        if self.output_transform:
            preds, target = self.output_transform(preds, target)
        tp, fp, tn, fn = stat_scores(preds, target, threshold=self.threshold)

        self.tp += tp
        self.tn += tn
        self.fp += fp
        self.fn += fn

    def compute(self):
        fscore = _fscore_compute(
            self.tp,
            self.fp,
            self.tn,
            self.fn,
        )
        opt_fscore, _ = torch.max(fscore, dim=0)

        return opt_fscore


class OptimalFScoreThreshold(OptimalFScore):
    def compute(self):
        fscore = _fscore_compute(
            self.tp,
            self.fp,
            self.tn,
            self.fn,
        )
        _, opt_threshold_idx = torch.max(fscore, dim=0)
        opt_threshold = self.threshold[opt_threshold_idx]

        return opt_threshold


def _compute_preds(
        preds: torch.Tensor,
        threshold: torch.Tensor
) -> torch.Tensor:
    scalar_threshold = isinstance(threshold, Number)
    if scalar_threshold:
        return (preds > threshold).int()
    else:
        preds_by_th = torch.zeros((threshold.size(0), preds.size(0)), device=preds.device)
        for index, th in enumerate(threshold):
            preds_by_th[index] = (preds > th).int()
        return preds_by_th


def _stat_scores_update(
        preds: torch.Tensor,
        target: torch.Tensor,
        reduce: str = "micro"
) -> List[torch.Tensor]:
    if preds.ndim == 1:
        dim = 0
    elif preds.ndim == 2:
        dim = 1

    true_pred, false_pred = target == preds, target != preds
    pos_pred, neg_pred = preds == 1, preds == 0

    tp = (true_pred * pos_pred).sum(dim=dim)
    fp = (false_pred * pos_pred).sum(dim=dim)

    tn = (true_pred * neg_pred).sum(dim=dim)
    fn = (false_pred * neg_pred).sum(dim=dim)

    return tp.long(), fp.long(), tn.long(), fn.long()


def stat_scores(
        preds: torch.Tensor,
        target: torch.Tensor,
        threshold: torch.Tensor
) -> torch.Tensor:
    preds = _compute_preds(preds, threshold)
    tp, fp, tn, fn = _stat_scores_update(
        preds,
        target,
        threshold
    )
    return tp, fp, tn, fn


def _fscore_compute(tp, fp, tn, fn):
    scalar = isinstance(tp, Number)
    if scalar:
        precision = tp / max((tp + fp), 1)
        recall = tp / max((tp + fn), 1)
        denom = precision + recall
        if denom == 0:
            denom = 1
        fscore = 2 * (precision * recall) / denom
    else:
        precision = _safe_divide(tp.float(), (tp + fp))
        recall = _safe_divide(tp.float(), (tp + fn))
        fscore = _safe_divide(2 * (precision * recall), (precision + recall))

    return fscore

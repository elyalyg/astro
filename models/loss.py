import warnings

import torch
import torch.nn as nn

from utils.stats import iqr, naniqr


class _MaskedLoss(nn.Module):
    """Base class for masked losses"""

    def __init__(self, reduction='mean', ignore_nans=True):
        super().__init__()
        self.reduction = reduction
        self.ignore_nans = ignore_nans

    def forward(self, input, target, mask=None):
        """
        Compute a loss between input and target for given mask.
        Automatically crops input/target/mask to the same time length if needed.
        """

        # --- ALIGN TIME DIMENSIONS ---
        # assume shape is (B, T, C) or similar, crop T to the shortest among them
        T_input = input.shape[1]
        T_target = target.shape[1] if isinstance(target, torch.Tensor) and target.dim() >= 2 else T_input
        T_mask = mask.shape[1] if (mask is not None and mask.dim() >= 2) else T_input

        T_min = min(T_input, T_target, T_mask)
        if T_min < T_input:
            input = input[:, :T_min, ...]
        if T_min < T_target:
            target = target[:, :T_min, ...]
        if mask is not None and T_min < T_mask:
            mask = mask[:, :T_min, ...]

        # --- END ALIGN ---

        if not (target.size() == input.size()):
            warnings.warn(
                f"MaskedLoss: input {input.size()} and target {target.size()} shapes differ "
                "even after cropping; results may be incorrect.",
                stacklevel=2,
            )

        if mask is None:
            mask = torch.ones_like(input, dtype=torch.bool)

        target_proxy = target
        if self.ignore_nans:
            target_proxy = target.clone()
            nans = torch.isnan(target)
            if nans.any():
                with torch.no_grad():
                    mask = mask & ~nans
                target_proxy[nans] = 0.0

        full_loss = self.criterion(input, target_proxy)

        if not mask.any():
            warnings.warn(
                "Evaluation mask is False everywhere; this might lead to incorrect results."
            )
        full_loss = full_loss.masked_fill(~mask, 0.0)

        if self.reduction == 'none':
            return full_loss
        if self.reduction == 'sum':
            return full_loss.sum()
        # mean over only the unmasked entries
        if self.reduction == 'mean':
            total = mask.to(full_loss.dtype).sum()
            return full_loss.sum() / (total if total > 0 else 1.0)


class MaskedMSELoss(_MaskedLoss):
    """Masked MSE loss"""

    def __init__(self, reduction='mean', ignore_nans=True):
        super().__init__(reduction=reduction, ignore_nans=ignore_nans)
        self.criterion = nn.MSELoss(reduction='none')


class MaskedL1Loss(_MaskedLoss):
    """Masked L1 loss."""

    def __init__(self, reduction='mean', ignore_nans=True):
        super().__init__(reduction=reduction, ignore_nans=ignore_nans)
        self.criterion = nn.L1Loss(reduction='none')


class MaskedHuberLoss(_MaskedLoss):
    """Masked Huber loss."""

    def __init__(self, reduction='mean', ignore_nans=True, delta=1.0):
        super().__init__(reduction=reduction, ignore_nans=ignore_nans)
        self.criterion = nn.HuberLoss(reduction='none', delta=delta)


class IQRLoss(nn.Module):
    """Interquartile‐range loss of the residuals."""

    def __init__(self, reduction='nanmean', ignore_nans=True):
        super().__init__()
        self.reduction = reduction
        self.ignore_nans = ignore_nans

    def forward(self, input, target=0.0):
        """
        Compute the IQR of (target - input). No mask here—
        simply crop to the shortest time length.
        """
        # align time dims if target is a tensor
        if isinstance(target, torch.Tensor):
            T_in = input.shape[1]
            T_tg = target.shape[1] if target.dim() >= 2 else T_in
            T_min = min(T_in, T_tg)
            if T_min < T_in:
                input = input[:, :T_min, ...]
            if T_min < T_tg:
                target = target[:, :T_min, ...]

        residual = target - input
        if self.ignore_nans:
            return naniqr(residual, reduction=self.reduction)
        else:
            return iqr(residual, reduction=self.reduction)

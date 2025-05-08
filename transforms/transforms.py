import warnings

import numpy as np
import torch

from . import functional_array as F_np
from . import functional_tensor as F_t
from utils.stats import nanstd


class Compose:
    """Composes several transforms together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, mask=None, info=None):
        out = x
        for t in self.transforms:
            out, mask, info = t(out, mask=mask, info=info)
        return out, mask, info

    def __repr__(self):
        fmt = self.__class__.__name__ + "("
        for t in self.transforms:
            fmt += f"\n    {t}"
        fmt += "\n)"
        return fmt


class FillNans:
    """Fill NaNs in the input with a constant value."""

    def __init__(self, value):
        self.value = value

    def __call__(self, x, mask=None, info=None):
        if isinstance(x, np.ndarray):
            out = F_np.fill_nans(x, self.value)
        else:
            out = F_t.fill_nans(x, self.value)
        return out, mask, info

    def __repr__(self):
        return f"FillNans(value={self.value})"


class Mask:
    """Mask a fraction of the input, optionally in blocks, works on np.ndarray or torch.Tensor."""

    def __init__(
        self,
        mask_ratio,
        block_len=None,
        block_mode="geom",
        interval_mode="geom",
        overlap_mode="random",
        value=np.nan,
        exclude_mask=True,
        max_ratio=None,
    ):
        self.mask_ratio = mask_ratio
        self.block_len = block_len
        self.overlap_mode = overlap_mode
        self.block_mode = block_mode
        self.interval_mode = interval_mode
        self.value = value
        self.exclude_mask = exclude_mask
        self.max_ratio = max_ratio

    def __call__(self, x, mask=None, info=None):
        # --- STEP 1: Prepare `out` as float array/tensor ---
        is_torch = isinstance(x, torch.Tensor)
        if is_torch:
            out = x.clone()
            arr = out.cpu().numpy()  # bring into NumPy
            orig_device = out.device
        else:
            out = x.copy()
            arr = out

        # --- STEP 2: Exclude existing masked region if requested ---
        temp_arr = arr
        if self.exclude_mask and mask is not None:
            # univariate only
            temp_arr = arr[~mask][:, None]

        # --- STEP 3: Create a new mask via NumPy backend ---
        temp_mask_np = F_np.create_mask_like(
            temp_arr,
            self.mask_ratio,
            block_len=self.block_len,
            block_mode=self.block_mode,
            interval_mode=self.interval_mode,
            overlap_mode=self.overlap_mode,
        )

        # --- STEP 4: Retry if we masked too much in one go ---
        if self.max_ratio is not None and temp_mask_np.mean() >= self.max_ratio:
            return self.__call__(x, mask=mask, info=info)

        # --- STEP 5: Apply the `value` to the masked positions ---
        temp_arr[temp_mask_np] = self.value

        # --- STEP 6: Convert mask back to torch if needed ---
        if is_torch:
            temp_mask = torch.from_numpy(temp_mask_np).to(orig_device)
        else:
            temp_mask = temp_mask_np

        # --- STEP 7: Merge with existing mask and rebuild `out` ---
        if mask is None:
            mask_out = temp_mask
            if is_torch:
                out = torch.from_numpy(temp_arr).to(orig_device)
            else:
                out = temp_arr
        else:
            if is_torch:
                out = x.clone()
                # fill in the newly unmasked slots
                out[~mask] = torch.from_numpy(temp_arr.squeeze()).to(orig_device)
                mask_out = mask.clone()
                mask_out[~mask] = temp_mask.squeeze()
            else:
                out = x.copy()
                out[~mask] = temp_arr.squeeze()
                mask_out = mask.copy()
                mask_out[~mask] = temp_mask.squeeze()

        return out, mask_out, info

    def __repr__(self):
        parts = [f"ratio={self.mask_ratio}", f"overlap={self.overlap_mode}"]
        if self.block_len:
            parts += [
                f"block_len={self.block_len}",
                f"block_mode={self.block_mode}",
                f"interval_mode={self.interval_mode}",
            ]
        return f"Mask({'; '.join(parts)})"



class AddGaussianNoise:
    """Add Gaussian noise to the input, optionally excluding or only on masked regions."""

    def __init__(self, sigma=1.0, exclude_mask=False, mask_only=False):
        self.sigma = sigma
        self.exclude_mask = exclude_mask
        self.mask_only = mask_only
        assert not (exclude_mask and mask_only), "Cannot both exclude and mask_only"

    def __call__(self, x, mask=None, info=None):
        exc = None
        if mask is not None:
            if self.exclude_mask:
                exc = mask
            elif self.mask_only:
                exc = ~mask

        if isinstance(x, np.ndarray):
            out = F_np.add_gaussian_noise(x, self.sigma, mask=exc)
        else:
            out = F_t.add_gaussian_noise(x, self.sigma, mask=exc)
        return out, mask, info

    def __repr__(self):
        return (
            f"AddGaussianNoise(sigma={self.sigma}; "
            f"exclude_mask={self.exclude_mask}; mask_only={self.mask_only})"
        )


class Scaler:
    """Base scaler with fit/transform interface."""

    def __init__(self, dim, centers=None, norms=None, eps=1e-10):
        super().__init__()
        self.dim = dim
        self.centers = centers
        self.norms = norms
        self.eps = eps

    def transform(self, x, mask=None):
        if mask is None:
            return (x - self.centers) / self.norms
        else:
            return (x - self.centers) / self.norms, mask

    def fit(self, x, mask=None):
        raise NotImplementedError

    def fit_transform(self, x, mask=None):
        self.fit(x, mask=mask)
        return self.transform(x, mask=mask)

    def inverse_transform(self, y):
        return (y * self.norms) + self.centers

    def __call__(self, x, mask=None, info=None):
        out, m = self.fit_transform(x, mask=mask)
        info["mu"] = self.centers
        info["sigma"] = self.norms
        return out, m, info


class StandardScaler(Scaler):
    """Scale inputs to zero mean and unit variance (ignoring NaNs)."""

    def fit(self, x, mask=None):
        xm = x.copy() if isinstance(x, np.ndarray) else x.clone()
        if mask is not None:
            xm[mask] = np.nan if isinstance(xm, np.ndarray) else float("nan")

        if isinstance(x, np.ndarray):
            self.centers = np.nanmean(xm, axis=self.dim, keepdims=True)
            self.norms = np.nanstd(xm, axis=self.dim, keepdims=True) + self.eps
        else:
            self.centers = torch.nanmean(xm, dim=self.dim, keepdim=True)
            self.norms = nanstd(xm, dim=self.dim, keepdim=True) + self.eps

    def __repr__(self):
        return f"StandardScaler(dim={self.dim})"


class DownSample:
    """Down-sample the sequence by an integer factor."""

    def __init__(self, factor=1):
        self.factor = factor

    def __call__(self, x, mask=None, info=None):
        out = x[:: self.factor]
        m = mask[:: self.factor] if mask is not None else None
        return out, m, info

    def __repr__(self):
        return f"DownSample(factor={self.factor})"


class RandomCrop:
    """Randomly crop a fixed-width window, retrying if too many NaNs."""

    def __init__(self, width, exclude_missing_threshold=None):
        self.width = width
        self.exclude_missing_threshold = exclude_missing_threshold
        assert (
            exclude_missing_threshold is None or 0 <= exclude_missing_threshold <= 1
        ), "exclude_missing_threshold must be in [0,1] or None"

    def __call__(self, x, mask=None, info=None):
        seq_len = x.shape[0]
        if seq_len < self.width:
            left = 0
            warnings.warn(
                f"RandomCrop: sequence length {seq_len} < width {self.width}; no crop"
            )
        else:
            left = np.random.randint(seq_len - self.width + 1)
        info = {} if info is None else info
        info["left_crop"] = left

        out_x = x[left : left + self.width]

        # compute NaN ratio safely on float array
        if self.exclude_missing_threshold is not None:
            arr = (
                out_x.cpu().numpy() if isinstance(out_x, torch.Tensor) else out_x
            )
            nan_ratio = np.isnan(arr).mean()
            if nan_ratio >= self.exclude_missing_threshold:
                return self.__call__(x, mask=mask, info=info)

        out_m = mask[left : left + self.width] if mask is not None else None
        return out_x, out_m, info

    def __repr__(self):
        return (
            f"RandomCrop(width={self.width}, "
            f"exclude_missing_threshold={self.exclude_missing_threshold})"
        )

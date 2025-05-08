# ────────────────────────────────────────────────────────────────
#  kepler_tess.py   –  merged official + custom (2025-05-01)
# ────────────────────────────────────────────────────────────────
"""
Unified PyTorch Dataset for TESS / Kepler light-curve FITS files.
Combines original repo's DatasetFolder functionality with a robust,
transform-compatible API.

Features:
• Recursively finds files under *root* matching Kepler or TESS patterns
• Supports raw FITS or cached .npy loading (load_processed)
• Caches to <root>/processed/ (save_processed)
• Optional one-shot caching of entire dataset (use_cache)
• Optional shuffling (shuffle, random_seed)
• Masking of missing values (mask_missing)
• Transforms: transform, transform_target, transform_both
• Subset that respects replace_transform(_target/_both) and full-dataset if no indices
• split_indices(dataset, val_ratio, test_ratio, seed)
"""
from __future__ import annotations
import glob, os, random
from pathlib import Path
from typing import Any, Callable, List, Tuple

import numpy as np
import torch
from astropy.io import fits
from torch.utils.data import Dataset, Subset as _TorchSubset

# Patterns for Kepler and TESS
KEPLER_LC_PATTERN = '**/kplr*-*lc.fits'
TESS_LC_PATTERN   = '**/tess*lc.fits'


def _fits_to_tensor(fp: str) -> torch.Tensor:
    """Load a FITS LC file, keep NaNs, return (T,1) float32 tensor."""
    with fits.open(fp, memmap=False) as hdul:
        data = hdul[1].data
        # try standard columns
        if 'FLUX' in data.columns.names:
            flux = data['FLUX']
        elif 'PDCSAP_FLUX' in data.columns.names:
            flux = data['PDCSAP_FLUX']
        elif 'SAP_FLUX' in data.columns.names:
            flux = data['SAP_FLUX']
        else:
            raise KeyError(f"No FLUX column in {fp}")
    arr = flux.astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(-1)


class TessDataset(Dataset):
    def __init__(
        self,
        root: str,
        *,
        files_list: str | None = None,
        transform: Callable[[torch.Tensor, torch.Tensor, dict], Tuple[Any, Any, dict]] | None = None,
        transform_target: Callable[[torch.Tensor, torch.Tensor, dict], Tuple[Any, Any, dict]] | None = None,
        transform_both: Callable[[torch.Tensor, torch.Tensor, dict], Tuple[Any, Any, dict]] | None = None,
        mask_missing: bool = True,
        max_samples: int | None = None,
        shuffle: bool = False,
        random_seed: int = 0,
        load_processed: bool = False,
        save_processed: bool | None = None,
        overwrite: bool = False,
        save_folder: str | None = None,
        use_cache: bool = False,
        is_tess: bool = True,
        **_ignored,
    ):
        # save args
        self.root = str(root)
        self.transform = transform
        self.transform_target = transform_target
        self.transform_both = transform_both
        self.mask_missing = mask_missing
        self.max_samples = max_samples
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.load_processed = load_processed
        # decide save_processed default
        self.save_processed = save_processed if save_processed is not None else not load_processed
        self.overwrite = overwrite
        self.use_cache = use_cache
        # setup save folder
        self.save_folder = save_folder or os.path.join(self.root, 'processed')
        if self.save_processed:
            os.makedirs(self.save_folder, exist_ok=True)
        # choose pattern
        self.file_pattern = TESS_LC_PATTERN if is_tess else KEPLER_LC_PATTERN
        # load file list
        if files_list is not None:
            with open(files_list, 'r') as f:
                lines = [l.strip() for l in f if l.strip()]
            self.files = [l if os.path.isabs(l) else os.path.join(self.root, l) for l in lines]
        else:
            self.files = sorted(glob.glob(os.path.join(self.root, self.file_pattern), recursive=True))
        # shuffle and sample
        if self.shuffle:
            random.seed(self.random_seed)
            random.shuffle(self.files)
        if self.max_samples is not None:
            self.files = self.files[: self.max_samples]
        if not self.files:
            raise RuntimeError(f"Found 0 files under {self.root} matching {self.file_pattern}")
        # optional caching
        if self.use_cache:
            self._cache_data()

    def __len__(self) -> int:
        return len(self.files)

    def _cache_data(self):
        """Cache raw FITS to .npy without transforms."""
        for path in self.files:
            y = _fits_to_tensor(path)
            outp = os.path.join(self.save_folder, os.path.basename(path).replace('.fits', '.npy'))
            if self.overwrite or not os.path.exists(outp):
                np.save(outp, y.numpy(), allow_pickle=False)
        print('data successfully cached')

    def __getitem__(self, idx: int):
        path = self.files[idx]
        # load
        # updated kepler_tess.py __getitem__ snippet

        if self.load_processed:
        # build the full path in <root>/processed/
            fname = os.path.basename(path).replace('.fits', '.npy')
            npy_path = os.path.join(self.save_folder, fname)
            # load it
            arr = np.load(npy_path)
            y = torch.from_numpy(arr)
        else:
            y = _fits_to_tensor(path)
            if self.save_processed:
                outp = os.path.join(self.save_folder, os.path.basename(path).replace('.fits', '.npy'))
                if self.overwrite or not os.path.exists(outp):
                    np.save(outp, y.numpy(), allow_pickle=False)

        # if self.load_processed:
        #     arr = np.load(path.replace('.fits', '.npy'))
        #     y = torch.from_numpy(arr)
        # else:
        #     y = _fits_to_tensor(path)
        #     if self.save_processed:
        #         outp = os.path.join(self.save_folder, os.path.basename(path).replace('.fits', '.npy'))
        #         if self.overwrite or not os.path.exists(outp):
        #             np.save(outp, y.numpy(), allow_pickle=False)
        # mask and info
        mask = y.isnan() if self.mask_missing else None
        info = {}
        # transforms
        if self.transform_both is not None:
            xm, m, info = self.transform_both(y.clone(), mask=mask, info=info)
            return xm, y, m, info
        if self.transform is not None:
            xm, m, info = self.transform(y.clone(), mask=mask, info=info)
            return xm, y, m, info
        if self.transform_target is not None:
            xt, m, info = self.transform_target(y.clone(), mask=mask, info=info)
            return y, xt, m, info
        # default
        return y.clone(), y, mask, info

# alias and helpers
KeplerDataset = lambda root, **kw: TessDataset(root, is_tess=False, **kw)

class Subset(_TorchSubset):
    """
    Wrap a dataset to apply optional transform overrides and custom indices.
    """
    def __init__(
        self,
        dataset,
        indices=None,
        replace_transform=None,
        replace_transform_target=None,
        replace_transform_both=None,
    ):
        # apply replacements
        if replace_transform is not None and hasattr(dataset, 'transform'):
            dataset.transform = replace_transform
        if replace_transform_target is not None and hasattr(dataset, 'transform_target'):
            dataset.transform_target = replace_transform_target
        if replace_transform_both is not None and hasattr(dataset, 'transform_both'):
            dataset.transform_both = replace_transform_both
        # default indices = full
        if indices is None:
            indices = list(range(len(dataset)))
        super().__init__(dataset, indices)


def split_indices(
    dataset,
    val_ratio: float = 0.2,
    test_ratio: float = 0.0,
    seed: int = 0,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Split dataset indices into train/val/test.
    """
    rng = np.random.default_rng(seed)
    n = len(dataset)
    perm = rng.permutation(n)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    test_idx = perm[:n_test]
    val_idx = perm[n_test : n_test + n_val]
    train_idx = perm[n_test + n_val :]
    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()

# smoke-test
if __name__ == '__main__':
    import sys, argparse
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='datasets')
    p.add_argument('--samples', type=int, default=3)
    args = p.parse_args()
    ds = TessDataset(
        args.root,
        load_processed=False,
        save_processed=False,
        max_samples=args.samples,
        use_cache=False,
    )
    print(
        f"[Smoke] Found {len(ds)} samples; first shape = {ds[0][0].shape}",
        file=sys.stderr,
    )

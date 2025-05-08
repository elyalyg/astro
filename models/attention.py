# models/attention.py
import torch
from torch import nn

class EyeAttention(nn.Module):
    """
    Builds a square (L × L) boolean attention mask whose diagonal stripes of
    width *eye* are un-masked (False) and everything else is masked (True).
        • eye = 0   → no mask   (full attention)
        • eye = 1   → true eye  (only self tokens attend to themselves)
        • eye > 1   → narrow band around the diagonal
    The mask is broadcastable to (num_heads, L, L).
    """

    def __init__(self, eye: int = 0):
        super().__init__()
        self.eye = int(eye)

    def forward(self, x: torch.Tensor) -> torch.Tensor | None:
        """
        x : (batch, seq_len, dim) – only seq_len is used
        returns
            None          if self.eye == 0   (no masking)
            Bool Tensor   shape (seq_len, seq_len) otherwise
                          True  = *masked / no attention*
                          False = attention allowed
        """
        if self.eye == 0:
            return None

        L = x.size(1)
        mask = torch.ones(L, L, dtype=torch.bool, device=x.device)
        idx = torch.arange(L, device=x.device)
        for offset in range(-self.eye + 1, self.eye):
            mask[idx, idx + offset] = False
        return mask

"""Time-series signature encoder: pre-onset trajectory -> unit-norm embedding.

This module is the input side of the retrieval framework. It maps a
standardised macro-financial *trajectory* --- shape ``(window, n_features)``,
as produced by :mod:`data` --- to a single L2-normalised embedding of dimension
``embed_dim``, which is exactly the representation :mod:`metric` consumes (its
``forward`` assumes unit-norm inputs) and the gate reads to infer the regime.

Architecture and rationale
--------------------------
A pre-onset window is a short multivariate series (here 10 annual steps x ~10
features), and what discriminates regimes is its *temporal shape* --- the pace
and profile of the credit/deposit/leverage build-up, not any single year. The
encoder is therefore deliberately compact and temporal:

  1. **Per-step feature projection** (``Linear(n_features -> hidden)``) lifts each
     year's feature vector into a hidden space.
  2. **Learned positional embeddings** (one per year in the window) let the model
     use *where* in the run-up an observation sits; the year just before onset
     is typically the most informative, and the model must be able to say so.
  3. **Residual dilated temporal convolutions** (kernel 3, dilations 1,2,4,...)
     give a receptive field spanning the whole window with few parameters, so
     multi-year acceleration patterns are captured cheaply.
  4. **Attention pooling** collapses the per-year features to one vector by a
     learned, interpretable weighting over years (which years mattered).
  5. **Projection head + L2 normalisation** produce the embedding; a separate
     projection head (discarded at inference, as in SupCon) maps the embedding
     into the contrastive space used by the training objective.

Normalisation. LayerNorm is used throughout, never BatchNorm: batches are small
and, more importantly, retrieval/contrastive training must not couple examples
through batch statistics, and per-example encoding must be *batch-independent*
and deterministic. Both properties are verified in the audit.

Capacity note (honest)
----------------------
The primary emerging-market corpus is small (~150 training cases). The encoder
is therefore intentionally small and regularised (modest ``hidden``/``embed_dim``,
dropout, and --- in ``train.py`` --- weight decay and early stopping on
validation retrieval), and the metric it feeds is low-rank. For this corpus a
smaller ``embed_dim`` (e.g. 32) may generalise better than the interface default
of 64; that is a tuning choice left to the configs. The supervised-contrastive
objective (implemented in ``train.py``) uses all same-regime pairs, which makes
efficient use of the limited data.

Contract
--------
``forward`` returns an ``(B, embed_dim)`` unit-norm tensor. With
``return_projection=True`` it additionally returns the ``(B, proj_dim)`` unit-norm
projection for the contrastive loss. Input must be ``(B, window, n_features)``
with the configured ``window`` and ``n_features``; a mismatch raises.

References
----------
Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y., Isola, P., Maschinot,
A., Liu, C., Krishnan, D. (2020). Supervised contrastive learning. NeurIPS. (The
projection head is trained for the contrastive objective and discarded at
inference, following this design.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["EncoderConfig", "SignatureEncoder"]

_EPS: Final[float] = 1e-12


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class EncoderConfig:
    """Hyper-parameters for :class:`SignatureEncoder`."""

    input_dim: int = 10            # n_features per timestep (GFDD primary = 10)
    window: int = 10               # trajectory length in years
    hidden: int = 64               # temporal hidden width
    embed_dim: int = 64            # output embedding dim (match MetricConfig)
    n_blocks: int = 3              # number of residual temporal-conv blocks
    kernel_size: int = 3           # temporal kernel (odd, for length-preserving)
    dilation_base: int = 2         # block i uses dilation dilation_base**i
    pool: str = "attention"        # "attention" | "mean"
    proj_dim: int = 32             # contrastive projection dimension
    proj_hidden: int = 64          # projection-head hidden width
    dropout: float = 0.10          # dropout in conv blocks and projection head

    def __post_init__(self) -> None:
        if min(self.input_dim, self.window, self.hidden, self.embed_dim,
               self.proj_dim, self.proj_hidden) < 1:
            raise ValueError("input_dim, window, hidden, embed_dim, proj_dim, "
                             "proj_hidden must be >= 1")
        if self.kernel_size < 1 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if self.dilation_base < 1:
            raise ValueError("dilation_base must be >= 1")
        if self.n_blocks < 0:
            raise ValueError("n_blocks must be >= 0")
        if self.pool not in ("attention", "mean"):
            raise ValueError("pool must be 'attention' or 'mean'")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in [0, 1)")


# --------------------------------------------------------------------------- #
# Building blocks
# --------------------------------------------------------------------------- #

class _TemporalBlock(nn.Module):
    """Residual, length-preserving dilated temporal-conv block.

    Operates on ``(B, T, hidden)`` tensors (LayerNorm-friendly layout),
    transposing internally for the ``Conv1d`` over the time axis. Length is
    preserved by symmetric padding ``dilation * (kernel-1) / 2`` (kernel odd).
    """

    def __init__(self, hidden: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.conv1 = nn.Conv1d(hidden, hidden, kernel, padding=pad, dilation=dilation)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel, padding=pad, dilation=dilation)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, T, hidden)
        h = self.conv1(x.transpose(1, 2)).transpose(1, 2)     # (B, T, hidden)
        h = self.drop(F.gelu(self.norm1(h)))
        h = self.conv2(h.transpose(1, 2)).transpose(1, 2)
        h = self.norm2(h)
        return F.gelu(x + h)                                   # residual


class _AttentionPool(nn.Module):
    """Learned attention pooling over the time axis.

    Scores each timestep against a learned query, softmax-normalises across time,
    and returns the weighted sum plus the attention weights (for interpretability
    --- e.g. which pre-onset years the encoder relies on).
    """

    def __init__(self, hidden: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(hidden) / (hidden ** 0.5))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = (x @ self.query) / (x.shape[-1] ** 0.5)      # (B, T)
        weights = torch.softmax(scores, dim=1)                # (B, T)
        pooled = torch.einsum("bt,bth->bh", weights, x)       # (B, hidden)
        return pooled, weights


# --------------------------------------------------------------------------- #
# Encoder
# --------------------------------------------------------------------------- #

class SignatureEncoder(nn.Module):
    """Encode a ``(window, n_features)`` trajectory into a unit-norm embedding."""

    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden

        self.input_proj = nn.Linear(cfg.input_dim, h)
        # Learned positional embedding, one vector per year in the window.
        self.pos = nn.Parameter(0.02 * torch.randn(cfg.window, h))

        self.blocks = nn.ModuleList([
            _TemporalBlock(h, cfg.kernel_size, cfg.dilation_base ** i, cfg.dropout)
            for i in range(cfg.n_blocks)
        ])

        # Attention pool has a learned query parameter; mean pooling is handled
        # inline in ``encode`` and needs no module (hence None).
        self.pool: nn.Module | None = (
            _AttentionPool(h) if cfg.pool == "attention" else None
        )

        self.head = nn.Linear(h, cfg.embed_dim)

        # Contrastive projection head (discarded at inference, SupCon-style).
        self.projection_head = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.proj_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity(),
            nn.Linear(cfg.proj_hidden, cfg.proj_dim),
        )

    # ------------------------------------------------------------------ #

    def _check(self, traj: torch.Tensor) -> None:
        if traj.ndim != 3:
            raise ValueError(f"trajectory must be 3-D (B, window, n_features); "
                             f"got shape {tuple(traj.shape)}")
        if traj.shape[1] != self.cfg.window or traj.shape[2] != self.cfg.input_dim:
            raise ValueError(
                f"expected (B, {self.cfg.window}, {self.cfg.input_dim}); "
                f"got (B, {traj.shape[1]}, {traj.shape[2]})"
            )

    def encode(self, traj: torch.Tensor, *, return_attention: bool = False):
        """Trajectory -> pre-normalisation embedding (and optional attention)."""
        self._check(traj)
        h = self.input_proj(traj) + self.pos.unsqueeze(0)     # (B, T, hidden)
        for blk in self.blocks:
            h = blk(h)
        if self.cfg.pool == "attention":
            pooled, attn = self.pool(h)                       # (B, hidden), (B,T)
        else:
            pooled = h.mean(dim=1)                            # (B, hidden)
            attn = h.new_full((h.shape[0], h.shape[1]), 1.0 / h.shape[1])
        emb_raw = self.head(pooled)                           # (B, embed_dim)
        return (emb_raw, attn) if return_attention else emb_raw

    def forward(self, traj: torch.Tensor, *, return_projection: bool = False,
                return_attention: bool = False):
        """Trajectory -> unit-norm embedding.

        Returns ``(B, embed_dim)`` on the unit sphere. With
        ``return_projection=True`` also returns the unit-norm ``(B, proj_dim)``
        contrastive projection; with ``return_attention=True`` also returns the
        ``(B, window)`` pooling weights. Extra outputs are appended in the order
        ``(embedding[, projection][, attention])``.
        """
        emb_raw, attn = self.encode(traj, return_attention=True)
        emb = F.normalize(emb_raw, p=2, dim=-1, eps=_EPS)

        outputs: list[torch.Tensor] = [emb]
        if return_projection:
            proj = self.projection_head(emb_raw)
            proj = F.normalize(proj, p=2, dim=-1, eps=_EPS)
            outputs.append(proj)
        if return_attention:
            outputs.append(attn)
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def extra_repr(self) -> str:
        return (f"input_dim={self.cfg.input_dim}, window={self.cfg.window}, "
                f"hidden={self.cfg.hidden}, embed_dim={self.cfg.embed_dim}, "
                f"n_blocks={self.cfg.n_blocks}, pool={self.cfg.pool}")

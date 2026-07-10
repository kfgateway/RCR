"""The regime-conditional similarity metric --- the core contribution.

This module implements the paper's central object: a similarity function whose
*geometry is conditioned on the query's inferred regime*. Rather than a single
fixed metric, it maintains one low-rank positive-semidefinite (PSD) metric per
regime and mixes them by a **calibrated posterior** over regimes produced from
the query alone:

    s(z_q, z_c | pi) = sum_tau  pi_tau(z_q) * ( z_q^T W_tau z_c )  +  gamma * z_q^T z_c,
        with   W_tau = W_shared + L_tau L_tau^T   (PSD),
               pi = gate(z_q) in the simplex,   gamma >= 0,   embeddings z unit-norm.

Each regime metric is a **shared PSD base** ``W_shared = L_0 L_0^T`` plus a low-rank
**per-regime deviation** ``D_tau = L_tau L_tau^T`` (both PSD, hence ``W_tau`` PSD). A
shrinkage penalty on ``L_tau`` (see :meth:`RegimeConditionalMetric.delta_penalty`)
drives ``D_tau -> 0``, collapsing every ``W_tau`` to the shared global metric when
the data does not support regime-specific geometry --- graceful degradation, and
Claim 1's boundary condition ("no divergence -> conditioning buys nothing") built
into the parameterisation. Regularising a learned metric toward a shared prior is
the information-theoretic metric-learning idea (Davis et al., 2007: ITML keeps the
metric close to a prior via a LogDet divergence); here the prior is the shared
metric and the penalty is Frobenius on the deviations.

Why this is the contribution (and how it maps to the theory)
------------------------------------------------------------
* **Conditional (Claim 1).** A single global metric provably cannot serve
  regimes whose discriminative geometries diverge; here each regime owns its own
  ``W_tau``. Setting the gate to uniform (``conditioning=False``) recovers the
  single-global-metric baseline exactly, which is the honest ablation.

* **Bayes form (Claim 2).** With the true posterior gate and the true per-regime
  metrics, ``s`` is the Bayes retrieval rule for the generative model in
  :mod:`synthetic`; this module is that architecture.

* **Calibrated (Claim 3).** The gate is a *probabilistic* regime classifier with
  an explicit temperature; its calibration --- not merely its top-1 accuracy ---
  is what the theory shows controls excess retrieval risk. Temperature scaling
  (Guo et al., 2017) is exposed as a first-class, post-hoc-fittable parameter.

PSD-by-construction guarantee
-----------------------------
For any fixed query (hence fixed gate ``pi`` on the simplex), the effective
metric is

    W(pi) = sum_tau pi_tau W_tau + gamma I,

a non-negative combination of PSD matrices plus ``gamma I >= 0``, hence PSD.
So *per query* the similarity is a genuine Mahalanobis inner product. The
overall scoring is intentionally **query-conditional and therefore asymmetric**
under swapping (q, c): the query sets the regime lens. That is the
conditional-similarity semantics, not a defect.

Efficiency
----------
``W_tau`` (d x d, per regime) is never materialised in the hot path. Scores use
the factorised identity ``z_q^T L_tau L_tau^T z_c = (L_tau^T z_q)^T (L_tau^T z_c)``,
i.e. project into each regime's rank-r space (cost O(d r T)) and take inner
products. ``effective_metric`` materialises ``W(pi)`` only for analysis/audit.

Contract
--------
Inputs are L2-normalised embeddings of shape ``(., embed_dim)`` (the encoder's
output). ``forward`` renormalises defensively when ``assume_normalized=False``.
The gate conditions over ``n_regimes`` regimes (default 5: the crisis regimes);
calm/negative cases are retrieval candidates, not conditioning regimes.

References
----------
Guo, C., Pleiss, G., Sun, Y., Weinberger, K.Q. (2017). On calibration of modern
neural networks. ICML. (Temperature scaling for the calibrated gate.)
Davis, J.V., Kulis, B., Jain, P., Sra, S., Dhillon, I.S. (2007).
Information-theoretic metric learning. ICML. (Regularising a metric toward a
prior --- the precedent for the shared-base shrinkage.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "REGIME_NAMES",
    "MetricConfig",
    "RegimeConditionalMetric",
]

#: Default conditioning regimes, in fixed order (matches the crisis regimes in
#: the data layer; excludes "none", which is a candidate label, not a condition).
REGIME_NAMES: Final[tuple[str, ...]] = (
    "banking", "currency", "sovereign", "twin", "triple",
)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class MetricConfig:
    """Hyper-parameters for :class:`RegimeConditionalMetric`."""

    embed_dim: int = 64            # dimension d of the input embeddings
    n_regimes: int = 5             # number of conditioning regimes T
    rank: int = 8                  # rank r of each per-regime DELTA factor L_tau
    shared_rank: int = 8           # rank of the shared base metric L_0 (0 disables)
    gamma_init: float = 0.5        # initial weight on the cosine baseline (>= 0)
    gate_hidden: int = 64          # hidden width of the gate MLP (0 = linear)
    factor_init_scale: float = 0.10  # std of the L_tau initialisation
    temperature_init: float = 1.0  # initial gate softmax temperature (> 0)
    learn_gamma: bool = True       # whether gamma is a trained parameter
    dropout: float = 0.0           # gate MLP dropout (regularisation)

    def __post_init__(self) -> None:
        if self.embed_dim < 1 or self.n_regimes < 1 or self.rank < 1:
            raise ValueError("embed_dim, n_regimes, rank must be >= 1")
        if self.shared_rank < 0:
            raise ValueError("shared_rank must be >= 0")
        if self.gamma_init < 0:
            raise ValueError("gamma_init must be >= 0")
        if self.temperature_init <= 0:
            raise ValueError("temperature_init must be > 0")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in [0, 1)")


# --------------------------------------------------------------------------- #
# The metric
# --------------------------------------------------------------------------- #

class RegimeConditionalMetric(nn.Module):
    """Mixture-of-Mahalanobis-metrics similarity gated by a calibrated posterior.

    Parameters are the per-regime factors ``L`` of shape ``(T, d, r)``, a gate
    network mapping an embedding to ``T`` regime logits, a non-negative cosine
    weight ``gamma``, and a positive softmax ``temperature``.

    Ablation switches (set at construction or toggled as attributes):
      * ``conditioning`` -- if ``False``, the gate is forced uniform, collapsing
        the metric to the single global metric ``mean_tau W_tau + gamma I``
        (the Claim-1 comparator).
      * ``use_cosine`` -- if ``False``, the ``gamma`` cosine term is dropped.
    """

    def __init__(self, cfg: MetricConfig,
                 conditioning: bool = True, use_cosine: bool = True) -> None:
        super().__init__()
        self.cfg = cfg
        self.conditioning = bool(conditioning)
        self.use_cosine = bool(use_cosine)

        d, T, r = cfg.embed_dim, cfg.n_regimes, cfg.rank

        # Per-regime low-rank DELTA factors L_tau: (T, d, r). Small random init;
        # D_tau = L_tau L_tau^T is PSD for any value of L_tau.
        L0 = cfg.factor_init_scale * torch.randn(T, d, r)
        self.L = nn.Parameter(L0)

        # Shared base factor L_0: (d, shared_rank). The per-regime metric is
        # W_tau = W_shared + D_tau with W_shared = L_0 L_0^T (PSD) --- a
        # hierarchical shrinkage parameterisation. Penalising ||L_tau|| (see
        # ``delta_penalty``) drives D_tau -> 0, collapsing every W_tau to the
        # shared global metric; this is Claim 1's boundary (no divergence ->
        # conditioning buys nothing) built into the model, giving graceful
        # degradation to a single global metric when the data does not support
        # regime-specific geometry.
        self.has_shared = cfg.shared_rank > 0
        if self.has_shared:
            self.L_shared = nn.Parameter(cfg.factor_init_scale
                                         * torch.randn(d, cfg.shared_rank))

        # Gate: embedding -> T regime logits. Linear if gate_hidden == 0.
        if cfg.gate_hidden > 0:
            self.gate_net: nn.Module = nn.Sequential(
                nn.Linear(d, cfg.gate_hidden),
                nn.ReLU(),
                nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity(),
                nn.Linear(cfg.gate_hidden, T),
            )
        else:
            self.gate_net = nn.Linear(d, T)

        # gamma >= 0 via softplus of a raw parameter/buffer.
        gamma_raw0 = _inverse_softplus(torch.tensor(float(cfg.gamma_init)))
        if cfg.learn_gamma:
            self._gamma_raw = nn.Parameter(gamma_raw0)
        else:
            self.register_buffer("_gamma_raw", gamma_raw0)

        # temperature > 0 via exp of a log-temperature parameter. Trained jointly
        # or fit post-hoc by calibrate_temperature.
        self._log_temp = nn.Parameter(
            torch.log(torch.tensor(float(cfg.temperature_init)))
        )

    # ------------------------------------------------------------------ #
    # Constrained scalars
    # ------------------------------------------------------------------ #

    @property
    def gamma(self) -> torch.Tensor:
        """Non-negative cosine weight."""
        return F.softplus(self._gamma_raw)

    @property
    def temperature(self) -> torch.Tensor:
        """Positive gate softmax temperature."""
        return torch.exp(self._log_temp)

    # ------------------------------------------------------------------ #
    # Gate (the calibrated regime posterior)
    # ------------------------------------------------------------------ #

    def gate_logits(self, z: torch.Tensor) -> torch.Tensor:
        """Raw (pre-temperature, pre-softmax) regime logits for embeddings ``z``.

        Shape ``(..., embed_dim) -> (..., n_regimes)``. These are the inputs to
        both the training classification loss and post-hoc calibration.
        """
        return self.gate_net(z)

    def gate(self, z: torch.Tensor, *, apply_temperature: bool = True
             ) -> torch.Tensor:
        """Regime posterior ``pi(z)`` on the simplex.

        With ``conditioning=False`` the posterior is forced uniform (the global-
        metric ablation). Otherwise it is ``softmax(logits / temperature)``.
        """
        if not self.conditioning:
            shape = z.shape[:-1] + (self.cfg.n_regimes,)
            return z.new_full(shape, 1.0 / self.cfg.n_regimes)
        logits = self.gate_logits(z)
        if apply_temperature:
            logits = logits / self.temperature
        return F.softmax(logits, dim=-1)

    # ------------------------------------------------------------------ #
    # Projections and scoring
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize(z: torch.Tensor, assume_normalized: bool) -> torch.Tensor:
        if assume_normalized:
            return z
        return F.normalize(z, p=2, dim=-1, eps=1e-12)

    def _project(self, z: torch.Tensor) -> torch.Tensor:
        """Project embeddings into every regime's rank-r space.

        ``z`` of shape ``(M, d)`` -> ``(M, T, r)`` where entry ``[m, tau]`` is
        ``L_tau^T z_m``. This is the factorised half of the bilinear form.
        """
        # einsum: (M,d) x (T,d,r) -> (M,T,r)
        return torch.einsum("md,tdr->mtr", z, self.L)

    def _project_shared(self, z: torch.Tensor) -> torch.Tensor:
        """Project into the shared base space: ``(M, d) -> (M, shared_rank)``."""
        return z @ self.L_shared

    def regime_scores(self, z_q: torch.Tensor, z_c: torch.Tensor, *,
                      assume_normalized: bool = True) -> torch.Tensor:
        """Per-regime DELTA bilinear scores, shape ``(B, T, N)``.

        Entry ``[b, tau, n] = z_q[b]^T D_tau z_c[n]`` with ``D_tau = L_tau L_tau^T``
        --- the *deviation* part of the per-regime metric ``W_tau = W_shared +
        D_tau``. :meth:`forward` mixes these over ``tau`` by the gate and adds the
        shared-base term once. Because the shared base is common to all regimes,
        ranking regimes by this delta score equals ranking by the full ``W_tau``
        contribution, which is what makes it the right quantity for per-regime
        attribution.
        """
        zq = self._normalize(z_q, assume_normalized)
        zc = self._normalize(z_c, assume_normalized)
        q_proj = self._project(zq)                 # (B, T, r)
        c_proj = self._project(zc)                 # (N, T, r)
        # (B,T,r) x (N,T,r) -> (B,T,N)
        return torch.einsum("btr,ntr->btn", q_proj, c_proj)

    def forward(self, z_q: torch.Tensor, z_c: torch.Tensor, *,
                assume_normalized: bool = True, return_gate: bool = False):
        """Conditional similarity matrix ``S`` of shape ``(B, N)``.

        ``S[b, n] = sum_tau pi(z_q[b])_tau * z_q[b]^T W_tau z_c[n]
                    + gamma * z_q[b]^T z_c[n]``.

        With ``return_gate=True`` also returns the gate posterior ``pi`` of shape
        ``(B, T)`` (needed by the calibration and classification losses).
        """
        zq = self._normalize(z_q, assume_normalized)
        zc = self._normalize(z_c, assume_normalized)

        per_regime = self.regime_scores(zq, zc, assume_normalized=True)  # (B,T,N)
        pi = self.gate(zq)                                               # (B,T)
        mixed = torch.einsum("bt,btn->bn", pi, per_regime)               # (B,N)

        if self.has_shared:
            # z_q^T W_shared z_c, regime-independent (sum_tau pi_tau = 1).
            qs, cs = self._project_shared(zq), self._project_shared(zc)
            mixed = mixed + qs @ cs.transpose(-1, -2)                    # (B,N)

        if self.use_cosine:
            cosine = zq @ zc.transpose(-1, -2)                           # (B,N)
            mixed = mixed + self.gamma * cosine

        return (mixed, pi) if return_gate else mixed

    def pairwise(self, z_q: torch.Tensor, z_c: torch.Tensor, *,
                 assume_normalized: bool = True) -> torch.Tensor:
        """Aligned scores ``s(z_q[i], z_c[i])`` for paired inputs, shape ``(B,)``.

        Avoids the ``B x N`` matrix for triple/contrastive losses over aligned
        (query, candidate) pairs.
        """
        zq = self._normalize(z_q, assume_normalized)
        zc = self._normalize(z_c, assume_normalized)
        q_proj = self._project(zq)                 # (B, T, r)
        c_proj = self._project(zc)                 # (B, T, r)
        per_regime = torch.einsum("btr,btr->bt", q_proj, c_proj)  # (B, T)
        pi = self.gate(zq)                                        # (B, T)
        out = torch.einsum("bt,bt->b", pi, per_regime)            # (B,)
        if self.has_shared:
            qs, cs = self._project_shared(zq), self._project_shared(zc)
            out = out + torch.einsum("br,br->b", qs, cs)
        if self.use_cosine:
            out = out + self.gamma * torch.einsum("bd,bd->b", zq, zc)
        return out

    # ------------------------------------------------------------------ #
    # Analysis / audit helpers (materialise the metric; not for hot paths)
    # ------------------------------------------------------------------ #

    def regime_metrics(self) -> torch.Tensor:
        """The per-regime PSD metrics ``W_tau = W_shared + L_tau L_tau^T``.

        Shape ``(T, d, d)``. Each is PSD (sum of two PSD matrices).
        """
        W = torch.einsum("tdr,ter->tde", self.L, self.L)       # D_tau, (T,d,d)
        if self.has_shared:
            W = W + (self.L_shared @ self.L_shared.t()).unsqueeze(0)
        return W

    def delta_penalty(self) -> torch.Tensor:
        """Squared Frobenius norm of the per-regime deltas, ``sum_tau ||L_tau||_F^2``.

        The shrinkage regulariser: ``train.py`` adds ``lambda_shrink *
        delta_penalty()`` to the loss. Larger weight -> deltas shrink -> every
        ``W_tau`` collapses to the shared global metric (graceful degradation).
        """
        return (self.L ** 2).sum()

    def effective_metric(self, pi: torch.Tensor) -> torch.Tensor:
        """The per-query effective metric ``W(pi) = sum_tau pi_tau W_tau + gamma I``.

        ``pi`` of shape ``(B, T)`` -> ``(B, d, d)``. Guaranteed PSD. Provided for
        verification and interpretability, not for scoring at scale.
        """
        W_tau = self.regime_metrics()                       # (T, d, d)
        W = torch.einsum("bt,tde->bde", pi, W_tau)          # (B, d, d)
        if self.use_cosine:
            eye = torch.eye(self.cfg.embed_dim, device=W.device, dtype=W.dtype)
            W = W + self.gamma * eye
        return W

    @torch.no_grad()
    def min_eigenvalue(self, pi: torch.Tensor) -> torch.Tensor:
        """Smallest eigenvalue of ``W(pi)`` per row (>= 0 iff PSD holds)."""
        W = self.effective_metric(pi)
        # Symmetrise to kill round-off asymmetry before eigvalsh.
        W = 0.5 * (W + W.transpose(-1, -2))
        return torch.linalg.eigvalsh(W)[..., 0]

    # ------------------------------------------------------------------ #
    # Post-hoc temperature calibration (Guo et al., 2017)
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _set_temperature(self, value: float) -> None:
        # In-place update of a leaf Parameter must run under no_grad; this is also
        # the portable idiom across torch versions (incl. the server's 2.1).
        with torch.no_grad():
            self._log_temp.copy_(torch.log(torch.tensor(
                float(value), device=self._log_temp.device)))

    def calibrate_temperature(self, val_logits: torch.Tensor,
                              val_labels: torch.Tensor, *,
                              max_iter: int = 200, lr: float = 0.05) -> float:
        """Fit the scalar gate temperature by NLL on held-out regime labels.

        Standard post-hoc calibration: freeze the logits, optimise a single
        temperature ``T`` minimising ``NLL(softmax(logits / T), labels)``. Only
        ``self._log_temp`` is updated; all other parameters are untouched. This
        is the operation that makes the "calibrated gate" of Claim 3 concrete.

        Parameters
        ----------
        val_logits : (M, T) raw gate logits on a validation set.
        val_labels : (M,) integer regime labels in ``[0, T)``.

        Returns the fitted temperature (float).
        """
        if val_logits.ndim != 2 or val_logits.shape[1] != self.cfg.n_regimes:
            raise ValueError(f"val_logits must be (M, {self.cfg.n_regimes})")
        if val_labels.ndim != 1 or val_labels.shape[0] != val_logits.shape[0]:
            raise ValueError("val_labels must be (M,) aligned with val_logits")

        logits = val_logits.detach()
        labels = val_labels.detach().long()
        log_t = torch.zeros(1, requires_grad=True, device=logits.device)
        opt = torch.optim.LBFGS([log_t], lr=lr, max_iter=max_iter,
                                line_search_fn="strong_wolfe")

        def _closure():
            opt.zero_grad()
            t = torch.exp(log_t)
            loss = F.cross_entropy(logits / t, labels)
            loss.backward()
            return loss

        opt.step(_closure)
        fitted = float(torch.exp(log_t.detach()))
        # Guard against pathological fits (empty/degenerate validation sets).
        if not (fitted > 0 and fitted < 1e3):
            fitted = 1.0
        self._set_temperature(fitted)
        return fitted

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def extra_repr(self) -> str:
        return (f"embed_dim={self.cfg.embed_dim}, n_regimes={self.cfg.n_regimes}, "
                f"rank={self.cfg.rank}, shared_rank={self.cfg.shared_rank}, "
                f"conditioning={self.conditioning}, use_cosine={self.use_cosine}")


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #

def _inverse_softplus(y: torch.Tensor) -> torch.Tensor:
    """Return ``x`` such that ``softplus(x) = y`` (for initialising gamma_raw).

    ``softplus^{-1}(y) = log(exp(y) - 1)``, computed stably. For large ``y`` this
    approaches ``y``; for small ``y`` it is dominated by ``log(y)``.
    """
    y = torch.as_tensor(y, dtype=torch.float32)
    # log(expm1(y)) with a floor to avoid log(0) at y -> 0.
    return torch.log(torch.clamp(torch.expm1(y), min=1e-12))

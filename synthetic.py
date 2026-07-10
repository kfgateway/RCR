"""Numerical validation of the regime-conditional-similarity theory.

This module is the empirical truth-check on the three theoretical claims that
underpin the paper. It is implemented in pure NumPy with hand-derived
gradients --- *independently* of the production metric (``metric.py``, which
is PyTorch) --- so that agreement between theory, this harness, and the
production model is a genuine three-way corroboration, not one implementation
validating itself.

The generative model (conditional-similarity form, faithful to the theory)
--------------------------------------------------------------------------
This is the crucial modelling choice. Items are *not* partitioned into
regimes. Instead:

  * A latent vector ``x_i`` is drawn isotropically on the unit sphere in
    :math:`\\mathbb{R}^d`; the *observed* signature is a noisy, renormalised
    version ``z_i = normalise(x_i + nu * noise)``. The observation noise gives
    retrieval an irreducible (Bayes) floor above zero, as in any real task.

  * A **regime** :math:`\\tau` is a *condition*, not a cluster: it owns a
    rank-``r`` subspace :math:`U_\\tau` with projector
    :math:`P_\\tau=B_\\tau B_\\tau^\\top`. Under condition :math:`\\tau`, two items
    are *similar* to the extent their **latent** projections onto
    :math:`U_\\tau` align: the ground-truth relevance of candidate ``c`` to a
    query ``(q, \\tau)`` is high when :math:`x_q^\\top P_\\tau x_c` is large.

  * A **query** carries its condition, ``(z_q, \\tau_q)``. The same pair of
    items can therefore be relevant under one regime and irrelevant under
    another --- which is precisely what a single fixed metric cannot represent
    (the conditional-similarity-network insight) and what makes regime
    conditioning provably necessary.

The true regime-conditional similarity is :math:`s(z_q,z_c)=z_q^\\top P_{\\tau_q} z_c`.
Relevance is defined on the *clean* latents while every scorer sees only the
*noisy* observations, so even the oracle has a positive, well-defined risk.

Controlled divergence. For the two-regime sweep, :math:`U_2` is :math:`U_1`
rotated by one angle :math:`\\theta` (all principal angles :math:`=\\theta`), so

    D(U_1,U_2) = (1/2)||P_1 - P_2||_F^2 = sum_i sin^2(theta_i) = r*sin^2(theta)

runs from 0 (identical conditions) to ``r`` (orthogonal conditions).

The three claims, and what this harness does about each
-------------------------------------------------------
* **Claim 1 (global-metric suboptimality).** The best *fitted* global metric
  cannot serve conflicting conditions; its retrieval risk is floored strictly
  above the oracle and the gap grows with ``D`` (vanishing at ``D=0``).

* **Claim 2 (Bayes-optimality of the oracle mixture).** With the true
  condition and true per-regime metrics, the mixture attains the information
  floor. Partly definitional; we corroborate it (oracle gap-to-floor ~ 0 while
  the global gap is positive) without claiming numerical novelty for it.

* **Claim 3 (calibration controls excess risk).** Degrading the gate by a
  controlled amount, we verify the L1 backbone
  ``R(pi_hat) - R* <= C * E||pi_hat - pi*||_1``: excess risk rises ~linearly in
  the gate's L1 deviation, with a slope (the constant ``C``) that itself grows
  with the geometry gap ``D``. The tighter "calibration-error" restatement is a
  separate theoretical step and is not claimed here.

Reproducibility
---------------
Every experiment threads one seeded :class:`numpy.random.Generator`; each
results blob is stamped with a SHA-256 of its resolved configuration (via
:mod:`provenance`), so any figure traces to the exact settings behind it.

References
----------
Veit, A., Belongie, S., Karaletsos, T. (2017). Conditional similarity networks.
CVPR. (The motivating observation that one fixed metric cannot represent
conditional similarity; the generative model here turns it into a provable,
divergence-indexed statement --- and, unlike their given-condition masks, the
condition is inferred through a calibrated gate.)
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np

try:
    import provenance as _prov
    _HAVE_PROV = True
except Exception:  # pragma: no cover
    _HAVE_PROV = False

__all__ = [
    "SyntheticConfig",
    "make_rotated_subspaces",
    "subspace_divergence",
    "generate_items",
    "sample_conditional_triples",
    "pairwise_retrieval_risk",
    "fit_global_metric",
    "oracle_projectors",
    "claim1_global_suboptimality",
    "claim2_bayes_gap",
    "claim3_calibration_controls_risk",
    "run_all",
]

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class SyntheticConfig:
    """Resolved settings for the synthetic validation."""

    # Geometry.
    dim: int = 32
    rank: int = 4
    n_regimes: int = 2
    # Items / observation.
    n_items: int = 1000
    obs_noise: float = 0.08          # nu: observation-noise scale -> Bayes floor
    # Relevance / triple construction.
    cand_pool: int = 64              # candidates screened per triple
    rel_margin: float = 0.05         # min latent-score gap for a usable triple
    n_risk_triples: int = 20_000
    n_bootstrap: int = 500
    # Global-metric comparator fitting.
    fit_rank: int = 8
    fit_steps: int = 800
    fit_lr: float = 5e-2
    fit_batch: int = 512
    fit_pool: int = 6000             # triples pre-sampled for fitting
    # Sweeps.
    n_theta: int = 13
    n_lambda: int = 11
    claim3_thetas: tuple[float, ...] = (0.35, 0.70, 1.20)
    # Reproducibility.
    seed: int = 20260517

    def __post_init__(self) -> None:
        if self.dim < 2 * self.rank:
            raise ValueError(
                f"dim ({self.dim}) must be >= 2*rank ({2*self.rank})"
            )
        if self.n_regimes < 2:
            raise ValueError("n_regimes must be >= 2")
        if self.obs_noise < 0:
            raise ValueError("obs_noise must be non-negative")
        if self.n_items < self.cand_pool + 2:
            raise ValueError("n_items too small for cand_pool")

    def fingerprint(self) -> str:
        if _HAVE_PROV:
            return _prov.hash_bytes(_prov.canonical_json(asdict(self)))
        import hashlib
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode("utf-8")
        ).hexdigest()


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #

def make_rotated_subspaces(
    dim: int, rank: int, theta: float
) -> tuple[np.ndarray, np.ndarray]:
    """Two rank-``r`` orthonormal bases separated by a single angle ``theta``.

    ``U_1 = span(e_0..e_{r-1})``; ``U_2`` rotates each ``e_i`` into the plane
    ``(e_i, e_{r+i})`` by ``theta``, so every principal angle equals ``theta``.
    """
    if not (0.0 <= theta <= np.pi / 2 + 1e-9):
        raise ValueError(f"theta in [0, pi/2]; got {theta}")
    B1 = np.zeros((dim, rank))
    B2 = np.zeros((dim, rank))
    c, s = np.cos(theta), np.sin(theta)
    for i in range(rank):
        B1[i, i] = 1.0
        B2[i, i] = c
        B2[rank + i, i] = s
    return B1, B2


def subspace_divergence(B1: np.ndarray, B2: np.ndarray) -> float:
    """D = (1/2)||P_1 - P_2||_F^2 = sum_i sin^2(theta_i)."""
    P1, P2 = B1 @ B1.T, B2 @ B2.T
    return 0.5 * float(np.sum((P1 - P2) ** 2))


def oracle_projectors(bases: list[np.ndarray]) -> list[np.ndarray]:
    """True per-regime projectors ``P_tau = B_tau B_tau^T``."""
    return [B @ B.T for B in bases]


# --------------------------------------------------------------------------- #
# Items and observation noise
# --------------------------------------------------------------------------- #

def _unit_rows(A: np.ndarray) -> np.ndarray:
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)


def generate_items(
    cfg: SyntheticConfig, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Draw latent items and their noisy observed signatures.

    Returns
    -------
    X : ndarray (N, dim)   -- clean latents (define ground-truth relevance)
    Z : ndarray (N, dim)   -- observed signatures the scorers actually see
    """
    X = _unit_rows(rng.standard_normal((cfg.n_items, cfg.dim)))
    Z = _unit_rows(X + cfg.obs_noise * rng.standard_normal(X.shape))
    return X, Z


# --------------------------------------------------------------------------- #
# Conditional triple sampling  (relevance defined on clean latents)
# --------------------------------------------------------------------------- #

def sample_conditional_triples(
    X: np.ndarray,
    projectors: list[np.ndarray],
    n: int,
    cfg: SyntheticConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample ``(q, c_plus, c_minus, tau)`` conditional triples.

    For each triple: draw a query item ``q`` and a condition ``tau``; screen a
    candidate pool; set ``c_plus`` = pool item with the *largest* latent
    conditional score ``x_q^T P_tau x_c`` and ``c_minus`` = the *smallest*.
    Triples whose latent score gap is below ``rel_margin`` are discarded so the
    ground-truth ordering is unambiguous. Relevance thus depends on the
    condition: the same items can swap roles under a different ``tau``.
    """
    N, T = X.shape[0], len(projectors)
    q = rng.integers(0, N, size=n)
    tau = rng.integers(0, T, size=n)
    cp = np.empty(n, dtype=np.int64)
    cm = np.empty(n, dtype=np.int64)
    keep = np.ones(n, dtype=bool)
    # Pre-project X by each regime once: XP[t] = X @ P_t  (N, dim)
    XP = [X @ P for P in projectors]
    for i in range(n):
        qi, ti = q[i], tau[i]
        cand = rng.integers(0, N, size=cfg.cand_pool)
        sc = XP[ti][qi] @ X[cand].T                 # latent scores, (pool,)
        hi, lo = int(np.argmax(sc)), int(np.argmin(sc))
        if sc[hi] - sc[lo] < cfg.rel_margin or cand[hi] == cand[lo]:
            keep[i] = False
            continue
        cp[i], cm[i] = cand[hi], cand[lo]
    return q[keep], cp[keep], cm[keep], tau[keep]


# --------------------------------------------------------------------------- #
# Scorers  (all condition-aware; global ignores the condition by design)
# --------------------------------------------------------------------------- #

def _make_oracle_scorer(Z: np.ndarray, projectors: list[np.ndarray]):
    """Score with the TRUE condition's projector: z_q^T P_{tau} z_c."""
    ZP = [Z @ P for P in projectors]               # (N, dim) per regime
    def _fn(iq, ic, tau):
        out = np.empty(iq.shape[0])
        for t in range(len(projectors)):
            m = tau == t
            if np.any(m):
                out[m] = np.einsum("md,md->m", ZP[t][iq[m]], Z[ic[m]])
        return out
    return _fn


def _make_gated_scorer(Z, projectors, gate: np.ndarray):
    """Score with a per-triple gate: sum_t gate[i,t] z_q^T P_t z_c.

    ``gate`` has shape (n_triples, T); it is the (possibly miscalibrated)
    posterior the retrieval uses instead of the true one-hot condition.
    """
    ZP = [Z @ P for P in projectors]
    def _fn(iq, ic, tau):
        out = np.zeros(iq.shape[0])
        for t in range(len(projectors)):
            s_t = np.einsum("md,md->m", ZP[t][iq], Z[ic])
            out += gate[:, t] * s_t
        return out
    return _fn


def _make_global_scorer(Z: np.ndarray, W: np.ndarray):
    """Single fixed metric, condition-blind: z_q^T W z_c."""
    ZW = Z @ W
    def _fn(iq, ic, tau):
        return np.einsum("md,md->m", ZW[iq], Z[ic])
    return _fn


# --------------------------------------------------------------------------- #
# Retrieval-risk estimator
# --------------------------------------------------------------------------- #

def pairwise_retrieval_risk(
    triples: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    score_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    cfg: SyntheticConfig,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Risk = P[ s(q, c-) >= s(q, c+) ] over the given triples, + bootstrap CI.

    Ties count as errors (conservative). ``score_fn(iq, ic, tau)`` returns the
    aligned score vector, so global and conditional scorers share one code path.
    """
    q, cp, cm, tau = triples
    if q.size == 0:
        raise ValueError("no valid triples")
    err = (score_fn(q, cm, tau) >= score_fn(q, cp, tau)).astype(np.float64)
    point = float(err.mean())
    n = err.shape[0]
    boot = np.empty(cfg.n_bootstrap)
    for b in range(cfg.n_bootstrap):
        boot[b] = err[rng.integers(0, n, size=n)].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"risk": point, "ci_lo": float(lo), "ci_hi": float(hi),
            "n_triples": int(n)}


# --------------------------------------------------------------------------- #
# Global-metric fitting  (the honest Claim-1 comparator)
# --------------------------------------------------------------------------- #

class _Adam:
    def __init__(self, shape, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = np.zeros(shape); self.v = np.zeros(shape); self.t = 0

    def step(self, p, g):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * (g * g)
        mh = self.m / (1 - self.b1 ** self.t)
        vh = self.v / (1 - self.b2 ** self.t)
        return p - self.lr * mh / (np.sqrt(vh) + self.eps)


def fit_global_metric(
    Z: np.ndarray,
    triples: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    cfg: SyntheticConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Fit the best global PSD metric ``W = L L^T`` by a contrastive objective.

    The global metric is trained on conditional triples but *without* the
    condition label --- it must rank ``c+`` above ``c-`` across all conditions
    with one metric. That is exactly the compromise Claim 1 predicts it cannot
    win. Objective: mean ``softplus(-(s_pos - s_neg))``, ``s(a,b)=(L^T a)^T(L^T b)``,
    with the fully analytic gradient ``ds/dL = a (L^T b)^T + b (L^T a)^T``.
    """
    d, k = cfg.dim, cfg.fit_rank
    L = 0.1 * rng.standard_normal((d, k))
    opt = _Adam(L.shape, cfg.fit_lr)
    q, cp, cm, _ = triples
    pool = q.size
    if pool == 0:
        raise ValueError("fit_global_metric: no triples")
    for _ in range(cfg.fit_steps):
        sel = rng.integers(0, pool, size=min(cfg.fit_batch, pool))
        zq, zp, zn = Z[q[sel]], Z[cp[sel]], Z[cm[sel]]
        Lq, Lp, Ln = zq @ L, zp @ L, zn @ L
        diff = np.einsum("mk,mk->m", Lq, Lp) - np.einsum("mk,mk->m", Lq, Ln)
        g = 1.0 / (1.0 + np.exp(diff))               # sigmoid(-diff)
        m = zq.shape[0]
        w = (-g / m)[:, None]
        grad = ((zq * w).T @ Lp + (zp * w).T @ Lq
                - (zq * w).T @ Ln - (zn * w).T @ Lq)
        L = opt.step(L, grad)
    return L @ L.T


# --------------------------------------------------------------------------- #
# Claim 1 --- global-metric suboptimality across a divergence sweep
# --------------------------------------------------------------------------- #

def _prep(cfg, theta, rng):
    B1, B2 = make_rotated_subspaces(cfg.dim, cfg.rank, float(theta))
    bases = [B1, B2]
    X, Z = generate_items(cfg, rng)
    Ps = oracle_projectors(bases)
    D = subspace_divergence(B1, B2)
    return bases, Ps, X, Z, D


def claim1_global_suboptimality(cfg: SyntheticConfig) -> dict:
    """Sweep ``D``; compare best-global vs oracle retrieval risk on shared triples."""
    if cfg.n_regimes != 2:
        raise ValueError("claim1 sweep is defined for n_regimes == 2")
    master = np.random.default_rng(cfg.seed)
    thetas = np.linspace(0.0, np.pi / 2, cfg.n_theta)
    rows = []
    for theta in thetas:
        rng = np.random.default_rng(int(master.integers(0, 2**63 - 1)))
        _, Ps, X, Z, D = _prep(cfg, theta, rng)
        triples = sample_conditional_triples(X, Ps, cfg.n_risk_triples, cfg, rng)

        r_oracle = pairwise_retrieval_risk(
            triples, _make_oracle_scorer(Z, Ps), cfg, rng)

        fit_tr = sample_conditional_triples(X, Ps, cfg.fit_pool, cfg, rng)
        W = fit_global_metric(Z, fit_tr, cfg, rng)
        r_global = pairwise_retrieval_risk(
            triples, _make_global_scorer(Z, W), cfg, rng)

        rows.append({
            "theta": float(theta), "divergence_D": float(D),
            "risk_oracle": r_oracle["risk"],
            "risk_oracle_ci": [r_oracle["ci_lo"], r_oracle["ci_hi"]],
            "risk_global": r_global["risk"],
            "risk_global_ci": [r_global["ci_lo"], r_global["ci_hi"]],
            "gap": r_global["risk"] - r_oracle["risk"],
        })
    Ds = np.array([r["divergence_D"] for r in rows])
    gaps = np.array([r["gap"] for r in rows])
    corr = float(np.corrcoef(Ds, gaps)[0, 1]) if len(rows) > 2 else float("nan")
    return {"claim": "global_metric_suboptimality",
            "config_fingerprint": cfg.fingerprint(),
            "rows": rows, "gap_vs_D_correlation": corr}


# --------------------------------------------------------------------------- #
# Claim 2 --- oracle mixture attains the information floor (scaffold)
# --------------------------------------------------------------------------- #

def claim2_bayes_gap(cfg: SyntheticConfig, theta: float = 1.0) -> dict:
    """Oracle gap-to-floor ~ 0 while the best global metric's gap is positive."""
    rng = np.random.default_rng(cfg.seed + 7)
    _, Ps, X, Z, _ = _prep(cfg, theta, rng)

    hi_cfg = SyntheticConfig(**{**asdict(cfg),
                                "n_risk_triples": max(cfg.n_risk_triples * 3, 60_000)})
    floor_tr = sample_conditional_triples(X, Ps, hi_cfg.n_risk_triples, hi_cfg, rng)
    floor = pairwise_retrieval_risk(floor_tr, _make_oracle_scorer(Z, Ps),
                                    hi_cfg, rng)["risk"]

    triples = sample_conditional_triples(X, Ps, cfg.n_risk_triples, cfg, rng)
    r_oracle = pairwise_retrieval_risk(triples, _make_oracle_scorer(Z, Ps),
                                       cfg, rng)["risk"]
    fit_tr = sample_conditional_triples(X, Ps, cfg.fit_pool, cfg, rng)
    W = fit_global_metric(Z, fit_tr, cfg, rng)
    r_global = pairwise_retrieval_risk(triples, _make_global_scorer(Z, W),
                                       cfg, rng)["risk"]
    return {"claim": "bayes_optimality_scaffold",
            "config_fingerprint": cfg.fingerprint(), "theta": float(theta),
            "information_floor": floor, "oracle_risk": r_oracle,
            "oracle_gap_to_floor": r_oracle - floor,
            "global_risk": r_global, "global_gap_to_floor": r_global - floor}


# --------------------------------------------------------------------------- #
# Claim 3 --- calibration (L1 backbone) controls excess risk
# --------------------------------------------------------------------------- #

def _degrade_gate(pi_star: np.ndarray, lam: float) -> tuple[np.ndarray, float]:
    """Blend one-hot posterior toward uniform by ``lam``; return (gate, mean L1)."""
    T = pi_star.shape[1]
    pi_hat = (1.0 - lam) * pi_star + lam * (1.0 / T)
    l1 = float(np.mean(np.sum(np.abs(pi_hat - pi_star), axis=1)))
    return pi_hat, l1


def claim3_calibration_controls_risk(cfg: SyntheticConfig) -> dict:
    """Verify excess risk rises ~linearly in gate L1 deviation, slope growing w/ D."""
    master = np.random.default_rng(cfg.seed + 13)
    series = []
    for theta in cfg.claim3_thetas:
        rng = np.random.default_rng(int(master.integers(0, 2**63 - 1)))
        _, Ps, X, Z, D = _prep(cfg, theta, rng)
        triples = sample_conditional_triples(X, Ps, cfg.n_risk_triples, cfg, rng)
        q, cp, cm, tau = triples
        pi_star = np.eye(cfg.n_regimes)[tau]         # (n_triples, T) one-hot

        r_star = pairwise_retrieval_risk(
            triples, _make_gated_scorer(Z, Ps, pi_star), cfg, rng)["risk"]

        pts = []
        for lam in np.linspace(0.0, 1.0, cfg.n_lambda):
            gate, l1 = _degrade_gate(pi_star, float(lam))
            r_hat = pairwise_retrieval_risk(
                triples, _make_gated_scorer(Z, Ps, gate), cfg, rng)["risk"]
            pts.append({"lambda": float(lam), "l1": l1,
                        "excess_risk": r_hat - r_star})
        l1s = np.array([p["l1"] for p in pts])
        exc = np.array([p["excess_risk"] for p in pts])
        slope = float(np.polyfit(l1s, exc, 1)[0]) if len(pts) > 1 else float("nan")
        series.append({"theta": float(theta), "divergence_D": float(D),
                       "risk_star": r_star, "points": pts,
                       "empirical_C": slope})
    Ds = np.array([s["divergence_D"] for s in series])
    Cs = np.array([s["empirical_C"] for s in series])
    corr = float(np.corrcoef(Ds, Cs)[0, 1]) if len(series) > 2 else float("nan")
    return {"claim": "calibration_controls_excess_risk_L1",
            "config_fingerprint": cfg.fingerprint(),
            "series": series, "C_vs_D_correlation": corr}


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def _figure_claim1(res, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = res["rows"]
    D = [r["divergence_D"] for r in rows]
    ro = [r["risk_oracle"] for r in rows]
    rg = [r["risk_global"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(D, rg, "o-", label="best global metric (fitted)")
    ax.plot(D, ro, "s-", label="regime-conditional (oracle)")
    ax.fill_between(D, ro, rg, alpha=0.15)
    ax.set_xlabel(r"geometry divergence $D=\sum_i \sin^2\theta_i$")
    ax.set_ylabel("pairwise retrieval risk")
    ax.set_title("Claim 1: global-metric suboptimality grows with divergence")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(path, dpi=200)
    plt.close(fig)


def _figure_claim3(res, path):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for s in res["series"]:
        l1 = [p["l1"] for p in s["points"]]
        exc = [p["excess_risk"] for p in s["points"]]
        ax.plot(l1, exc, "o-",
                label=fr"$D={s['divergence_D']:.2f}$ ($\hat C={s['empirical_C']:.3f}$)")
    ax.set_xlabel(r"gate deviation $\mathbb{E}\|\hat\pi-\pi^\star\|_1$")
    ax.set_ylabel(r"excess retrieval risk $R(\hat\pi)-R^\star$")
    ax.set_title("Claim 3: excess risk controlled (linearly) by gate deviation")
    ax.legend(frameon=False, fontsize=8); fig.tight_layout(); fig.savefig(path, dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

def run_all(cfg: SyntheticConfig, outdir: str, make_figures: bool = True) -> dict:
    os.makedirs(outdir, exist_ok=True)
    results = {
        "config": asdict(cfg),
        "config_fingerprint": cfg.fingerprint(),
        "claim1": claim1_global_suboptimality(cfg),
        "claim2": claim2_bayes_gap(cfg),
        "claim3": claim3_calibration_controls_risk(cfg),
    }
    with open(os.path.join(outdir, "synthetic_results.json"), "w",
              encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    if make_figures:
        _figure_claim1(results["claim1"],
                       os.path.join(outdir, "fig_claim1_divergence.png"))
        _figure_claim3(results["claim3"],
                       os.path.join(outdir, "fig_claim3_calibration.png"))
    return results


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Regime-conditional theory validation.")
    p.add_argument("--outdir", default="results/synthetic")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--no-figures", action="store_true")
    return p


def main(argv=None) -> None:
    args = _build_argparser().parse_args(argv)
    overrides = {}
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.quick:
        overrides.update(n_items=400, n_risk_triples=4000, n_bootstrap=100,
                         fit_steps=250, fit_pool=3000, n_theta=7, n_lambda=6)
    cfg = SyntheticConfig(**overrides)
    res = run_all(cfg, args.outdir, make_figures=not args.no_figures)
    print("Claim 1  gap~D correlation :",
          round(res["claim1"]["gap_vs_D_correlation"], 4))
    print("Claim 2  oracle gap-to-floor:",
          round(res["claim2"]["oracle_gap_to_floor"], 4),
          "| global gap-to-floor:",
          round(res["claim2"]["global_gap_to_floor"], 4))
    print("Claim 3  C~D correlation   :",
          round(res["claim3"]["C_vs_D_correlation"], 4))


if __name__ == "__main__":
    main()

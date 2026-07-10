"""Held-out evaluation with confidence intervals --- the paper's empirical crux.

This module measures the trained framework on the *test* split with the rigour a
top venue expects, and it is deliberately built to report the truth rather than
a favourable slice of it:

  * **Full metric suite.** Recall@k, MRR, nDCG@k (ranking quality), the *pairwise
    retrieval risk* the theory bounds, and the gate's Expected Calibration Error
    (ECE), which is the operational face of Claim 3.

  * **Multi-seed with confidence intervals.** Every configuration is trained
    across many seeds; metrics are aggregated with bootstrap 95% CIs. On a small
    corpus this is essential --- a single run is noise.

  * **Ablations, paired.** The conditional model is compared against the
    single-global-metric ablation (``conditioning=False``) and the no-cosine
    ablation (``use_cosine=False``). The conditioning comparison is reported both
    unpaired (per-config CIs) and **paired by seed** (delta per seed + CI), which
    controls seed variance and is the honest test of "does conditioning help?".

  * **Cross-market transfer.** A secondary, explicitly-scoped check: train on the
    emerging-market corpus and evaluate queries from the advanced-economy (JST)
    corpus on the shared :data:`data.COMMON_FEATURES`, reporting the same metrics.

Retrieval protocol
------------------
Test queries (crisis cases) retrieve against the library of *all non-test* cases
(train + validation folded back in after model selection). Relevance is same
crisis regime; calm cases are real hard negatives. This is the realistic
deployment setup and it never leaks test cases into the library.

Honest framing
--------------
As diagnosed during training, the real-data conditioning advantage on this small
corpus is modest and concentrated on the pairwise-risk metric; it is not a large
effect on MRR. This module is designed to quantify exactly that --- with CIs, so
the paper can state whether the effect is distinguishable from zero rather than
over-claim it.

References
----------
Musgrave, K., Belongie, S., Lim, S.-N. (2020). A metric learning reality check.
ECCV. (The evaluation protocol --- validation-based model selection, fixed
training budget across methods, multi-seed reporting with confidence intervals
--- follows their recommendations, so an honest null reads as rigour.)
Guo, C., Pleiss, G., Sun, Y., Weinberger, K.Q. (2017). On calibration of modern
neural networks. ICML. (ECE and MCE definitions for the gate.)
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass, replace
from typing import Sequence

import numpy as np
import torch

from data import DataConfig, gfdd_source, jst_source, load_laeven_valencia, \
    build_library, COMMON_FEATURES
from encoder import EncoderConfig
from metric import MetricConfig, REGIME_NAMES
from train import Corpus, TrainConfig, load_snapshot, train

__all__ = [
    "EvalConfig", "ndcg_at_k", "expected_calibration_error", "bootstrap_ci",
    "evaluate_model", "run_multiseed", "evaluate_transfer",
    "maximum_calibration_error",
]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class EvalConfig:
    """Evaluation protocol settings."""

    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)   # paper: 20-50; more = tighter CIs
    ks: tuple[int, ...] = (1, 5, 10)
    ece_bins: int = 10
    bootstrap_n: int = 2000
    alpha: float = 0.05                        # 95% CI
    train_epochs: int = 200

    def __post_init__(self) -> None:
        if len(self.seeds) < 1:
            raise ValueError("need at least one seed")
        if self.ece_bins < 1:
            raise ValueError("ece_bins must be >= 1")


# --------------------------------------------------------------------------- #
# Metric helpers
# --------------------------------------------------------------------------- #

def ndcg_at_k(relevance_ranked: torch.Tensor, n_relevant: int, k: int) -> float:
    """Binary nDCG@k from a boolean relevance vector ordered by descending score.

    ``DCG = sum_{i<=k} rel_i / log2(i+1)``; the ideal DCG places all relevant
    items first. Returns 0 if there are no relevant items.
    """
    if n_relevant <= 0:
        return 0.0
    topk = relevance_ranked[:k].float()
    if topk.numel() == 0:
        return 0.0
    discounts = 1.0 / torch.log2(torch.arange(2, 2 + topk.numel(),
                                              dtype=torch.float32))
    dcg = float((topk * discounts).sum())
    ideal_n = min(k, n_relevant)
    idcg = float((1.0 / torch.log2(torch.arange(2, 2 + ideal_n,
                                                dtype=torch.float32))).sum())
    return dcg / idcg if idcg > 0 else 0.0


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray,
                               n_bins: int = 10) -> float:
    """Expected Calibration Error (Guo et al., 2017) with equal-width bins.

    ``ECE = sum_b (|B_b|/N) * |acc(B_b) - conf(B_b)|`` over confidence bins.
    """
    n = len(confidence)
    if n == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (confidence > lo) & (confidence <= hi) if b > 0 else \
               (confidence >= lo) & (confidence <= hi)
        m = int(mask.sum())
        if m == 0:
            continue
        acc = float(correct[mask].mean())
        conf = float(confidence[mask].mean())
        ece += (m / n) * abs(acc - conf)
    return float(ece)


def maximum_calibration_error(confidence: np.ndarray, correct: np.ndarray,
                             n_bins: int = 10) -> float:
    """Maximum Calibration Error (Guo et al., 2017): the worst-case bin gap.

    ``MCE = max_b |acc(B_b) - conf(B_b)|``. Reported alongside ECE because for a
    high-risk application (crisis analogy retrieval) the worst-case bin
    miscalibration matters, not only the sample-weighted average.
    """
    n = len(confidence)
    if n == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    worst = 0.0
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (confidence > lo) & (confidence <= hi) if b > 0 else \
               (confidence >= lo) & (confidence <= hi)
        if int(mask.sum()) == 0:
            continue
        worst = max(worst, abs(float(correct[mask].mean())
                               - float(confidence[mask].mean())))
    return float(worst)


def bootstrap_ci(values: Sequence[float], *, n: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of ``values`` (over seeds)."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return (0.0, 0.0)
    if v.size == 1:
        return (float(v[0]), float(v[0]))
    rng = np.random.default_rng(seed)
    boots = np.array([v[rng.integers(0, v.size, v.size)].mean()
                      for _ in range(n)])
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return lo, hi


def _mean(x: list[float]) -> float:
    return float(np.mean(x)) if x else 0.0


# --------------------------------------------------------------------------- #
# Single-model evaluation
# --------------------------------------------------------------------------- #

@torch.no_grad()
def evaluate_model(encoder, metric, corpus: Corpus,
                   library_idx: np.ndarray, query_idx: np.ndarray,
                   eval_cfg: EvalConfig, device: str = "cpu") -> dict:
    """Evaluate a trained model: test crisis queries retrieved against a library.

    Returns a dict with Recall@k, nDCG@k, MRR, pairwise_risk, gate ECE and
    accuracy, and the effective query count.
    """
    encoder.eval(); metric.eval()
    dev = torch.device(device)
    lib_idx = torch.as_tensor(library_idx, device=dev)
    q_idx = torch.as_tensor(query_idx, device=dev)

    lib_emb = encoder(corpus.traj[lib_idx])
    lib_lab = corpus.regime_id[lib_idx]

    # Queries are the crisis cases within query_idx.
    q_is_cr = corpus.is_crisis[q_idx]
    q_sel = q_idx[q_is_cr]
    if q_sel.numel() == 0:
        return {"n_queries": 0}
    q_emb = encoder(corpus.traj[q_sel])
    q_lab = corpus.regime_id[q_sel]

    S = metric(q_emb, lib_emb)                    # (Q, L)
    ks = eval_cfg.ks
    recall = {k: [] for k in ks}
    ndcg = {k: [] for k in ks}
    mrr: list[float] = []
    risk: list[float] = []

    for i in range(q_sel.numel()):
        rel = (lib_lab == q_lab[i])
        n_rel = int(rel.sum())
        if n_rel == 0:
            continue
        order = torch.argsort(S[i], descending=True)
        rr = rel[order]
        first = int(torch.nonzero(rr, as_tuple=False)[0]) + 1
        mrr.append(1.0 / first)
        for k in ks:
            topk = rr[:k]
            recall[k].append(float(topk.sum()) / n_rel)
            ndcg[k].append(ndcg_at_k(rr, n_rel, k))
        pos_s, neg_s = S[i][rel], S[i][~rel]
        if neg_s.numel() > 0:
            risk.append(float((neg_s.unsqueeze(0) >= pos_s.unsqueeze(1))
                              .float().mean()))

    # Gate calibration on the same queries.
    pi = metric.gate(q_emb)
    conf, pred = pi.max(dim=1)
    correct = (pred == q_lab).float()
    conf_np, corr_np = conf.cpu().numpy(), correct.cpu().numpy()
    ece = expected_calibration_error(conf_np, corr_np, eval_cfg.ece_bins)
    mce = maximum_calibration_error(conf_np, corr_np, eval_cfg.ece_bins)

    out = {"mrr": _mean(mrr), "pairwise_risk": _mean(risk),
           "ece": ece, "mce": mce, "gate_acc": float(correct.mean()),
           "n_queries": len(mrr)}
    for k in ks:
        out[f"recall@{k}"] = _mean(recall[k])
        out[f"ndcg@{k}"] = _mean(ndcg[k])
    return out


# --------------------------------------------------------------------------- #
# Multi-seed evaluation with ablations
# --------------------------------------------------------------------------- #

_CONFIGS: dict[str, dict] = {
    "conditional": {"conditioning": True, "use_cosine": True},
    "global": {"conditioning": False, "use_cosine": True},
    "no_cosine": {"conditioning": True, "use_cosine": False},
}


def _aggregate(per_seed: list[dict], eval_cfg: EvalConfig, metric_keys: list[str]
               ) -> dict:
    """Mean + bootstrap CI over seeds for each metric key."""
    agg = {}
    for key in metric_keys:
        vals = [d[key] for d in per_seed if key in d]
        lo, hi = bootstrap_ci(vals, n=eval_cfg.bootstrap_n, alpha=eval_cfg.alpha)
        agg[key] = {"mean": _mean(vals), "std": float(np.std(vals)) if vals else 0.0,
                    "ci": [lo, hi], "n_seeds": len(vals)}
    return agg


def run_multiseed(corpus: Corpus, enc_cfg: EncoderConfig, met_cfg: MetricConfig,
                  train_cfg: TrainConfig, eval_cfg: EvalConfig, *,
                  device: str = "cpu", verbose: bool = True) -> dict:
    """Train + evaluate every configuration across seeds; aggregate with CIs.

    For each seed and configuration, the model is trained (early-stopped on
    validation MRR), then evaluated on the test split with the library = all
    non-test cases. Returns per-config aggregates and the paired conditioning
    delta (conditional - global) per seed.
    """
    per_seed: dict[str, list[dict]] = {name: [] for name in _CONFIGS}
    metric_keys = (["mrr", "pairwise_risk", "ece", "mce", "gate_acc"]
                   + [f"recall@{k}" for k in eval_cfg.ks]
                   + [f"ndcg@{k}" for k in eval_cfg.ks])

    for seed in eval_cfg.seeds:
        tc = replace(train_cfg, seed=seed, epochs=eval_cfg.train_epochs)
        for name, flags in _CONFIGS.items():
            res = train(corpus, enc_cfg, met_cfg, tc,
                        conditioning=flags["conditioning"],
                        use_cosine=flags["use_cosine"], verbose=False)
            splits = res["splits"]
            library_idx = np.array(splits["train_proper"] + splits["val"], dtype=int)
            query_idx = np.array(splits["test"], dtype=int)
            m = evaluate_model(res["encoder"], res["metric"], corpus,
                               library_idx, query_idx, eval_cfg, device)
            per_seed[name].append(m)
        if verbose:
            c = per_seed["conditional"][-1]; g = per_seed["global"][-1]
            print(f"  seed {seed}: cond risk={c['pairwise_risk']:.3f} "
                  f"mrr={c['mrr']:.3f} | glob risk={g['pairwise_risk']:.3f} "
                  f"mrr={g['mrr']:.3f}")

    aggregates = {name: _aggregate(rows, eval_cfg, metric_keys)
                  for name, rows in per_seed.items()}

    # Paired conditioning delta (conditional - global), per seed.
    paired = {}
    for key in ("pairwise_risk", "mrr", "ndcg@5", "recall@5"):
        deltas = [per_seed["conditional"][i][key] - per_seed["global"][i][key]
                  for i in range(len(eval_cfg.seeds))
                  if key in per_seed["conditional"][i] and key in per_seed["global"][i]]
        lo, hi = bootstrap_ci(deltas, n=eval_cfg.bootstrap_n, alpha=eval_cfg.alpha)
        # For risk, "helps" = delta < 0; for mrr/ndcg/recall, "helps" = delta > 0.
        wins = (sum(d < 0 for d in deltas) if key == "pairwise_risk"
                else sum(d > 0 for d in deltas))
        paired[key] = {"mean_delta": _mean(deltas), "ci": [lo, hi],
                       "n_seeds": len(deltas), "seeds_conditioning_wins": wins,
                       "ci_excludes_zero": (hi < 0 or lo > 0)}

    return {"per_seed": per_seed, "aggregates": aggregates,
            "paired_conditioning_delta": paired,
            "config": {"seeds": list(eval_cfg.seeds), "ks": list(eval_cfg.ks),
                       "epochs": eval_cfg.train_epochs},
            "corpus_root": corpus.merkle_root}


# --------------------------------------------------------------------------- #
# Cross-market transfer (secondary, common-feature)
# --------------------------------------------------------------------------- #

def evaluate_transfer(source: Corpus, target: Corpus, enc_cfg: EncoderConfig,
                      met_cfg: MetricConfig, train_cfg: TrainConfig,
                      eval_cfg: EvalConfig, *, device: str = "cpu") -> dict:
    """Train on ``source``, evaluate ``target`` crisis queries (shared features).

    Both corpora must share the feature space (built on COMMON_FEATURES). The
    library is the source's non-test cases; queries are the target's crisis
    cases. This is the explicitly-scoped emerging-market -> advanced-economy
    generalisation check; it is reported on the restricted common features only.
    """
    if source.feature_set != target.feature_set:
        raise ValueError("transfer requires source and target to share features")
    per_seed = []
    for seed in eval_cfg.seeds:
        tc = replace(train_cfg, seed=seed, epochs=eval_cfg.train_epochs)
        res = train(source, enc_cfg, met_cfg, tc, conditioning=True,
                    use_cosine=True, verbose=False)
        # Library = source non-test; queries = ALL target crisis cases.
        src_lib = np.array(res["splits"]["train_proper"] + res["splits"]["val"],
                           dtype=int)
        lib_emb = res["encoder"](source.traj[torch.as_tensor(src_lib)])
        # Build a merged corpus view for evaluate_model by scoring target queries
        # against the source library directly.
        enc, met = res["encoder"], res["metric"]
        enc.eval(); met.eval()
        with torch.no_grad():
            tgt_cr = target.is_crisis
            q_sel = torch.nonzero(tgt_cr, as_tuple=False).squeeze(1)
            if q_sel.numel() == 0:
                continue
            q_emb = enc(target.traj[q_sel])
            q_lab = target.regime_id[q_sel]
            lib_lab = source.regime_id[torch.as_tensor(src_lib)]
            S = met(q_emb, lib_emb)
            mrr, risk = [], []
            rec5 = []
            for i in range(q_sel.numel()):
                rel = (lib_lab == q_lab[i]); n_rel = int(rel.sum())
                if n_rel == 0:
                    continue
                order = torch.argsort(S[i], descending=True)
                rr = rel[order]
                mrr.append(1.0 / (int(torch.nonzero(rr, as_tuple=False)[0]) + 1))
                rec5.append(float(rr[:5].sum()) / n_rel)
                pos_s, neg_s = S[i][rel], S[i][~rel]
                if neg_s.numel() > 0:
                    risk.append(float((neg_s.unsqueeze(0) >= pos_s.unsqueeze(1))
                                      .float().mean()))
        per_seed.append({"mrr": _mean(mrr), "pairwise_risk": _mean(risk),
                         "recall@5": _mean(rec5), "n_queries": len(mrr)})
    agg = _aggregate(per_seed, eval_cfg, ["mrr", "pairwise_risk", "recall@5"])
    return {"per_seed": per_seed, "aggregates": agg,
            "common_features": list(source.feature_set)}


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def _print_table(result: dict, eval_cfg: EvalConfig) -> None:
    print("\n=== Held-out test evaluation "
          f"({len(eval_cfg.seeds)} seeds, 95% bootstrap CI) ===")
    keys = ["pairwise_risk", "mrr", "recall@5", "ndcg@5", "ece", "gate_acc"]
    header = f"{'config':<12}" + "".join(f"{k:>16}" for k in keys)
    print(header)
    for name in _CONFIGS:
        a = result["aggregates"][name]
        row = f"{name:<12}"
        for k in keys:
            m = a[k]["mean"]; lo, hi = a[k]["ci"]
            row += f"{m:.3f}[{lo:.2f},{hi:.2f}]".rjust(16)
        print(row)
    print("\n--- Paired conditioning effect (conditional - global) ---")
    for k, p in result["paired_conditioning_delta"].items():
        direction = "lower=better" if k == "pairwise_risk" else "higher=better"
        sig = "SIGNIFICANT" if p["ci_excludes_zero"] else "n.s."
        print(f"  {k:<14} delta={p['mean_delta']:+.4f} "
              f"CI[{p['ci'][0]:+.4f},{p['ci'][1]:+.4f}] "
              f"wins={p['seeds_conditioning_wins']}/{p['n_seeds']} "
              f"({direction}) -> {sig}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Held-out evaluation of the RCR model.")
    p.add_argument("--snapshot", required=True, help="primary (GFDD) snapshot dir")
    p.add_argument("--out", default="results/evaluation.json")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--embed-dim", type=int, default=48)
    p.add_argument("--rank", type=int, default=6)
    p.add_argument("--shared-rank", type=int, default=8)
    p.add_argument("--lambda-shrink", type=float, default=1.0)
    p.add_argument("--device", default="cpu")
    # Transfer arm (optional): raw files to build source (GFDD) + target (JST).
    p.add_argument("--transfer-gfdd", default=None)
    p.add_argument("--transfer-lv", default=None)
    p.add_argument("--transfer-jst", default=None)
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_argparser().parse_args(argv)
    corpus = load_snapshot(args.snapshot, device=args.device)
    n_feat = len(corpus.feature_set)
    enc_cfg = EncoderConfig(input_dim=n_feat, window=corpus.window,
                            embed_dim=args.embed_dim)
    met_cfg = MetricConfig(embed_dim=args.embed_dim, n_regimes=len(REGIME_NAMES),
                           rank=args.rank, shared_rank=args.shared_rank)
    eval_cfg = EvalConfig(seeds=tuple(range(args.seeds)), train_epochs=args.epochs)
    train_cfg = TrainConfig(device=args.device, lambda_shrink=args.lambda_shrink)

    print(f"Evaluating on {corpus.traj.shape[0]} cases "
          f"(root {corpus.merkle_root[:12]}...), {args.seeds} seeds")
    result = run_multiseed(corpus, enc_cfg, met_cfg, train_cfg, eval_cfg,
                           device=args.device)
    _print_table(result, eval_cfg)

    if args.transfer_gfdd and args.transfer_lv and args.transfer_jst:
        print("\n=== Cross-market transfer (EM -> advanced, common features) ===")
        dcfg = DataConfig()
        s_src = gfdd_source(args.transfer_gfdd, dcfg, features=COMMON_FEATURES)
        lv_s, _ = load_laeven_valencia(args.transfer_lv, s_src.name_to_iso)
        src_bundle = build_library(s_src, lv_s, dcfg)
        s_tgt = jst_source(args.transfer_jst, dcfg, features=COMMON_FEATURES)
        lv_t, _ = load_laeven_valencia(args.transfer_lv, s_tgt.name_to_iso)
        tgt_bundle = build_library(s_tgt, lv_t, dcfg)
        src_corpus = _bundle_to_corpus(src_bundle, args.device)
        tgt_corpus = _bundle_to_corpus(tgt_bundle, args.device)
        tec = EncoderConfig(input_dim=len(COMMON_FEATURES), window=src_corpus.window,
                            embed_dim=args.embed_dim)
        tr = evaluate_transfer(src_corpus, tgt_corpus, tec, met_cfg, train_cfg,
                               eval_cfg, device=args.device)
        result["transfer"] = tr
        a = tr["aggregates"]
        print(f"  transfer MRR={a['mrr']['mean']:.3f} "
              f"risk={a['pairwise_risk']['mean']:.3f} "
              f"recall@5={a['recall@5']['mean']:.3f} "
              f"(common features: {tr['common_features']})")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nresults -> {args.out}")


def _bundle_to_corpus(bundle: dict, device: str) -> Corpus:
    """Adapt a data.build_library bundle into a train.Corpus (for transfer)."""
    recs = bundle["records"]
    traj = np.stack([r.pre_onset for r in recs]).astype(np.float32)
    regime = np.array([REGIME_NAMES.index(r.regime) if r.regime in REGIME_NAMES
                       else -1 for r in recs], dtype=np.int64)
    is_cr = np.array([r.is_crisis for r in recs], dtype=bool)
    ids = [r.case_id for r in recs]
    split = [bundle["splits"][cid] for cid in ids]
    return Corpus(traj=torch.tensor(traj, device=device),
                  regime_id=torch.tensor(regime, device=device),
                  is_crisis=torch.tensor(is_cr, device=device),
                  split=split, case_ids=ids, feature_set=bundle["feature_set"],
                  window=traj.shape[1], merkle_root=bundle["manifest"].root)


if __name__ == "__main__":
    main()

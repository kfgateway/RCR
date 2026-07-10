"""Joint training of the encoder and the regime-conditional metric.

This module fits the two learnable components --- :class:`encoder.SignatureEncoder`
and :class:`metric.RegimeConditionalMetric` --- end to end on the sealed corpus
produced by :mod:`data`, and produces a checkpoint bound to that corpus's Merkle
root.

Objectives
----------
The loss is a weighted sum of three terms, each with a clear purpose:

  * **Metric supervised-contrastive loss (primary).** In the *conditional-metric
    score* space, same-regime cases are pulled together and different-regime (and
    calm) cases pushed apart, via a supervised NT-Xent / SupCon objective. This
    is what trains the per-regime geometries ``W_tau`` and the encoder jointly,
    and it is the empirical analogue of minimising the pairwise retrieval risk
    the theory analyses.

  * **Encoder SupCon loss (auxiliary).** The same objective on the encoder's
    contrastive *projection* (cosine space), which regularises the representation
    and stabilises learning on a small corpus.

  * **Gate cross-entropy.** The gate is a regime classifier; cross-entropy against
    the true regime label (crisis queries only) trains it. Its *calibration* ---
    what Claim 3 shows controls excess retrieval risk --- is then fixed
    **post-hoc** by temperature scaling on the validation split (the gate
    temperature is frozen during training and fitted afterwards).

Leakage-safe retrieval setup
----------------------------
Retrieval is always *against the training library*. Cases are partitioned into
``train_proper`` (the candidate library and the gradient-carrying queries),
``val`` (early-stopping queries, retrieved against ``train_proper``), and
``test`` (untouched here; used by ``evaluate.py``). Relevance is same crisis
regime; calm cases are real hard negatives and are never relevant. Queries are
crisis cases (they carry a conditioning regime); calm cases appear only as
candidates.

Honest note on scale
--------------------
The training library is small (~190 cases after the validation carve). Training
uses weight decay, dropout, and early stopping on validation MRR, and the metric
is low-rank, but generalisation is the binding constraint. ``train.py`` therefore
also trains the ``conditioning=False`` global-metric ablation under identical
settings so the *comparison* --- does conditioning help on real data? --- is
reported honestly regardless of the absolute numbers.

References
----------
Khosla, P., et al. (2020). Supervised contrastive learning. NeurIPS. (The
retrieval loss uses their superior L_out form --- the sum over positives outside
the log --- at temperature 0.1, as recommended.)
Guo, C., Pleiss, G., Sun, Y., Weinberger, K.Q. (2017). On calibration of modern
neural networks. ICML. (Post-hoc temperature scaling of the gate, fit on the
validation split after training.)
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from typing import Final

import numpy as np
import torch
import torch.nn.functional as F

from encoder import EncoderConfig, SignatureEncoder
from metric import MetricConfig, RegimeConditionalMetric, REGIME_NAMES

__all__ = [
    "TrainConfig", "load_snapshot", "supcon_loss",
    "retrieval_metrics", "train", "Corpus",
]

_REGIME_TO_ID: Final[dict[str, int]] = {r: i for i, r in enumerate(REGIME_NAMES)}
_CALM_ID: Final[int] = -1


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class TrainConfig:
    """Training hyper-parameters."""

    epochs: int = 300
    lr: float = 1e-3
    weight_decay: float = 1e-4
    contrastive_temp: float = 0.10     # SupCon temperature (loss scale)
    w_metric: float = 1.0              # weight on the metric SupCon term
    w_encoder: float = 0.5             # weight on the encoder-projection SupCon
    w_gate: float = 0.5                # weight on the gate cross-entropy
    val_fraction: float = 0.20         # fraction of train carved for validation
    patience: int = 40                 # early-stopping patience (epochs)
    eval_every: int = 5                # validate every N epochs
    min_epochs: int = 40               # never stop before this many epochs
    grad_clip: float = 5.0             # gradient-norm clip
    lambda_shrink: float = 0.0         # weight on the metric delta shrinkage penalty
    seed: int = 20260517
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if not (0.0 < self.val_fraction < 0.9):
            raise ValueError("val_fraction must be in (0, 0.9)")
        if self.contrastive_temp <= 0:
            raise ValueError("contrastive_temp must be > 0")


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Favour determinism; harmless on CPU.
    torch.use_deterministic_algorithms(True, warn_only=True)


# --------------------------------------------------------------------------- #
# Corpus loading
# --------------------------------------------------------------------------- #

@dataclass
class Corpus:
    """Tensors and metadata for one snapshot, partitioned into splits."""

    traj: torch.Tensor            # (N, window, n_features)
    regime_id: torch.Tensor       # (N,)  crisis id 0..T-1, or _CALM_ID for calm
    is_crisis: torch.Tensor       # (N,)  bool
    split: list[str]              # per-case split tag from the snapshot ("train"/"test")
    case_ids: list[str]
    feature_set: list[str]
    window: int
    merkle_root: str


def load_snapshot(snapshot_dir: str, device: str = "cpu") -> Corpus:
    """Load a :mod:`data` snapshot into tensors + metadata.

    Reads ``case_library.json`` (sealed records), ``splits.json`` (case_id ->
    split), and ``manifest.json`` (for the Merkle root the checkpoint binds to).
    """
    with open(os.path.join(snapshot_dir, "case_library.json"), encoding="utf-8") as f:
        lib = json.load(f)
    with open(os.path.join(snapshot_dir, "splits.json"), encoding="utf-8") as f:
        splits = json.load(f)
    with open(os.path.join(snapshot_dir, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)

    if not lib:
        raise ValueError("empty case library")
    feats = list(lib[0]["feature_names"])
    window = len(lib[0]["pre_onset"])

    traj = np.array([c["pre_onset"] for c in lib], dtype=np.float32)
    regime = np.array([_REGIME_TO_ID.get(c["regime"], _CALM_ID) for c in lib],
                      dtype=np.int64)
    is_cr = np.array([bool(c["is_crisis"]) for c in lib], dtype=bool)
    ids = [c["case_id"] for c in lib]
    split = [splits.get(cid, "train") for cid in ids]

    return Corpus(
        traj=torch.tensor(traj, device=device),
        regime_id=torch.tensor(regime, device=device),
        is_crisis=torch.tensor(is_cr, device=device),
        split=split, case_ids=ids, feature_set=feats, window=window,
        merkle_root=manifest.get("root", ""),
    )


def _partition(corpus: Corpus, cfg: TrainConfig
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return index arrays (train_proper, val, test).

    ``test`` is exactly the snapshot's test split (onset >= split_year). ``val``
    is a random ``val_fraction`` carved from the snapshot's train split; the
    remainder is ``train_proper``. Stratification is by crisis/calm so both val
    and train_proper contain queries.
    """
    rng = np.random.default_rng(cfg.seed)
    split = np.array(corpus.split)
    is_cr = corpus.is_crisis.cpu().numpy()
    test_idx = np.where(split == "test")[0]
    train_all = np.where(split != "test")[0]

    val_idx_parts, tp_idx_parts = [], []
    for mask in (is_cr[train_all], ~is_cr[train_all]):
        grp = train_all[mask]
        rng.shuffle(grp)
        n_val = max(1, int(round(cfg.val_fraction * len(grp)))) if len(grp) else 0
        val_idx_parts.append(grp[:n_val])
        tp_idx_parts.append(grp[n_val:])
    val_idx = np.concatenate(val_idx_parts) if val_idx_parts else np.array([], int)
    tp_idx = np.concatenate(tp_idx_parts) if tp_idx_parts else np.array([], int)
    return np.sort(tp_idx), np.sort(val_idx), np.sort(test_idx)


# --------------------------------------------------------------------------- #
# Supervised-contrastive loss (over a query x candidate score matrix)
# --------------------------------------------------------------------------- #

def supcon_loss(scores: torch.Tensor, query_col: torch.Tensor,
                cand_labels: torch.Tensor, temp: float
                ) -> tuple[torch.Tensor, float]:
    """Supervised NT-Xent over a ``(Q, C)`` query-vs-candidate score matrix.

    Parameters
    ----------
    scores : (Q, C) similarity of each query to each candidate.
    query_col : (Q,) column index of each query within the candidate set, so its
        own entry can be excluded (queries are a subset of candidates).
    cand_labels : (C,) regime id per candidate (crisis id >= 0; calm = _CALM_ID).
    temp : contrastive temperature.

    A query's positives are same-regime crisis candidates other than itself; the
    denominator runs over all candidates except itself (so calm and other-regime
    cases act as negatives). Queries with no positive are ignored. Returns the
    mean loss and the fraction of queries that had >= 1 positive.
    """
    Q, C = scores.shape
    device = scores.device
    z = scores / temp
    ar = torch.arange(Q, device=device)

    self_mask = torch.zeros(Q, C, dtype=torch.bool, device=device)
    self_mask[ar, query_col] = True

    q_labels = cand_labels[query_col]                              # (Q,)
    same = cand_labels.unsqueeze(0) == q_labels.unsqueeze(1)       # (Q, C)
    pos = same & (cand_labels.unsqueeze(0) >= 0) & (~self_mask)    # positives
    valid = ~self_mask                                            # denominator set

    z = z.masked_fill(~valid, float("-inf"))
    log_prob = z - torch.logsumexp(z, dim=1, keepdim=True)        # (Q, C)
    pos_count = pos.sum(dim=1)                                    # (Q,)
    has_pos = pos_count > 0
    # Mean log-prob over each query's positives (guard empty).
    per_q = -(log_prob.masked_fill(~pos, 0.0).sum(dim=1)) / pos_count.clamp(min=1)
    if has_pos.any():
        loss = per_q[has_pos].mean()
    else:
        loss = (scores * 0.0).sum()  # keeps graph; no positives this batch
    return loss, float(has_pos.float().mean())


# --------------------------------------------------------------------------- #
# Retrieval metrics (for validation / reporting)
# --------------------------------------------------------------------------- #

@torch.no_grad()
def retrieval_metrics(query_emb: torch.Tensor, lib_emb: torch.Tensor,
                      query_labels: torch.Tensor, lib_labels: torch.Tensor,
                      met: RegimeConditionalMetric,
                      ks: tuple[int, ...] = (1, 5, 10)) -> dict[str, float]:
    """Compute MRR, Recall@k, and pairwise retrieval risk on a query set.

    Each query retrieves against the library (``lib_emb``); relevant = same crisis
    regime. Queries with no relevant library case are skipped. Pairwise risk is
    the mean over (positive, negative) library pairs of ``[s(neg) >= s(pos)]``,
    the same quantity the theory bounds.
    """
    if query_emb.shape[0] == 0 or lib_emb.shape[0] == 0:
        return {"mrr": 0.0, "pairwise_risk": 1.0, **{f"recall@{k}": 0.0 for k in ks}}
    S = met(query_emb, lib_emb)                        # (Q, L)
    L = lib_emb.shape[0]
    mrr, risk = [], []
    rec = {k: [] for k in ks}
    for i in range(query_emb.shape[0]):
        rel = (lib_labels == query_labels[i])
        n_rel = int(rel.sum())
        if n_rel == 0:
            continue
        s = S[i]
        order = torch.argsort(s, descending=True)
        ranked_rel = rel[order]
        # MRR: reciprocal rank of the first relevant.
        first = int(torch.nonzero(ranked_rel, as_tuple=False)[0]) + 1
        mrr.append(1.0 / first)
        for k in ks:
            rec[k].append(float(ranked_rel[:k].any()))
        # Pairwise risk over all (pos, neg) pairs.
        pos_s = s[rel]
        neg_s = s[~rel]
        if neg_s.numel() > 0:
            diff = (neg_s.unsqueeze(0) >= pos_s.unsqueeze(1)).float()
            risk.append(float(diff.mean()))
    if not mrr:
        return {"mrr": 0.0, "pairwise_risk": 1.0, **{f"recall@{k}": 0.0 for k in ks}}
    out = {"mrr": float(np.mean(mrr)),
           "pairwise_risk": float(np.mean(risk)) if risk else 0.0}
    for k in ks:
        out[f"recall@{k}"] = float(np.mean(rec[k]))
    return out


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def _encode_all(enc: SignatureEncoder, traj: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a trajectory batch, returning (embeddings, projections)."""
    emb, proj = enc(traj, return_projection=True)
    return emb, proj


def train(corpus: Corpus, enc_cfg: EncoderConfig, met_cfg: MetricConfig,
          cfg: TrainConfig, *, conditioning: bool = True,
          use_cosine: bool = True, verbose: bool = True) -> dict:
    """Train encoder + metric on ``corpus``; return the fitted modules + history.

    Set ``conditioning=False`` to train the global-metric ablation under
    otherwise-identical settings (for the honest comparison).
    """
    _seed_everything(cfg.seed)
    device = torch.device(cfg.device)

    enc = SignatureEncoder(enc_cfg).to(device)
    met = RegimeConditionalMetric(met_cfg, conditioning=conditioning,
                                  use_cosine=use_cosine).to(device)
    # Freeze the gate temperature during training; fit it post-hoc.
    met._log_temp.requires_grad_(False)

    params = list(enc.parameters()) + [p for p in met.parameters()
                                       if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    tp_idx, val_idx, test_idx = _partition(corpus, cfg)
    tp = torch.as_tensor(tp_idx, device=device)
    labels = corpus.regime_id.to(device)
    is_cr = corpus.is_crisis.to(device)

    # Query columns within the train_proper library (crisis cases only).
    tp_is_cr = is_cr[tp]
    q_cols = torch.nonzero(tp_is_cr, as_tuple=False).squeeze(1)   # cols into tp
    tp_labels = labels[tp]
    if q_cols.numel() == 0:
        raise ValueError("no crisis queries in train_proper; check the split")

    history: list[dict] = []
    best = {"mrr": -1.0, "epoch": -1, "enc": None, "met": None}
    patience = 0

    for epoch in range(1, cfg.epochs + 1):
        enc.train(); met.train()
        opt.zero_grad()

        emb, proj = _encode_all(enc, corpus.traj[tp])            # (Ntp, d)
        q_emb = emb[q_cols]                                      # (Q, d)

        # (1) Metric SupCon over conditional scores.
        S = met(q_emb, emb)                                     # (Q, Ntp)
        L_metric, cov = supcon_loss(S, q_cols, tp_labels, cfg.contrastive_temp)

        # (2) Encoder SupCon over projection cosine similarity.
        S_proj = proj[q_cols] @ proj.t()                        # (Q, Ntp)
        L_enc, _ = supcon_loss(S_proj, q_cols, tp_labels, cfg.contrastive_temp)

        # (3) Gate cross-entropy on crisis queries (raw logits, no temperature).
        gate_logits = met.gate_logits(q_emb)                    # (Q, T)
        L_gate = F.cross_entropy(gate_logits, tp_labels[q_cols])

        loss = cfg.w_metric * L_metric + cfg.w_encoder * L_enc + cfg.w_gate * L_gate
        if cfg.lambda_shrink > 0:
            loss = loss + cfg.lambda_shrink * met.delta_penalty()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
        opt.step()

        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            enc.eval(); met.eval()
            with torch.no_grad():
                lib_emb, _ = _encode_all(enc, corpus.traj[tp])
                val_tr = corpus.traj[torch.as_tensor(val_idx, device=device)]
                val_lab = labels[torch.as_tensor(val_idx, device=device)]
                val_cr = is_cr[torch.as_tensor(val_idx, device=device)]
                if val_cr.any():
                    v_emb, _ = _encode_all(enc, val_tr[val_cr])
                    vm = retrieval_metrics(v_emb, lib_emb, val_lab[val_cr],
                                           tp_labels, met)
                else:
                    vm = {"mrr": 0.0, "pairwise_risk": 1.0}
            rec = {"epoch": epoch, "loss": float(loss.item()),
                   "L_metric": float(L_metric.item()), "L_gate": float(L_gate.item()),
                   "val_mrr": vm["mrr"], "val_risk": vm["pairwise_risk"],
                   "pos_coverage": cov}
            history.append(rec)
            if verbose:
                print(f"  ep{epoch:3d} loss={rec['loss']:.4f} "
                      f"val_mrr={rec['val_mrr']:.3f} val_risk={rec['val_risk']:.3f}")
            # Early stopping on validation MRR (after a warmup).
            if vm["mrr"] > best["mrr"] + 1e-4:
                best = {"mrr": vm["mrr"], "epoch": epoch,
                        "enc": {k: v.detach().clone() for k, v in enc.state_dict().items()},
                        "met": {k: v.detach().clone() for k, v in met.state_dict().items()}}
                patience = 0
            elif epoch >= cfg.min_epochs:
                patience += 1
                if patience * cfg.eval_every >= cfg.patience:
                    if verbose:
                        print(f"  early stop at epoch {epoch} "
                              f"(best val_mrr={best['mrr']:.3f} @ ep{best['epoch']})")
                    break

    # Restore best-by-val-MRR weights.
    if best["enc"] is not None:
        enc.load_state_dict(best["enc"]); met.load_state_dict(best["met"])

    # Post-hoc temperature calibration on the validation gate logits.
    fitted_temp = 1.0
    enc.eval(); met.eval()
    with torch.no_grad():
        val_tr = corpus.traj[torch.as_tensor(val_idx, device=device)]
        val_lab = labels[torch.as_tensor(val_idx, device=device)]
        val_cr = is_cr[torch.as_tensor(val_idx, device=device)]
    if conditioning and val_cr.any():
        with torch.no_grad():
            v_emb, _ = _encode_all(enc, val_tr[val_cr])
            v_logits = met.gate_logits(v_emb)
        fitted_temp = met.calibrate_temperature(v_logits, val_lab[val_cr])

    return {"encoder": enc, "metric": met, "history": history,
            "best_val_mrr": best["mrr"], "best_epoch": best["epoch"],
            "fitted_temperature": fitted_temp,
            "splits": {"train_proper": tp_idx.tolist(), "val": val_idx.tolist(),
                       "test": test_idx.tolist()},
            "conditioning": conditioning, "use_cosine": use_cosine}


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #

def save_checkpoint(result: dict, corpus: Corpus, enc_cfg: EncoderConfig,
                    met_cfg: MetricConfig, cfg: TrainConfig, path: str) -> None:
    """Save model weights, configs, splits, and the corpus Merkle root."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "encoder_state": result["encoder"].state_dict(),
        "metric_state": result["metric"].state_dict(),
        "encoder_config": asdict(enc_cfg),
        "metric_config": asdict(met_cfg),
        "train_config": asdict(cfg),
        "fitted_temperature": result["fitted_temperature"],
        "best_val_mrr": result["best_val_mrr"],
        "splits": result["splits"],
        "conditioning": result["conditioning"],
        "corpus_merkle_root": corpus.merkle_root,
        "feature_set": corpus.feature_set,
    }, path)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the RCR encoder + metric.")
    p.add_argument("--snapshot", required=True, help="data.py snapshot directory")
    p.add_argument("--out", default="results/checkpoint.pt")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--embed-dim", type=int, default=48)
    p.add_argument("--rank", type=int, default=6)
    p.add_argument("--shared-rank", type=int, default=8)
    p.add_argument("--lambda-shrink", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--compare-global", action="store_true",
                   help="also train the conditioning=False ablation and compare")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_argparser().parse_args(argv)
    over: dict = {"device": args.device}
    if args.epochs is not None:
        over["epochs"] = args.epochs
    if args.seed is not None:
        over["seed"] = args.seed
    over["lambda_shrink"] = args.lambda_shrink
    cfg = TrainConfig(**over)

    corpus = load_snapshot(args.snapshot, device=args.device)
    n_feat = len(corpus.feature_set)
    enc_cfg = EncoderConfig(input_dim=n_feat, window=corpus.window,
                            embed_dim=args.embed_dim)
    met_cfg = MetricConfig(embed_dim=args.embed_dim, n_regimes=len(REGIME_NAMES),
                           rank=args.rank, shared_rank=args.shared_rank)

    print(f"Corpus: {corpus.traj.shape[0]} cases, {n_feat} features, "
          f"window {corpus.window}, root {corpus.merkle_root[:12]}...")
    print("Training regime-conditional model:")
    cond = train(corpus, enc_cfg, met_cfg, cfg, conditioning=True)
    save_checkpoint(cond, corpus, enc_cfg, met_cfg, cfg, args.out)
    print(f"  best val MRR (conditional): {cond['best_val_mrr']:.3f} "
          f"@ epoch {cond['best_epoch']}, temp={cond['fitted_temperature']:.3f}")
    print(f"  checkpoint -> {args.out}")

    if args.compare_global:
        print("Training global-metric ablation (conditioning=False):")
        glob = train(corpus, enc_cfg, met_cfg, cfg, conditioning=False)
        print(f"  best val MRR (global):      {glob['best_val_mrr']:.3f} "
              f"@ epoch {glob['best_epoch']}")
        print(f"\n  CONDITIONING EFFECT on val MRR: "
              f"{cond['best_val_mrr']:.3f} (conditional) vs "
              f"{glob['best_val_mrr']:.3f} (global)  "
              f"-> delta {cond['best_val_mrr'] - glob['best_val_mrr']:+.3f}")


if __name__ == "__main__":
    main()

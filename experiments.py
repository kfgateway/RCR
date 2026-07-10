"""Orchestration: reproduce every result in the paper from one entry point.

This module ties the pieces together and writes a single, hash-stamped results
bundle so that a reviewer can regenerate the paper's numbers with one command.
It delegates to the audited modules rather than re-implementing anything:

  * :mod:`synthetic` --- the controlled validation of Claims 1-3 (global-metric
    suboptimality, Bayes-optimality of the oracle mixture, calibration-controls-
    excess-risk). This is the primary evidence for the conditional advantage.

  * :mod:`data` + :mod:`train` + :mod:`evaluate` --- the real-data study:
    multi-seed training and held-out test evaluation of the conditional model and
    its ablations (global-metric, no-cosine), with bootstrap CIs and paired
    conditioning deltas. Reported honestly: on the emerging-market corpus
    conditioning does *not* improve retrieval --- it is statistically
    indistinguishable from a single global metric on pairwise risk, nDCG, and
    recall, and (without sufficient shrinkage) worse on MRR from overfitting the
    extra per-regime capacity. The hierarchical-shrinkage parameterisation keeps
    the degradation graceful; the conditional advantage itself is a theoretical
    and synthetic result (the gate nonetheless recovers real regime structure).

  * optional cross-market transfer (EM -> advanced economies, common features).

Every sub-result is stamped with a SHA-256 fingerprint of its resolved
configuration (via :mod:`provenance`), and the consolidated bundle records the
corpus Merkle root, so the whole reproduction chain is auditable.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from dataclasses import asdict, dataclass

import numpy as np
import torch

import provenance as prov
from data import (COMMON_FEATURES, DataConfig, build_library, gfdd_source,
                  jst_source, load_laeven_valencia)
from encoder import EncoderConfig
from evaluate import EvalConfig, evaluate_transfer, run_multiseed, _bundle_to_corpus
from metric import MetricConfig, REGIME_NAMES
from synthetic import SyntheticConfig, run_all as run_synthetic
from train import TrainConfig, load_snapshot

__all__ = ["ExperimentConfig", "reproduce_synthetic", "reproduce_real",
           "reproduce_all"]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level reproduction settings."""

    out_dir: str = "results"
    snapshot_dir: str = "data_snapshot"
    seeds: int = 20                    # real-data evaluation seeds (paper: 20-50)
    epochs: int = 200
    embed_dim: int = 48
    rank: int = 6
    shared_rank: int = 8
    lambda_shrink: float = 1.0         # graceful-degradation shrinkage
    device: str = "cpu"
    quick: bool = False                # small/fast settings for a smoke run

    def resolved(self) -> "ExperimentConfig":
        """Return a fast variant when ``quick`` is set."""
        if not self.quick:
            return self
        return ExperimentConfig(
            out_dir=self.out_dir, snapshot_dir=self.snapshot_dir,
            seeds=3, epochs=60, embed_dim=self.embed_dim, rank=self.rank,
            shared_rank=self.shared_rank, lambda_shrink=self.lambda_shrink,
            device=self.device, quick=True,
        )

    def fingerprint(self) -> str:
        return prov.hash_bytes(prov.canonical_json(asdict(self)))


# --------------------------------------------------------------------------- #
# Sub-experiments
# --------------------------------------------------------------------------- #

def reproduce_synthetic(cfg: ExperimentConfig) -> dict:
    """Run the controlled theory validation (Claims 1-3)."""
    outdir = os.path.join(cfg.out_dir, "synthetic")
    scfg = SyntheticConfig()
    if cfg.quick:
        scfg = SyntheticConfig(n_items=400, n_risk_triples=4000, n_bootstrap=100,
                               fit_steps=250, n_theta=7, n_lambda=6)
    res = run_synthetic(scfg, outdir, make_figures=True)
    return {
        "claim1_gap_vs_D_correlation": res["claim1"]["gap_vs_D_correlation"],
        "claim2_oracle_gap_to_floor": res["claim2"]["oracle_gap_to_floor"],
        "claim2_global_gap_to_floor": res["claim2"]["global_gap_to_floor"],
        "claim3_C_vs_D_correlation": res["claim3"]["C_vs_D_correlation"],
        "config_fingerprint": scfg.fingerprint(),
        "artifacts_dir": outdir,
    }


def reproduce_real(cfg: ExperimentConfig) -> dict:
    """Run the real-data multi-seed evaluation with ablations."""
    corpus = load_snapshot(cfg.snapshot_dir, device=cfg.device)
    n_feat = len(corpus.feature_set)
    enc_cfg = EncoderConfig(input_dim=n_feat, window=corpus.window,
                            embed_dim=cfg.embed_dim)
    met_cfg = MetricConfig(embed_dim=cfg.embed_dim, n_regimes=len(REGIME_NAMES),
                           rank=cfg.rank, shared_rank=cfg.shared_rank)
    eval_cfg = EvalConfig(seeds=tuple(range(cfg.seeds)), train_epochs=cfg.epochs)
    train_cfg = TrainConfig(device=cfg.device, lambda_shrink=cfg.lambda_shrink)
    result = run_multiseed(corpus, enc_cfg, met_cfg, train_cfg, eval_cfg,
                           device=cfg.device, verbose=False)
    return result


def reproduce_transfer(cfg: ExperimentConfig, gfdd: str, lv: str, jst: str
                       ) -> dict:
    """Cross-market transfer (EM -> advanced economies) on common features."""
    dcfg = DataConfig()
    s_src = gfdd_source(gfdd, dcfg, features=COMMON_FEATURES)
    lv_s, _ = load_laeven_valencia(lv, s_src.name_to_iso)
    src = _bundle_to_corpus(build_library(s_src, lv_s, dcfg), cfg.device)
    s_tgt = jst_source(jst, dcfg, features=COMMON_FEATURES)
    lv_t, _ = load_laeven_valencia(lv, s_tgt.name_to_iso)
    tgt = _bundle_to_corpus(build_library(s_tgt, lv_t, dcfg), cfg.device)
    enc_cfg = EncoderConfig(input_dim=len(COMMON_FEATURES), window=src.window,
                            embed_dim=cfg.embed_dim)
    met_cfg = MetricConfig(embed_dim=cfg.embed_dim, n_regimes=len(REGIME_NAMES),
                           rank=cfg.rank, shared_rank=cfg.shared_rank)
    eval_cfg = EvalConfig(seeds=tuple(range(cfg.seeds)), train_epochs=cfg.epochs)
    train_cfg = TrainConfig(device=cfg.device, lambda_shrink=cfg.lambda_shrink)
    return evaluate_transfer(src, tgt, enc_cfg, met_cfg, train_cfg, eval_cfg,
                             device=cfg.device)


# --------------------------------------------------------------------------- #
# Full reproduction
# --------------------------------------------------------------------------- #

def reproduce_all(cfg: ExperimentConfig, *, run_transfer: bool = False,
                  transfer_paths: dict | None = None, verbose: bool = True
                  ) -> dict:
    """Run synthetic + real (+ optional transfer); write a consolidated bundle."""
    cfg = cfg.resolved()
    os.makedirs(cfg.out_dir, exist_ok=True)

    if verbose:
        print("[1/2] synthetic theory validation ...")
    synth = reproduce_synthetic(cfg)
    if verbose:
        print(f"      Claim1 corr={synth['claim1_gap_vs_D_correlation']:.3f}  "
              f"Claim3 corr={synth['claim3_C_vs_D_correlation']:.3f}")

    if verbose:
        print(f"[2/2] real-data evaluation ({cfg.seeds} seeds) ...")
    real = reproduce_real(cfg)
    paired = real["paired_conditioning_delta"]["pairwise_risk"]
    if verbose:
        print(f"      conditioning paired risk delta={paired['mean_delta']:+.4f} "
              f"CI={[round(x,4) for x in paired['ci']]} "
              f"(significant={paired['ci_excludes_zero']})")

    bundle = {
        "experiment_fingerprint": cfg.fingerprint(),
        "config": asdict(cfg),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "corpus_root": real.get("corpus_root", ""),
        "synthetic": synth,
        "real_data": real,
    }
    if run_transfer and transfer_paths:
        if verbose:
            print("[+] cross-market transfer ...")
        bundle["transfer"] = reproduce_transfer(cfg, **transfer_paths)

    path = os.path.join(cfg.out_dir, "reproduction.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)
    if verbose:
        print(f"consolidated results -> {path}")
    return bundle


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Reproduce all RCR results.")
    p.add_argument("--snapshot", default="data_snapshot")
    p.add_argument("--out", default="results")
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lambda-shrink", type=float, default=1.0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--transfer-gfdd", default=None)
    p.add_argument("--transfer-lv", default=None)
    p.add_argument("--transfer-jst", default=None)
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_argparser().parse_args(argv)
    cfg = ExperimentConfig(
        out_dir=args.out, snapshot_dir=args.snapshot, seeds=args.seeds,
        epochs=args.epochs, lambda_shrink=args.lambda_shrink,
        device=args.device, quick=args.quick,
    )
    tpaths = None
    if args.transfer_gfdd and args.transfer_lv and args.transfer_jst:
        tpaths = {"gfdd": args.transfer_gfdd, "lv": args.transfer_lv,
                  "jst": args.transfer_jst}
    reproduce_all(cfg, run_transfer=tpaths is not None, transfer_paths=tpaths)


if __name__ == "__main__":
    main()

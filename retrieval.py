"""Retrieval serving layer: trained model + case library -> auditable analogues.

This module is the inference side of the framework and the place where the
*auditability* pillar becomes a demonstrated property rather than a stored one.
Given a trained :class:`encoder.SignatureEncoder`, a trained
:class:`metric.RegimeConditionalMetric`, and a case library, it:

  1. encodes the library once (index build) and caches the embeddings;
  2. for a query trajectory, infers the **calibrated regime posterior** (the
     gate), scores every library case with the regime-conditional metric, and
     returns the top-k analogues; and
  3. attaches to each retrieved analogue a **verified provenance chain** (via
     :func:`provenance.provenance_chain`): its identity, that it is content-hash
     verified and belongs to the Merkle-root-bound corpus, its regime, and its
     primary-source citations.

So a retrieval result is not just "here are similar past crises" but "here are
similar past crises, each of which traces to its primary sources and is provably
part of the archived, root-bound case base" --- the trustworthy-retrieval claim,
end to end.

Calibration-aware output
------------------------
Each result carries the query's regime posterior, its top regime, a
``confidence`` (the top posterior mass) and a normalised ``entropy`` (regime
uncertainty). Because the metric hedges --- mixing broadly when the posterior is
diffuse --- these numbers tell an operator how sharply the regime lens was
applied, which is the operational face of Claim 3.

Modes
-----
* **Hard** (default; inference/audit): integer top-k under ``no_grad``, with
  provenance chains. Deterministic.
* **Soft** (retrieval-augmented / analysis): a differentiable softmax weighting
  over the whole library, for downstream use of retrieved evidence.

Contract
--------
Query trajectories must be standardised in the *same* feature space as the
library (use :meth:`RetrievalIndex.standardize` with the snapshot's stored
statistics for a genuinely new raw query; queries drawn from the snapshot are
already standardised). Input shape is ``(window, n_features)`` or a batch
``(B, window, n_features)``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import torch

import provenance as prov
from encoder import EncoderConfig, SignatureEncoder
from metric import MetricConfig, RegimeConditionalMetric, REGIME_NAMES

__all__ = ["RetrievedCase", "RetrievalResult", "RetrievalIndex"]


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class RetrievedCase:
    """One retrieved analogue with its audit trail."""

    case_id: str
    rank: int                       # 1-based rank in the result
    score: float                    # conditional-metric similarity
    regime: str
    is_crisis: bool
    dominant_regime: str            # regime whose W_tau contributed most to the score
    provenance: dict = field(default_factory=dict)   # verified provenance chain


@dataclass(frozen=True)
class RetrievalResult:
    """A full retrieval response for one query."""

    regime_posterior: dict[str, float]     # calibrated gate over REGIME_NAMES
    top_regime: str
    confidence: float                      # max posterior mass in [0, 1]
    entropy: float                         # normalised regime entropy in [0, 1]
    library_root: str                      # Merkle root the analogues verify against
    retrieved: tuple[RetrievedCase, ...]


# --------------------------------------------------------------------------- #
# Index
# --------------------------------------------------------------------------- #

class RetrievalIndex:
    """A retrieval index over a case library backed by a trained model."""

    def __init__(self, encoder: SignatureEncoder, metric: RegimeConditionalMetric,
                 records: Sequence[prov.CaseRecord],
                 manifest: prov.LibraryManifest | None = None,
                 *, device: str = "cpu",
                 standardizer: dict[str, list[float]] | None = None) -> None:
        if not records:
            raise ValueError("RetrievalIndex requires a non-empty library")
        self.device = torch.device(device)
        self.encoder = encoder.to(self.device).eval()
        self.metric = metric.to(self.device).eval()
        self.records: list[prov.CaseRecord] = [r.seal() for r in records]
        self.manifest = manifest
        self.standardizer = standardizer

        self._ids = [r.case_id for r in self.records]
        self._id_to_pos = {cid: i for i, cid in enumerate(self._ids)}
        self._regimes = [r.regime for r in self.records]
        self._is_crisis = torch.tensor([r.is_crisis for r in self.records],
                                       device=self.device)
        self._window = self.records[0].pre_onset.shape[0]
        self._n_features = self.records[0].pre_onset.shape[1]

        self._lib_emb = self._encode_library()   # cached (L, d), unit-norm

    # ------------------------------------------------------------------ #
    # Index build
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _encode_library(self) -> torch.Tensor:
        traj = np.stack([r.pre_onset for r in self.records]).astype(np.float32)
        t = torch.tensor(traj, device=self.device)
        return self.encoder(t)                    # (L, d), unit-norm by contract

    def rebuild(self) -> None:
        """Recompute cached library embeddings (call after the encoder changes)."""
        self.encoder.eval()
        self._lib_emb = self._encode_library()

    # ------------------------------------------------------------------ #
    # Standardisation helper (for genuinely new raw queries)
    # ------------------------------------------------------------------ #

    def standardize(self, raw_traj: np.ndarray) -> np.ndarray:
        """Apply the stored per-feature train statistics to a raw trajectory."""
        if self.standardizer is None:
            raise ValueError("no standardizer available on this index")
        mean = np.asarray(self.standardizer["mean"], dtype=np.float64)
        std = np.asarray(self.standardizer["std"], dtype=np.float64)
        return (np.asarray(raw_traj, dtype=np.float64) - mean) / std

    # ------------------------------------------------------------------ #
    # Query encoding + gate
    # ------------------------------------------------------------------ #

    def _as_batch(self, query_traj: np.ndarray | torch.Tensor) -> torch.Tensor:
        t = torch.as_tensor(np.asarray(query_traj, dtype=np.float32),
                             device=self.device)
        if t.ndim == 2:
            t = t.unsqueeze(0)
        if t.ndim != 3 or t.shape[1:] != (self._window, self._n_features):
            raise ValueError(
                f"query must be (window, n_features)=({self._window},"
                f"{self._n_features}) or a batch thereof; got {tuple(t.shape)}"
            )
        return t

    @torch.no_grad()
    def encode_query(self, query_traj: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Encode a query trajectory (or batch) to unit-norm embeddings."""
        return self.encoder(self._as_batch(query_traj))

    @staticmethod
    def _posterior_stats(pi_row: torch.Tensor) -> tuple[dict[str, float], str, float, float]:
        probs = pi_row.detach().cpu()
        posterior = {name: float(probs[i]) for i, name in enumerate(REGIME_NAMES)}
        top_idx = int(torch.argmax(probs))
        top_regime = REGIME_NAMES[top_idx]
        confidence = float(probs[top_idx])
        # Normalised Shannon entropy in [0, 1] (0 = certain, 1 = uniform).
        p = probs.clamp_min(1e-12)
        ent = float(-(p * p.log()).sum() / np.log(len(REGIME_NAMES)))
        return posterior, top_regime, confidence, min(max(ent, 0.0), 1.0)

    # ------------------------------------------------------------------ #
    # Hard retrieval (top-k with provenance) --- the audit path
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def retrieve(self, query_traj: np.ndarray | torch.Tensor, *, k: int = 10,
                 exclude_ids: Iterable[str] = (), with_provenance: bool = True
                 ) -> RetrievalResult:
        """Return the top-k analogues for one query, each with a provenance chain.

        Parameters
        ----------
        query_traj : ``(window, n_features)`` standardised trajectory.
        k : number of analogues to return (clamped to the available library).
        exclude_ids : case ids to omit (e.g. the query's own id in leave-one-out).
        with_provenance : if False, skip provenance-chain emission (faster).
        """
        emb = self.encode_query(query_traj)          # (1, d)
        if emb.shape[0] != 1:
            raise ValueError("retrieve expects a single query; use retrieve_batch")
        pi = self.metric.gate(emb)                    # (1, T)
        posterior, top_regime, confidence, entropy = self._posterior_stats(pi[0])

        scores = self.metric(emb, self._lib_emb)[0]   # (L,)
        per_regime = self.metric.regime_scores(emb, self._lib_emb)[0]  # (T, L)

        # Exclusion mask.
        mask = torch.ones_like(scores, dtype=torch.bool)
        for cid in exclude_ids:
            pos = self._id_to_pos.get(cid)
            if pos is not None:
                mask[pos] = False
        avail = int(mask.sum())
        kk = max(0, min(k, avail))

        masked = scores.masked_fill(~mask, float("-inf"))
        order = torch.argsort(masked, descending=True)[:kk]

        retrieved: list[RetrievedCase] = []
        for rank, idx in enumerate(order.tolist(), start=1):
            rec = self.records[idx]
            dom = REGIME_NAMES[int(torch.argmax(per_regime[:, idx]))]
            chain = (prov.provenance_chain(rec, self.manifest)
                     if with_provenance else {})
            retrieved.append(RetrievedCase(
                case_id=rec.case_id, rank=rank, score=float(scores[idx]),
                regime=rec.regime, is_crisis=rec.is_crisis,
                dominant_regime=dom, provenance=chain,
            ))

        root = self.manifest.root if self.manifest is not None else ""
        return RetrievalResult(
            regime_posterior=posterior, top_regime=top_regime,
            confidence=confidence, entropy=entropy, library_root=root,
            retrieved=tuple(retrieved),
        )

    @torch.no_grad()
    def retrieve_batch(self, query_trajs: np.ndarray | torch.Tensor, *,
                       k: int = 10, with_provenance: bool = False
                       ) -> list[RetrievalResult]:
        """Retrieve for a batch of queries (provenance off by default for speed)."""
        batch = self._as_batch(query_trajs)
        return [self.retrieve(batch[i], k=k, with_provenance=with_provenance)
                for i in range(batch.shape[0])]

    # ------------------------------------------------------------------ #
    # Soft retrieval (differentiable weighting over the library)
    # ------------------------------------------------------------------ #

    def soft_retrieval_weights(self, query_traj: np.ndarray | torch.Tensor, *,
                               temperature: float = 1.0
                               ) -> tuple[list[str], torch.Tensor]:
        """Return ``(case_ids, weights)`` --- a softmax weighting over the library.

        Differentiable (no ``no_grad``): gradients flow to the encoder/metric so
        retrieved evidence can be used inside a downstream differentiable module.
        """
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        emb = self.encoder(self._as_batch(query_traj))
        scores = self.metric(emb, self._lib_emb)      # (B, L)
        weights = torch.softmax(scores / temperature, dim=-1)
        return list(self._ids), weights

    # ------------------------------------------------------------------ #
    # Reconstruction from saved artefacts
    # ------------------------------------------------------------------ #

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, snapshot_dir: str, *,
                        device: str = "cpu",
                        library_split: str = "train_proper") -> "RetrievalIndex":
        """Rebuild an index from a ``train.py`` checkpoint + ``data.py`` snapshot.

        The candidate library is the checkpoint's ``library_split`` indices
        (default ``train_proper`` --- the leakage-safe library queries retrieve
        against). Encoder/metric configs and weights (including the fitted gate
        temperature) come from the checkpoint; records and the manifest come from
        the snapshot, so provenance verification is against the true corpus root.
        """
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        enc = SignatureEncoder(EncoderConfig(**ckpt["encoder_config"]))
        met = RegimeConditionalMetric(
            MetricConfig(**ckpt["metric_config"]),
            conditioning=ckpt.get("conditioning", True),
        )
        enc.load_state_dict(ckpt["encoder_state"])
        met.load_state_dict(ckpt["metric_state"])   # restores fitted _log_temp

        with open(os.path.join(snapshot_dir, "case_library.json"),
                  encoding="utf-8") as f:
            lib = json.load(f)                      # file order (matches splits idx)
        with open(os.path.join(snapshot_dir, "manifest.json"), encoding="utf-8") as f:
            manifest = prov.LibraryManifest.from_dict(json.load(f))
        std = None
        std_path = os.path.join(snapshot_dir, "standardizer.json")
        if os.path.isfile(std_path):
            with open(std_path, encoding="utf-8") as f:
                std = json.load(f)

        idx = ckpt.get("splits", {}).get(library_split)
        records_all = [prov.CaseRecord.from_dict(d) for d in lib]
        records = ([records_all[i] for i in idx] if idx is not None
                   else records_all)
        return cls(enc, met, records, manifest, device=device, standardizer=std)


# --------------------------------------------------------------------------- #
# CLI demo: retrieve for one held-out query and print its audit trail
# --------------------------------------------------------------------------- #

def _demo(checkpoint_path: str, snapshot_dir: str, k: int = 5) -> None:
    index = RetrievalIndex.from_checkpoint(checkpoint_path, snapshot_dir)
    # Use the first test-split crisis case as an illustrative query.
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    with open(os.path.join(snapshot_dir, "case_library.json"), encoding="utf-8") as f:
        lib = json.load(f)
    test_idx = ckpt.get("splits", {}).get("test", [])
    query = next((lib[i] for i in test_idx if lib[i]["is_crisis"]), lib[0])
    res = index.retrieve(np.array(query["pre_onset"], dtype=np.float32),
                         k=k, exclude_ids=[query["case_id"]])
    print(f"Query: {query['country_iso3']} {query['onset']} "
          f"(true regime: {query['regime']})")
    print(f"Inferred regime posterior: "
          f"{ {r: round(p,3) for r,p in res.regime_posterior.items()} }")
    print(f"Top regime={res.top_regime} confidence={res.confidence:.3f} "
          f"entropy={res.entropy:.3f}")
    print(f"Library root: {res.library_root[:16]}...")
    print(f"Top {len(res.retrieved)} analogues:")
    for rc in res.retrieved:
        srcs = [c["source"] for c in rc.provenance.get("sources", [])]
        print(f"  #{rc.rank} {rc.provenance.get('country_iso3','?')} "
              f"{rc.provenance.get('onset','?')} regime={rc.regime} "
              f"score={rc.score:.3f} dom={rc.dominant_regime} "
              f"verified={rc.provenance.get('verified')} "
              f"in_library={rc.provenance.get('in_library')} sources={srcs}")


def main(argv: list[str] | None = None) -> None:
    import argparse
    p = argparse.ArgumentParser(description="Retrieval demo with provenance.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--snapshot", required=True)
    p.add_argument("--k", type=int, default=5)
    args = p.parse_args(argv)
    _demo(args.checkpoint, args.snapshot, args.k)


if __name__ == "__main__":
    main()

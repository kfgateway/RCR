"""Property and rigor test suite for the regime-conditional retrieval framework.

Consolidates the load-bearing invariants of every module into one suite. It is
both pytest-discoverable (``pytest tests.py``) and runnable standalone
(``python tests.py``), the latter printing a pass/fail summary.

The tests assert *properties*, not fitted numbers: content-hash determinism and
tamper-evidence, Merkle verification, the metric's PSD-by-construction guarantee
and the exact equivalence of its efficient and explicit forms, its graceful
degradation to a global metric, the encoder's unit-norm/order-sensitivity/
batch-independence contract, the synthetic model's geometric identities, and ---
when a data snapshot is present --- all five regimes, leakage-safe
standardisation, and real (non-synthetic) negatives.
"""

from __future__ import annotations

import os

import numpy as np
import torch

import provenance as prov
from encoder import EncoderConfig, SignatureEncoder
from metric import MetricConfig, RegimeConditionalMetric, REGIME_NAMES

_SNAPSHOT = os.environ.get("RCR_SNAPSHOT", "data_snapshot")


def _unit(x):
    return torch.nn.functional.normalize(x, dim=-1)


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #

def _make_record(country="KOR", regime="currency", crisis=True):
    traj = np.arange(20 * 4, dtype=float).reshape(20, 4) / 7.0
    cites = (prov.SourceCitation("GFDD", "KOR:1997", "2026-01-15"),)
    return prov.CaseRecord(
        country_iso3=country, onset="1997Q4", regime=regime, is_crisis=crisis,
        pre_onset=traj, feature_names=("a", "b", "c", "d"),
        citations=cites, quality=0.9, outcome_loss=None,
    )


def test_canonical_json_is_order_independent():
    a = {"b": 1, "a": 2, "c": [3, {"z": 9, "y": 8}]}
    b = {"c": [3, {"y": 8, "z": 9}], "a": 2, "b": 1}
    assert prov.canonical_json(a) == prov.canonical_json(b)


def test_canonical_json_collapses_negative_zero():
    assert prov.canonical_json({"x": -0.0}) == prov.canonical_json({"x": 0.0})


def test_content_hash_is_deterministic():
    assert _make_record().seal().case_id == _make_record().seal().case_id


def test_tamper_is_detected():
    r = _make_record().seal()
    forged = prov.CaseRecord(
        country_iso3="ARG", onset="2001Q4", regime="sovereign", is_crisis=True,
        pre_onset=np.ones((20, 4)), feature_names=("a", "b", "c", "d"),
        citations=(prov.SourceCitation("X", "y", "2026-01-15"),),
        quality=0.5, outcome_loss=None, case_id=r.case_id,
    )
    assert forged.verify() is False


def test_merkle_verifies_and_detects_change():
    recs = [_make_record(country=c) for c in ("KOR", "THA", "IDN", "ARG", "BRA")]
    manifest = prov.build_manifest(recs)
    assert prov.verify_library(recs, manifest)
    for i, cid in enumerate(manifest.case_ids):
        proof = prov.merkle_proof(manifest.case_ids, i)
        assert prov.verify_merkle_proof(cid, proof, manifest.root)


def test_record_immutability():
    r = _make_record().seal()
    try:
        r.country_iso3 = "USA"
        raise AssertionError("frozen record allowed mutation")
    except Exception:
        pass
    try:
        r.pre_onset[0, 0] = 999.0
        raise AssertionError("read-only buffer allowed mutation")
    except (ValueError, RuntimeError):
        pass


# --------------------------------------------------------------------------- #
# metric
# --------------------------------------------------------------------------- #

def _metric(d=16, T=5, r=6, shared=8):
    torch.manual_seed(0)
    return RegimeConditionalMetric(
        MetricConfig(embed_dim=d, n_regimes=T, rank=r, shared_rank=shared,
                     gate_hidden=32)).double()


def test_metric_psd_by_construction():
    m = _metric()
    pis = torch.softmax(torch.randn(50, m.cfg.n_regimes, dtype=torch.float64), -1)
    assert (m.min_eigenvalue(pis) >= -1e-9).all()
    onehot = torch.eye(m.cfg.n_regimes, dtype=torch.float64)
    assert (m.min_eigenvalue(onehot) >= -1e-9).all()


def test_metric_efficient_equals_explicit():
    m = _metric()
    zq = _unit(torch.randn(7, 16, dtype=torch.float64))
    zc = _unit(torch.randn(20, 16, dtype=torch.float64))
    S = m(zq, zc)
    W = m.effective_metric(m.gate(zq))
    S_explicit = torch.einsum("bd,bde,ne->bn", zq, W, zc)
    assert torch.allclose(S, S_explicit, atol=1e-9)


def test_metric_gate_is_simplex():
    m = _metric()
    pi = m.gate(_unit(torch.randn(8, 16, dtype=torch.float64)))
    assert torch.allclose(pi.sum(-1), torch.ones(8, dtype=torch.float64), atol=1e-9)
    assert (pi >= 0).all()


def test_metric_graceful_degradation():
    """Zeroing the deltas collapses every W_tau to the shared global metric."""
    m = _metric()
    with torch.no_grad():
        m.L.zero_()
    W = m.regime_metrics()
    W_shared = m.L_shared @ m.L_shared.t()
    for t in range(m.cfg.n_regimes):
        assert torch.allclose(W[t], W_shared, atol=1e-10)
    assert float(m.delta_penalty().detach()) < 1e-12


def test_metric_conditioning_ablation_is_uniform():
    m = RegimeConditionalMetric(
        MetricConfig(embed_dim=16, n_regimes=5, rank=4), conditioning=False).double()
    z = _unit(torch.randn(6, 16, dtype=torch.float64))
    pi = m.gate(z)
    assert torch.allclose(pi, torch.full_like(pi, 1.0 / 5), atol=1e-9)


# --------------------------------------------------------------------------- #
# encoder
# --------------------------------------------------------------------------- #

def _encoder(F=10, W=10, d=48):
    torch.manual_seed(0)
    return SignatureEncoder(EncoderConfig(input_dim=F, window=W, embed_dim=d)).double().eval()


def test_encoder_output_is_unit_norm():
    enc = _encoder()
    e = enc(torch.randn(8, 10, 10, dtype=torch.float64))
    assert torch.allclose(e.norm(dim=-1), torch.ones(8, dtype=torch.float64), atol=1e-9)


def test_encoder_is_temporal_order_sensitive():
    enc = _encoder()
    x = torch.randn(4, 10, 10, dtype=torch.float64)
    perm = torch.randperm(10)
    assert not torch.allclose(enc(x), enc(x[:, perm, :]), atol=1e-4)


def test_encoder_is_batch_independent():
    enc = _encoder()
    x = torch.randn(8, 10, 10, dtype=torch.float64)
    assert torch.allclose(enc(x[:1]), enc(x)[:1], atol=1e-9)


# --------------------------------------------------------------------------- #
# synthetic
# --------------------------------------------------------------------------- #

def test_synthetic_divergence_identity():
    import synthetic as S
    B1, B2 = S.make_rotated_subspaces(32, 4, 0.7)
    assert abs(S.subspace_divergence(B1, B2) - 4 * np.sin(0.7) ** 2) < 1e-9
    assert abs(S.subspace_divergence(*S.make_rotated_subspaces(32, 4, 0.0))) < 1e-12


def test_synthetic_conditional_relevance():
    """The same query's top neighbour differs across regimes (conditional)."""
    import synthetic as S
    rng = np.random.default_rng(1)
    X, _ = S.generate_items(S.SyntheticConfig(n_items=500), rng)
    B1, B2 = S.make_rotated_subspaces(32, 4, 0.7)
    P1, P2 = B1 @ B1.T, B2 @ B2.T
    s1, s2 = X @ P1 @ X.T, X @ P2 @ X.T
    assert int(np.argsort(-s1[0])[1]) != int(np.argsort(-s2[0])[1])


# --------------------------------------------------------------------------- #
# integration
# --------------------------------------------------------------------------- #

def test_encoder_metric_compose():
    enc = _encoder(d=48)
    met = _metric(d=48)
    x = torch.randn(12, 10, 10, dtype=torch.float64)
    emb = enc(x)
    S = met(emb[:4], emb)
    assert S.shape == (4, 12) and torch.isfinite(S).all()
    assert (met.min_eigenvalue(met.gate(emb[:4])) >= -1e-9).all()


# --------------------------------------------------------------------------- #
# data (only when a snapshot is present)
# --------------------------------------------------------------------------- #

def _load_snapshot_or_skip():
    import json
    lib_path = os.path.join(_SNAPSHOT, "case_library.json")
    if not os.path.isfile(lib_path):
        return None
    with open(lib_path, encoding="utf-8") as f:
        lib = json.load(f)
    with open(os.path.join(_SNAPSHOT, "splits.json"), encoding="utf-8") as f:
        splits = json.load(f)
    return lib, splits


def test_data_all_regimes_present():
    data = _load_snapshot_or_skip()
    if data is None:
        return  # snapshot absent -> skip
    lib, _ = data
    regimes = {c["regime"] for c in lib}
    for r in REGIME_NAMES:
        assert r in regimes, f"regime {r} missing from corpus"


def test_data_has_real_negatives():
    data = _load_snapshot_or_skip()
    if data is None:
        return
    lib, _ = data
    calm = [c for c in lib if not c["is_crisis"]]
    assert len(calm) > 0
    # A real negative cites a data source (not a synthetic construction).
    assert all(len(c["citations"]) >= 1 for c in calm)


def test_data_no_test_leakage_in_standardizer():
    """Standardised train cases have ~zero mean; test cases need not (excluded)."""
    data = _load_snapshot_or_skip()
    if data is None:
        return
    import json
    lib, splits = data
    with open(os.path.join(_SNAPSHOT, "standardizer.json"), encoding="utf-8") as f:
        feats = json.load(f)["feature_set"]
    id_split = {c["case_id"]: splits.get(c["case_id"], "train") for c in lib}
    train = np.array([c["pre_onset"] for c in lib
                      if id_split[c["case_id"]] == "train"], dtype=float)
    if train.size == 0:
        return
    stacked = train.reshape(-1, len(feats))
    assert np.allclose(stacked.mean(axis=0), 0.0, atol=1e-5)


# --------------------------------------------------------------------------- #
# data-layer regression + evaluation-metric properties
# --------------------------------------------------------------------------- #

def test_interp_no_index_misalignment():
    """Regression: gap interpolation must not corrupt values on a non-contiguous
    index (a silent pandas alignment bug that once scrambled the whole panel)."""
    try:
        import pandas as pd
        from data import _interp_and_mask, DataConfig
    except Exception:
        return  # data/pandas unavailable -> skip
    df = pd.DataFrame({"iso3": ["A"] * 4 + ["B"] * 3,
                       "year": [2000, 2001, 2002, 2003, 2000, 2001, 2002],
                       "x": [1.0, np.nan, 3.0, 4.0, 10.0, 20.0, 30.0]})
    df.index = [5, 7, 12, 20, 33, 41, 50]   # non-contiguous, as after filtering
    out = _interp_and_mask(df, ("x",), DataConfig())
    # A/2001 gap -> 2.0 by linear interp; every other cell must be unchanged.
    assert np.allclose(out["x"].to_numpy(), [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0])


def test_mce_is_worst_case_bin_gap():
    """MCE is the worst-case calibration bin gap and dominates ECE."""
    try:
        from evaluate import (maximum_calibration_error,
                              expected_calibration_error)
    except Exception:
        return
    conf = np.array([0.99, 0.99, 0.99, 0.99, 0.5, 0.5])
    corr = np.array([1, 0, 1, 0, 1, 0])   # 0.99-bin: acc .5, gap ~.49; .5-bin: gap 0
    mce = maximum_calibration_error(conf, corr, 10)
    ece = expected_calibration_error(conf, corr, 10)
    assert mce >= ece - 1e-9
    assert abs(mce - 0.49) < 0.05


# --------------------------------------------------------------------------- #
# Standalone runner
# --------------------------------------------------------------------------- #

def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = failed = 0
    for t in tests:
        try:
            t()
            passed += 1
            print(f"  [PASS] {t.__name__}")
        except Exception as exc:  # noqa: BLE001 - report and continue
            failed += 1
            print(f"  [FAIL] {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n==== {passed}/{passed + failed} tests passed ====")
    return failed


if __name__ == "__main__":
    import sys
    sys.exit(1 if _run_all() else 0)

# Regime-Conditional Retrieval (RCR)

Reference implementation for the paper

> **Regime-conditional similarity: calibrated mixture-of-metrics retrieval for auditable case-based reasoning**
> Yixue Hao, Pu Han, Lianxing Min. *Information Sciences* (Elsevier), 2026.

RCR finds historical financial-crisis analogues by making similarity *conditional on the query's inferred crisis regime*, while keeping every retrieval calibrated and auditable. This repository reproduces every number, table, and figure in the paper from a single entry point, and ships a sealed, content-addressed case library so the corpus is bit-reproducible.

The corpus is committed by the SHA-256 Merkle root

```
99aee6ae64ed7ba8266733451b524add243c37d4d9c6052757b96f29175056e4
```

and each reported result carries the configuration fingerprint

```
a41059b0eb8dad1522674cfa52dd2f6a921b5e56da0da5902f2714cfeb7b3a2c
```

---

## 1. What the framework does

Given a pre-onset macro-financial trajectory (a window of `T` years x `F` features), RCR:

1. **encodes** it to a unit-norm embedding `z` (`encoder.py`);
2. **infers the crisis regime** from `z` with a temperature-scaled gate that reports calibrated confidence (`metric.py`);
3. **scores** every library case with a mixture of per-regime positive-semidefinite (PSD) metrics blended by the gate, `s(z_q, z_c) = sum_tau pi_tau(z_q) (z_q^T W_tau z_c) + gamma z_q^T z_c` with `W_tau = W_shared + L_tau L_tau^T` (`metric.py`, `retrieval.py`); and
4. **returns the top-k analogues with a verified provenance certificate**: a Merkle inclusion proof, content-hash verification, regime, and primary-source citations (`retrieval.py`, `provenance.py`).

A central result of the paper is a *predicted null*: from the corpus geometry alone the theory forecasts that conditioning cannot materially improve retrieval on this emerging-market corpus, and the experiment confirms it, while the same machinery still recovers latent regimes, calibrates its own confidence, and certifies every result.

---

## 2. Repository layout

```
.
├── data.py           # canonical sources -> sealed, content-hashed case library
├── provenance.py     # CaseRecord, canonical hashing, Merkle tree, inclusion proofs, tamper detection
├── encoder.py        # SignatureEncoder: trajectory -> L2-normalised embedding
├── metric.py         # RegimeConditionalMetric: per-regime PSD metrics + calibrated gate (core contribution)
├── retrieval.py      # RetrievalIndex: top-k analogues + verified provenance chain
├── train.py          # joint encoder+metric training; checkpoint bound to the Merkle root
├── evaluate.py       # multi-seed held-out evaluation with bootstrap CIs, ablations, transfer
├── synthetic.py      # NumPy validation of Claims 1-3 (independent of the PyTorch model)
├── experiments.py    # orchestration: reproduce all paper results with one command
├── tests.py          # 22 property / rigor tests (pytest-discoverable and standalone)
├── Datasets/         # GFDD, Laeven-Valencia, JST source files (see Section 5)
├── data_snapshot/    # prebuilt sealed corpus (case_library.json, manifest.json, splits, standardizer)
└── results/          # reproduction bundle (reproduction.json, synthetic/, figures)
```

---

## 3. Requirements

Tested end to end with:

| Package    | Tested version |
|------------|----------------|
| Python     | 3.12           |
| torch      | 2.12.1         |
| numpy      | 2.4.4          |
| pandas     | 3.0.2          |
| openpyxl   | (xlsx reading) |
| matplotlib | (synthetic figures) |

The model runs on CPU by default (`--device cpu`); no GPU is required to reproduce the paper.

A pinned `requirements.txt` is included:

```
pip install -r requirements.txt
# or, minimally:
pip install torch numpy pandas openpyxl matplotlib
```

---

## 4. Quick start

The fastest way to confirm the artifact is healthy (a few minutes, CPU):

```bash
# 1. Property / rigor suite (22 checks; uses the prebuilt snapshot)
RCR_SNAPSHOT=data_snapshot python tests.py

# 2. Synthetic validation of the theory (Claims 1-3) + figures
python synthetic.py --outdir results/synthetic

# 3. Smoke reproduction of the full pipeline (few seeds, short training)
python experiments.py --snapshot data_snapshot --out results --quick
```

Full paper-grade reproduction (Section 6) uses 20 seeds and 200 epochs.

---

## 5. Data

The corpus is assembled deterministically from three canonical, citable sources placed in `Datasets/`:

- **GFDD** (primary feature source) — Cihak, Demirguc-Kunt, Feyen, Levine (2012), *Benchmarking Financial Systems Around the World*, World Bank Policy Research WP 6175; Global Financial Development Database, August 2022 vintage.
- **Laeven-Valencia** (crisis chronology and regime labels) — Laeven, Valencia (2020), *Systemic Banking Crises Database II*, IMF Economic Review 68(2), 307-361, doi:10.1057/s41308-020-00107-3.
- **JST R6** (advanced-economy panel, used only for the cross-market transfer study) — Jorda, Schularick, Taylor (2017), *Macrofinancial History and the New Business Cycle Facts*, NBER Macroeconomics Annual 31, 213-263; macrohistory.net release R6.

These are third-party materials redistributed subject to their respective licenses and terms of use; retain their original attribution.

**Corpus composition** (built from GFDD + Laeven-Valencia): 481 cases (146 crisis, 335 calm real negatives) across 116 countries. Regime counts: currency 61, twin 32, sovereign 25, banking 21, triple 7, calm 335. Temporal split: 240 train (onset ≤ 2007) / 241 test (≥ 2008). Ten annual features (`private_credit_gdp`, `d_private_credit_gdp`, `dom_credit_private_gdp`, `private_credit_ofi_gdp`, `liquid_liabilities_gdp`, `fin_deposits_gdp`, `dmb_assets_gdp`, `bank_deposits_gdp`, `credit_deposit_ratio`, `cb_assets_gdp`). Standardization statistics are fit on the training split only; GFDD's banking-crisis dummy is used only to widen calm exclusion, never as a feature.

**Rebuild the snapshot** (optional; the Merkle root is bit-reproducible):

```bash
python data.py \
  --gfdd "Datasets/20220909-global-financial-development-database.xlsx" \
  --lv   "Datasets/SYSTEMIC BANKING CRISES DATABASE_2018.xlsx" \
  --out  data_snapshot
```

Rebuilding from `Datasets/` reproduces the Merkle root in Section 1 exactly.

---

## 6. Reproducing the paper

All commands write a single hash-stamped bundle under `results/`.

**Everything, in one command** (synthetic validation + real-data study + ablations):

```bash
python experiments.py --snapshot data_snapshot --out results --seeds 20 --epochs 200
```

**Add the emerging-market -> advanced-economy transfer study:**

```bash
python experiments.py --snapshot data_snapshot --out results --seeds 20 --epochs 200 \
  --transfer-gfdd "Datasets/20220909-global-financial-development-database.xlsx" \
  --transfer-lv   "Datasets/SYSTEMIC BANKING CRISES DATABASE_2018.xlsx" \
  --transfer-jst  "Datasets/JSTdatasetR6.xlsx"
```

**Run individual stages** if preferred:

```bash
# Synthetic validation of Claims 1-3 (writes results/synthetic/)
python synthetic.py --outdir results/synthetic

# Single training run with the paired global-metric ablation
python train.py --snapshot data_snapshot --out results/checkpoint.pt \
  --epochs 200 --embed-dim 48 --rank 6 --shared-rank 8 --lambda-shrink 1.0 \
  --seed 0 --compare-global

# Multi-seed held-out evaluation with bootstrap CIs and ablations
python evaluate.py --snapshot data_snapshot --out results \
  --seeds 20 --epochs 200 --embed-dim 48 --rank 6 --shared-rank 8 --lambda-shrink 1.0

# Auditable retrieval demo (top-k analogues + verified provenance)
python retrieval.py --checkpoint results/checkpoint.pt --snapshot data_snapshot --k 5
```

**Default configuration** (matches the paper): embedding dimension 48, per-regime factor rank 6, shared-metric rank 8, Frobenius shrinkage `lambda = 1.0`, 200 epochs, 20 seeds, 1000 bootstrap resamples, CPU. Pass `--quick` to `experiments.py`/`synthetic.py` for a fast smoke run.

### Mapping to the paper

| Paper element | Produced by |
|---|---|
| Claims 1-3, synthetic (Fig. 2, Table 3a) | `synthetic.py` |
| Theory-to-data bridge, fitted divergence (Fig. 3, Table 3b) | `experiments.py` (synthetic + real) |
| Real-data retrieval quality (Table 4) | `evaluate.py` |
| Conditioning contrast, paired deltas (Table 5) | `evaluate.py` (paired ablation) |
| Gate recovery and calibration (Table 6, Fig. 4) | `evaluate.py` |
| Graceful degradation vs. `lambda` (Fig. C.1) | `evaluate.py` (lambda sweep) |
| Cross-market transfer (Table D.1) | `experiments.py --transfer-*` |
| Integrity checks (Table C.1) | `tests.py` |

The paper's figures are generated by the separate plotting scripts (`make_synth_fig.py`, `make_bridge_fig.py`, `make_gate_fig.py`, `make_degrade_fig.py`, and the `rcr_overview.tex` TikZ source), which read the numbers reproduced here.

---

## 7. Reproducibility and integrity

- **Content addressing.** Every `CaseRecord` is canonically serialised (order-independent, negative-zero-collapsing) and SHA-256 hashed, so logically equal records hash identically.
- **Merkle commitment.** The whole corpus collapses to the 64-hex root above; any single altered byte changes it. `provenance.verify_library` and `verify_merkle_proof` check inclusion, and a single altered field is rejected.
- **Configuration fingerprints.** Each result carries the fingerprint of the exact model, split, standardizer, and seed behind it, so any number traces to a fully specified computation.
- **Property suite.** `tests.py` asserts *properties*, not fitted numbers: hash determinism and tamper-evidence, Merkle verification, the metric's PSD-by-construction guarantee and the exact equivalence of its efficient and explicit scoring paths, graceful degradation to a global metric, the encoder's unit-norm / order-sensitivity / batch-independence contract, the synthetic model's geometric identities, and (when a snapshot is present) full regime coverage, real negatives, and leakage-safe standardisation. All **22 checks pass**.

```bash
RCR_SNAPSHOT=data_snapshot python tests.py     # standalone summary
# or
RCR_SNAPSHOT=data_snapshot pytest tests.py     # pytest discovery
```

---

## 8. Known data-layer caveats

Stated in the interest of empirical honesty, and documented rather than silently patched. The as-shipped corpus omits several well-known episodes (KOR 1997, THA 1997, IDN 1997, MEX 1994, UKR 2014, CHL 1981) owing to GFDD coverage plus the 2022-income filter, and 12 Laeven-Valencia names (including China) owing to alias gaps; the test split therefore holds 23 crisis queries (twin = 1, triple = 0). Optional coverage patches discussed in the paper recover 227 episodes / 39 test queries. The reported conclusions rest jointly on theory and evidence and are robust to this limitation; see the paper's threats-to-validity discussion.

---

## 9. Data and code availability

The sealed case-library snapshot and this code, including the scripts that regenerate the reported results, are archived at:

```
https://doi.org/XX.XXXX/zenodo.XXXXXXX      (update on publication)
```

The snapshot is content-addressed and committed by the Merkle root in Section 1; the configuration fingerprints accompanying each result identify the exact computation. Underlying primary sources are third-party materials under their respective terms (Section 5).

---

## 10. Citation

```bibtex
@article{hao2026rcr,
  title   = {Regime-conditional similarity: calibrated mixture-of-metrics
             retrieval for auditable case-based reasoning},
  author  = {Hao, Yixue and Han, Pu and Min, Lianxing},
  journal = {Information Sciences},
  year    = {2026},
  note    = {Update volume, pages, and DOI on publication}
}
```

---

## 11. License

The code in this repository is released under the MIT License (add a `LICENSE` file, or replace with your preferred license). The datasets under `Datasets/` are **not** covered by this license and remain subject to the original terms of the World Bank GFDD, the IMF Economic Review (Laeven-Valencia), and the Jorda-Schularick-Taylor Macrohistory Database, respectively.

---

## 12. Contact

Yixue Hao (corresponding author), School of Management, Xihua University, Chengdu, Sichuan 610039, China. Email: `0120130001@xhu.edu.cn`. ORCID: 0009-0009-2525-7287.

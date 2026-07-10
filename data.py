"""Real-data layer: canonical sources -> a sealed, content-hashed case library.

This module turns three canonical, citable datasets into the corpus of
:class:`provenance.CaseRecord` artefacts the retrieval framework reasons over,
and it is where the paper's *empirical honesty* is enforced in code.

Why the primary source is GFDD + Laeven--Valencia (not JST)
-----------------------------------------------------------
The framework's thesis is *regime-conditional* similarity, which is only
demonstrable on a corpus in which multiple regimes are well populated. The
Laeven--Valencia (2020) chronology, clustered into episodes, yields (globally)
77 banking, 148 currency, 63 sovereign, 78 twin, and 26 triple episodes --- but
this diversity lives in *emerging markets*. The Jorda--Schularick--Taylor panel
covers only 18 advanced economies, whose post-1970 crises are almost all
banking; on JST alone the corpus collapses to a single populated regime, which
cannot support the thesis.

The **primary** corpus is therefore built from the World Bank Global Financial
Development Database (GFDD) --- financial-system ratios for ~135 emerging
markets --- with regimes composed from Laeven--Valencia. GFDD carries
financial-*development* features (private-credit/GDP, liquid-liabilities/GDP,
loan-to-deposit, ...) rather than fast macro-*cycle* variables; for a
*retrieval/analogy* task (find past episodes with a similar financial-system
configuration under the same regime) these are appropriate descriptors, and the
regime-conditional metric learns which of them matter per regime.

Feature scope, and a central honest limitation
----------------------------------------------
GFDD's 4x2 conceptual matrix (Cihak et al., 2012) spans depth, access,
efficiency, and stability, each for financial institutions and markets. The
features used here (see :data:`GFDD_CODE_MAP`) sit almost entirely in the
*depth-of-institutions* cell --- credit and deposit ratios. They deliberately do
**not** include the slope of the yield curve, which Bluwstein et al. (2023) find
to be, alongside credit growth, one of the two most important predictors of
systemic crises. GFDD has no interest-rate term structure, so the slope cannot be
constructed here; this is a genuine limitation of the primary corpus and a
plausible reason the retrieval signal is modest. It also reflects an unavoidable
tension in the available data: JST *has* the yield-curve slope but (post-1970)
only banking crises, while GFDD *has* the multi-regime diversity the thesis needs
but not the slope --- no single public panel offers both. Credit growth, the
predictor the two datasets agree on, *is* captured (``d_private_credit_gdp``).

JST retains a role as the **advanced-economy source** for the cross-market
transfer check (train the geometry on emerging markets, test transfer to
advanced economies) on the shared feature subset :data:`COMMON_FEATURES`. The
loader machinery is generic over the source (:class:`FeatureSource`), so both
corpora flow through one pipeline.

Enforced invariants
--------------------
  * **Real crises, real types.** Episodes and regimes come from L-V's separate
    banking / currency / sovereign columns, clustered by co-occurrence.
  * **Real negatives.** Calm cases are genuine non-crisis country-years, never
    synthetic or time-shuffled windows.
  * **No label leakage.** GFDD's banking-crisis dummy (``oi19``) is used only to
    *exclude* calm anchors, never as a feature.
  * **No split leakage.** Standardisation is fit on the training split only.
  * **Auditable identity.** Every case is content-addressed via
    :mod:`provenance`; the corpus collapses to one Merkle root.

Conventions
-----------
Sources are annual; the :class:`provenance.CaseRecord` schema encodes onsets as
``YYYYQq``, so annual episodes are anchored at year-end (``onset -> "YYYYQ4"``)
and trajectories are annual arrays ``(window_years, n_features)``. Splits are
stored in a *separate* index; a case's identity is its content, so re-splitting
must never change its hash.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

import provenance as prov

__all__ = [
    "DataConfig", "FeatureSource",
    "GFDD_FEATURES", "COMMON_FEATURES", "GFDD_CODE_MAP",
    "load_laeven_valencia", "gfdd_source", "jst_source",
    "cluster_onsets", "compose_regime",
    "build_crisis_cases", "build_calm_cases",
    "assign_splits", "standardize_cases",
    "build_library", "save_snapshot",
]

# --------------------------------------------------------------------------- #
# Source keys and citations (match provenance.SourceCitation.source)
# --------------------------------------------------------------------------- #

SRC_JST, SRC_LV, SRC_GFDD = "JST_R6", "LaevenValencia_2020", "WorldBank_GFDD_2022"
ACCESSED = "2026-01-15"

CITATION_TEXT: dict[str, str] = {
    SRC_JST: ("Jorda, O., Schularick, M., Taylor, A.M. (2017). Macrofinancial "
              "History and the New Business Cycle Facts. NBER Macroeconomics "
              "Annual 31, 213-263. Database release R6, macrohistory.net."),
    SRC_LV: ("Laeven, L., Valencia, F. (2020). Systemic Banking Crises Database "
             "II. IMF Economic Review 68(2), 307-361. "
             "doi:10.1057/s41308-020-00107-3."),
    SRC_GFDD: ("Cihak, M., Demirguc-Kunt, A., Feyen, E., Levine, R. (2012). "
               "Benchmarking Financial Systems Around the World. World Bank "
               "Policy Research Working Paper 6175. GFDD, August 2022 vintage."),
}

# --------------------------------------------------------------------------- #
# Feature sets
# --------------------------------------------------------------------------- #

#: GFDD indicator code -> engineered feature name. All are %-of-GDP ratios or
#: rates with good emerging-market coverage. The banking-crisis dummy (oi19) is
#: deliberately absent: it is a label, and using it as a feature would leak.
GFDD_CODE_MAP: dict[str, str] = {
    "di01": "private_credit_gdp",       # private credit by deposit money banks
    "di14": "dom_credit_private_gdp",   # domestic credit to private sector
    "di12": "private_credit_ofi_gdp",   # private credit incl. other fin. insts
    "di05": "liquid_liabilities_gdp",   # financial depth
    "di08": "fin_deposits_gdp",         # financial-system deposits
    "di02": "dmb_assets_gdp",           # deposit money bank assets
    "oi02": "bank_deposits_gdp",        # bank deposits
    "si04": "credit_deposit_ratio",     # bank credit / bank deposits (leverage)
    "di06": "cb_assets_gdp",            # central bank assets
}

#: Primary feature set (GFDD). The change in private credit is appended in the
#: source adapter to capture pre-onset credit dynamics.
GFDD_FEATURES: tuple[str, ...] = (
    "private_credit_gdp", "d_private_credit_gdp",
    "dom_credit_private_gdp", "private_credit_ofi_gdp",
    "liquid_liabilities_gdp", "fin_deposits_gdp",
    "dmb_assets_gdp", "bank_deposits_gdp",
    "credit_deposit_ratio", "cb_assets_gdp",
)

#: GFDD-and-JST common subset for the secondary transfer check (thin, honest).
COMMON_FEATURES: tuple[str, ...] = (
    "private_credit_gdp", "d_private_credit_gdp", "liquid_liabilities_gdp",
)

#: L-V country-name aliases -> GFDD country-name spelling (only mismatches).
_LV_ALIASES: dict[str, str] = {
    "Korea": "Korea, Rep.", "Korea, Republic of": "Korea, Rep.",
    "Russia": "Russian Federation", "Egypt": "Egypt, Arab Rep.",
    "Iran": "Iran, Islamic Rep.", "Venezuela": "Venezuela, RB",
    "Turkey": "Turkiye", "Turkey, Republic of": "Turkiye",
    "Slovak Republic": "Slovak Republic", "Kyrgyz Republic": "Kyrgyz Republic",
    "Yemen": "Yemen, Rep.", "Congo, Dem. Rep.": "Congo, Dem. Rep.",
    "Congo, Democratic Republic of": "Congo, Dem. Rep.",
    "Lao PDR": "Lao PDR", "Macedonia": "North Macedonia",
    "Macedonia, FYR": "North Macedonia", "Cape Verde": "Cabo Verde",
    "Gambia, The": "Gambia, The", "Gambia": "Gambia, The",
    "Hong Kong": "Hong Kong SAR, China", "Slovakia": "Slovak Republic",
    "United States": "United States", "United Kingdom": "United Kingdom",
}


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class DataConfig:
    """Resolved settings for corpus construction."""

    window_years: int = 10
    cluster_window: int = 2       # years within which L-V onsets fuse into one episode
    exclusion_years: int = 4      # calm anchors must be >= this far from any crisis
    min_quality: float = 0.70     # reject windows below this coverage score
    interp_limit: int = 2         # max consecutive years linearly interpolated
    calm_per_country: int = 3     # calm anchors sampled per country (upper bound)
    split_year: int = 2008        # onsets before -> 'train', on/after -> 'test'
    seed: int = 20260517

    def __post_init__(self) -> None:
        if self.window_years < 2:
            raise ValueError("window_years must be >= 2")
        if not (0.0 <= self.min_quality <= 1.0):
            raise ValueError("min_quality must be in [0, 1]")
        if self.cluster_window < 0 or self.exclusion_years < 0:
            raise ValueError("cluster_window/exclusion_years must be non-negative")

    def fingerprint(self) -> str:
        return prov.hash_bytes(prov.canonical_json(asdict(self)))


# --------------------------------------------------------------------------- #
# Feature source abstraction
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class FeatureSource:
    """A tidy feature panel plus the metadata the case builders need.

    Attributes
    ----------
    key : str            provenance source key (SRC_GFDD / SRC_JST).
    panel : DataFrame     columns ``iso3, year`` + ``features``.
    features : tuple[str] engineered feature names in the panel.
    name_to_iso : dict    country-name -> iso3 (for matching L-V names).
    extra_crisis_years : dict[iso3, set[int]]
        Source-reported crisis years used *only* to widen calm exclusion
        (e.g. GFDD's banking-crisis dummy). Never used as a feature.
    """

    key: str
    panel: pd.DataFrame
    features: tuple[str, ...]
    name_to_iso: dict[str, str]
    extra_crisis_years: dict[str, set[int]] = field(default_factory=dict)


def _interp_and_mask(panel: pd.DataFrame, feats: tuple[str, ...],
                     cfg: DataConfig) -> pd.DataFrame:
    """Linearly interpolate short internal gaps within each country.

    Uses ``groupby(...).transform`` rather than ``apply().reset_index(drop=True)``:
    ``transform`` returns a result aligned to the input's original index, so it
    is correct regardless of whether ``panel`` carries a contiguous index. (The
    ``apply`` + ``reset_index`` idiom silently misaligns feature values to rows
    whenever the incoming index is non-contiguous, which it is here after the
    upstream filtering/sorting.)
    """
    out = panel.copy()
    out[list(feats)] = out.groupby("iso3")[list(feats)].transform(
        lambda g: g.interpolate(method="linear", limit=cfg.interp_limit,
                                limit_area="inside")
    )
    return out


def gfdd_source(path: str, cfg: DataConfig,
                code_map: dict[str, str] = GFDD_CODE_MAP,
                features: tuple[str, ...] = GFDD_FEATURES,
                emerging_only: bool = True) -> FeatureSource:
    """Build the primary emerging-market feature source from GFDD."""
    df = pd.read_excel(path, sheet_name="Data - August 2022", engine="openpyxl")
    need = {"iso3", "country", "year", "income"}
    if not need.issubset(df.columns):
        raise ValueError(f"GFDD missing id columns {need - set(df.columns)}")
    have = {c: n for c, n in code_map.items() if c in df.columns}
    if "di01" not in have:
        raise ValueError("GFDD: private-credit code di01 not found; check vintage.")
    name_to_iso = (df[["country", "iso3"]].dropna().drop_duplicates()
                   .set_index("country")["iso3"].to_dict())
    if emerging_only:
        df = df[df["income"].astype(str) != "High income"].copy()
    df = df.rename(columns=have)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    for n in have.values():
        df[n] = pd.to_numeric(df[n], errors="coerce")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    df = df.sort_values(["iso3", "year"])
    # Dynamics: change in private credit / GDP.
    df["d_private_credit_gdp"] = df.groupby("iso3")["private_credit_gdp"].diff()
    # Extra crisis years for calm exclusion: GFDD banking-crisis dummy (oi19).
    extra: dict[str, set[int]] = {}
    if "oi19" in df.columns:
        dd = df[["iso3", "year", "oi19"]].dropna()
        for iso3, sub in dd[dd["oi19"] == 1].groupby("iso3"):
            extra[iso3] = set(int(y) for y in sub["year"])
    panel = _interp_and_mask(df[["iso3", "year", *features]], features, cfg)
    return FeatureSource(SRC_GFDD, panel.reset_index(drop=True), tuple(features),
                         name_to_iso, extra)


def jst_source(path: str, cfg: DataConfig,
               features: tuple[str, ...] = COMMON_FEATURES) -> FeatureSource:
    """Build the advanced-economy source from JST, on the COMMON feature set.

    Only the features shared with GFDD are constructed, so JST cases live in the
    same space as GFDD cases for the transfer check.
    """
    df = pd.read_excel(path, sheet_name="Sheet1", engine="openpyxl")
    df = df.rename(columns={"iso": "iso3"})
    df["year"] = df["year"].astype(int)
    df = df.sort_values(["iso3", "year"]).copy()
    grp = df["iso3"]
    df["private_credit_gdp"] = 100.0 * df["tloans"] / df["gdp"].where(df["gdp"] != 0)
    df["d_private_credit_gdp"] = df["private_credit_gdp"].groupby(grp).diff()
    df["liquid_liabilities_gdp"] = 100.0 * df["money"] / df["gdp"].where(df["gdp"] != 0)
    name_to_iso = (df[["country", "iso3"]].dropna().drop_duplicates()
                   .set_index("country")["iso3"].to_dict())
    extra: dict[str, set[int]] = {}
    if "crisisJST" in df.columns:
        cc = df[["iso3", "year", "crisisJST"]].dropna()
        for iso3, sub in cc[cc["crisisJST"] == 1].groupby("iso3"):
            extra[iso3] = set(int(y) for y in sub["year"])
    feats = tuple(f for f in features if f in df.columns)
    panel = _interp_and_mask(df[["iso3", "year", *feats]], feats, cfg)
    return FeatureSource(SRC_JST, panel.reset_index(drop=True), feats,
                         name_to_iso, extra)


# --------------------------------------------------------------------------- #
# Laeven--Valencia loader (global; keyed by ISO-3 via a source name map)
# --------------------------------------------------------------------------- #

def _parse_year_list(cell: object) -> list[int]:
    """Extract all four-digit years from a messy L-V cell (robust to noise)."""
    if cell is None:
        return []
    s = str(cell)
    if s.strip().lower() in {"none", "n.a.", "na", "nan", ""}:
        return []
    return sorted(set(int(y) for y in
                      re.findall(r"(?<!\d)(1[89]\d{2}|20\d{2})(?!\d)", s)))


def load_laeven_valencia(path: str, name_to_iso: dict[str, str],
                         ) -> tuple[dict[str, dict[str, list[int]]], dict[str, int]]:
    """Load global L-V crisis years keyed by ISO-3, using a name->iso map.

    Returns ``(lv, match_report)`` where ``lv[iso3] = {"banking":[...],
    "currency":[...], "sovereign":[...]}`` and ``match_report`` counts matched
    vs unmatched country names (unmatched countries are simply skipped).
    """
    ws = pd.read_excel(path, sheet_name="Crisis Years", engine="openpyxl", header=0)
    cols = list(ws.columns)
    if len(cols) < 5:
        raise ValueError("L-V 'Crisis Years' needs >= 5 columns")
    lv: dict[str, dict[str, list[int]]] = {}
    matched = unmatched = 0
    for _, row in ws.iterrows():
        raw = str(row[cols[0]]).strip()
        if raw.lower() in {"none", "nan", ""}:
            continue
        cand = _LV_ALIASES.get(raw, raw)
        iso3 = name_to_iso.get(cand) or name_to_iso.get(raw)
        if iso3 is None:
            unmatched += 1
            continue
        matched += 1
        banking = _parse_year_list(row[cols[1]])
        currency = _parse_year_list(row[cols[2]])
        sovereign = sorted(set(_parse_year_list(row[cols[3]])
                               + _parse_year_list(row[cols[4]])))
        lv[iso3] = {"banking": banking, "currency": currency, "sovereign": sovereign}
    return lv, {"matched": matched, "unmatched": unmatched}


# --------------------------------------------------------------------------- #
# Regime composition
# --------------------------------------------------------------------------- #

def cluster_onsets(lv_country: dict[str, list[int]], cluster_window: int
                   ) -> list[tuple[int, frozenset[str]]]:
    """Cluster a country's typed onset years into composite episodes."""
    tagged = [(y, t) for t in ("banking", "currency", "sovereign")
              for y in lv_country.get(t, [])]
    if not tagged:
        return []
    tagged.sort()
    episodes: list[tuple[int, set[str]]] = []
    cur_year, cur_types, last = tagged[0][0], {tagged[0][1]}, tagged[0][0]
    for y, t in tagged[1:]:
        if y - last <= cluster_window:
            cur_types.add(t); last = y
        else:
            episodes.append((cur_year, cur_types))
            cur_year, cur_types, last = y, {t}, y
    episodes.append((cur_year, cur_types))
    return [(y, frozenset(t)) for y, t in episodes]


def compose_regime(types: Iterable[str]) -> str:
    """Map a set of active crisis types to one of the five regimes."""
    t = set(types)
    if len(t) >= 3:
        return "triple"
    if len(t) == 2:
        return "twin"
    if t == {"banking"}:
        return "banking"
    if t == {"currency"}:
        return "currency"
    if t == {"sovereign"}:
        return "sovereign"
    raise ValueError(f"cannot compose regime from {t!r}")


# --------------------------------------------------------------------------- #
# Window extraction
# --------------------------------------------------------------------------- #

def _window(panel_c: pd.DataFrame, onset_year: int, feats: tuple[str, ...],
            w: int) -> tuple[np.ndarray | None, float]:
    """Extract the ``w``-year pre-onset window ending at ``onset_year - 1``.

    Returns ``(array, quality)`` or ``(None, quality)``. A window is admissible
    only if it has exactly ``w`` rows and *every* feature cell is finite (after
    the upstream short-gap interpolation); admissible windows therefore report
    ``quality == 1.0``. The ``min_quality`` config is thus a secondary guard --
    the binding requirement is full finiteness, which keeps standardisation and
    hashing free of missing values.
    """
    years = list(range(onset_year - w, onset_year))
    sub = panel_c[panel_c["year"].isin(years)].sort_values("year")
    if len(sub) != w:
        return None, 0.0
    arr = sub[list(feats)].to_numpy(dtype=float)
    finite = np.isfinite(arr)
    quality = float(finite.mean())
    if not finite.all():
        return None, quality
    return arr, quality


# --------------------------------------------------------------------------- #
# Case construction
# --------------------------------------------------------------------------- #

def build_crisis_cases(src: FeatureSource, lv: dict[str, dict[str, list[int]]],
                       cfg: DataConfig) -> list[prov.CaseRecord]:
    """Build unsealed, unstandardised crisis CaseRecords from L-V episodes."""
    feats = src.features
    cases: list[prov.CaseRecord] = []
    for iso3 in sorted(lv):
        pc = src.panel[src.panel["iso3"] == iso3]
        if pc.empty:
            continue
        for onset_year, types in cluster_onsets(lv[iso3], cfg.cluster_window):
            traj, q = _window(pc, onset_year, feats, cfg.window_years)
            if traj is None or q < cfg.min_quality:
                continue
            regime = compose_regime(types)
            cites = (
                prov.SourceCitation(src.key, f"{iso3}:{onset_year-cfg.window_years}"
                                    f"-{onset_year-1}:{','.join(feats)}", ACCESSED),
                prov.SourceCitation(SRC_LV, f"{iso3}:onset{onset_year}:"
                                    f"{'+'.join(sorted(types))}", ACCESSED),
            )
            cases.append(prov.CaseRecord(
                country_iso3=iso3, onset=f"{onset_year}Q4", regime=regime,
                is_crisis=True, pre_onset=traj, feature_names=feats,
                citations=cites, quality=q, outcome_loss=None,
            ))
    return cases


def build_calm_cases(src: FeatureSource, lv: dict[str, dict[str, list[int]]],
                     cfg: DataConfig, rng: np.random.Generator
                     ) -> list[prov.CaseRecord]:
    """Sample REAL calm-period negatives from non-crisis country-years."""
    feats = src.features
    cases: list[prov.CaseRecord] = []
    for iso3 in sorted(src.panel["iso3"].dropna().unique()):
        pc = src.panel[src.panel["iso3"] == iso3]
        d = lv.get(iso3, {})
        crisis_years = (set(d.get("banking", [])) | set(d.get("currency", []))
                        | set(d.get("sovereign", []))
                        | src.extra_crisis_years.get(iso3, set()))
        admissible = []
        for Y in sorted(int(y) for y in pc["year"].unique()):
            if any(abs(Y - c) <= cfg.exclusion_years for c in crisis_years):
                continue
            traj, q = _window(pc, Y, feats, cfg.window_years)
            if traj is None or q < cfg.min_quality:
                continue
            admissible.append((Y, traj, q))
        if not admissible:
            continue
        take = min(cfg.calm_per_country, len(admissible))
        for j in rng.choice(len(admissible), size=take, replace=False):
            Y, traj, q = admissible[j]
            cites = (prov.SourceCitation(
                src.key, f"{iso3}:calm:{Y-cfg.window_years}-{Y-1}", ACCESSED,
                note="non-crisis window (real negative)"),)
            cases.append(prov.CaseRecord(
                country_iso3=iso3, onset=f"{Y}Q4", regime="none", is_crisis=False,
                pre_onset=traj, feature_names=feats, citations=cites,
                quality=q, outcome_loss=None,
            ))
    return cases


# --------------------------------------------------------------------------- #
# Splits and leakage-safe standardisation
# --------------------------------------------------------------------------- #

def _natural_key(rec: prov.CaseRecord) -> str:
    return f"{rec.country_iso3}|{rec.onset}|{int(rec.is_crisis)}|{rec.regime}"


def assign_splits(cases: list[prov.CaseRecord], cfg: DataConfig) -> dict[str, str]:
    """Temporal split by natural key: year < split_year -> train, else test."""
    return {_natural_key(r): ("train" if int(r.onset[:4]) < cfg.split_year
                              else "test") for r in cases}


def standardize_cases(cases: list[prov.CaseRecord], split: dict[str, str],
                      feats: tuple[str, ...]
                      ) -> tuple[list[prov.CaseRecord], dict[str, list[float]]]:
    """Z-score every trajectory using per-feature TRAIN statistics only."""
    train = [r.pre_onset for r in cases if split.get(_natural_key(r)) == "train"]
    if not train:
        raise ValueError("no training cases to fit the standardiser")
    stack = np.concatenate(train, axis=0)
    mean = stack.mean(axis=0)
    std = np.where(stack.std(axis=0) < 1e-8, 1.0, stack.std(axis=0))
    out = [prov.CaseRecord(
        country_iso3=r.country_iso3, onset=r.onset, regime=r.regime,
        is_crisis=r.is_crisis, pre_onset=(r.pre_onset - mean) / std,
        feature_names=feats, citations=r.citations, quality=r.quality,
        outcome_loss=r.outcome_loss) for r in cases]
    return out, {"mean": mean.tolist(), "std": std.tolist()}


# --------------------------------------------------------------------------- #
# Library assembly
# --------------------------------------------------------------------------- #

def build_library(src: FeatureSource, lv: dict[str, dict[str, list[int]]],
                  cfg: DataConfig) -> dict:
    """Build a sealed, verified case library from a source + L-V labels."""
    rng = np.random.default_rng(cfg.seed)
    cases = (build_crisis_cases(src, lv, cfg)
             + build_calm_cases(src, lv, cfg, rng))
    if not cases:
        raise ValueError("no usable cases were constructed")
    split_key = assign_splits(cases, cfg)
    std_cases, stats = standardize_cases(cases, split_key, src.features)
    sealed = [r.seal() for r in std_cases]
    manifest = prov.build_manifest(sealed)
    prov.verify_library(sealed, manifest)
    split_id = {r.case_id: split_key[_natural_key(r)] for r in sealed}
    split_counts: dict[str, int] = {}
    for v in split_id.values():
        split_counts[v] = split_counts.get(v, 0) + 1
    return {
        "records": sealed, "manifest": manifest, "splits": split_id,
        "standardizer": stats, "feature_set": list(src.features),
        "source_key": src.key,
        "summary": {
            "n_cases": len(sealed),
            "n_crisis": int(sum(r.is_crisis for r in sealed)),
            "n_calm": int(sum(not r.is_crisis for r in sealed)),
            "regime_counts": manifest.regime_counts,
            "split_counts": split_counts,
            "n_countries": len({r.country_iso3 for r in sealed}),
            "merkle_root": manifest.root,
            "data_fingerprint": cfg.fingerprint(),
        },
    }


# --------------------------------------------------------------------------- #
# Snapshot writing
# --------------------------------------------------------------------------- #

def _emit_data_sources_md(bundle: dict) -> str:
    s = bundle["summary"]
    lines = ["# Data sources and reproducibility", "",
             "Built deterministically from the canonical sources below. Every "
             "case is content-addressed (SHA-256); the corpus is bound to Merkle "
             f"root `{s['merkle_root']}`.", "", "## Sources"]
    for k in (SRC_GFDD, SRC_LV, SRC_JST):
        lines.append(f"- **{k}** — {CITATION_TEXT[k]}")
    lines += ["", "## Corpus composition",
              f"- Source: {bundle['source_key']}",
              f"- Cases: {s['n_cases']} (crisis {s['n_crisis']}, calm {s['n_calm']}) "
              f"across {s['n_countries']} countries",
              f"- Regimes: {json.dumps(s['regime_counts'])}",
              f"- Splits: {json.dumps(s['split_counts'])}",
              f"- Features: {', '.join(bundle['feature_set'])}",
              f"- Config fingerprint: `{s['data_fingerprint']}`", "",
              "Negatives are real non-crisis country-years. GFDD's banking-crisis "
              "dummy is used only to widen calm exclusion, never as a feature. "
              "Standardisation is fit on the training split only."]
    return "\n".join(lines) + "\n"


def save_snapshot(bundle: dict, out_dir: str) -> None:
    """Write sealed corpus, manifest, splits, standardiser, and sources doc."""
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "case_library.json"), "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in bundle["records"]], f)
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(bundle["manifest"].to_dict(), f, indent=2)
    with open(os.path.join(out_dir, "splits.json"), "w", encoding="utf-8") as f:
        json.dump(bundle["splits"], f, indent=2)
    with open(os.path.join(out_dir, "standardizer.json"), "w", encoding="utf-8") as f:
        json.dump({"feature_set": bundle["feature_set"],
                   **bundle["standardizer"]}, f, indent=2)
    with open(os.path.join(out_dir, "DATA_SOURCES.md"), "w", encoding="utf-8") as f:
        f.write(_emit_data_sources_md(bundle))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build the RCR case library.")
    p.add_argument("--gfdd", required=True, help="path to GFDD xlsx (primary)")
    p.add_argument("--lv", required=True, help="path to Laeven-Valencia xlsx")
    p.add_argument("--jst", default=None, help="path to JST xlsx (transfer source)")
    p.add_argument("--out", default="data/processed", help="snapshot dir")
    p.add_argument("--source", choices=["gfdd", "jst"], default="gfdd")
    p.add_argument("--window", type=int, default=None)
    p.add_argument("--split-year", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_argparser().parse_args(argv)
    over: dict = {}
    if args.window is not None:
        over["window_years"] = args.window
    if args.split_year is not None:
        over["split_year"] = args.split_year
    if args.seed is not None:
        over["seed"] = args.seed
    cfg = DataConfig(**over)

    if args.source == "gfdd":
        src = gfdd_source(args.gfdd, cfg)
    else:
        if not args.jst:
            raise SystemExit("--source jst requires --jst")
        src = jst_source(args.jst, cfg)

    lv, report = load_laeven_valencia(args.lv, src.name_to_iso)
    bundle = build_library(src, lv, cfg)
    save_snapshot(bundle, args.out)
    s = bundle["summary"]
    print("Case library built and verified.")
    print(f"  L-V name match: {report}")
    print(f"  cases        : {s['n_cases']} (crisis {s['n_crisis']}, "
          f"calm {s['n_calm']}) / {s['n_countries']} countries")
    print(f"  regimes      : {s['regime_counts']}")
    print(f"  splits       : {s['split_counts']}")
    print(f"  merkle root  : {s['merkle_root']}")
    print(f"  snapshot ->  : {args.out}")


if __name__ == "__main__":
    main()

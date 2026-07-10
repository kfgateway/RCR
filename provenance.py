"""Auditability layer for regime-conditional case-based retrieval.

This module is the trust foundation of the framework. It defines the
atomic knowledge artefact --- the :class:`CaseRecord` --- and the
machinery that makes a corpus of such artefacts *tamper-evident* and
*reproducibly identifiable*:

    * a **canonical serialisation** so that logically-equal records hash
      identically regardless of field-insertion order or floating-point
      formatting incidentals;
    * **content addressing** via SHA-256, so every record carries a
      stable identity derived from its content alone;
    * a **Merkle tree** over the whole corpus, so the entire case base
      collapses to a single 64-hex-character root that changes if *any*
      byte of *any* record changes;
    * a **manifest** binding that root to the library's metadata, which
      downstream training/evaluation artefacts reference so that every
      reported number traces to one archived corpus checksum;
    * a **provenance chain** emitter, so any retrieved case can be opened
      back to its primary sources for human audit.

Design contract
---------------
The guarantees this module provides are only as strong as three
invariants, all enforced here rather than assumed:

    (I1) *Determinism.* Serialising the same logical record on any
         machine, any Python build, any dict ordering yields byte-identical
         output. Hashes are therefore portable and archivable.
    (I2) *Immutability.* A :class:`CaseRecord` cannot be mutated after
         construction; any "edit" is a new record with a new identity.
         This is what makes the Merkle root meaningful as a version key.
    (I3) *Verifiability.* Given a manifest and a corpus, :func:`verify_library`
         recomputes every hash and the root from scratch and checks them,
         so corruption or tampering is detected, not trusted-away.

The module depends only on the Python standard library (``hashlib``,
``json``, ``dataclasses``, ``datetime``, ``re``) so that the trust root has
no third-party surface. ``numpy`` is imported *only* for typing/coercion of
trajectory arrays and is not part of the hashing trust boundary.

References
----------
Merkle, R.C. (1988). A digital signature based on a conventional
encryption function. CRYPTO '87, LNCS 293, 369--378.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Final, Mapping, Sequence

import numpy as np

__all__ = [
    "PROVENANCE_SCHEMA_VERSION",
    "HASH_ALGORITHM",
    "FLOAT_DECIMALS",
    "CRISIS_REGIMES",
    "ProvenanceError",
    "IntegrityError",
    "SchemaError",
    "SourceCitation",
    "CaseRecord",
    "canonical_json",
    "hash_bytes",
    "hash_record",
    "merkle_root",
    "merkle_proof",
    "verify_merkle_proof",
    "LibraryManifest",
    "build_manifest",
    "verify_library",
    "provenance_chain",
]

# --------------------------------------------------------------------------- #
# Module-level constants (the trust parameters; changing any is a schema bump)
# --------------------------------------------------------------------------- #

#: Bumped whenever the canonical serialisation or record schema changes in a
#: way that would alter hashes. Recorded in every manifest so a stored root
#: can never be silently compared across incompatible schema versions.
PROVENANCE_SCHEMA_VERSION: Final[str] = "1.0.0"

#: The single hash primitive used everywhere. Named in the manifest so the
#: verifier and the archive agree; hard-coded (not caller-configurable) so a
#: weaker algorithm cannot be substituted at call sites.
HASH_ALGORITHM: Final[str] = "sha256"

#: Decimal places to which every float is rounded *before* serialisation.
#: This defends invariant (I1): it removes cross-platform float-repr drift
#: (e.g. 0.1000000000000000055 vs 0.1) and IEEE-754 last-bit noise from the
#: hash while preserving all economically-meaningful precision. 10 dp on
#: standardised macro-financial ratios is far below any signal scale.
FLOAT_DECIMALS: Final[int] = 10

#: The five crisis regimes, in fixed canonical order. Fixing the order here
#: (rather than inferring it from data) means the regime axis of every
#: downstream tensor is stable across runs and machines.
CRISIS_REGIMES: Final[tuple[str, ...]] = (
    "banking",
    "currency",
    "sovereign",
    "twin",
    "triple",
)

#: A leaf-vs-node domain-separation tag. Prepended to the pre-image before
#: hashing so a leaf hash can never be reinterpreted as an internal-node hash
#: (defends against second-preimage / "leaf-as-node" Merkle attacks).
_LEAF_TAG: Final[bytes] = b"\x00"
_NODE_TAG: Final[bytes] = b"\x01"

#: ISO-3166 alpha-3 shape check (three uppercase letters). Deliberately a
#: shape check, not a membership check: the canonical country set lives in the
#: data layer, and hard-coding a closed list here would couple the trust layer
#: to a particular panel vintage.
_ISO3_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Z]{3}$")

#: Accepted onset-quarter shape, e.g. "1997Q3". Year 1000--2999, quarter 1--4.
_QUARTER_RE: Final[re.Pattern[str]] = re.compile(r"^[12]\d{3}Q[1-4]$")

#: Hex-digest shape for any field that claims to be a SHA-256 hash.
_SHA256_HEX_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #

class ProvenanceError(Exception):
    """Base class for all errors raised by this module."""


class SchemaError(ProvenanceError):
    """A record or manifest violates the schema (bad field, shape, or type)."""


class IntegrityError(ProvenanceError):
    """A recomputed hash or Merkle root does not match its stored value."""


# --------------------------------------------------------------------------- #
# Canonical serialisation  (invariant I1)
# --------------------------------------------------------------------------- #

def _canonicalise(obj: Any) -> Any:
    """Recursively coerce ``obj`` into a JSON-canonical, hash-stable form.

    Rules, chosen so that logically-equal inputs map to byte-identical JSON:

    * ``float`` (and numpy floats) are rounded to :data:`FLOAT_DECIMALS` and
      normalised so that ``-0.0`` becomes ``0.0``. Non-finite floats (NaN,
      +/-Inf) are rejected: they have no canonical JSON form and must never
      silently enter a hash.
    * numpy scalars/arrays are converted to Python scalars/nested lists.
    * ``dict`` keys are coerced to ``str`` and the mapping is *not* sorted
      here (``json.dumps(..., sort_keys=True)`` performs the ordering); we
      only recurse into values.
    * ``tuple``/``list`` preserve order (order is semantic for trajectories
      and timelines).
    * ``bool`` (and ``numpy.bool_``) is coerced to a Python ``bool``. This
      must be checked *before* ``int``, since ``bool`` is a subclass of
      ``int`` in Python; ``numpy.bool_`` is *not* a subclass of either, so it
      is matched explicitly (otherwise it would fall through to the
      unserialisable-type error).

    Raises
    ------
    SchemaError
        On a non-finite float or an unserialisable type.
    """
    # bool must precede int: isinstance(True, int) is True. numpy.bool_ is not
    # a subclass of bool or int, so match it explicitly; coerce to Python bool
    # because json cannot serialise numpy.bool_.
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        if not np.isfinite(f):
            raise SchemaError(
                f"non-finite float {f!r} cannot be canonically serialised; "
                "clean or impute it in the data layer before hashing"
            )
        r = round(f, FLOAT_DECIMALS)
        # Collapse negative zero so 0.0 and -0.0 hash identically.
        return r + 0.0 if r == 0.0 else r
    if isinstance(obj, str):
        return obj
    if obj is None:
        return None
    if isinstance(obj, np.ndarray):
        return _canonicalise(obj.tolist())
    if isinstance(obj, Mapping):
        return {str(k): _canonicalise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_canonicalise(v) for v in obj]
    raise SchemaError(
        f"type {type(obj).__name__!r} is not canonically serialisable"
    )


def canonical_json(obj: Any) -> bytes:
    """Serialise ``obj`` to canonical UTF-8 JSON bytes (invariant I1).

    Determinism is achieved by (a) canonicalising values via
    :func:`_canonicalise`, (b) sorting object keys, (c) eliminating
    insignificant whitespace, and (d) disabling non-ASCII escaping ambiguity
    by emitting ``ensure_ascii=False`` UTF-8. ``allow_nan=False`` is a
    belt-and-braces guard; :func:`_canonicalise` has already rejected non-finite
    floats.

    Returns
    -------
    bytes
        UTF-8 encoded canonical JSON.
    """
    return json.dumps(
        _canonicalise(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


# --------------------------------------------------------------------------- #
# Hash primitives  (invariant I3 building blocks)
# --------------------------------------------------------------------------- #

def hash_bytes(data: bytes, *, tag: bytes = b"") -> str:
    """Return the SHA-256 hex digest of ``tag + data``.

    The optional domain-separation ``tag`` lets callers make the pre-image of
    a hash unambiguous across contexts (e.g. Merkle leaf vs node).
    """
    h = hashlib.new(HASH_ALGORITHM)
    h.update(tag)
    h.update(data)
    return h.hexdigest()


def hash_record(record: "CaseRecord") -> str:
    """Compute the content hash of ``record`` from its canonical payload.

    The hash is taken over the record's *content* payload only --- the
    ``case_id`` field is excluded, because ``case_id`` *is* this hash and
    including it would be circular. This is what makes ``case_id`` a true
    content address: two records with identical content necessarily share an
    identity, and any content change necessarily changes it.
    """
    payload = record._hash_payload()
    return hash_bytes(canonical_json(payload), tag=_LEAF_TAG)


# --------------------------------------------------------------------------- #
# SourceCitation --- one primary-source pointer  (invariant I2)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, slots=True)
class SourceCitation:
    """An immutable pointer to one primary source backing a case field.

    Attributes
    ----------
    source : str
        Short canonical source key, e.g. ``"JST_R6"``, ``"LaevenValencia_2020"``,
        ``"WorldBank_GFDD_2022"``.
    locator : str
        A within-source locator: a table/sheet name, a row key, a variable
        code, or a page --- whatever lets a human re-find the exact datum.
    accessed : str
        ISO-8601 date (YYYY-MM-DD) on which the source was accessed/pinned.
    note : str
        Optional free-text clarification (may be empty).

    Notes
    -----
    Frozen and slotted: instances are hashable and cannot be mutated, so a
    citation cannot drift after a record is sealed.
    """

    source: str
    locator: str
    accessed: str
    note: str = ""

    def __post_init__(self) -> None:
        for name in ("source", "locator", "accessed"):
            val = getattr(self, name)
            if not isinstance(val, str) or not val.strip():
                raise SchemaError(
                    f"SourceCitation.{name} must be a non-empty string; "
                    f"got {val!r}"
                )
        # Validate the access date is a real calendar date in ISO form.
        try:
            _dt.date.fromisoformat(self.accessed)
        except ValueError as exc:
            raise SchemaError(
                f"SourceCitation.accessed must be ISO-8601 (YYYY-MM-DD); "
                f"got {self.accessed!r}"
            ) from exc

    def to_dict(self) -> dict[str, str]:
        """Return a plain-dict view (used in canonical payloads and export)."""
        return {
            "source": self.source,
            "locator": self.locator,
            "accessed": self.accessed,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "SourceCitation":
        """Rebuild from a mapping, ignoring unknown keys defensively."""
        try:
            return cls(
                source=str(d["source"]),
                locator=str(d["locator"]),
                accessed=str(d["accessed"]),
                note=str(d.get("note", "")),
            )
        except KeyError as exc:
            raise SchemaError(
                f"SourceCitation.from_dict missing required key: {exc.args[0]!r}"
            ) from exc


# --------------------------------------------------------------------------- #
# CaseRecord --- the atomic knowledge artefact  (invariants I1, I2)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CaseRecord:
    """An immutable, content-addressed record of one historical episode.

    A :class:`CaseRecord` is the unit the entire retrieval framework reasons
    over. It bundles the *evidence* (pre-onset trajectory, regime label,
    outcome) with its *provenance* (per-field primary-source citations and a
    data-quality score) and its *identity* (a content hash).

    Fields
    ------
    country_iso3 : str
        ISO-3166 alpha-3 code (shape-validated: three uppercase letters).
    onset : str
        Onset quarter, ``"YYYYQq"`` (e.g. ``"1997Q3"``). For non-crisis
        ("calm") records this is the as-of quarter of the sampled window.
    regime : str
        One of :data:`CRISIS_REGIMES` for crisis records, or ``"none"`` for a
        calm/negative record. Kept as a plain string (not an enum) so it
        serialises transparently; membership is validated in ``__post_init__``.
    is_crisis : bool
        ``True`` for a crisis episode, ``False`` for a calm/negative window.
        Cross-checked against ``regime`` (``regime == "none"`` iff not crisis).
    pre_onset : np.ndarray
        Real-valued array of shape ``(window_len, n_features)`` --- the
        pre-onset (or calm) macro-financial trajectory. Stored as float; the
        canonical payload rounds to :data:`FLOAT_DECIMALS`.
    feature_names : tuple[str, ...]
        Column names for ``pre_onset``, length ``n_features``. Persisted so a
        stored trajectory is never silently re-columned.
    outcome_loss : float | None
        Cumulative output loss over the post-onset horizon (percent of GDP),
        or ``None`` if not applicable (e.g. calm records).
    citations : tuple[SourceCitation, ...]
        Primary-source pointers. Must be non-empty: a record with no
        provenance is not admissible in an auditable library.
    quality : float
        Data-quality score in ``[0, 1]`` (coverage x reliability x
        winsorisation). The data layer filters on this; it is persisted so the
        filter threshold used is itself auditable.
    schema_version : str
        The provenance schema version this record was built under.
    case_id : str
        The content hash (SHA-256 hex). Left empty at construction and filled
        by :meth:`seal`; *excluded* from the hash pre-image (see
        :func:`hash_record`).

    Immutability
    ------------
    ``frozen=True`` blocks attribute reassignment. The one controlled
    exception is sealing the ``case_id`` after construction, done via
    ``object.__setattr__`` inside :meth:`seal` and nowhere else.
    """

    country_iso3: str
    onset: str
    regime: str
    is_crisis: bool
    pre_onset: np.ndarray
    feature_names: tuple[str, ...]
    citations: tuple[SourceCitation, ...]
    quality: float
    outcome_loss: float | None = None
    schema_version: str = PROVENANCE_SCHEMA_VERSION
    case_id: str = ""

    # -- validation -------------------------------------------------------- #

    def __post_init__(self) -> None:
        # Country code shape.
        if not (isinstance(self.country_iso3, str)
                and _ISO3_RE.match(self.country_iso3)):
            raise SchemaError(
                f"country_iso3 must be 3 uppercase letters; "
                f"got {self.country_iso3!r}"
            )
        # Onset quarter shape.
        if not (isinstance(self.onset, str) and _QUARTER_RE.match(self.onset)):
            raise SchemaError(
                f"onset must match YYYYQq (e.g. '1997Q3'); got {self.onset!r}"
            )
        # Regime membership and crisis/regime consistency.
        if self.is_crisis:
            if self.regime not in CRISIS_REGIMES:
                raise SchemaError(
                    f"crisis record regime must be one of {CRISIS_REGIMES}; "
                    f"got {self.regime!r}"
                )
        else:
            if self.regime != "none":
                raise SchemaError(
                    f"non-crisis record must have regime 'none'; "
                    f"got {self.regime!r}"
                )
        # Trajectory: coerce to a contiguous float64 ndarray of shape (T, F).
        arr = np.asarray(self.pre_onset, dtype=np.float64)
        if arr.ndim != 2:
            raise SchemaError(
                f"pre_onset must be 2-D (window_len, n_features); "
                f"got shape {arr.shape}"
            )
        if arr.size == 0:
            raise SchemaError("pre_onset must be non-empty")
        if not np.all(np.isfinite(arr)):
            raise SchemaError(
                "pre_onset contains non-finite values (NaN/Inf); impute or "
                "drop in the data layer before sealing"
            )
        # feature_names must be a tuple matching the column count.
        fnames = tuple(str(x) for x in self.feature_names)
        if len(fnames) != arr.shape[1]:
            raise SchemaError(
                f"feature_names length {len(fnames)} != n_features "
                f"{arr.shape[1]}"
            )
        # Quality in [0, 1].
        q = float(self.quality)
        if not (0.0 <= q <= 1.0 and np.isfinite(q)):
            raise SchemaError(f"quality must be in [0, 1]; got {self.quality!r}")
        # Outcome loss finite-or-None.
        if self.outcome_loss is not None:
            ol = float(self.outcome_loss)
            if not np.isfinite(ol):
                raise SchemaError(
                    f"outcome_loss must be finite or None; got {self.outcome_loss!r}"
                )
        # Citations non-empty and correctly typed.
        if not self.citations:
            raise SchemaError(
                "a CaseRecord must carry at least one SourceCitation; a record "
                "without provenance is not admissible in an auditable library"
            )
        for c in self.citations:
            if not isinstance(c, SourceCitation):
                raise SchemaError(
                    f"every citation must be a SourceCitation; got "
                    f"{type(c).__name__}"
                )
        # case_id, if present, must look like a SHA-256 digest.
        if self.case_id and not _SHA256_HEX_RE.match(self.case_id):
            raise SchemaError(
                f"case_id must be a 64-char lowercase hex SHA-256 digest; "
                f"got {self.case_id!r}"
            )
        # Freeze canonical, coerced forms in place (frozen dataclass ->
        # object.__setattr__ is the sanctioned mechanism).
        object.__setattr__(self, "pre_onset", np.ascontiguousarray(arr))
        object.__setattr__(self, "feature_names", fnames)
        object.__setattr__(self, "citations", tuple(self.citations))
        object.__setattr__(self, "quality", q)
        # Coerce is_crisis to a plain Python bool so the field always matches
        # its annotation and never carries a numpy.bool_ into the hash payload.
        object.__setattr__(self, "is_crisis", bool(self.is_crisis))
        # Make the array read-only so the "immutable" promise also holds for
        # the buffer, not just the attribute binding.
        self.pre_onset.setflags(write=False)

    # -- hashing / sealing ------------------------------------------------- #

    def _hash_payload(self) -> dict[str, Any]:
        """The content payload that defines this record's identity.

        Excludes ``case_id`` (which *is* the hash). Includes everything else
        that a change in would make this a different case: the trajectory, the
        labels, the outcome, the provenance, the quality, and the schema
        version. ``pre_onset`` is emitted as a nested list so the canonical
        float-rounding rule applies element-wise.
        """
        return {
            "schema_version": self.schema_version,
            "country_iso3": self.country_iso3,
            "onset": self.onset,
            "regime": self.regime,
            "is_crisis": self.is_crisis,
            "pre_onset": self.pre_onset.tolist(),
            "feature_names": list(self.feature_names),
            "outcome_loss": self.outcome_loss,
            "quality": self.quality,
            "citations": [c.to_dict() for c in self.citations],
        }

    def compute_id(self) -> str:
        """Return this record's content hash without mutating it."""
        return hash_record(self)

    def seal(self) -> "CaseRecord":
        """Return a copy of this record with ``case_id`` set to its content hash.

        Idempotent and verifying: if ``case_id`` is already set, it is checked
        against a freshly recomputed hash and :class:`IntegrityError` is raised
        on mismatch (so ``seal`` doubles as a tamper check).
        """
        digest = self.compute_id()
        if self.case_id:
            if self.case_id != digest:
                raise IntegrityError(
                    f"case_id {self.case_id!r} does not match recomputed "
                    f"content hash {digest!r}: record has been altered"
                )
            return self
        # frozen dataclass: sanctioned single-field seal.
        object.__setattr__(self, "case_id", digest)
        return self

    def verify(self) -> bool:
        """Return ``True`` iff ``case_id`` matches the recomputed content hash.

        Raises :class:`IntegrityError` if the record was never sealed, so a
        caller cannot mistake an unsealed record for a verified one.
        """
        if not self.case_id:
            raise IntegrityError("record is unsealed; call seal() first")
        return self.case_id == self.compute_id()

    # -- (de)serialisation ------------------------------------------------- #

    def to_dict(self) -> dict[str, Any]:
        """Full dict view including ``case_id`` (for JSON export / snapshots)."""
        d = self._hash_payload()
        d["case_id"] = self.case_id
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CaseRecord":
        """Rebuild a record from a :meth:`to_dict` mapping.

        The rebuilt record is *not* auto-sealed; the caller decides whether to
        ``seal()`` (which re-derives and re-checks the hash). This keeps
        loading and verification as separate, explicit steps.
        """
        try:
            rec = cls(
                country_iso3=str(d["country_iso3"]),
                onset=str(d["onset"]),
                regime=str(d["regime"]),
                is_crisis=bool(d["is_crisis"]),
                pre_onset=np.asarray(d["pre_onset"], dtype=np.float64),
                feature_names=tuple(str(x) for x in d["feature_names"]),
                citations=tuple(
                    SourceCitation.from_dict(c) for c in d["citations"]
                ),
                quality=float(d["quality"]),
                outcome_loss=(
                    None if d.get("outcome_loss") is None
                    else float(d["outcome_loss"])
                ),
                schema_version=str(d.get("schema_version",
                                         PROVENANCE_SCHEMA_VERSION)),
                case_id=str(d.get("case_id", "")),
            )
        except KeyError as exc:
            raise SchemaError(
                f"CaseRecord.from_dict missing required key: {exc.args[0]!r}"
            ) from exc
        return rec


# --------------------------------------------------------------------------- #
# Merkle tree over the corpus  (invariant I3)
# --------------------------------------------------------------------------- #

def _merkle_parent(left: str, right: str) -> str:
    """Hash two child hex digests into their parent (node-tagged)."""
    return hash_bytes(bytes.fromhex(left) + bytes.fromhex(right), tag=_NODE_TAG)


def merkle_root(leaf_hashes: Sequence[str]) -> str:
    """Compute the Merkle root over an ordered sequence of leaf hex digests.

    Construction details, each load-bearing:

    * Leaves are consumed **in the given order**. Callers (the manifest
      builder) must pass a canonical order --- here, leaves are sorted by
      ``case_id`` --- so the root is order-independent of *insertion* but
      well-defined given the corpus.
    * Odd levels **duplicate the last node** (``h`` pairs with itself). This
      is the standard promotion rule and keeps the tree binary without
      inventing padding leaves. The known ambiguity of this rule (distinct leaf
      multisets sharing a root) is neutralised here by (a) the leaf/node domain
      tags and (b) the manifest recording the exact leaf count, so the tree
      shape a root was built from is pinned and re-checked by
      :func:`verify_library`.
    * Leaf and node hashes use **distinct domain tags** (see :data:`_LEAF_TAG`
      / :data:`_NODE_TAG`), which prevents the classic Merkle second-preimage
      attack where an internal node is passed off as a leaf.

    Parameters
    ----------
    leaf_hashes : Sequence[str]
        Non-empty sequence of 64-char lowercase hex SHA-256 digests.

    Returns
    -------
    str
        The 64-char hex Merkle root.

    Raises
    ------
    IntegrityError
        If ``leaf_hashes`` is empty or any element is not a valid digest.
    """
    if not leaf_hashes:
        raise IntegrityError("merkle_root: empty leaf set has no root")
    for h in leaf_hashes:
        if not _SHA256_HEX_RE.match(h):
            raise IntegrityError(f"merkle_root: invalid leaf digest {h!r}")

    level: list[str] = list(leaf_hashes)
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])  # promote/duplicate the last node
        level = [
            _merkle_parent(level[i], level[i + 1])
            for i in range(0, len(level), 2)
        ]
    return level[0]


def merkle_proof(leaf_hashes: Sequence[str], index: int) -> list[tuple[str, str]]:
    """Return an audit path proving membership of leaf ``index``.

    The proof is a list of ``(sibling_hash, side)`` steps from leaf to root,
    where ``side`` is ``"L"`` if the sibling sits on the left, ``"R"`` on the
    right. Replaying the proof (see :func:`verify_merkle_proof`) reconstructs
    the root from the single leaf, so a verifier can confirm one case belongs
    to a corpus without holding the whole corpus.

    Raises
    ------
    IndexError
        If ``index`` is out of range.
    IntegrityError
        If the leaf set is empty or malformed.
    """
    if not leaf_hashes:
        raise IntegrityError("merkle_proof: empty leaf set")
    n = len(leaf_hashes)
    if not (0 <= index < n):
        raise IndexError(f"merkle_proof: index {index} out of range [0,{n})")

    level: list[str] = list(leaf_hashes)
    idx = index
    proof: list[tuple[str, str]] = []
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        if idx % 2 == 0:  # current node is a left child; sibling on right
            proof.append((level[idx + 1], "R"))
        else:              # current node is a right child; sibling on left
            proof.append((level[idx - 1], "L"))
        level = [
            _merkle_parent(level[i], level[i + 1])
            for i in range(0, len(level), 2)
        ]
        idx //= 2
    return proof


def verify_merkle_proof(leaf: str, proof: Sequence[tuple[str, str]],
                        root: str) -> bool:
    """Return ``True`` iff replaying ``proof`` from ``leaf`` reproduces ``root``."""
    if not _SHA256_HEX_RE.match(leaf):
        raise IntegrityError(f"verify_merkle_proof: invalid leaf {leaf!r}")
    acc = leaf
    for sibling, side in proof:
        if side == "L":
            acc = _merkle_parent(sibling, acc)
        elif side == "R":
            acc = _merkle_parent(acc, sibling)
        else:
            raise IntegrityError(f"proof step side must be 'L'/'R'; got {side!r}")
    return acc == root


# --------------------------------------------------------------------------- #
# Library manifest --- binds the corpus to a single archivable root
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, slots=True)
class LibraryManifest:
    """A tamper-evident summary of a sealed case library.

    Downstream training/evaluation artefacts store ``root`` so that every
    reported number is bound to exactly one corpus state. If a single byte of
    a single case changes, ``root`` changes, and the binding breaks loudly.

    Attributes
    ----------
    schema_version : str
        Provenance schema version of the records (must match the runtime).
    hash_algorithm : str
        Hash primitive name (must match :data:`HASH_ALGORITHM`).
    n_cases : int
        Number of records in the library.
    root : str
        Merkle root over the case IDs (sorted).
    case_ids : tuple[str, ...]
        The sorted case IDs, in the exact order the root was built from.
    regime_counts : dict[str, int]
        Count of records per regime label (including ``"none"``), for a quick
        composition audit without re-reading the corpus.
    created_utc : str
        ISO-8601 UTC timestamp of manifest creation (informational; excluded
        from any hash so it does not perturb reproducibility comparisons).
    """

    schema_version: str
    hash_algorithm: str
    n_cases: int
    root: str
    case_ids: tuple[str, ...]
    regime_counts: dict[str, int]
    created_utc: str

    def __post_init__(self) -> None:
        # Light internal-consistency checks so a corrupt manifest is rejected
        # at construction/load. (Full corpus agreement is checked separately by
        # :func:`verify_library`; this only guards the manifest's own fields.)
        if not _SHA256_HEX_RE.match(self.root):
            raise IntegrityError(
                f"manifest root must be a 64-char SHA-256 hex digest; "
                f"got {self.root!r}"
            )
        if self.n_cases != len(self.case_ids):
            raise IntegrityError(
                f"manifest n_cases ({self.n_cases}) != number of case_ids "
                f"({len(self.case_ids)})"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "hash_algorithm": self.hash_algorithm,
            "n_cases": self.n_cases,
            "root": self.root,
            "case_ids": list(self.case_ids),
            "regime_counts": dict(self.regime_counts),
            "created_utc": self.created_utc,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "LibraryManifest":
        try:
            return cls(
                schema_version=str(d["schema_version"]),
                hash_algorithm=str(d["hash_algorithm"]),
                n_cases=int(d["n_cases"]),
                root=str(d["root"]),
                case_ids=tuple(str(x) for x in d["case_ids"]),
                regime_counts={str(k): int(v)
                               for k, v in dict(d["regime_counts"]).items()},
                created_utc=str(d["created_utc"]),
            )
        except KeyError as exc:
            raise SchemaError(
                f"LibraryManifest.from_dict missing key: {exc.args[0]!r}"
            ) from exc


def build_manifest(records: Sequence[CaseRecord]) -> LibraryManifest:
    """Seal, order, and summarise a set of records into a manifest.

    Steps:
      1. Seal every record (idempotent; verifies any pre-set ``case_id``).
      2. Reject duplicate content (identical ``case_id``) --- a duplicated
         episode would double-count in evaluation and is treated as an error,
         not silently deduplicated.
      3. Sort case IDs lexicographically to fix a canonical leaf order.
      4. Compute the Merkle root over that order.

    Raises
    ------
    IntegrityError
        On an empty corpus or a duplicate case.
    """
    if not records:
        raise IntegrityError("build_manifest: cannot build a manifest for an "
                             "empty library")
    sealed = [r.seal() for r in records]

    ids = [r.case_id for r in sealed]
    dupes = {h for h in ids if ids.count(h) > 1}
    if dupes:
        raise IntegrityError(
            f"build_manifest: {len(dupes)} duplicate case(s) detected "
            f"(identical content hashes): {sorted(dupes)[:3]}..."
        )

    ordered_ids = tuple(sorted(ids))
    root = merkle_root(ordered_ids)

    regime_counts: dict[str, int] = {}
    for r in sealed:
        regime_counts[r.regime] = regime_counts.get(r.regime, 0) + 1

    return LibraryManifest(
        schema_version=PROVENANCE_SCHEMA_VERSION,
        hash_algorithm=HASH_ALGORITHM,
        n_cases=len(sealed),
        root=root,
        case_ids=ordered_ids,
        regime_counts=regime_counts,
        created_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(
            timespec="seconds"
        ),
    )


def verify_library(records: Sequence[CaseRecord],
                   manifest: LibraryManifest) -> bool:
    """Recompute everything and confirm ``records`` match ``manifest`` (I3).

    This is the function a reviewer or a re-runner calls to establish that a
    corpus on disk is *exactly* the corpus a result was computed on. It
    trusts nothing: it re-seals, re-hashes, re-sorts, and re-roots, then
    checks schema version, algorithm, count, the id set, and the root.

    Returns
    -------
    bool
        ``True`` iff every check passes.

    Raises
    ------
    IntegrityError
        With a specific message identifying the first check that failed.
    """
    if manifest.schema_version != PROVENANCE_SCHEMA_VERSION:
        raise IntegrityError(
            f"schema mismatch: manifest {manifest.schema_version!r} vs runtime "
            f"{PROVENANCE_SCHEMA_VERSION!r}"
        )
    if manifest.hash_algorithm != HASH_ALGORITHM:
        raise IntegrityError(
            f"hash-algorithm mismatch: manifest {manifest.hash_algorithm!r} vs "
            f"runtime {HASH_ALGORITHM!r}"
        )
    if manifest.n_cases != len(records):
        raise IntegrityError(
            f"case-count mismatch: manifest {manifest.n_cases} vs corpus "
            f"{len(records)}"
        )
    # Recompute ids from scratch (this also verifies any pre-set case_id).
    recomputed = sorted(r.seal().case_id for r in records)
    if tuple(recomputed) != tuple(manifest.case_ids):
        raise IntegrityError(
            "case-id set/order does not match manifest: the corpus is not the "
            "one this manifest was built from"
        )
    recomputed_root = merkle_root(recomputed)
    if recomputed_root != manifest.root:
        raise IntegrityError(
            f"Merkle-root mismatch: recomputed {recomputed_root!r} vs manifest "
            f"{manifest.root!r}: corpus has been altered"
        )
    return True


# --------------------------------------------------------------------------- #
# Provenance chain --- open a retrieved result back to its sources
# --------------------------------------------------------------------------- #

def provenance_chain(record: CaseRecord,
                     manifest: LibraryManifest | None = None) -> dict[str, Any]:
    """Emit a human-auditable provenance chain for one (retrieved) record.

    The returned mapping is what a monitoring desk would open when a retrieval
    surfaces ``record``: the case identity, its verification status, its
    labels and outcome, and --- the point --- the full list of primary-source
    citations that back it. If a ``manifest`` is supplied, the chain also
    asserts (and reports) membership of the record's ``case_id`` in that
    corpus, so the reviewer sees not just "here are the sources" but "this
    exact artefact is part of the archived, root-bound library".

    This function performs no network or disk I/O; it operates purely on the
    in-memory record and manifest, so it is safe to call inside a serving loop.

    Raises
    ------
    IntegrityError
        If ``record`` carries a ``case_id`` that does not match its content
        (i.e. it has been tampered with): sealing re-derives and re-checks the
        hash, so a corrupted record fails loudly rather than being surfaced as
        a trustworthy analogue.
    """
    sealed = record.seal()  # ensures case_id present + consistent
    chain: dict[str, Any] = {
        "case_id": sealed.case_id,
        "verified": sealed.verify(),
        "country_iso3": sealed.country_iso3,
        "onset": sealed.onset,
        "regime": sealed.regime,
        "is_crisis": sealed.is_crisis,
        "outcome_loss": sealed.outcome_loss,
        "quality": sealed.quality,
        "schema_version": sealed.schema_version,
        "sources": [c.to_dict() for c in sealed.citations],
    }
    if manifest is not None:
        chain["in_library"] = sealed.case_id in set(manifest.case_ids)
        chain["library_root"] = manifest.root
    return chain

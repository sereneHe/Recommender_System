#!/usr/bin/env python3
"""Explicit W adapter contract for lag-augmented synthetic feature sets.

A lag column (``X3_lag1``) is a *predictor*, not a DAG node: it has no
structural coefficient, so it can never carry a learned edge.  The preflight's
tier ``t1``/``t2`` therefore need a W matrix that is the base node W plus an
**explicit zero extension** over the lag columns.

This module deliberately defines the *interface, the validation and the
manifest* only.  ``build_augmented_w`` raises ``NotImplementedError`` on
purpose: the alignment must be reviewed against this contract before it is
wired into training, otherwise a silent mis-alignment would masquerade as a
model result (or, worse, trip the missing-column MILP fallback and quietly
replace the true DAG with an estimated one).

Contract
--------
1. ``X{i}_lag{k}`` is a predictor feature, never a DAG node.
2. A lag column may enter W only through an explicit all-zero row/column pair.
3. The result must not require a MILP recomputation (no missing-column
   fallback): ``recalculate_dag=false`` must find every active feature in
   ``row_and_col_names``.
4. ``alignment_manifest`` must record the W shape, the name order and
   ``w_source``.
5. Every tier must use the same target column and the same row mask.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

LAG_COLUMN_RE = re.compile(r"^(?P<node>.+)_lag(?P<k>[0-9]+)$")

# Keys every alignment manifest must carry (contract point 4).
REQUIRED_MANIFEST_KEYS = (
    "w_shape",
    "row_and_col_names",
    "w_source",
    "node_columns",
    "lag_columns",
    "target",
)

VALID_W_SOURCES = ("supplied_true_w", "estimated_milp", "supplied_true_w_zero_extended")


def split_lag_column(name: str) -> tuple[str, int] | None:
    """Return ``(node, k)`` for ``X3_lag1`` style names, else ``None``."""
    match = LAG_COLUMN_RE.match(str(name))
    if match is None:
        return None
    return match.group("node"), int(match.group("k"))


def classify_columns(base_names, feature_names, target: str) -> dict:
    """Partition ``feature_names`` into node columns, lag columns and unknown.

    Raises ``ValueError`` on a column that is neither a declared base node, nor
    a lag column of a declared base node, nor the target.  Refusing here is what
    keeps contract point 1 honest: an unrecognised column must never be folded
    into the node W by accident.
    """
    base = [str(n) for n in base_names]
    target = str(target)
    nodes, lagged, unknown = [], [], []
    for raw in feature_names:
        name = str(raw)
        if name == target:
            continue
        if name in base:
            nodes.append(name)
            continue
        parsed = split_lag_column(name)
        if parsed is not None and parsed[0] in base:
            lagged.append(name)
            continue
        unknown.append(name)
    if unknown:
        raise ValueError(
            "feature names must be base nodes or '<node>_lag<k>' columns; "
            f"unrecognised: {unknown}"
        )
    return {"node_columns": nodes, "lag_columns": lagged, "target": target}


@dataclass
class WAlignment:
    """The (not yet produced) aligned W plus its audit record."""

    W_aug: object
    row_and_col_names: list[str]
    alignment_manifest: dict = field(default_factory=dict)


def validate_alignment_manifest(manifest: dict) -> list[str]:
    """Return the contract violations in ``manifest`` (empty means valid)."""
    problems: list[str] = []
    if not isinstance(manifest, dict):
        return ["manifest is not a mapping"]
    for key in REQUIRED_MANIFEST_KEYS:
        if key not in manifest:
            problems.append(f"missing manifest key {key!r}")
    source = manifest.get("w_source")
    if source is not None and source not in VALID_W_SOURCES:
        problems.append(f"w_source {source!r} not in {VALID_W_SOURCES}")
    names = manifest.get("row_and_col_names")
    shape = manifest.get("w_shape")
    if names is not None and shape is not None:
        try:
            rows, cols = int(shape[0]), int(shape[1])
        except (TypeError, IndexError, ValueError):
            problems.append(f"w_shape {shape!r} is not a (rows, cols) pair")
        else:
            if rows != len(names) or cols != len(names):
                problems.append(
                    f"w_shape {shape} does not match {len(names)} names")
    node_columns = manifest.get("node_columns")
    lag_columns = manifest.get("lag_columns")
    if node_columns is not None and lag_columns is not None:
        overlap = set(map(str, node_columns)) & set(map(str, lag_columns))
        if overlap:
            problems.append(f"columns classified both as node and lag: {sorted(overlap)}")
    return problems


def build_augmented_w(base_w, base_names, feature_names, target: str) -> WAlignment:
    """Return the W aligned to ``feature_names``.

    NOT IMPLEMENTED YET -- this is the contract placeholder.  When it is
    implemented it must: place ``base_w`` entries by node name, add an all-zero
    row/column pair for every lag column, and return the manifest described in
    :data:`REQUIRED_MANIFEST_KEYS` with ``w_source`` from
    :data:`VALID_W_SOURCES`.

    The input checks below run first, so a caller that passes an inconsistent
    feature set gets a precise error instead of a mis-aligned matrix.
    """
    classification = classify_columns(base_names, feature_names, target)
    base = [str(n) for n in base_names]
    if str(target) not in base:
        raise ValueError(f"target {target!r} is not one of the base names")
    base_w = getattr(base_w, "shape", None)
    if base_w is not None and tuple(base_w) != (len(base), len(base)):
        raise ValueError(
            f"base_w shape {tuple(base_w)} does not match {len(base)} base names")
    raise NotImplementedError(
        "build_augmented_w is an interface only: the zero-extension of lag "
        "columns must be implemented and reviewed against the contract in "
        "scripts/evidence_tree/w_adapter.py (points 1-5) before it is wired "
        "into training.  Received a valid request with "
        f"{len(classification['node_columns'])} node and "
        f"{len(classification['lag_columns'])} lag columns."
    )

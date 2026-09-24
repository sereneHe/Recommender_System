#!/usr/bin/env python3
"""Strict evidence builder v3 (nuisance-stratified, replicate-aware).

Fixes over v2
-------------
* nuisance_config_hash is now BLACKLIST-based: full resolved config minus the
  declared treatment keys minus `experiment`.  Any uncontrolled difference
  (dropout, grad_clip_norm, data fingerprint, graph-estimation keys, ...)
  therefore breaks the pair instead of being silently ignored.
* runs are stratified by nuisance_config_hash; candidate/reference are paired
  ONLY inside the same stratum, by cluster (graph x noise / target x seed).
* each explicit evidence batch (or legacy PBS-job cohort) is analysed
  separately.  A later rerun of the same graph/noise seed is never averaged
  into an earlier code/configuration snapshot; bootstrap resamples CLUSTERS,
  never folds or reruns.
* per-comparison `require` predicates drop known-bad arms (e.g. e6 runs with
  use_w_constraints=false from the old bug).
* any result with < MIN_CLUSTERS independent units is suffixed `_low_power`.

Outputs: reports/evidence_pairs.csv, evidence_index.csv, evidence_exclusions.csv
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "progress"
PAIRS = ROOT / "reports" / "evidence_pairs.csv"
INDEX = ROOT / "reports" / "evidence_index.csv"
EXCL = ROOT / "reports" / "evidence_exclusions.csv"
POOLED = ROOT / "reports" / "evidence_pooled.csv"
REGISTRY_CSV = ROOT / "reports" / "evidence_registry.csv"
REGISTRY_MD = ROOT / "docs" / "evidence_tree_registry.md"
RNG = np.random.default_rng(20260923)
N_BOOT = 20000
PRACTICAL = 0.05
EQUIV_MARGIN = 0.05
MIN_CLUSTERS = 5

# Comparisons that must share an exact split + fold-W cache.  Empty until the
# fold-W cache / exact-split receipts exist for every data family; the contract
# is recorded but non-blocking in the meantime (see pair_contract).
REQUIRES_SAME_FOLD_W: set[str] = set()
VIOLATION_ABS_TOL = 1e-6
VIOLATION_REL_TOL = 0.01

# Comparisons whose protocol is frozen in experiment_registry.yaml.  These get
# the registry's own practical threshold and minimum independent unit count, and
# their split/fold-W contract becomes BLOCKING (a missing receipt excludes the
# pair instead of being silently recorded).
REGISTRY_HYP_FOR_COMPARISON = {
    "H1.ce_vs_nn.ER": "H1",
    "A8.nn_vs_xgb.SmoothER": "A8",
    "A8.ce_vs_nn.SmoothER": "A8",
}
REGISTRY_PATH = ROOT / "experiment_registry.yaml"


def _load_registry() -> dict:
    try:
        import sys as _sys
        _sys.path.insert(0, str(ROOT / "scripts"))
        from experiment_registry import load as _load, validate as _validate
        reg = _load(REGISTRY_PATH)
        problems = _validate(reg)
        if problems:
            raise ValueError("; ".join(problems))
        return reg
    except Exception as exc:  # registry is optional for legacy inspection only
        print(f"[build_evidence_index] registry unavailable, using legacy defaults: {exc}",
              file=__import__("sys").stderr)
        return {}


REGISTRY = _load_registry()


def _cohort_eligibility() -> tuple[set[str] | None, dict]:
    """Cohorts allowed into STRICT (frozen-protocol) inference.

    Returns (eligible_set, registry).  eligible_set is None when the registry
    itself is unavailable (legacy inspection mode: no frozen gating at all).
    Otherwise only cohorts with an ACTIVE reservation that match no
    diagnostic-only pattern are eligible.  Legacy cohorts without manifests,
    retired cohorts, and diagnostic patterns are excluded with an auditable
    reason instead of being silently averaged in.
    """
    if not REGISTRY:
        return None, {}
    try:
        import sys as _sys
        _sys.path.insert(0, str(ROOT / "scripts"))
        from experiment_registry import (
            COHORT_MANIFESTS_DIR, COHORT_REGISTRY_DIR,
            is_cohort_eligible_for_strict,
        )
        import yaml as _yaml
        known: set[str] = set()
        reg_dir = Path(COHORT_REGISTRY_DIR)
        if reg_dir.exists():
            known.update(p.name for p in reg_dir.iterdir() if p.is_dir())
        man_dir = Path(COHORT_MANIFESTS_DIR)
        if man_dir.exists():
            for m in man_dir.glob("*.yaml"):
                known.add(m.stem)
        eligible = {c for c in known
                    if is_cohort_eligible_for_strict(c, REGISTRY, reg_dir)}
        return eligible, REGISTRY
    except Exception as exc:
        print(f"[build_evidence_index] eligibility unavailable: {exc}",
              file=__import__("sys").stderr)
        return None, REGISTRY


ELIGIBLE_STRICT_COHORTS, _ = _cohort_eligibility()


def _exec_key_norm(v) -> str:
    try:
        import math
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return "na"
    except Exception:
        pass
    if pd.isna(v):
        return "na"
    return str(v)


def _duplicate_execution_keys(full: pd.DataFrame) -> set[tuple]:
    """Full-df (pre-canonical-filter) rerun detection for strict cohorts.

    Two rows with the same (cohort, experiment, seeds, target, layout) are two
    executions of one unit, not two units.  Mirror LAYOUT copies of a single
    execution (mlruns vs multirun) share everything except `layout` and are NOT
    flagged here; same-layout reruns are, and the cluster is excluded.
    """
    cols = ["cohort", "experiment", "graph_seed", "noise_seed", "target", "seed", "layout"]
    for c in cols:
        if c not in full.columns:
            full[c] = "na"
    keys = full[cols].fillna("na").astype(str).apply(tuple, axis=1)
    counts = keys.value_counts()
    return set(counts[counts > 1].index)


def _seed_units(h: dict) -> int:
    try:
        import sys as _sys
        _sys.path.insert(0, str(ROOT / "scripts"))
        from experiment_registry import effective_seed_table, seed_table_units
        return seed_table_units(effective_seed_table(h))
    except Exception:
        return 0


# Frozen protocols make their split + fold-W contract blocking.
for _cid, _hyp in REGISTRY_HYP_FOR_COMPARISON.items():
    _h = (REGISTRY.get("hypotheses") or {}).get(_hyp) or {}
    if _h.get("fold_w_contract") == "same" and _h.get("split_contract") == "same":
        REQUIRES_SAME_FOLD_W.add(_cid)


def protocol_for(cid: str) -> dict:
    """Return the frozen protocol for a comparison, or legacy defaults."""
    hyp = REGISTRY_HYP_FOR_COMPARISON.get(cid)
    h = (REGISTRY.get("hypotheses") or {}).get(hyp) if hyp else None
    if not h:
        return {
            "min_units": MIN_CLUSTERS,
            "practical": PRACTICAL,
            "strong": PRACTICAL,
            "strict_contract": False,
            "frozen": False,
            "expected_units": 0,
        }
    return {
        "min_units": int(h["min_independent_units"]),
        "practical": float(h["practical_threshold"]),
        "strong": float(h.get("strong_threshold", h["practical_threshold"])),
        "strict_contract": cid in REQUIRES_SAME_FOLD_W,
        "frozen": True,
        "expected_units": _seed_units(h),
    }

GRAPH_SUF = __import__("re").compile(r"^(?P<base>.+?)_graph(?P<g>\d+)_noise(?P<n>\d+)$")
SEED_SUF = __import__("re").compile(r"^(?P<base>.+?)_seed(?P<s>\d+)$")
CONFIG_CACHE: dict[str, dict] = {}


def load_cfg(run_dir: str) -> dict:
    if run_dir not in CONFIG_CACHE:
        try:
            CONFIG_CACHE[run_dir] = yaml.safe_load((ROOT / run_dir / "config.yaml").read_text()) or {}
        except Exception:
            CONFIG_CACHE[run_dir] = {}
    return CONFIG_CACHE[run_dir]


def flatten(o, p=""):
    out = {}
    if isinstance(o, dict):
        for k, v in o.items():
            out.update(flatten(v, f"{p}.{k}" if p else str(k)))
    elif isinstance(o, list):
        out[p] = json.dumps(o, default=str)
    else:
        out[p] = o
    return out


def sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


# Blacklist: keys REMOVED before hashing the nuisance config, per treatment.
TREATMENT_PREFIXES = {
    "ce": ["solver.ce_", "solver.ci_", "solver.constrained", "solver.constraint_audit",
           "solver.use_ci_penalty", "solver.w_matrix_space", "solver.weights_bound",
           "solver.use_w_constraints", "solver.w_constraint_mode", "solver.w_prediction",
           "solver.rho0", "solver.rho_mult", "solver.lambda0", "solver.lambda1", "solver.lambda2",
           "solver.lambda_update_rate", "solver.use_stochastic_constrained_optimizer"],
    "w": ["solver.use_w_constraints", "solver.w_constraint_mode", "solver.w_prediction",
          "solver.weights_bound", "solver.constraints_mode", "solver.constrained",
          "solver.constraint_audit", "solver.w_matrix_space", "solver.rho0", "solver.rho_mult",
          "solver.w_bias_calibration"],
    "trees": ["solver.n_estimators"],
    # A4 varies the graph estimate / MIP budget.  The DAG/W estimate is the
    # intervention here, so every graph key is dropped before the nuisance
    # hash; otherwise the two arms land in different strata and can never pair.
    "graph": ["solver.recalculate_dag", "solver.dag_solver_backend", "solver.time_limit",
              "solver.target_mip_gap", "solver.edge_penalty", "solver.max_parents",
              "solver.enable_clique_constraints", "solver.max_clique_size",
              "solver.weights_bound", "solver.lambda1", "solver.lambda2",
              "solver.nonzero_threshold", "solver.loss_type", "solver.reg_type",
              "solver.a_reg_type", "solver.constraints_mode", "solver.callback_mode",
              "solver.robust", "solver.tabu_edges", "solver.clique_", "solver.gurobi_"],
    # A6 data-representation / robust-loss treatments.
    "data": ["problem.feature_lag", "problem.add_time_trend", "problem.regime_break_date"],
    "loss": ["solver.prediction_loss", "solver.huber_delta"],
    # A5 capacity / training-budget treatments.
    "budget": ["solver.n_outer", "solver.n_inner"],
    "capacity": ["solver.hidden_dim", "solver.depth"],
    "lr_wd": ["solver.learning_rate", "solver.weight_decay"],
    "none": [],
    "end_to_end": [],
}

def nuisance_hash(cfg: dict, treat: str) -> str:
    flat = flatten(cfg)
    drops = ["experiment", "problem.evidence_node", "problem.evidence_arm",
             "problem.evidence_scope", "problem.evidence_batch_id"] + TREATMENT_PREFIXES.get(treat, [])
    sel = {k: v for k, v in flat.items() if not any(k == d or k.startswith(d) for d in drops)}
    return sha(sel)


def _truthy(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _hash_text(value) -> str:
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value or "").strip()
    return "" if text.lower() in {"nan", "none", "nat"} else text


# Constraint-violation columns.  A positive verdict requires that the candidate
# is not worse on ANY recorded violation, not only the CE one: a W arm can add a
# W-prediction violation that the CE-only comparison would otherwise miss.
VIOLATION_COLUMNS = ("ce_prediction_violation", "w_prediction_l2")


def violation_value(run) -> float:
    """Largest recorded constraint violation for a run (NaN when none)."""
    vals = []
    for col in VIOLATION_COLUMNS:
        try:
            v = float(getattr(run, col, np.nan))
        except (TypeError, ValueError):
            v = np.nan
        if np.isfinite(v):
            vals.append(v)
    return max(vals) if vals else float("nan")


def fold_w_policy(comparison_type: str, treatment: str) -> str:
    """Return the predeclared fold-W contract for a comparison.

    Equality is required only when W/DAG is a frozen nuisance.  A W treatment
    requires only the candidate's receipt; a graph/data treatment requires a
    receipt from each arm but intentionally permits different hashes.
    """
    if comparison_type != "mechanism":
        return "none"
    if treatment == "w":
        return "candidate"
    if treatment in {"graph", "data"}:
        return "both"
    return "same"


def pair_contract(reference, candidate, w_policy, strict: bool = False):
    """Check the reproducibility contract for one paired run.

    The comparison contract is explicit and per-comparison:

    * the outer split is recorded, and enforced only when the comparison opts
      in (``strict=True``) AND both runs carry a split receipt;
    * fold-W policy is explicit: equality for a shared nuisance, an arm-local
      receipt when W/DAG is the intervention, and no W requirement for
      end-to-end baselines that do not estimate a graph.

    While the exact-split / fold-W-cache receipts are not yet implemented for
    every family, the contract is recorded but NOT blocking (strict=False), so
    the tree does not silently drop every legacy comparison.  G0.3/G0.5 report
    the remaining gap separately.
    """
    if isinstance(w_policy, bool):
        w_policy = "same" if w_policy else "none"
    ref_split = _hash_text(getattr(reference, "split_hash", ""))
    cand_split = _hash_text(getattr(candidate, "split_hash", ""))
    split_present = bool(ref_split and cand_split)
    split_equal = bool(split_present and ref_split == cand_split)
    ref_wh = _hash_text(getattr(reference, "fold_w_cache_hash", ""))
    cand_wh = _hash_text(getattr(candidate, "fold_w_cache_hash", ""))
    w_present = bool(ref_wh and cand_wh)
    w_equal = bool(w_present and ref_wh == cand_wh)
    if w_policy == "same":
        w_valid = w_equal
    elif w_policy == "candidate":
        w_valid = bool(cand_wh)
    elif w_policy == "both":
        w_valid = bool(ref_wh and cand_wh)
    elif w_policy == "none":
        w_valid = True
    else:
        raise ValueError(f"unknown fold-W policy: {w_policy}")
    contract_valid = bool(split_equal and w_valid) if strict else True
    return {
        "split_hash_present": split_present,
        "split_hash_equal": split_equal,
        "fold_w_cache_present": w_present,
        "fold_w_cache_hash_equal": w_equal,
        "fold_w_cache_policy": w_policy,
        "fold_w_cache_applicable": w_policy != "none",
        "fold_w_contract_valid": w_valid,
        "pair_contract_strict": bool(strict),
        "pair_contract_valid": contract_valid,
    }


def duplicate_reason(R: list, C: list) -> str | None:
    """Return a reason when a cluster/nuisance/cohort key holds >1 record.

    More than one run with the same cluster key inside one cohort is a duplicate
    execution, not an extra independent unit; the whole pair must be rejected.
    """
    if len(R) != 1 or len(C) != 1:
        return "invalid_pairing:duplicate_within_cohort"
    return None


def parse_arm(exp: str):
    m = GRAPH_SUF.match(exp)
    if m:
        return m["base"], int(m["g"]), int(m["n"]), None
    m = SEED_SUF.match(exp)
    if m:
        return m["base"], None, None, int(m["s"])
    return exp, None, None, None


_MECHANISM_SCOPE = {
    "smooth_additive": "synthetic/SmoothER",
    "compositional": "synthetic/CompositionalER",
    "highdim_smooth": "synthetic/HighDim",
    "periodic": "synthetic/Periodic",
    "temporal_smooth": "synthetic/Temporal",
}


def scope_of(r):
    if r["problem"] == "synthetic":
        # The A7 mechanisms are distinct data families: never record them as
        # plain ER or their evidence would be mixed with the linear results.
        try:
            mech = r.get("synthetic_mechanism")
        except Exception:
            mech = None
        mech = "" if mech is None else str(mech).strip().lower()
        if mech in _MECHANISM_SCOPE:
            return _MECHANISM_SCOPE[mech]
        return f"synthetic/{r['graph_type']}"
    if r["problem"] == "industry_eu":
        return "FRED"
    if r["problem"] == "codiet":
        return "CoDiet"
    return str(r["problem"])


def cluster_of(r):
    if r["problem"] == "synthetic":
        return f"{r['graph_seed']}|{r['noise_seed']}"
    seed = r["s"] if pd.notna(r["s"]) else r["seed"]
    return f"{r['target']}|{seed}"


def boot_ci(d):
    n = len(d)
    if n < 2:
        return (np.nan, np.nan)
    idx = RNG.integers(0, n, size=(N_BOOT, n))
    return tuple(np.percentile(d[idx].mean(axis=1), [2.5, 97.5]))


def req(**kw):
    """require predicate over the runs dataframe."""
    def f(d):
        m = pd.Series(True, index=d.index)
        for k, v in kw.items():
            m &= d[k].eq(v)
        return m
    return f


# id, root, scope, ref_arm, cand_arm, type, treatment, candidate require-predicate
COMPARISONS = [
    ("C0b.ce_vs_nn.ER", "C0b", "synthetic/ER", "PLAN_CE_NEW_ER_e0_no_constraint",
     "PLAN_CE_NEW_ER_e2_pure_ce_upgraded", "mechanism", "ce",
     req(use_ci_penalty=True, use_w_constraints=False)),
    ("C0b.ce_vs_nn.SF", "C0b", "synthetic/SF", "PLAN_CE_NEW_SF_e0_no_constraint",
     "PLAN_CE_NEW_SF_e2_pure_ce_upgraded", "mechanism", "ce",
     req(use_ci_penalty=True, use_w_constraints=False)),
    ("A7.codiet_cmi_vs_w", "A7", "CoDiet", "PLAN_CE_NEW_CODIET_V2_hce_w_only",
     "PLAN_CE_NEW_CODIET_V2_hce_cmi_upgraded", "mechanism", "ce",
     req(use_ci_penalty=True, ci_penalty_kind="discrete_conditional_independence",
         use_w_constraints=True)),
    ("C0b.w_vs_nn.ER", "C0b", "synthetic/ER", "PLAN_CE_NEW_ER_e0_no_constraint",
     "PLAN_CE_NEW_ER_e1_true_w", "mechanism", "w",
     req(use_w_constraints=True, use_ci_penalty=False)),
    ("C0b.wce_vs_w.ER", "C0b", "synthetic/ER", "PLAN_CE_NEW_ER_e1_true_w",
     "PLAN_CE_NEW_ER_e3_w_plus_ce_upgraded", "mechanism", "ce",
     req(use_w_constraints=True, use_ci_penalty=True)),
    ("C0b.wce_vs_ce.ER", "C0b", "synthetic/ER", "PLAN_CE_NEW_ER_e2_pure_ce_upgraded",
     "PLAN_CE_NEW_ER_e3_w_plus_ce_upgraded", "mechanism", "w",
     req(use_w_constraints=True, use_ci_penalty=True)),
    ("C0b.w_ablation.ER", "C0b", "synthetic/ER", "PLAN_CE_NEW_ER_m0_xgb100_no_w",
     "PLAN_CE_NEW_ER_m1_xgb100_w", "mechanism", "w", None),
    # ---- H1: the frozen confirmatory CE-only vs matched NN cohort -----------
    # Protocol lives in experiment_registry.yaml (min 20 units, 5% threshold,
    # split + fold-W contracts blocking).  Kept separate from every legacy C0b
    # cohort so a pilot run can never be promoted into the H1 claim.
    ("H1.ce_vs_nn.ER", "H1", "synthetic/ER", "EV:H1:nn",
     "EV:H1:ce_only", "mechanism", "ce",
     req(use_ci_penalty=True, use_w_constraints=False)),
    # ---- B0 baseline rail ----
    ("B0.tree_count.ER", "B0", "synthetic/ER", "PLAN_CE_NEW_ER_b1_mark",
     "PLAN_CE_NEW_ER_a1_mark_100", "end_to_end", "trees", None),
    ("B0.cc_vs_mark100.ER", "B0", "synthetic/ER", "PLAN_CE_NEW_ER_a1_mark_100",
     "PLAN_CE_NEW_ER_b2_mark_with_cc", "end_to_end", "none", None),
    # ---- nonlinear boundary ----
    ("P1.3.nonlin_ce.ER", "P1.3", "synthetic/ER", "PLAN_CE_NEW_ER_b2nl_mark_with_cc",
     "PLAN_CE_NEW_ER_e4_nonlinear_pure_ce", "end_to_end", "end_to_end",
     req(use_ci_penalty=True)),
    ("P1.3.nonlin_wce.ER", "P1.3", "synthetic/ER", "PLAN_CE_NEW_ER_b2nl_mark_with_cc",
     "PLAN_CE_NEW_ER_e6_nonlinear_w_plus_ce", "end_to_end", "end_to_end",
     req(use_w_constraints=True, use_ci_penalty=True)),
    # Direct nonlinear mechanism contrasts.  These use the same NN rather
    # than an out-of-cohort Mark-CC baseline, so they are the relevant test of
    # whether the CE statistics add value under misspecification.
    ("P1.3.nonlin_ce_vs_nn.ER", "P1.3", "synthetic/ER", "PLAN_CE_NEW_ER_e7_nonlinear_no_constraint",
     "PLAN_CE_NEW_ER_e4_nonlinear_pure_ce", "mechanism", "ce",
     req(use_ci_penalty=True, use_w_constraints=False)),
    ("P1.3.quantile_vs_linear.ER", "P1.3", "synthetic/ER", "PLAN_CE_NEW_ER_e4_nonlinear_pure_ce",
     "PLAN_CE_NEW_ER_e5_nonlinear_quantile_ce", "mechanism", "ce",
     req(use_ci_penalty=True, use_w_constraints=False)),
    ("P1.3.nonlin_wce_vs_ce.ER", "P1.3", "synthetic/ER", "PLAN_CE_NEW_ER_e4_nonlinear_pure_ce",
     "PLAN_CE_NEW_ER_e6_nonlinear_w_plus_ce", "mechanism", "w",
     req(use_ci_penalty=True, use_w_constraints=True)),
    # ---- product (provisional) ----
    ("C0a.hcce_vs_markcc.ER", "C0a", "synthetic/ER", "PLAN_CE_NEW_ER_b2_mark_with_cc",
     "PLAN_CE_NEW_ER_e2_pure_ce_upgraded", "end_to_end", "end_to_end",
     req(use_ci_penalty=True, use_w_constraints=False)),
    # ---- Clean configuration-repair batch ----
    ("C0b.ce_vs_nn.repair_ER", "C0b", "synthetic/ER", "PLAN_EVIDENCE_REPAIR_ER_lin_nn",
     "PLAN_EVIDENCE_REPAIR_ER_lin_ce", "mechanism", "ce",
     req(use_ci_penalty=True, use_w_constraints=False)),
    ("C0b.w_vs_nn.repair_ER", "C0b", "synthetic/ER", "PLAN_EVIDENCE_REPAIR_ER_lin_nn",
     "PLAN_EVIDENCE_REPAIR_ER_lin_w", "mechanism", "w",
     req(use_w_constraints=True, use_ci_penalty=False)),
    ("C0b.wce_vs_ce.repair_ER", "C0b", "synthetic/ER", "PLAN_EVIDENCE_REPAIR_ER_lin_ce",
     "PLAN_EVIDENCE_REPAIR_ER_lin_w_ce", "mechanism", "w",
     req(use_w_constraints=True, use_ci_penalty=True)),
    ("P1.3.nonlin_ce_vs_nn.repair_ER", "P1.3", "synthetic/ER", "PLAN_EVIDENCE_REPAIR_ER_nonlin_nn",
     "PLAN_EVIDENCE_REPAIR_ER_nonlin_ce", "mechanism", "ce",
     req(use_ci_penalty=True, use_w_constraints=False)),
    ("P1.3.quantile_vs_linear.repair_ER", "P1.3", "synthetic/ER", "PLAN_EVIDENCE_REPAIR_ER_nonlin_ce",
     "PLAN_EVIDENCE_REPAIR_ER_nonlin_quantile_ce", "mechanism", "ce",
     req(use_ci_penalty=True, use_w_constraints=False)),
    ("P1.3.nonlin_wce_vs_ce.repair_ER", "P1.3", "synthetic/ER", "PLAN_EVIDENCE_REPAIR_ER_nonlin_ce",
     "PLAN_EVIDENCE_REPAIR_ER_nonlin_w_ce", "mechanism", "w",
     req(use_ci_penalty=True, use_w_constraints=True)),
    # ---- Powered CE confirmation, then explicitly gated optimizer sweep ----
    ("C0b.ce_vs_nn.priority_ER", "C0b", "synthetic/ER", "PLAN_EVIDENCE_PRIORITY_ER_nn",
     "PLAN_EVIDENCE_PRIORITY_ER_ce", "mechanism", "ce",
     req(use_ci_penalty=True, use_w_constraints=False)),
    ("A1.alm_vs_pbm.priority_ER", "A1", "synthetic/ER", "PLAN_EVIDENCE_PRIORITY_ER_opt_alm_all",
     "PLAN_EVIDENCE_PRIORITY_ER_opt_pbm_all", "mechanism", "ce",
     req(use_ci_penalty=True, use_w_constraints=False)),
    ("A1.pbm_vs_spbm.priority_ER", "A1", "synthetic/ER", "PLAN_EVIDENCE_PRIORITY_ER_opt_pbm_all",
     "PLAN_EVIDENCE_PRIORITY_ER_opt_spbm_all", "mechanism", "ce",
     req(use_ci_penalty=True, use_w_constraints=False)),
    ("A1.spbm_vs_sco.priority_ER", "A1", "synthetic/ER", "PLAN_EVIDENCE_PRIORITY_ER_opt_spbm_all",
     "PLAN_EVIDENCE_PRIORITY_ER_sco_spbm_all", "mechanism", "ce",
     req(use_ci_penalty=True, use_w_constraints=False)),

    # ---- Field-matched EV:<node>:<arm> comparisons --------------------------
    # These match the explicit evidence_node/evidence_arm fields written by
    # scripts/evidence_tree/*.sh, so the builder does not depend on the
    # experiment-name string.
    # A1 optimizer backends (ref_ce is alm_pbm = ALM under independent-only).
    ("A1.ALM_ALL.er", "A1", "synthetic/ER", "EV:ref:ref_ce",
     "EV:A1.ALM_ALL:ce_alm_all", "mechanism", "ce", req(use_ci_penalty=True, use_w_constraints=False)),
    ("A1.PBM_ALL.er", "A1", "synthetic/ER", "EV:ref:ref_ce",
     "EV:A1.PBM_ALL:ce_pbm_all", "mechanism", "ce", req(use_ci_penalty=True, use_w_constraints=False)),
    ("A1.STOCHASTIC_PBM.er", "A1", "synthetic/ER", "EV:A1.PBM_ALL:ce_pbm_all",
     "EV:A1.STOCHASTIC_PBM:ce_spbm_all", "mechanism", "ce", req(use_ci_penalty=True, use_w_constraints=False)),
    ("A1.SCO_LAYER.er", "A1", "synthetic/ER", "EV:A1.STOCHASTIC_PBM:ce_spbm_all",
     "EV:A1.SCO_LAYER:ce_sco", "mechanism", "ce", req(use_ci_penalty=True, use_w_constraints=False)),
    # A2 constraint embedding.  The ordinary legacy-global W arm is the
    # canonical B0.hc_w_only baseline and is not rerun under A2.
    ("A2.W_target_residual.er", "A2", "synthetic/ER", "EV:ref:ref_nn",
     "EV:A2.W_target_residual:w_target_residual", "mechanism", "w", req(use_w_constraints=True, use_ci_penalty=False)),
    ("A2.W_mask.er", "A2", "synthetic/ER", "EV:ref:ref_nn",
     "EV:A2.W_mask:w_mask", "mechanism", "w", req(use_w_constraints=True, use_ci_penalty=False)),
    ("A2.W_bias_calibration.er", "A2", "synthetic/ER", "EV:ref:ref_nn",
     "EV:A2.W_bias_calibration:w_bias_calibration", "mechanism", "w", req(use_w_constraints=False, use_ci_penalty=False)),
    ("A2.balanced_batch.er", "A2", "synthetic/ER", "EV:ref:ref_ce",
     "EV:A2.balanced_batch:ce_balanced_batch", "mechanism", "ce", req(use_ci_penalty=True, use_w_constraints=False)),
    ("A2.pruning.er", "A2", "synthetic/ER", "EV:ref:ref_ce",
     "EV:A2.pruning:ce_pruning", "mechanism", "ce", req(use_ci_penalty=True, use_w_constraints=False)),
    # A3 CI/CE statistic.
    ("A3.covariance.er", "A3", "synthetic/ER", "EV:ref:ref_ce",
     "EV:A3.covariance:ce_covariance", "mechanism", "ce", req(use_ci_penalty=True, use_w_constraints=False)),
    # Single window-vs-HAC contrast (ref_ce is the window arm); the previous
    # two reverse-direction entries double-counted the same comparison.
    ("A3.window_vs_hac.er", "A3", "synthetic/ER", "EV:ref:ref_ce",
     "EV:A3.HAC_SE:ce_hac_se", "mechanism", "ce", req(use_ci_penalty=True, use_w_constraints=False)),
    ("A3.sign_flip_filter.er", "A3", "synthetic/ER", "EV:A3.sign_flip_filter:dep_filter_off",
     "EV:A3.sign_flip_filter:dep_filter_on", "mechanism", "ce", req(use_ci_penalty=True, use_w_constraints=False)),
    # A4 graph estimation & causal prior.  One shared MIP control arm serves
    # both the time and the gap contrast.
    ("A4.mip_time.er", "A4", "synthetic/ER", "EV:A4.mip_control:mip_control",
     "EV:A4.mip_time:mip_time_300", "mechanism", "graph", req(recalculate_dag=True)),
    ("A4.mip_gap.er", "A4", "synthetic/ER", "EV:A4.mip_control:mip_control",
     "EV:A4.mip_gap:mip_gap_1e4", "mechanism", "graph", req(recalculate_dag=True)),
    ("A4.edge_penalty.er", "A4", "synthetic/ER", "EV:A4.edge_penalty:edge_penalty_0",
     "EV:A4.edge_penalty:edge_penalty_0p05", "mechanism", "graph", req(recalculate_dag=True)),
    ("A4.parents_limit_3.er", "A4", "synthetic/ER", "EV:A4.parents_limit:parents_none",
     "EV:A4.parents_limit:parents_3", "mechanism", "graph", req(recalculate_dag=True)),
    ("A4.parents_limit_4.er", "A4", "synthetic/ER", "EV:A4.parents_limit:parents_none",
     "EV:A4.parents_limit:parents_4", "mechanism", "graph", req(recalculate_dag=True)),
    ("A4.parents_limit_5.er", "A4", "synthetic/ER", "EV:A4.parents_limit:parents_none",
     "EV:A4.parents_limit:parents_5", "mechanism", "graph", req(recalculate_dag=True)),
    ("A4.clique_cap.er", "A4", "synthetic/ER", "EV:A4.clique_cap:clique_off",
     "EV:A4.clique_cap:clique_on", "mechanism", "graph", req(recalculate_dag=True)),
    # A5 capacity & budget.
    ("A5.hidden_depth.er", "A5", "synthetic/ER", "EV:ref:ref_nn",
     "EV:A5.hidden_depth:hicap_64x3", "mechanism", "capacity", None),
    ("A5.lr_wd.er", "A5", "synthetic/ER", "EV:ref:ref_nn",
     "EV:A5.lr_wd:lr0p05_wd0p10", "mechanism", "lr_wd", None),
    ("A5.n_outer_inner.er", "A5", "synthetic/ER", "EV:ref:ref_nn",
     "EV:A5.n_outer_inner:updates_half_5x50", "mechanism", "budget", None),
    ("A5.n_outer_inner_matched.er", "A5", "synthetic/ER", "EV:ref:ref_nn",
     "EV:A5.n_outer_inner:updates_matched_20x50", "mechanism", "budget", None),
    # A6 data / time / robustness (FRED).
    ("A6.lag.fred", "A6", "FRED", "EV:ref:fred_ref",
     "EV:A6.lag:feature_lag_3", "mechanism", "data", None),
    ("A6.trend.fred", "A6", "FRED", "EV:ref:fred_ref",
     "EV:A6.trend:add_time_trend", "mechanism", "data", None),
    ("A6.regime.fred", "A6", "FRED", "EV:ref:fred_ref",
     "EV:A6.regime:regime_break_2020", "mechanism", "data", None),
    ("A6.huber.fred", "A6", "FRED", "EV:ref:fred_ref",
     "EV:A6.huber:huber_loss", "mechanism", "loss", None),
    # B0 classical baselines.
    ("B0.mark_100.er", "B0", "synthetic/ER", "EV:B0.mark_10:mark_10",
     "EV:B0.mark_100:mark_100", "end_to_end", "trees", None),
    ("B0.mark_cc_100.er", "B0", "synthetic/ER", "EV:B0.mark_cc_10:mark_cc_10",
     "EV:B0.mark_cc_100:mark_cc_100", "end_to_end", "trees", None),
    ("B0.hc_nn.er", "B0", "synthetic/ER", "EV:B0.mark_100:mark_100",
     "EV:B0.hc_nn_no_constraint:hc_nn_no_constraint", "end_to_end", "none", None),
    ("B0.hc_w_only.er", "B0", "synthetic/ER", "EV:B0.hc_nn_no_constraint:hc_nn_no_constraint",
     "EV:B0.hc_w_only:hc_w_only", "mechanism", "w", req(use_w_constraints=True, use_ci_penalty=False)),
    ("B0.hc_ce.er", "B0", "synthetic/ER", "EV:B0.hc_nn_no_constraint:hc_nn_no_constraint",
     "EV:B0.hc_ce_only:hc_ce_only", "mechanism", "ce", req(use_w_constraints=False, use_ci_penalty=True)),
    ("B0.hc_w_ce.er", "B0", "synthetic/ER", "EV:B0.hc_w_only:hc_w_only",
     "EV:B0.hc_w_ce:hc_w_ce", "mechanism", "ce", req(use_w_constraints=True, use_ci_penalty=True)),
    ("B0.hc_w_ce_vs_ce.er", "B0", "synthetic/ER", "EV:B0.hc_ce_only:hc_ce_only",
     "EV:B0.hc_w_ce:hc_w_ce", "mechanism", "w", req(use_w_constraints=True, use_ci_penalty=True)),
]

# ---- A8: NN-favourable mechanisms (own scopes; never mixed with linear ER) --
# Each mechanism gets: NN vs the fixed XGB reference (the pre-registered NN
# advantage claim), plus CE/W/W+CE vs the NN baseline to separate "NN bias" from
# "the constraints add information".
_A8_MECHANISMS = [
    ("smooth", "synthetic/SmoothER"),
    ("compos", "synthetic/CompositionalER"),
    ("highdim", "synthetic/HighDim"),
    ("periodic", "synthetic/Periodic"),
    ("temporal", "synthetic/Temporal"),
]
for _short, _scope in _A8_MECHANISMS:
    COMPARISONS += [
        (f"A8.{_short}.nn_vs_xgb", "A8", _scope, "EV:A8:xgb100",
         "EV:A8:nn0", "end_to_end", "none", None),
        (f"A8.{_short}.ce_vs_nn", "A8", _scope, "EV:A8:nn0",
         "EV:A8:nn_ce_true", "mechanism", "ce",
         req(use_ci_penalty=True, use_w_constraints=False)),
        (f"A8.{_short}.w_vs_nn", "A8", _scope, "EV:A8:nn0",
         "EV:A8:nn_w_true", "mechanism", "w",
         req(use_ci_penalty=False, use_w_constraints=True)),
        (f"A8.{_short}.wce_vs_nn", "A8", _scope, "EV:A8:nn0",
         "EV:A8:nn_w_ce_true", "mechanism", "w",
         req(use_ci_penalty=True, use_w_constraints=True)),
    ]


def cohort_of(r) -> str:
    """Explicit batch id wins; otherwise use the containing PBS-job mirror.

    Distinct PBS jobs often contain the same synthetic graph/noise seeds but
    were made after a code or configuration change.  They are replications for
    diagnosis, not independent observations to be averaged together.
    """
    batch = getattr(r, "evidence_batch_id", None)
    if pd.notna(batch) and str(batch).strip():
        return str(batch).strip()
    legacy = getattr(r, "execution_cohort", None)
    if pd.notna(legacy) and str(legacy).strip():
        return str(legacy).strip()
    parts = Path(str(r.run_dir)).parts
    try:
        return parts[parts.index("metacentrum_runs") + 1]
    except (ValueError, IndexError):
        return "legacy_unknown_cohort"


def main():
    df = pd.read_csv(OUT / "runs_metrics.csv")
    if "is_canonical" in df.columns:
        df = df[df.is_canonical.fillna(True).astype(bool)].copy()
    df["arm"], df["g"], df["n"], df["s"] = zip(*df["experiment"].map(parse_arm))
    df["scope"] = df.apply(scope_of, axis=1)
    df["cluster"] = df.apply(cluster_of, axis=1)
    val = df[df.metric_validity == "valid"].copy()
    # The scanner added these fields in v4.  Keep a fallback so an older
    # runs_metrics.csv can still be inspected before a refresh.
    if "evidence_batch_id" not in val:
        val["evidence_batch_id"] = np.nan
    if "execution_cohort" not in val:
        val["execution_cohort"] = np.nan
    for col in ("evidence_node", "evidence_arm", "evidence_scope"):
        if col not in val:
            val[col] = np.nan
    val["cohort"] = val.apply(cohort_of, axis=1)

    # Full-df (pre-canonical-filter) rerun keys for strict cohorts.  The
    # canonical filter keeps one mirror copy per logical run; a second
    # same-layout execution of the same unit must exclude the cluster, not
    # silently collapse into the survivor.
    _full = df[df.metric_validity == "valid"].copy()
    for _c in ("evidence_batch_id", "execution_cohort", "evidence_node",
               "evidence_arm", "evidence_scope", "graph_seed", "noise_seed",
               "target", "seed", "layout"):
        if _c not in _full:
            _full[_c] = np.nan
    _full["cohort"] = _full.apply(cohort_of, axis=1)
    _full["cluster"] = _full.apply(cluster_of, axis=1)
    DUP_EXEC_KEYS = _duplicate_execution_keys(_full)
    del _full

    now = datetime.now().isoformat(timespec="seconds")
    try:
        git_commit = (ROOT / ".git" / "HEAD").read_text().strip()[:40]
    except Exception:
        git_commit = ""

    pair_rows, idx_rows, excl_rows = [], [], []

    def _select(sub, token):
        """Resolve an arm token.

        ``EV:<node>:<arm>`` matches the explicit evidence_node/evidence_arm
        fields written into the problem config, so the builder no longer
        depends on the experiment-name string.  A trailing ``*`` on the arm is
        a prefix match (used where the arm label encodes a tunable such as the
        MIP time limit).  Any other token is a legacy experiment-name prefix
        match.
        """
        if token.startswith("EV:"):
            _, node, arm = token.split(":", 2)
            node_match = sub.evidence_node.astype(str) == node
            if arm.endswith("*"):
                arm_match = sub.evidence_arm.astype(str).str.startswith(arm[:-1])
            else:
                arm_match = sub.evidence_arm.astype(str) == arm
            return sub[node_match & arm_match]
        return sub[sub.arm == token]

    for cid, root, scope, ref_arm, cand_arm, ctype, treat, require in COMPARISONS:
        proto = protocol_for(cid)
        sub = val[val.scope == scope]
        ref_all = _select(sub, ref_arm)
        cand_all = _select(sub, cand_arm)
        if proto["frozen"] and ELIGIBLE_STRICT_COHORTS is not None:
            # Strict inference admits only ACTIVE, registered cohorts.  Retired,
            # unregistered (legacy), and diagnostic-pattern cohorts stay visible
            # as diagnostic records but can never back the frozen claim.
            for _side, _rows in (("reference", ref_all), ("candidate", cand_all)):
                _bad = _rows[~_rows.cohort.astype(str).isin(ELIGIBLE_STRICT_COHORTS)]
                for _b in _bad.itertuples():
                    excl_rows.append(dict(
                        comparison_id=cid, cohort=_b.cohort, cluster=_b.cluster,
                        reason="diagnostic_ineligible_cohort",
                        **{f"{_side}_run_id": _b.run_id,
                           f"{_side}_run_dir": _b.run_dir},
                        updated_at=now,
                    ))
            ref_all = ref_all[ref_all.cohort.astype(str).isin(ELIGIBLE_STRICT_COHORTS)]
            cand_all = cand_all[cand_all.cohort.astype(str).isin(ELIGIBLE_STRICT_COHORTS)]
        if require is not None and len(cand_all):
            keep = require(cand_all)
            # Preserve an auditable record of a misconfigured arm instead of
            # silently filtering it.  This is how the historical E6 W=false
            # artifact remains visible while being excluded from inference.
            for bad in cand_all[~keep].itertuples():
                excl_rows.append(dict(
                    comparison_id=cid, cohort=cohort_of(bad), cluster=bad.cluster,
                    reason="invalid_configuration:treatment_requirement",
                    candidate_run_id=bad.run_id, candidate_run_dir=bad.run_dir,
                    candidate_config_hash=bad.resolved_config_hash,
                    updated_at=now,
                ))
            cand_all = cand_all[keep]
        if ref_all.empty or cand_all.empty:
            idx_rows.append(dict(comparison_id=cid, root=root, scope=scope, reference=ref_arm,
                                 candidate=cand_arm, comparison_type=ctype,
                                 treatment=treat,
                                 n_clusters_expected=proto["expected_units"], n_clusters_used=0,
                                 n_excluded=0,
                                 power="low", rel_improvement=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                                 statistical_win=False, practical_win=False,
                                 strongly_supported=False,
                                 practical_threshold=proto["practical"], strong_threshold=proto["strong"],
                                 min_independent_units=proto["min_units"],
                                 protocol_frozen=proto["frozen"],
                                 status="out_of_scope", note="missing valid arm", updated_at=now))
            continue

        # ---- cohort + nuisance stratification ----
        # A cohort is one explicit evidence batch (new scripts) or one PBS job
        # mirror (legacy artifacts).  Never average an old and a new config
        # simply because they used the same synthetic seeds.
        def stratify(rows):
            out = {}
            for r in rows.itertuples():
                nh = nuisance_hash(load_cfg(r.run_dir), treat) if ctype == "mechanism" else "n/a"
                out.setdefault((r.cohort, nh, r.cluster), []).append(r)
            return out

        rs, cs = stratify(ref_all), stratify(cand_all)
        cohorts = sorted({k[0] for k in rs} & {k[0] for k in cs})
        if not cohorts:
            idx_rows.append(dict(
                comparison_id=cid, root=root, scope=scope, cohort="", reference=ref_arm,
                candidate=cand_arm, comparison_type=ctype, treatment=treat,
                n_clusters_expected=proto["expected_units"], n_clusters_used=0, n_excluded=0,
                power="low",
                rel_improvement=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                statistical_win=False, practical_win=False, strongly_supported=False,
                practical_threshold=proto["practical"], strong_threshold=proto["strong"],
                min_independent_units=proto["min_units"], protocol_frozen=proto["frozen"],
                status="invalid_pairing",
                note="no common cohort after candidate requirements", updated_at=now))
            continue

        for cohort in cohorts:
            # Matching configurations must appear in both treatment arms.
            strata = sorted(
                {k[1] for k in rs if k[0] == cohort}
                & {k[1] for k in cs if k[0] == cohort}
            )
            cluster_vals: dict[str, list[float]] = {}
            excluded = 0
            expected_clusters = set()
            ref_ch = cand_ch = ""
            nh_used = "n/a"
            nh_equal = False
            for nh in strata:
                ref_clusters = {k[2] for k in rs if k[0] == cohort and k[1] == nh}
                cand_clusters = {k[2] for k in cs if k[0] == cohort and k[1] == nh}
                expected_clusters |= ref_clusters & cand_clusters
                for cl in sorted(ref_clusters & cand_clusters):
                    R, C = rs.get((cohort, nh, cl), []), cs.get((cohort, nh, cl), [])
                    # More than one record with a cluster/nuisance/cohort key
                    # is a duplicate execution, not an extra independent unit.
                    dup = duplicate_reason(R, C)
                    if dup is not None:
                        excluded += max(len(R), len(C))
                        excl_rows.append(dict(
                            comparison_id=cid, cohort=cohort, cluster=cl,
                            reason=dup,
                            reference_run_dirs=";".join(str(x.run_dir) for x in R),
                            candidate_run_dirs=";".join(str(x.run_dir) for x in C),
                            nuisance_hash=nh, updated_at=now))
                        continue
                    rr, cc = R[0], C[0]
                    if proto["frozen"]:
                        # Same-layout reruns of one unit (visible only in the
                        # pre-canonical scan) exclude the cluster: the retry
                        # must be a new cohort, never a second execution inside
                        # this one.
                        def _exec_key(r):
                            return tuple(_exec_key_norm(getattr(r, c, "na"))
                                         for c in ("cohort", "experiment", "graph_seed",
                                                   "noise_seed", "target", "seed", "layout"))
                        if _exec_key(rr) in DUP_EXEC_KEYS or _exec_key(cc) in DUP_EXEC_KEYS:
                            excluded += 1
                            excl_rows.append(dict(
                                comparison_id=cid, cohort=cohort, cluster=cl,
                                reason="invalid_pairing:duplicate_execution",
                                reference_run_id=rr.run_id, candidate_run_id=cc.run_id,
                                reference_run_dir=rr.run_dir, candidate_run_dir=cc.run_dir,
                                nuisance_hash=nh, updated_at=now))
                            continue
                    contract = pair_contract(
                        rr, cc, fold_w_policy(ctype, treat),
                        strict=proto["strict_contract"],
                    )
                    if not contract["pair_contract_valid"]:
                        excluded += 1
                        reasons = []
                        if not contract["split_hash_equal"]:
                            reasons.append("split_hash_mismatch_or_missing")
                        if contract["fold_w_cache_applicable"] and not contract["fold_w_cache_hash_equal"]:
                            reasons.append("fold_w_cache_hash_mismatch_or_missing")
                        excl_rows.append(dict(
                            comparison_id=cid, cohort=cohort, cluster=cl,
                            reason="invalid_pairing:" + ",".join(reasons),
                            reference_run_id=rr.run_id, candidate_run_id=cc.run_id,
                            reference_run_dir=rr.run_dir, candidate_run_dir=cc.run_dir,
                            reference_split_hash=getattr(rr, "split_hash", ""),
                            candidate_split_hash=getattr(cc, "split_hash", ""),
                            reference_fold_w_cache_hash=getattr(rr, "fold_w_cache_hash", ""),
                            candidate_fold_w_cache_hash=getattr(cc, "fold_w_cache_hash", ""),
                            nuisance_hash=nh, updated_at=now))
                        continue
                    rv, cv = float(rr.nmse), float(cc.nmse)
                    if not np.isfinite(rv) or not np.isfinite(cv) or rv == 0:
                        excluded += 1
                        excl_rows.append(dict(
                            comparison_id=cid, cohort=cohort, cluster=cl,
                            reason="invalid_metric:nonfinite_or_zero",
                            reference_run_id=rr.run_id, candidate_run_id=cc.run_id,
                            reference_run_dir=rr.run_dir, candidate_run_dir=cc.run_dir,
                            nuisance_hash=nh, updated_at=now))
                        continue
                    rel = (rv - cv) / rv
                    ref_violation = violation_value(rr)
                    cand_violation = violation_value(cc)
                    violation_comparable = bool(np.isfinite(ref_violation) and np.isfinite(cand_violation))
                    violation_not_worse = (
                        bool(cand_violation <= ref_violation + VIOLATION_ABS_TOL
                             + VIOLATION_REL_TOL * abs(ref_violation))
                        if violation_comparable else None
                    )
                    cluster_vals.setdefault(cl, []).append(rel)
                    ref_ch = str(rr.resolved_config_hash)
                    cand_ch = str(cc.resolved_config_hash)
                    nh_used = nh
                    nh_equal = True
                    pair_rows.append(dict(
                        comparison_id=cid, node_id=cid, scope_id=scope, root=root,
                        cohort=cohort,
                        dataset_id=(load_cfg(rr.run_dir).get("problem", {}) or {}).get("name"),
                        sem_type=(load_cfg(rr.run_dir).get("problem", {}) or {}).get("sem_type"),
                        target_id=(load_cfg(rr.run_dir).get("problem", {}) or {}).get("target"),
                        cluster=cl, graph_seed=cl.split("|")[0] if "|" in cl else cl,
                        noise_seed=cl.split("|")[1] if "|" in cl else "",
                        reference_arm=ref_arm, candidate_arm=cand_arm,
                        reference_run_id=rr.run_id, candidate_run_id=cc.run_id,
                        reference_run_dir=rr.run_dir, candidate_run_dir=cc.run_dir,
                        reference_config_hash=rr.resolved_config_hash,
                        candidate_config_hash=cc.resolved_config_hash,
                        reference_artifact_fingerprint=rr.artifact_fingerprint,
                        candidate_artifact_fingerprint=cc.artifact_fingerprint,
                        nuisance_hash=nh, nmse_ref=rv, nmse_cand=cv,
                        relative_improvement=rel, metric_validity="valid",
                        ce_prediction_violation_ref=ref_violation,
                        ce_prediction_violation_cand=cand_violation,
                        violation_comparable=violation_comparable,
                        violation_not_worse=violation_not_worse,
                        split_hash=getattr(rr, "split_hash", ""),
                        reference_split_hash=getattr(rr, "split_hash", ""),
                        candidate_split_hash=getattr(cc, "split_hash", ""),
                        split_hash_equal=contract["split_hash_equal"],
                        reference_fold_w_cache_hash=getattr(rr, "fold_w_cache_hash", ""),
                        candidate_fold_w_cache_hash=getattr(cc, "fold_w_cache_hash", ""),
                        fold_w_cache_hash_equal=contract["fold_w_cache_hash_equal"],
                        fold_w_cache_applicable=contract["fold_w_cache_applicable"],
                        pair_contract_valid=contract["pair_contract_valid"],
                        git_commit=git_commit, timestamp=now,
                    ))

            # Bootstrap only independent graph x noise (or target x seed)
            # clusters within this one execution cohort.
            d = np.asarray([np.mean(v) for v in cluster_vals.values()], dtype=float)
            n_clusters = len(d)
            lo, hi = boot_ci(d)
            rel_mean = float(np.mean(d)) if n_clusters else np.nan
            cohort_pair_rows = [x for x in pair_rows
                                if x.get("comparison_id") == cid and x.get("cohort") == cohort]
            comparable_violations = (
                [x for x in cohort_pair_rows if x.get("violation_comparable")]
                if treat in ("ce", "w", "graph") else []
            )
            violation_guard = (
                all(bool(x.get("violation_not_worse")) for x in comparable_violations)
                if comparable_violations else None
            )
            min_units = proto["min_units"]
            practical = proto["practical"]
            strong = proto["strong"]
            enough = n_clusters >= min_units
            stat_win = bool(enough and lo > 0)
            # Two-tier practical verdict (researcher decision 2026-09-24):
            #   supported          : point estimate >= threshold AND CI_lo > 0
            #   strongly_supported : CI_lo >= threshold
            # A positive verdict requires the violation to be VERIFIED not worse.
            # An unverifiable (missing) violation cannot support a claim.
            prac_win = bool(enough and rel_mean >= practical and violation_guard is True)
            strong_win = bool(enough and lo >= strong and violation_guard is True)
            power = "low" if not enough else "ok"

            def _present(x, field) -> bool:
                v = x.get(field, "")
                return bool(v) and str(v).strip().lower() not in ("", "nan", "none")

            split_receipts_present = bool(cohort_pair_rows) and all(
                _present(x, "reference_split_hash") and _present(x, "candidate_split_hash")
                for x in cohort_pair_rows)
            fold_w_receipts_present = bool(cohort_pair_rows) and all(
                (not x.get("fold_w_cache_applicable"))
                or (_present(x, "reference_fold_w_cache_hash")
                    and _present(x, "candidate_fold_w_cache_hash"))
                for x in cohort_pair_rows)
            pair_contract_all_valid = bool(cohort_pair_rows) and all(
                bool(x.get("pair_contract_valid")) for x in cohort_pair_rows)

            if n_clusters == 0:
                status = "invalid_pairing" if excluded or expected_clusters else "out_of_scope"
            elif violation_guard is False:
                status = "constraint_violation_worse"
            elif strong_win:
                status = "strongly_supported"
            elif prac_win:
                status = "supported"
            elif stat_win and violation_guard is not True:
                status = "constraint_violation_unverified"
            elif hi < 0:
                status = "contradicted"
            elif stat_win:
                status = "below_practical_threshold"
            elif lo > -EQUIV_MARGIN and hi < EQUIV_MARGIN:
                status = "equivalent_within_margin"
            else:
                status = "inconclusive"
            if power == "low" and status in ("supported", "strongly_supported", "contradicted",
                                             "below_practical_threshold", "equivalent_within_margin"):
                status = status + "_low_power"
            idx_rows.append(dict(
                comparison_id=cid, root=root, scope=scope, cohort=cohort,
                reference=ref_arm, candidate=cand_arm, comparison_type=ctype,
                treatment=treat,
                # For frozen protocols, expected units come from the registered
                # seed table, not the observed intersection. Otherwise a
                # missing arm/seed could shrink the denominator and look complete.
                n_clusters_expected=(proto["expected_units"] if proto["frozen"]
                                     else len(expected_clusters)),
                n_clusters_used=n_clusters, n_excluded=excluded, power=power,
                rel_improvement=rel_mean, ci_lo=lo, ci_hi=hi,
                statistical_win=stat_win, practical_win=prac_win,
                strongly_supported=strong_win, status=status,
                practical_threshold=practical, strong_threshold=strong,
                min_independent_units=min_units, protocol_frozen=proto["frozen"],
                violation_comparable=bool(comparable_violations),
                violation_not_worse=violation_guard,
                reference_config_hash=ref_ch, candidate_config_hash=cand_ch,
                nuisance_config_hash=nh_used, nuisance_hash_equal=nh_equal,
                split_hash_equal=bool(cohort_pair_rows) and all(
                    bool(x.get("split_hash_equal")) for x in cohort_pair_rows
                ),
                split_receipts_present=split_receipts_present,
                fold_w_cache_hash_equal=bool(cohort_pair_rows) and all(
                    bool(x.get("fold_w_cache_hash_equal")) for x in cohort_pair_rows
                ),
                fold_w_receipts_present=fold_w_receipts_present,
                pair_contract_valid=pair_contract_all_valid,
                note=(f"{ctype}; cohort+nuisance stratified; practical>={practical:.0%}"),
                updated_at=now))

    pd.DataFrame(pair_rows).to_csv(PAIRS, index=False)
    pd.DataFrame(idx_rows).to_csv(INDEX, index=False)
    pd.DataFrame(excl_rows).to_csv(EXCL, index=False)

    # ---- protocol registry export (auto-generated: the docs must not drift) ----
    reg_rows = [{
        "comparison_id": cid, "axis": root, "scope": scope,
        "reference": ref_arm, "candidate": cand_arm,
        "comparison_type": ctype, "treatment": treat,
        "candidate_requirement": "none" if require is None else "custom_predicate",
    } for cid, root, scope, ref_arm, cand_arm, ctype, treat, require in COMPARISONS]
    reg = pd.DataFrame(reg_rows)
    reg.to_csv(REGISTRY_CSV, index=False)
    try:
        REGISTRY_MD.parent.mkdir(parents=True, exist_ok=True)
        rl = ["# Evidence-tree comparison registry (AUTO-GENERATED)", "",
              "> Generated by `scripts/build_evidence_index.py` from its COMPARISONS table.",
              "> DO NOT EDIT BY HAND. This registry is the source of truth for the",
              "> evidence-tree axis / phase mapping. Regenerate with",
              "> `python scripts/build_evidence_index.py`.", "",
              "| comparison_id | axis | scope | reference | candidate | type | treatment |",
              "|---|---|---|---|---|---|---|"]
        for r in reg_rows:
            rl.append(f"| {r['comparison_id']} | {r['axis']} | {r['scope']} | "
                      f"{r['reference']} | {r['candidate']} | {r['comparison_type']} | {r['treatment']} |")
        rl.append("")
        REGISTRY_MD.write_text("\n".join(rl), encoding="utf-8")
    except Exception:
        pass
    print(f"wrote {REGISTRY_CSV} ({len(reg_rows)} comparisons)")
    print(f"wrote {REGISTRY_MD}")

    # ---- pooled direction (NON-STRICT): pool every cohort, signal only ----
    pooled_rows = []
    if pair_rows:
        pr = pd.DataFrame(pair_rows)
        for cid, g in pr.groupby("comparison_id"):
            vals = g.groupby("cluster")["relative_improvement"].mean().to_numpy(float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            lo, hi = boot_ci(vals)
            mean_rel = float(np.mean(vals))
            if np.isfinite(lo) and lo > 0:
                direction = "positive"
            elif np.isfinite(hi) and hi < 0:
                direction = "negative"
            else:
                direction = "flat"
            pooled_rows.append(dict(
                comparison_id=cid, pooled_units=len(vals),
                pooled_rel=mean_rel, pooled_ci_lo=lo, pooled_ci_hi=hi,
                pooled_direction=direction,
                pooled_practical=bool(np.isfinite(lo) and lo > 0 and mean_rel >= PRACTICAL),
                note="pooled across cohorts; non-strict, signal only",
                updated_at=now))
    pd.DataFrame(pooled_rows).to_csv(POOLED, index=False)
    print(f"wrote {POOLED} ({len(pooled_rows)} pooled comparisons)")
    print(f"wrote {PAIRS} ({len(pair_rows)} pairs)")
    print(f"wrote {INDEX} ({len(idx_rows)} comparisons)")
    print(f"wrote {EXCL} ({len(excl_rows)} exclusions)")
    idx = pd.DataFrame(idx_rows)
    print()
    print(idx[["comparison_id", "n_clusters_used", "n_excluded", "power",
               "rel_improvement", "ci_lo", "ci_hi", "status"]].to_string(index=False))


if __name__ == "__main__":
    main()

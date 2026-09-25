# Evidence Tree — execution & inference contract (single source of truth)

> **This document is the authoritative execution and inference contract for the
> evidence tree.** It supersedes the phase-plan document
> `docs/causal_predictor_staged_experiment_plan.md`, which is now a historical
> methodology / scope reference only.
>
> Machine-readable sources (never hand-maintain what can be exported):
> - `experiment_registry.yaml` — the frozen protocol registry (source of truth).
> - `scripts/experiment_registry.py` — its loader / validator.
> - `reports/evidence_registry.csv` and `docs/evidence_tree_registry.md` —
>   **AUTO-GENERATED** from `scripts/build_evidence_index.py` (comparison table).
> - `reports/cohort_registry.jsonl` — append-only cohort log.
>
> If this document and an auto-generated artifact disagree, the artifact wins.

---

## 1. What the evidence tree is

A closed loop that turns cluster runs into an auditable claim tree:

```
experiment_registry.yaml        (frozen hypotheses, arms, thresholds, seeds)
        │
submit_cohort.sh ──► cohort id (minted once) + cohort manifest + PBS job
        │
run on MetaCentrum (scripts/evidence_tree/*.sh)
        │
sync artifacts back  →  metacentrum_runs/<job>/…
        │
scan_progress_tree.py      → runs_metrics.csv (+ metric_validity, provenance)
build_evidence_index.py    → evidence_pairs.csv / evidence_index.csv /
                             evidence_exclusions.csv / evidence_pooled.csv /
                             evidence_registry.csv
build_evidence_nodes.py    → evidence_nodes.csv  (axis A1–A6 / B0 / G0 / F0)
render_evidence_tree.py    → progress_tree.md / evidence_tree_review.md
        │
G0 contract gate (g0_check.py)  →  blocks publication if violated
        │
dashboard (serve_evidence_dashboard.py, port 8770)
```

## 2. Protocol registry (source of truth)

`experiment_registry.yaml` freezes, **before any compute is submitted**:

| Field | Meaning |
|---|---|
| `reference_arm` / `candidate_arm` | the pair every hypothesis tests |
| `scope`, `comparison_type`, `treatment` | where and what the intervention is |
| `primary_metric`, `effect_metric` | `nmse`, `relative_improvement` |
| `practical_threshold`, `strong_threshold` | 5% (see §6) |
| `min_independent_units`, `unit_definition` | e.g. 20 units, `graph_x_noise` |
| `seed_table` + `seed_table_hash` | the exact seeds, hash-pinned |
| `split_contract`, `fold_w_contract` | `same` / `none` / `candidate` / `both` |
| `reserved_holdout_seed_table` (F0) | disjoint, hash-pinned holdout seeds |
| `diagnostic_records` | runs that may **never** back a strict claim |

`scripts/experiment_registry.py` loads and validates it (`load`, `validate`,
`effective_seed_table`, `seed_table_units`). **If this loader is missing, the
builder silently falls back to legacy defaults and every frozen protocol is
disabled** — so the loader is part of the contract, not an optional helper.

Seed-table hash = `sha256(json.dumps(table, sort_keys=True, separators=(",", ":")))`.

## 3. Cohort identity

A cohort is one immutable execution batch. Rules:

- a cohort id is minted **exactly once** by `scripts/evidence_tree/submit_cohort.sh`
  and recorded in the append-only `reports/cohort_registry.jsonl`;
- re-submitting the same id **must fail** (no local state proves a prior use);
- every run in one submission carries `problem.evidence_batch_id` = that id;
- the builder **never** merges runs from different cohorts (different PBS job or
  different `evidence_batch_id`) into one comparison.

`problem.evidence_node` / `evidence_arm` / `evidence_scope` label a run; they are
metadata and are stripped from the nuisance hash.

## 4. Pairing contract

For each candidate/reference comparison, pairing happens **within one cohort**,
stratified by nuisance configuration:

- **pairing unit**: synthetic = `graph_seed × noise_seed`; FRED = `target × seed`.
  Folds are never independent samples; reruns are replicates within a cluster.
- **nuisance hash** = resolved config minus the declared treatment keys minus
  metadata (`experiment`, `problem.evidence_*`).  Any uncontrolled difference
  (dropout, grad_clip, data fingerprint, …) breaks the pair instead of being
  averaged.
- **split / fold-W contract** per `experiment_registry.yaml`:
  - `same` → both arms must share the receipt;
  - `candidate` → only the candidate needs a receipt;
  - `none` → not applicable (end-to-end baselines, data/loss treatments).
  A frozen comparison (`***_contract: same`) is **BLOCKING**: a missing/different
  receipt excludes the pair (it appears in `evidence_exclusions.csv`).  Legacy
  comparisons are recorded but non-blocking.
- **exclusions are explicit**, never silently dropped: every excluded pair is
  written to `reports/evidence_exclusions.csv` with a machine-readable reason
  (`invalid_configuration:treatment_requirement`,
  `invalid_pairing:duplicate_within_cohort`, `invalid_metric:…`, …).

## 5. Metrics & validity

- Primary metric: **fold-normalized macro-NMSE** = mean over folds of
  `test_err_fold / score_normalizer_fold`.
- `metric_validity`:
  - `valid` — `cv_fold_metrics.csv:test_nmse` present (preferred);
  - `legacy_unverified` — only `cv_errors.yaml` (schema unverifiable);
  - `invalid_double_normalized` — the retired double-normalization (kept, never used).
- Strict inference reads **only `valid`** rows.
- Run identity: deduplicated by a logical key
  (`experiment │ graph_seed │ noise_seed │ target │ seed │ batch │ config_hash │ job`),
  preferring the multirun artifact that carries `cv_fold_metrics.csv`.

## 6. Inference rule

For a candidate vs its reference, on the declared pairing units:

| Verdict | Condition |
|---|---|
| `supported` | point ≥ `practical_threshold` **and** bootstrap CI lower bound > 0 |
| `strongly_supported` | CI lower bound ≥ `strong_threshold` |
| `contradicted` | CI upper bound < 0 |
| `inconclusive` | CI crosses 0 |
| `equivalent_within_margin` | CI inside ±`EQUIV_MARGIN` (5%) |
| `constraint_violation_worse` | candidate violates a recorded constraint more than the reference (constraint comparisons only) |
| `*_low_power` | independent units < `min_independent_units` (suffix, not a verdict) |

Additional rules:

- **Significance is computed per cohort**, over clusters, never per fold.
- **Relative improvement**, not an absolute NMSE delta. The historical absolute
  `Δ ≥ 0.02` threshold is **retired** (it does not transfer across ER/FRED/CoDiet).
- A `pooled` column (`evidence_pooled.csv`) is **descriptive only** — pooled
  across cohorts for direction, explicitly non-strict; it is never a verdict.
- Claims are **scope-specific**: `supported(ER)`, `contradicted(SF)`,
  `inconclusive(FRED)`, …

## 7. G0 — fair-comparison contract gates

`scripts/evidence_tree/g0_check.py` audits `reports/evidence_index.csv` against
the registry and exits non-zero when a **frozen** comparison violates any of:

| Gate | Checks |
|---|---|
| G0.1 | metric-schema columns exist |
| G0.2 | artifact complete (expected == used units, no missing arm) |
| G0.3 | every pair shares the **exact split receipt** |
| G0.4 | every pair shares the **nuisance configuration** |
| G0.5 | every W-applicable pair shares the **fold-W cache receipt** |
| G0.6 | a **frozen-selection receipt** exists |
| — | no duplicate/excluded pairing inside the frozen cohort |

Legacy comparisons are reported but **not** gated (diagnostic records only).

## 8. Frozen selection & F0 locked holdout

- `make_frozen_selection_receipt.py` writes a **validation-only** schema-v2 receipt
  (protocol version, pre-declared selection rule, candidate set, selected config
  hash, validation report hash, code commit, reserved holdout seeds + hash,
  registry hash).  It never selects a winner and never reads a holdout metric.
- `f0_guard.py` lets F0 run **only** when: the receipt is schema-v2, hash-intact,
  produced from a validation-only artifact, matches the current registry, every
  seed lies inside the reserved holdout table, and the runner is a plain
  executable path.
- F0 must reject any run outside the reserved seed table.

## 9. Hard gates vs scope limits

| Kind | Item | Status source |
|---|---|---|
| **Hard gate** | `G0.1` metric schema valid | `g0_check.py` |
| **Hard gate** | `G0.2` complete artifact | `g0_check.py` |
| **Hard gate** | `G0.3` exact split hash | `g0_check.py` |
| **Hard gate** | `G0.5` fold-W cache hash | `g0_check.py` |
| **Hard gate** | `G0.6` frozen-selection receipt | `make_frozen_selection_receipt.py` |
| **Hard gate** | `F0` locked holdout | `f0_guard.py` |
| Scope limit | external population-level oracle | not implemented |
| Scope limit | CLR/ILR, Gaussian-copula, mixed-variable statistic (CoDiet) | not implemented |
| Scope limit | count / zero-inflated SEM | not implemented |
| Scope limit | edge-trace ESS / split-R-hat for BD chains | not implemented |

Scope limits restrict what a result may *claim*; hard gates block *publication*
of a frozen result.

## 10. Dashboard publication conditions

`serve_evidence_dashboard.py` (port 8770) is the publication surface. It shows a
node's three dimensions — **implementation / evidence / claim** — and must:

1. read the **strict** tables (`evidence_index.csv`, `evidence_nodes.csv`) plus
   the descriptive `evidence_pooled.csv`, and never present pooled as a verdict;
2. display the **cohort** and **k = independent units** next to every verdict;
3. show `metric_integrity.md` status; if the metric integrity gate fails, the
   dashboard shows **`invalid`** and emits **no** supported/contradicted verdicts;
4. show hard gates as `←` (open) and scope limits as `⊘`;
5. rather than a single percentage, report per-node `implementation / evidence /
   claim`, and gate root completion only on nodes marked required;
6. refresh via `SKIP_SYNC=0 bash scripts/refresh_progress_tree.sh`, which runs the
   whole chain in §1.

## 11. Axis map — registry-driven, do not hand-maintain

The node/axis tree and the phase → axis correspondence live in:

- **`experiment_registry.yaml`** — hypotheses (`H1`, `F0`, …) with their arms;
- **`docs/evidence_tree_registry.md`** — the auto-generated comparison table
  (regenerate with `python scripts/build_evidence_index.py`);
- **`reports/evidence_nodes.csv`** — the per-axis node registry
  (`scripts/build_evidence_nodes.py`).

Axis meanings (structure only; membership is registry-driven):

```
A1 optimizer mechanics        A4 graph estimation & causal prior
A2 constraint embedding       A5 model capacity & training budget
A3 CI/CE statistic            A6 data representation / time / robustness
B0 baseline rail              G0 fair-comparison contract
C0 CoDiet pre-audit           F0 frozen selection + locked holdout
```

> Do not copy the comparison list into prose. When a comparison is added or
> removed, regenerate `docs/evidence_tree_registry.md` and the docs follow.

## 12. Legacy documents

- `docs/causal_predictor_staged_experiment_plan.md` — **historical** methodology
  and scope reference; its "absolute Δ ≥ 0.02" rule is retired. Only the shared
  helpers it documents (`scripts/causal_predictor_plan/_common.sh`,
  `09_ce_preaudit.sh`, `audit_nonlinear_ci.py`) are still used by the evidence
  tree.
- `docs/optimal_method_funnel_plan.md` — strategy note (search-space funnel);
  not an execution contract.

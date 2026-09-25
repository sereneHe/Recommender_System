# A8.5 lag-channel preflight (diagnostic-only, pre-registered)

Status: **pre-registered, not yet run**.  This document fixes the design and the
decision gates *before* any number is produced, so the preflight cannot be
re-interpreted after the fact.

The 23998597 A8.5 smoke is **not** a "NN vs XGB on a dynamic SEM" comparison:
it used a static feature matrix (`X_t` only) and, because the synthetic frame
has no `site`/`gender` column, the CV silently fell back to shuffled `KFold`
(`recommender_utils.py:193`, manifest `cv_kind: kfold`).  It measured shuffled
static regression risk.  This preflight fixes the input and the evaluation
target first; only then can a model difference be interpreted.

## Scope

* Diagnostic only.  Results are recorded as `diagnostic`, never as strict
  evidence, and cannot back an A8 claim.
* No CE/W arms.  CE correctness is uninterpretable until a lag-aware oracle is
  in place (see the oracle section).
* No changes to A8.5 run *as a smoke*; the preflight is a new, separate batch.

## Input tiers

| Tier | Feature set given to the model | Purpose |
|---|---|---|
| `t0` | `X_t` | reproduce the current smoke baseline |
| `t1` | `X_t` + `X_{t-1}` | measure the value of the available lag information |
| `t2` | `P_t` + `P_{t-1}` for the target's **true parents** | upper reference for a lag-aware tabular model |

The lag-aware oracle is evaluated on the same `P_t + P_{t-1}` block and is the
achievable lower bound for `t1`/`t2`.

## Evaluation

* Outer CV: **expanding-window / rolling-origin** (`solver.cv_strategy=time_series`
  plus `solver.cv_time_test_size`), never shuffled KFold.
* Inner validation: **last training time block**
  (`solver.validation_split_strategy=time`), never a random split.
* Rows must be time-sorted; the splitter already refuses unsorted input.
* Aggregation and any interval use `graph x noise` units.  Folds are **not**
  independent replicates.

## Models

* `xgb100` — the fixed 100-round XGB reference.
* `nn0` — the current MLP (no CE, no W).
* Both models get identical inputs, splits and budgets per tier.  Any budget
  tuning must happen inside each training window and be recorded; the current
  fixed-100 vs default-MLP contrast is *not* a pure architecture contrast.

## Oracle

* `synthetic_utils.structural_conditional_mean(..., require_lag=True)` must
  reproduce the generator mean on `P_t + P_{t-1}` (verified: `max|Δ| = 0`).
* Called with a static feature matrix it must **refuse** (`ValueError`), not
  silently return `tanh(z)`.  Confirmed by test.
* An oracle number may only be reported next to the tier that produced it.

## Data

* Reuse the smoke/pilot units already inspected (graph 42-44 x noise 101-102).
  They are flagged `strict_inference_allowed: false` in
  `experiment_registry.yaml`; the preflight is allowed to use them because it
  makes no claim.
* The A8 confirmation table is graph 52-61 x noise 101-102 and must not appear
  in the preflight.

## Pre-registered decision gates

Let `excess(t) = NMSE(model, tier t) - NMSE(lag_oracle)`.

1. **No lag value.**  If `t1` does not materially reduce the excess risk of
   *both* XGB and the MLP relative to `t0`, the lag signal exploitable at
   `rho = 0.5` is insufficient.  **Stop A8.5; no confirm cohort.**
2. **Lag helps, architecture does not.**  If `t1` reduces the excess for both
   models but the NN-XGB difference is not directionally stable across the
   `graph x noise` units, the conclusion is "lag information helps, the current
   MLP has no extra architectural advantage".  **Do not run CE/W; no confirm.**
3. **Promising.**  Only if `t1` has real lag value *and* the NN-XGB difference
   is directionally stable across units does a new 20-unit confirm cohort
   (graph 52-61 x noise 101-102, frozen) become justified.

"Materially reduce" and "directionally stable" are fixed as: mean excess
reduction `>= 0.05` relative to the `t0` excess, and the same sign in `>= 60%`
of the independent units.

## Burn-in

* Deferred for this preflight; it is a strict-confirm prerequisite, not a
  preflight requirement.  Until it is frozen the DGP may only be called a
  *dynamic SEM*, not strictly stationary sampling.
* Determined by `scripts/evidence_tree/burnin_mixing_diagnostic.py`
  (non-training; mean / variance / lag-1 ACF + a first-`K` decay profile), then
  written into the problem config.
* Current measurement at `rho = 0.5` (graph 42-44 x noise 101-102,
  `n_samples = 20000`): `burn_in = 0`, `all_converged = true` — the transient
  is below tolerance.  But the sequential generator **never assigns row 0**, so
  the series contains one all-zero row (`n_all_zero_rows = 1`); that is an
  impossible observation, not a mixing transient.  Strict confirm must set
  `burn_in >= 1` (or fix the initialisation) before claiming stationarity.

## Out of scope (tracked separately)

* A8.2 compositional: an independent pilot.  A8.5 says nothing about it.
* A8.1 smooth / A8.4 periodic: boundary tests, no presumption the NN wins.

# HC predictor staged server experiment plan

```text
STATUS: Historical methodology and scope reference; not the current execution contract.

Current execution and audit contract:
- scripts/evidence_tree/
- scripts/test/
- docs/evidence_tree.md

Current inference rule:
relative improvement, cohort-separated pairing, predeclared contracts,
minimum independent-unit gate, and locked holdout.
The historical absolute macro-NMSE delta threshold is retired.
```

> Superseded by [`docs/evidence_tree.md`](evidence_tree.md), which is the single
> execution and inference contract. Keep this file for its methodology notes and
> its phase arm definitions (S0–S4, I0–I5, B1/B2) and for the list of things that
> are intentionally **not** implemented. Of the scripts in
> `scripts/causal_predictor_plan/`, only the shared helper `_common.sh`,
> `09_ce_preaudit.sh` and `audit_nonlinear_ci.py` are still used by the evidence
> tree; the phase scripts `01`–`08` are historical.

---

This document was the execution contract for the eight scripts in
`scripts/causal_predictor_plan/`.  They were deliberately staged: complete the
review gate for a phase before submitting the next one.  A lower cross-validation
number is development evidence, not a license to continue every branch.

## Scope and decision rule

The primary development metric is macro normalized MSE, with each fold
normalized by the mean-only baseline fitted on that fold's training rows.  Raw
MSE, MAE, per-target NMSE, constraint counts, and failure/runtime information
remain required secondary outputs.

Use the same seed list, targets, time splits, training-update budget, and input
information set within a comparison.  Decide only from development folds.  The
final locked holdout is not part of these scripts and must not be inspected
until the configuration is frozen.

Before the first phase, freeze the smallest effect worth pursuing.  The initial
recommendation is an absolute macro-NMSE reduction of `0.02`.  For a candidate
against its reference:

- advance when the paired validation/development estimate improves by at least
  `0.02`, its paired interval excludes zero, and no prespecified target group
  worsens by more than `0.05`;
- if the interval crosses zero, run only the two best candidates with more
  seeds;
- stop the branch after two consecutive rounds without a qualifying change;
- never use the final held-out period to choose a parameter, a constraint type,
  a graph threshold, or a BD support threshold.

## What this change adds

The launch scripts use only implemented settings.  The code now also provides:

- `solver.cv_strategy=time_series`: expanding-window `TimeSeriesSplit`;
- `solver.validation_split_strategy=time`: last inner-training block for
  HC-CE early stopping;
- `solver.dag_fit_scope=inner_train`: learn an HC-CE DAG without reading the
  inner validation block;
- `solver.prediction_loss=mse|huber` and `solver.huber_delta` while reporting
  MSE/NMSE unchanged;
- Industry options `+problem.feature_lag`, `+problem.add_time_trend`, and
  `+problem.regime_break_date`;
- Gaussian, variance-matched Laplace, and variance-matched Student-t linear
  synthetic SEM noise;
- `HC_CE_BD_CHAINS`: equal-mass aggregation of multiple approximate
  birth-death chains.

The following are **not** implemented and are intentionally not faked by the
scripts:

- a true final independent FRED holdout runner;
- fold-W caching shared between separately launched method arms;
- an external population-level oracle for observational data (the phase-1
  Gaussian synthetic runner does save held-out `Y`, analytic
  `E[Y|X]`, and `Y_hat` statistics);
- CLR/ILR, Gaussian-copula, or a valid mixed continuous/discrete constraint
  statistic for CoDiet;
- a count or zero-inflated structural equation generator;
- edge-trace ESS or split-R-hat for birth-death chains.

Therefore, phase-1 through phase-8 output directories are development results.
They must not be described as final test results or converged Bayesian posterior
results.

## Server submission

The PBS launcher now accepts `EXPERIMENT_SCRIPT`; its default remains the
historical HC stability script.  A typical submission is:

```sh
qsub -v EXPERIMENT_SCRIPT=scripts/causal_predictor_plan/01_synthetic_constraint_screen.sh \
  cluster_computing/run_metacentrum.pbs
```

The phase scripts execute their seed list sequentially in one PBS job.  The
default is `SEEDS="42 43 44"`.  To make jobs smaller, submit individual seeds:

```sh
qsub -v EXPERIMENT_SCRIPT=scripts/causal_predictor_plan/02_industry_hc_tuning.sh,SEEDS=42,STAGE=optimizer \
  cluster_computing/run_metacentrum.pbs
```

Use a local command preview before submission:

```sh
DRY_RUN=1 bash scripts/causal_predictor_plan/03_industry_constraint_ablation.sh
```

Every script accepts trailing Hydra overrides, for example:

```sh
SEEDS="42 43 44" TIME_LIMIT=600 \
  bash scripts/causal_predictor_plan/02_industry_hc_tuning.sh
```

Do not put comma-separated seed lists in Hydra overrides; the scripts loop over
seeds specifically to avoid a hidden Cartesian sweep.

## Phase 1 — Gaussian synthetic constraint screen

Script: `01_synthetic_constraint_screen.sh`

Runs Gaussian ER and SF data with:

| Arm | Graph source | Constraint |
|---|---|---|
| S0 | none | no W, no CE |
| S1 | true synthetic W | W constraint |
| S2 | true synthetic W | independent d-separation CE only |
| S3 | true synthetic W | target-terminal negative control: no eligible dependent CE |
| S4 | fold-estimated DAG (W penalty off) | independent d-separation CE only |

All phase-1 arms use `hc_predictor_ce`, the same outer/inner budget, feature
set, validation behavior, full-batch setting, and paired model seed. S1 differs
from S0 only by enabling the W penalty. Its supplied raw synthetic SEM
coefficients are converted to the model-standardized coordinates before the W
term is evaluated; the raw matrix is retained separately for the oracle
structural-equation diagnostic. S2 and S4 explicitly set
`ci_add_collider_conditional_dependence=false`; their
`dependent_constraints` count must therefore be zero. The target is required
to be an endpoint rather than merely a conditioning/collider variable.

For every enabled CI/W constraint, the runner saves:

- `constraint_metadata.csv`: exact relation, variables, conditioning set,
  source, margin/tolerance, and posterior support if applicable;
- `constraint_stat_audit.csv`: held-out outer-test statistic for observed
  `Y`, analytic synthetic oracle `E[Y|X]` from the generator's retained true
  structural matrix, and `Y_hat`;
- `w_constraint_audit.csv`: each coordinate and L2 norm of the W moment
  residual for observed `Y`, oracle, and `Y_hat`; the true-W arm also records the
  raw target structural-equation residual. The latter is not the same as the
  legacy standardized first-moment penalty, so S1 is an audit of that existing
  penalty rather than proof of a full structural-equation constraint.

These outer-test rows are independent of each fold's training data but are
development diagnostics, not a final locked test. Do not use them to tune
per-run hyperparameters.

After each completed phase-1 launcher, `results/phase1/` contains the
following summary artifacts (and the PBS launcher copies that directory back):

- `phase1_run_summary.csv`: one newest completed row for every
  `ER/SF × graph_seed × noise_seed × arm`; it includes the fold-mean NMSE,
  actual independent/dependent/W constraint counts, the three required CI
  statistics for observed `Y`, oracle `E[Y|X]`, and `Y_hat`, plus W diagnostics;
- `phase1_summary.csv`: the corresponding `graph type × arm` aggregate,
  including paired arm-minus-S0 NMSE and its descriptive bootstrap interval;
- `phase1_missing_pairs.csv`: an explicit grid-completeness check.  A missing
  CI audit means that the estimated/true graph generated no eligible target
  endpoint constraint; it must never be interpreted as a zero violation;
- `phase1_pair_consistency.csv`: for every graph/noise realization, verifies
  the shared training budget, arm-matched model seed, byte-identical `W_true`,
  and matching S2/S4 independent-CI generation settings before the paired
  result is called comparable;
- `phase1_summary.md`: a compact human-readable version of the grouped table.

The summary also records the actual training budget (`n_outer`, `n_inner`,
batch size, seed settings) and the constraint-generation settings, and marks
each arm's interpretation boundary.  It therefore checks that comparisons are
paired and that an accidental setting mismatch is visible rather than silently
averaged.  `S4` is **estimated-DAG CE only**: it does not enable a re-estimated
W penalty, so it isolates graph-estimation error for the CI screen rather than
testing an estimated-W constraint.  S1's legacy W term remains a
standardized mean-moment penalty, not a full structural-equation constraint.

When phase-1 has been split across PBS jobs, combine their copied multirun
roots only after all required arms have finished, for example:

```sh
python scripts/causal_predictor_plan/summarize_phase1.py \
  --scan-root /storage/brno2/home/hexiaoyu/Recommender_Pavel/metacentrum_runs/JOB_A/multirun \
  --scan-root /storage/brno2/home/hexiaoyu/Recommender_Pavel/metacentrum_runs/JOB_B/multirun \
  --output-dir results/phase1/final
```

`S3` is disabled by default. It uses an absolute conditional-residual-
covariance dependence statistic, because a positive signed covariance is not a
direction-free dependence claim. In the current generator `X9` is always the
last (terminal) node, while the implemented dependent rule is only the parent
pair in `x -> collider <- y`. With `ci_target_constraint_role=endpoint`, X9
therefore cannot occur in a dependent pair. S3 is intentionally retained as a
negative control and its summary must report zero dependent constraints; it
does **not** test whether dependent constraints work. Do not use it to support
a dependent-constraint claim. A separate non-terminal-target, collider-aware
synthetic design with an analytic conditional-mean oracle is required first.
It can be run only to verify this expected inactive behavior:

```sh
INCLUDE_DEPENDENT=1 bash scripts/causal_predictor_plan/01_synthetic_constraint_screen.sh
```

The default `GRAPH_SEEDS="42 43 44"` and `NOISE_SEEDS="101"` is a smoke
screen only. The requested formal design is 20 independent graph/structural
coefficient seeds and five innovation-noise seeds, for example:

```sh
GRAPH_SEEDS="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20" \
NOISE_SEEDS="101 102 103 104 105" \
bash scripts/causal_predictor_plan/01_synthetic_constraint_screen.sh
```

That formal grid is 800 Hydra runs before opt-in S3, so do not submit it as
one sequential PBS job. Submit one graph seed (or a small fixed graph-seed
batch) per job with the same five `NOISE_SEEDS`, then aggregate only after all
paired arms finish.

Compare arms as paired differences within the same graph/noise realization.
Advance only if independent constraints are feasible in both observed and
oracle audits, do not make prediction collapse, and the held-out constraint
set is nonempty and interpretable. If observed `Y` is feasible but the oracle
is not, stop applying that CE constraint class to a predictor. If only `Y_hat`
fails, investigate optimization/capacity before advancing. S4 must be read as
an estimated-graph development arm because its DAG is fit inside each
outer-training fold, and phase 1 additionally sets `dag_fit_scope=inner_train`
so the graph does not read the inner validation block used for checkpointing.

## Phase 2 — Industry HC tuning

Script: `02_industry_hc_tuning.sh`

The script uses expanding-window monthly FRED validation with four origins and
a 12-month horizon by default.  It runs one stage at a time:

```sh
STAGE=optimizer bash scripts/causal_predictor_plan/02_industry_hc_tuning.sh
STAGE=capacity SELECTED_LR=0.03 SELECTED_WEIGHT_DECAY=0.01 \
  bash scripts/causal_predictor_plan/02_industry_hc_tuning.sh
STAGE=budget SELECTED_LR=0.03 SELECTED_WEIGHT_DECAY=0.01 \
  SELECTED_HIDDEN_DIM=32 SELECTED_DEPTH=2 \
  bash scripts/causal_predictor_plan/02_industry_hc_tuning.sh
```

`optimizer` tests six chosen learning-rate/weight-decay pairs. `capacity`
tests five network sizes. `budget` changes `n_outer/n_inner` only after those
are frozen.  Because those values also change dual-update cadence, describe
this as a training-schedule comparison rather than a pure architecture sweep.

After selecting one setting, run:

```sh
STAGE=confirm WINNER_LR=0.03 WINNER_WEIGHT_DECAY=0.01 \
WINNER_HIDDEN_DIM=32 WINNER_DEPTH=2 WINNER_N_OUTER=10 WINNER_N_INNER=100 \
bash scripts/causal_predictor_plan/02_industry_hc_tuning.sh
```

## Phase 3 — Industry W/CE/batch ablation

Script: `03_industry_constraint_ablation.sh`

It runs six arms with fixed phase-2 settings:

| Arm | W | independent CE | balanced batch |
|---|---|---|---|
| I0 | no | no | no |
| I1 | yes | no | no |
| I2 | no | yes | no |
| I3 | no | yes | yes |
| I4 | yes | yes | no |
| I5 | yes | yes | yes |

For continuous FRED this explicitly selects
`ci_penalty_kind=conditional_expectation`, which in the current code is a
linear residual-covariance statistic.  Shielded-collider dependence is off.
`CE_BACKEND=alm_pbm` is the default; setting `CE_BACKEND=alm_all` or `pbm_all`
creates an end-to-end algorithm comparison, not a pure optimizer comparison.

Each method launch currently recomputes the fold graph.  Treat this as a
development ablation until a fold-W cache has been implemented and verified.

## Phase 4 — time, trend, regime, and robust loss

Script: `04_industry_time_robustness.sh`

Run stages serially with the winning phase-2 predictor:

```sh
STAGE=lag bash scripts/causal_predictor_plan/04_industry_time_robustness.sh
STAGE=trend SELECTED_LAG=3 bash scripts/causal_predictor_plan/04_industry_time_robustness.sh
STAGE=loss SELECTED_LAG=3 bash scripts/causal_predictor_plan/04_industry_time_robustness.sh
STAGE=regime SELECTED_LAG=3 REGIME_BREAK_DATE=2020-03-01 \
  bash scripts/causal_predictor_plan/04_industry_time_robustness.sh
```

`feature_lag=k` replaces every predictor at time `t` by its observed value at
`t-k`; the target remains at `t`.  It does not create a causal claim about
publication timing. Before reporting a forecast result, verify externally that
each lagged feature was actually available at the forecast origin.

The regime date is an author-specified historical cut; it must be fixed before
examining results.  Do not search crisis dates against the final holdout.

## Phase 5 — Mark and Mark-CC comparison

Script: `05_mark_markcc_comparison.sh`

This runs Mark, Mark-CC, HC-W-ALM, and HC-CE independent constraints across all
16 FRED targets with the same time ordering and seeds.  It is an end-to-end
comparison because Mark and HC use different graph-learning/prediction paths.
Do not label a difference as “only the optimizer changed.”

## Phase 6 — CoDiet screen

Script: `06_codiet_type_screen.sh`

```sh
STAGE=baseline bash scripts/causal_predictor_plan/06_codiet_type_screen.sh
STAGE=discrete_cmi bash scripts/causal_predictor_plan/06_codiet_type_screen.sh
```

This phase deliberately runs only raw predictive baselines and the current
quantile-binned discrete-CMI arm.  Before a copula, CLR/ILR, or mixed-variable
constraint experiment, create and review a feature-type table containing:

```text
column, semantic_type, storage_dtype, cardinality, zero_fraction,
is_compositional, transformation, constraint_statistic
```

Do not integer-encode nominal categories and interpret their quantile bins as
distances.  Do not apply CLR/ILR to microbiome labels or embedding coordinates.

## Phase 7 — continuous non-Gaussian synthetic stress test

Script: `07_synthetic_noise_robustness.sh`

The script compares Gaussian, Laplace, and Student-t noise, with matched noise
variance.  For each distribution it runs true-W, true-W independent CE, and
estimated-W independent CE arms.  Change the Student-t degrees of freedom with
`STUDENT_T_DF`, which must exceed two.

Poisson, negative-binomial, and zero-inflated SEMs are intentionally absent:
a linear additive SEM would not be a valid substitute. Add a separate count
generator before opening that branch.

## Phase 8 — approximate posterior-stable BD constraints

Script: `08_bd_posterior_constraints.sh`

This phase is restricted to continuous Industry/FRED data. It launches both:

| Arm | `HC_CE_BD_MCMC` | Graph constraints |
|---|---:|---|
| B1 | 0 | single thresholded DAG |
| B2 | 1 | equal-mass aggregation of multiple birth-death chains |

Default sampler controls are:

```text
HC_CE_BD_CHAINS=4
HC_CE_BD_STEPS=2000
HC_CE_BD_BURN_IN=500
HC_CE_BD_SEED_BASE=20260916
HC_CE_BD_INDEPENDENCE_SUPPORT=0.8
HC_CE_BD_MAX_CONFLICT_SUPPORT=0.2
```

The implementation saves aggregate per-chain diagnostics and aggregates each
chain with equal posterior mass. It does **not** retain traces, so it cannot
yet establish ESS or split-R-hat. Report this method as approximate
posterior-stable constraints, not a verified converged posterior.

## Final confirmation, after phase 8

Freeze only the methods that passed their prior gates. Then create a separate
final-holdout runner with:

1. a calendar cutoff fixed before its first execution;
2. transformations and DAG fit only on data before that cutoff;
3. one refit for each frozen configuration;
4. no further parameter or threshold changes afterward.

The final table must include raw MSE, fold-normalized MSE, MAE, target-level
results, constraint counts/violations, runtime, and failure rate. Any result
that improves constraint values but not prediction error is a constraint-fidelity
result, not a predictive-performance result.

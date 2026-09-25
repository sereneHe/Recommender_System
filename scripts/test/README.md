# Evidence-tree test batches

These wrappers cover the current **runnable** Evidence Tree, with fixed cohort
IDs and data-family guards.  They are intentionally separate PBS jobs: a
single 24-hour sequential driver would mix MILP tails with FRED runs and make
an interrupted cohort difficult to audit.

## Estimated duration of each Evidence Tree script

The counts below assume the current defaults (`GRAPH_SEEDS=42 43 44`,
`NOISE_SEEDS=101`, `n_outer=10`, `n_inner=100`).  They are planning ranges,
not a scheduler guarantee; queue time is excluded.  The synthetic estimates
use the recent 7--13 minute jobs as a reference, while A4 is dominated by the
Gurobi time limit.

| Evidence Tree script | Workload at default seeds | Approx. compute time |
|---|---:|---:|
| `A1_optimizer.sh` | 5 arms × 3 = 15 ER runs | 20–60 min |
| `A2_constraint_embedding.sh` | 7 arms × 3 = 21 ER runs | 30–90 min |
| `A3_ci_statistic.sh` | 5 arms × 3 = 15 ER runs | 20–60 min |
| `A4_graph_prior.sh` | 11 arms × 3 = 33 ER MILP runs | 7–10 h with `MIP_TIME_LIMIT=300`; potentially 25 h+ at 1800 s |
| `A5_capacity_budget.sh` | 5 arms × 3 = 15 ER runs | 20–60 min |
| `A6_data_time_robustness.sh` | 5 arms × 3 per FRED problem | 0.5–3 h per country (use the pair shards) |
| `B0_baselines.sh` | 8 arms × 3 = 24 ER runs | 30–90 min |
| `G0_contract_audit.sh` | no training | <5 min |
| `F0_locked_holdout.sh` | guarded preflight; requires receipt + reviewed runner | blocked until freeze |
| `batch1_synthetic_light.sh` | A1+A2+A3+A5+B0 = 90 ER runs | about 2–6 h |
| `batch2_milp_fred.sh` | A4 (33) + A6 default one country (15) | about 8–12 h with MIP limit 300; not a safe 12 h job at 1800 |

`A6` is intentionally split into eight two-country scripts because its actual
runtime depends strongly on the country and DAG fit.  Each pair expands to
`5 × 3 × 2 = 30` Hydra jobs and is expected to remain below one 12-hour
allocation after the first shard is measured.  If the first shard exceeds the
estimate, split each pair into single-country jobs before submitting the rest.

| Script | Scope | Default commands | Expected wall time |
|---|---|---:|---:|
| `01_synthetic_core.sh` | A1, A2, A3, A5, B0, G0; synthetic ER | 90 model runs | ~2–6 h |
| `02_graph_prior_er.sh` | A4; ER ExDBN/MILP | 33 MILP runs | ~7–10 h with `MIP_TIME_LIMIT=300` |
| `03`–`10_fred_*.sh` | A6; two FRED countries per shard | 30 model runs each | ~2–8 h each, monitor first shard |
| `99_refresh_evidence_tree.sh` | scan/index/tree only | no training | minutes |
| `11_unverified_scope_audits.sh` | nonlinear/SF/noise/CoDiet extensions | data-family-specific | ~1–8 h plus CoDiet |

The FRED shards cover these 16 problems exactly once: AUT, BEL, DEU, ESP, EST,
FIN, FRA, GRC, IRL, ITA, LTU, LUX, NLD, PRT, SVK, SVN.  The
`11_unverified_scope_audits.sh` wave covers the runnable extensions:
nonlinear ER/SF CI-statistic audit, independent SF and nonlinear-ER screens,
supported Laplace/Student-t noise, and the CoDiet baseline/CMI pre-audit.  It
explicitly reports classical bootstrap stability selection and a
non-Gaussian-moment ExDBN prior as blocked rather than relabelling the existing
BD approximation.

The full current-tree workload is therefore roughly **25–80 compute hours**
(synthetic core + A4 + the eight FRED shards), before queue time.  The shards
are designed to be submitted independently or in parallel; they are not a
promise that the whole study finishes in one 12-hour PBS job.

## Submission order

Submit `01` and `02` separately.  Submit `03`–`10` as separate jobs (or at
most two concurrently if the allocation permits).  After all jobs are
complete, run `99_refresh_evidence_tree.sh` once.

```bash
qsub -l walltime=12:00:00 -v EXPERIMENT_SCRIPT=scripts/test/01_synthetic_core.sh cluster_computing/run_metacentrum.pbs
qsub -l walltime=12:00:00 -v EXPERIMENT_SCRIPT=scripts/test/02_graph_prior_er.sh,MIP_TIME_LIMIT=300 cluster_computing/run_metacentrum.pbs
qsub -l walltime=12:00:00 -v EXPERIMENT_SCRIPT=scripts/test/03_fred_aut_bel.sh cluster_computing/run_metacentrum.pbs
# repeat for 04--10
qsub -l walltime=12:00:00 -v EXPERIMENT_SCRIPT=scripts/test/11_unverified_scope_audits.sh cluster_computing/run_metacentrum.pbs
qsub -l walltime=01:00:00 -v EXPERIMENT_SCRIPT=scripts/test/99_refresh_evidence_tree.sh cluster_computing/run_metacentrum.pbs
```

For a larger synthetic pilot, override `GRAPH_SEEDS` and `NOISE_SEEDS` on all
related jobs.  Do not mix different seed lists under one `et_*_v2` batch ID.

## What the refresh can and cannot conclude

The refresh generates the evidence index and updates the tree; it does not
turn all completed runs into a positive result.  A method conclusion is valid
only when:

1. the same split, nuisance settings, and fold-W cache are shared by paired
   arms;
2. configuration is selected from validation only, followed by one untouched
   locked holdout;
3. the comparison uses cluster units (graph × noise × target for synthetic,
   target × model seed for FRED), not individual folds;
4. the pre-declared effect/CI rule passes and constraint violation is not
   worse; and
5. the result is stated separately for ER, SF/nonlinear ER, FRED, and CoDiet.

New runs now write `cv_split_manifest.yaml` and `fold_w_manifest.yaml`; the
evidence builder rejects old or mismatched pairs.  `G0.3/G0.5` therefore become
passable only after rerunning the paired arms.  `G0.6` is created explicitly
with `scripts/evidence_tree/make_frozen_selection_receipt.py`, and `F0` remains
blocked until a reviewed independent holdout runner is supplied.  Therefore
“all shards finished” means the tree is updated and auditable, not
automatically that HC-CE has proved superior or that a universally valid method
was found.

## Task 12–19（A8 / scope extensions / CoDiet，独立作业）

先前 task1–3 只覆盖 A1–A6 的部分节点。以下任务补齐可运行的扩展，**每个机制单独一个 PBS 作业**，不把 A8 的全部命令塞进一个 allocation：

| 脚本 | 覆盖 | 说明 |
|---|---|---|
| `12_a8_smooth.sh` | A8.1 smooth nonlinear ER | true-DAG + generator oracle（`synthetic_generator_oracle`） |
| `13_a8_compositional.sh` | A8.2 deep compositional ER | 同上 |
| `14_a8_highdim.sh` | A8.3 high-dim smooth interaction | 同上（p=50） |
| `15_a8_periodic.sh` | A8.4 periodic / multiscale | 同上 |
| `16_a8_temporal.sh` | A8.5 temporal smooth SEM | **oracle 审计被显式禁用**；仅预测结果，不得称“约束数学已验证” |
| `17_nonlinear_sf_audit.sh` | 非线性 ER/SF CI 统计量审计 | 从 task11 拆出，只做统计量质量，不训练 KCI/HSIC |
| `18_noise_robustness.sh` | Laplace / Student-t 噪声鲁棒性 | 仅支持的噪声族 |
| `19_codiet_cmi.sh` | CoDiet baseline + discrete-CMI | 独立报告，不并入 ER/FRED 均值 |

顺序：task1–3 → 12–19 → refresh Evidence Tree + G0 审核 → H1 → 冻结配置 → F0。

### 当前 task1 的恢复模式

`23944989` 已完成 A1 与 A2，但在 A3 covariance 的旧源码快照上中断。修复后的
`task1_synthetic.sh` 默认 `TASK1_MODE=resume`，只运行缺失的 A3、A5、B0、A4 与
G0，并使用新的 `v3` cohort ID；它不会重跑已完成的 A1/A2。`A3` 会先执行
`test_ce_tolerances.py`，若 covariance + window/HAC standard-error 支持缺失则在
任何训练开始前停止。

### 仍是缺口、加 `.sh` 解决不了的（必须实现方法或改节点定义）

1. **A3.discrete_CMI**：节点当前要求 synthetic ER，但现有 CoDiet CMI 不能充当它的证据。需新增“离散/量化 synthetic ER + CMI”实验，或把该节点改为 A7 的类型匹配 CMI。
2. **A6.standardization**：尚无严格 raw / standardised / Gaussianized 配对消融；task2–3 不能证明标准化本身有效。
3. **A3.KCI_HSIC / A4.stability_selection / A4.non_gaussian_moments**：算法未实现，不能用现有日志填成已验证。

### H1 / F0 不混入 `task*.sh`

- **H1** 必须经 cohort 注册器（`scripts/evidence_tree/submit_cohort.sh --hyp H1 --remote`）投递**新的 20-unit cohort**，保证同 split、同训练预算、同 fold-W。失败/中断的 attempt 一律 `retire` 为 `diagnostic_ineligible`，新 attempt 重跑完整 20 对。
- **F0** 只有在 H1、比较合同与 frozen-selection receipt 全部通过后，才允许触发一次 locked holdout。
- **`99_refresh_evidence_tree.sh`** 应在所有任务结果同步回项目后运行；它更新树的显示，但不会把未覆盖节点误标为已证。

## Mark-CC 树数公平对照（修正）

旧 `mark_cc_10`(n_outer=1) 与 `mark_cc_100`(n_outer=10) **同时改变了树数和 n_outer**，
因此**不能**用 `mark_cc_100 − mark_cc_10` 证明树数效应，也不能证明 Mark-CC 优势。

`B0_baselines.sh` 已改为 **固定 `CC_N_OUTER`（默认 = `N_OUTER`），只变 `n_estimators` 10 → 100**。
配套索引对照：`B0.cc_tree_count.er`（`EV:B0.mark_cc_10` vs `EV:B0.mark_cc_100`）。

必须**重跑一个新的 B0 cohort** 才能得到公平证据；旧 cohort 仅作诊断，不参与树数/Mark-CC 结论。
（注意 `mark_with_cc` 总树数 = `n_outer × n_estimators`；若要做“总预算 10 vs 100”，应设 `CC_N_OUTER=1`。）

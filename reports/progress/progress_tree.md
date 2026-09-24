# Evidence Tree（搜索空间 v3；严格配对）

> 六个实验轴：**A1 优化器与后端 / A2 约束嵌入 / A3 条件独立统计量 / A4 图估计与先验 / A5 模型容量与预算 / A6 数据、时间与鲁棒损失**。
> S0 是注册表 + fallback，不宣称科学最优性。状态分为：实现、证据、scope-specific 主张。
> 证据来自 `metacentrum_runs/`（严格配对，仅 valid 指标）；`mark_cc_100` 缺失时锚点用 `mark_100` 兜底。
> 数据族和变量类型合同见 `reports/progress/evidence_tree_data_contract.md`；B0 负责 canonical 基线，A2 不重复运行 W-only/CE-only/W+CE。
> 脚本与节点逐项对照见 `reports/progress/evidence_tree_script_audit.md`。

```text
S0  实验搜索空间与可运行锚点  [注册✓ · 证据missing · -]  runs u1010/v525 · pairs 0  (代码)
├─ B0  B0 端到端基线轨（可比较 fallback）  [注册✓ · 证据missing · -]  runs u295/v188 · pairs 0  (实验)
│  ├─ B0.mark_100  B0.1 Mark-100（100 棵树）  [实现✓ · 证据complete · inconclusive(ER)]  runs u3/v3 · pairs 3  (实验)
│  │     ◇ cohort: 23919696.pbs-m1.meta=inconclusive · [diagnostic_only: 不可用于严格推断]
│  ├─ B0.mark_cc_total100  B0.2 Mark-CC-100（总预算 100 棵，待产物核验）  [实现✓ · 证据missing · open_hard_gate]  runs u27/v24 · pairs 0  (代码)
│  ├─ B0.hc_nn_no_constraint  B0.3 HC-NN 无约束基线  [实现✓ · 证据missing · implemented_unverified]  runs u70/v59 · pairs 0  (实验)
│  ├─ B0.hc_w_only  B0.4 HC-NN + W（基线臂）  [实现✓ · 证据missing · implemented_unverified]  runs u47/v39 · pairs 0  (实验)
│  ├─ B0.hc_ce_only  B0.5 HC-NN + CE（基线臂）  [实现✓ · 证据missing · implemented_unverified]  runs u151/v72 · pairs 0  (实验)
│  └─ B0.hc_w_ce  B0.6 HC-NN + W + CE（基线臂）  [实现✓ · 证据missing · implemented_unverified]  runs u27/v18 · pairs 0  (实验)
├─ A1  A1 优化器与约束求解后端  [注册✓ · 证据missing · -]  runs u1010/v525 · pairs 0  (代码)
│  ├─ A1.ALM_ALL  A1.1 CE 全部使用 ALM  [实现✓ · 证据complete · equivalent_within_margin_low_power(ER)]  runs u14/v14 · pairs 3  (代码)
│  │     ◇ cohort: et_a1_er_v2=equivalent_within_margin_low_power · [diagnostic_only: 不可用于严格推断]
│  ├─ A1.PBM_ALL  A1.2 CE 全部使用确定性 PBM  [实现✓ · 证据complete · constraint_violation_worse(ER)]  runs u3/v3 · pairs 3  (代码)
│  │     ◇ cohort: et_a1_er_v2=constraint_violation_worse · [diagnostic_only: 不可用于严格推断]
│  ├─ A1.HYBRID_ALM_PBM  A1.3 Hybrid：独立约束→ALM、依赖约束→PBM  [实现✗ · 证据missing · inactive_under_independence_only]  runs u0/v0 · pairs 0  (实验)
│  │     📚 随机/不精确增广拉格朗日 (Li et al. PMLR v130)
│  ├─ A1.STOCHASTIC_PBM  A1.4 CE 全部使用 SPBM  [实现✓ · 证据complete · constraint_violation_worse(ER)]  runs u13/v3 · pairs 3  (代码)
│  │     📚 Adam-SPBM 随机惩罚-障碍 (spbm/Adam_SPBM.pdf)
│  │     ◇ cohort: et_a1_er_v2=constraint_violation_worse · [diagnostic_only: 不可用于严格推断]
│  └─ A1.SCO_LAYER  A1.5 SPBM + SCO/proximal 训练层  [实现✓ · 证据complete · contradicted_low_power(ER)]  runs u35/v32 · pairs 3  (代码)
│        📚 SCO/proximal 层与 ALM/PBM 不等价，需独立比较
│        ◇ cohort: et_a1_er_v2=contradicted_low_power · [diagnostic_only: 不可用于严格推断]
├─ A2  A2 约束嵌入与组合方式  [注册✓ · 证据missing · -]  runs u1010/v525 · pairs 0  (代码)
│  ├─ A2.W_target_residual  A2.1 目标相关残差 W（区别于基线 W）  [实现✓ · 证据complete · inconclusive(ER)]  runs u6/v6 · pairs 3  (代码)
│  │     ◇ cohort: et_a2_er_v2=inconclusive · [diagnostic_only: 不可用于严格推断]
│  ├─ A2.W_mask  A2.2 预测相关 mask W  [实现✓ · 证据missing · implemented_unverified]  runs u0/v0 · pairs 0  (代码)
│  ├─ A2.W_bias_calibration  A2.3 仅 W 偏置校准（不加 W 惩罚）  [实现✓ · 证据missing · implemented_unverified]  runs u0/v0 · pairs 0  (代码)
│  ├─ A2.balanced_batch  A2.4 平衡批次（CE 采样修饰）  [实现✓ · 证据missing · implemented_unverified ×crossed]  runs u12/v0 · pairs 0  (代码)
│  │     ▤ n=0; mean_indep=nan; mean_dep=nan; mean_signflip=nan
│  ├─ A2.pruning  A2.5 冗余约束剪枝（CE 修饰）  [实现✓ · 证据missing · implemented_unverified ×crossed]  runs u5/v0 · pairs 0  (代码)
│  │     ▤ n=0; mean_indep=nan; mean_dep=nan; mean_signflip=nan
│  └─ A2.W_adds_to_CE  A2.6 W 在 CE 之上是否提供新增信息  [实现✓ · 证据missing · implemented_unverified]  runs u27/v18 · pairs 0  (实验)
├─ A3  A3 条件独立统计量与不确定性  [注册✓ · 证据missing · -]  runs u1010/v525 · pairs 0  (代码)
│  ├─ A3.covariance  A3.1 残差协方差 CE 统计量  [实现✗ · 证据missing · out_of_scope(ER) ×crossed]  runs u0/v0 · pairs 0  (实验)
│  ├─ A3.partial_correlation  A3.2 标准化偏相关 CE 统计量  [实现✓ · 证据missing · - ×crossed]  runs u179/v176 · pairs 0  (实验)
│  │  ├─ A3.pcorr.linear_ER  A3.2a 线性 ER 数据  [实现✓ · 证据complete · inconclusive(ER) ×crossed]  runs u179/v176 · pairs 3  (实验)
│  │  │     ◇ cohort: 23896397.pbs-m1.meta=inconclusive; 23896397.pbs-m1.meta=inconclusive; 23908664.pbs-m1.meta=inconclusive; 23908664.pbs-m1.meta=inconclusive; 23918638.pbs-m1.meta=inconclusive; repair_er_20260923T1=inconclusive · [diagnostic_only: 不可用于严格推断]
│  │  ├─ A3.pcorr.nonlinear_ER  A3.2b 非线性 ER 数据  [实现✓ · 证据complete · contradicted_low_power(ER); inconclusive(ER) ×crossed]  runs u60/v60 · pairs 3  (实验)
│  │  │     ◇ cohort: 23908664.pbs-m1.meta=contradicted_low_power; 23908664.pbs-m1.meta=contradicted_low_power; 23918638.pbs-m1.meta=inconclusive; repair_er_20260923T1=inconclusive · [diagnostic_only: 不可用于严格推断]
│  │  └─ A3.pcorr.SF  A3.2c SF 数据  [实现✓ · 证据complete · contradicted_low_power(SF) ×crossed]  runs u26/v24 · pairs 3  (实验)
│  │        ◇ cohort: 23908664.pbs-m1.meta=contradicted_low_power; 23908664.pbs-m1.meta=contradicted_low_power · [diagnostic_only: 不可用于严格推断]
│  ├─ A3.discrete_CMI  A3.3 离散/分箱 CMI  [实现✓ · 证据missing · - ×crossed]  runs u97/v69 · pairs 0  (实验)
│  ├─ A3.quantile_residual  A3.4 分位数残差化  [实现✓ · 证据complete · constraint_violation_worse(ER) ×crossed]  runs u0/v0 · pairs 3  (代码)
│  │     ◇ cohort: 23896397.pbs-m1.meta=constraint_violation_worse; 23896397.pbs-m1.meta=constraint_violation_worse; 23908664.pbs-m1.meta=constraint_violation_worse; 23908664.pbs-m1.meta=constraint_violation_worse; 23918638.pbs-m1.meta=constraint_violation_worse; repair_er_20260923T1=constraint_violation_worse · [diagnostic_only: 不可用于严格推断]
│  │     ▤ n=0; mean_indep=nan; mean_dep=nan; mean_signflip=nan
│  ├─ A3.window_SE  A3.5 跨时间窗 SE  [实现✓ · 证据missing · implemented_unverified ×crossed]  runs u122/v119 · pairs 0  (代码)
│  │     ▤ n=119; mean_indep=4.8; mean_dep=0.0; mean_signflip=0.30
│  ├─ A3.HAC_SE  A3.6 HAC/Newey-West SE  [实现✓ · 证据missing · implemented_unverified ×crossed]  runs u0/v0 · pairs 0  (代码)
│  │     ▤ n=0; mean_indep=nan; mean_dep=nan; mean_signflip=nan
│  ├─ A3.sign_flip_filter  A3.7 有符号依赖的 sign-flip 过滤  [实现✓ · 证据missing · implemented_unverified ×crossed]  runs u0/v0 · pairs 0  (代码)
│  └─ A3.KCI_HSIC  A3.8 KCI/HSIC 非线性 CI（待接入训练）  [假设 · 证据missing · literature_hypothesis ×crossed]  runs u0/v0 · pairs 0  (文献)
│        📚 零偏相关依赖线性/高斯；非线性需核方法 KCI/HSIC (Zhang et al. arXiv:1202.3775)
├─ A4  A4 图估计与因果先验  [注册✓ · 证据missing · -]  runs u1010/v525 · pairs 0  (代码)
│  ├─ A4.edge_penalty  A4.1 边数惩罚  [实现✓ · 证据missing · implemented_unverified ×crossed]  runs u0/v0 · pairs 0  (代码)
│  ├─ A4.parents_limit  A4.2 最大父节点数限制  [实现✓ · 证据missing · implemented_unverified ×crossed]  runs u0/v0 · pairs 0  (代码)
│  ├─ A4.clique_cap  A4.3 Clique cuts/大小上限  [实现✓ · 证据missing · implemented_unverified ×crossed]  runs u0/v0 · pairs 0  (代码)
│  ├─ A4.MIP_time_gap  A4.4 MIP 时间限制与 gap  [实现✓ · 证据missing · implemented_unverified ×crossed]  runs u25/v0 · pairs 0  (代码)
│  ├─ A4.stability_selection  A4.5 Bootstrap 稳定性选择（待实现）  [假设 · 证据missing · literature_hypothesis ×crossed]  runs u0/v0 · pairs 0  (文献)
│  │     📚 子抽样结构选择概率可控制错误选择 (Meinshausen & Bühlmann 2010)
│  ├─ A4.non_gaussian_moments  A4.6 非高斯高阶矩先验（待实现）  [假设 · 证据missing · literature_hypothesis ×crossed]  runs u0/v0 · pairs 0  (文献)
│  │     📚 非高斯扰动可识别方向 (LiNGAM, Shimizu et al. 2006)
│  ├─ A4.graph_quality  A4.7 估计图质量（F1/SHD/边数）  [实现✓ · 证据missing · implemented_unverified ×crossed]  runs u145/v117 · pairs 0  (代码)
│  ├─ A4.constraint_quality  A4.8 生成约束质量（独立/依赖数量）  [实现✓ · 证据missing · implemented_unverified ×crossed]  runs u270/v176 · pairs 0  (代码)
│  └─ A4.solver_stability  A4.9 求解稳定性（状态/gap/时间）  [实现✓ · 证据missing · implemented_unverified ×crossed]  runs u270/v176 · pairs 0  (代码)
├─ A5  A5 模型容量与训练预算  [注册✓ · 证据missing · -]  runs u1010/v525 · pairs 0  (代码)
│  ├─ A5.tree_count  A5.1 Boosting 树预算：10 → 100  [实现✓ · 证据complete · inconclusive(ER) ×crossed]  runs u45/v39 · pairs 3  (实验)
│  │     ◇ cohort: 23919696.pbs-m1.meta=inconclusive · [diagnostic_only: 不可用于严格推断]
│  ├─ A5.hidden_depth  A5.2 NN 隐藏层宽度与深度  [实现✓ · 证据missing · implemented_unverified ×crossed]  runs u295/v188 · pairs 0  (代码)
│  ├─ A5.lr_wd  A5.3 NN 学习率与权重衰减  [实现✓ · 证据missing · implemented_unverified ×crossed]  runs u340/v227 · pairs 0  (代码)
│  └─ A5.n_outer_inner  A5.4 外层/内层训练预算  [实现✓ · 证据missing · implemented_unverified ×crossed]  runs u322/v212 · pairs 0  (代码)
├─ A6  A6 数据表示、时间结构与鲁棒损失  [注册✓ · 证据missing · -]  runs u1010/v525 · pairs 0  (代码)
│  ├─ A6.standardization  A6.1 数据尺度：raw / 标准化 / Gaussianization  [实现✓ · 证据missing · - ×crossed]  runs u408/v162 · pairs 0  (代码)
│  ├─ A6.lag  A6.2 预测变量滞后  [实现✓ · 证据complete · inconclusive(FRED) ×crossed]  runs u30/v30 · pairs 30  (代码)
│  │     ◇ cohort: et_a6_fred_v2=inconclusive · [diagnostic_only: 不可用于严格推断]
│  ├─ A6.trend  A6.3 时间趋势项  [实现✓ · 证据complete · inconclusive(FRED) ×crossed]  runs u26/v25 · pairs 25  (代码)
│  │     ◇ cohort: et_a6_fred_v2=inconclusive · [diagnostic_only: 不可用于严格推断]
│  ├─ A6.regime  A6.4 Regime 断点处理  [实现✓ · 证据complete · inconclusive(FRED) ×crossed]  runs u24/v24 · pairs 24  (代码)
│  │     ◇ cohort: et_a6_fred_v2=inconclusive · [diagnostic_only: 不可用于严格推断]
│  └─ A6.huber  A6.5 Huber 预测损失  [实现✓ · 证据complete · inconclusive(FRED) ×crossed]  runs u24/v24 · pairs 24  (代码)
│        ◇ cohort: et_a6_fred_v2=inconclusive · [diagnostic_only: 不可用于严格推断]
├─ A7  A7 CoDiet 类型匹配约束  [实现✓ · 证据missing · -]  runs u22/v16 · pairs 0  (代码)
│  └─ A7.codiet_cmi  A7.1 CoDiet 离散 CMI 相对 W-only  [实现✓ · 证据missing · implemented_unverified ×crossed]  runs u22/v16 · pairs 0  (实验)
├─ H1  H1 CE-only 相对 NN（合成 ER 确认）  [注册✓ · 证据missing · open]  runs u3/v3 · pairs 0  (实验)
│  └─ H1.ce_vs_nn  H1.1 CE-only vs 匹配 NN（20 graph×noise）  [实现✓ · 证据missing · open]  runs u3/v3 · pairs 0  (实验)
├─ A8  A8 NN 有利的数据机制与规模  [注册✓ · 证据missing · -]  runs u571/v347 · pairs 0  (实验)
│  ├─ A8.smooth_er  A8.1 平滑非线性 ER（tanh/sin/softplus）  [实现✓ · 证据missing · implemented_unverified]  runs u0/v0 · pairs 0  (实验)
│  ├─ A8.compositional_er  A8.2 深层组合函数 ER  [实现✓ · 证据missing · implemented_unverified]  runs u0/v0 · pairs 0  (实验)
│  ├─ A8.highdim  A8.3 高维稠密平滑交互  [实现✓ · 证据missing · implemented_unverified]  runs u0/v0 · pairs 0  (实验)
│  ├─ A8.periodic  A8.4 平滑周期/多尺度函数  [实现✓ · 证据missing · implemented_unverified]  runs u0/v0 · pairs 0  (实验)
│  └─ A8.temporal  A8.5 平滑时间序列 SEM  [实现✓ · 证据missing · unverified]  runs u0/v0 · pairs 0  (实验)
├─ G0  G0 公平比较与可追溯性合同  [注册✓ · 证据missing · -]  runs u1010/v525 · pairs 0  (代码)
│  ├─ G0.1_metric_schema_valid  G0.1 指标表结构有效  [实现✓ · 证据missing · checker_implemented]  runs u525/v525 · pairs 0  (代码)
│  ├─ G0.2_complete_artifact  G0.2 运行产物完整  [实现✓ · 证据missing · checker_implemented]  runs u422/v422 · pairs 0  (代码)
│  ├─ G0.3_exact_split_hash  G0.3 精确 split hash  [实现✓ · 证据missing · open]  runs u0/v0 · pairs 0  (代码)
│  ├─ G0.4_nuisance_config_hash  G0.4 非处理参数 hash  [实现✓ · 证据missing · open]  runs u0/v0 · pairs 0  (代码)
│  ├─ G0.5_fold_W_cache_hash  G0.5 每折 W-cache hash  [实现✓ · 证据missing · open]  runs u0/v0 · pairs 0  (代码)
│  └─ G0.6_frozen_selection_receipt  G0.6 冻结选择凭证  [实现✓ · 证据missing · open]  runs u0/v0 · pairs 0  (代码)
└─ F0  F0 冻结配置与最终 locked holdout  [实现✗ · 证据missing · open]  runs u0/v0 · pairs 0  (代码)
```

## 状态总表

| node | axis | source | crossed | 实现 | 证据 | 主张 | unique | valid | pairs |
|---|---|---|---|---|---|---|---|---:|---:|---:|
| S0 | S0 | 代码 | False | 注册✓ | missing | - | 1010 | 525 | 0 |
| B0 | B0 | 实验 | False | 注册✓ | missing | - | 295 | 188 | 0 |
| B0.mark_100 | B0 | 实验 | False | 实现✓ | complete | inconclusive(ER) | 3 | 3 | 3 |
| B0.mark_cc_total100 | B0 | 代码 | False | 实现✓ | missing | open_hard_gate | 27 | 24 | 0 |
| B0.hc_nn_no_constraint | B0 | 实验 | False | 实现✓ | missing | implemented_unverified | 70 | 59 | 0 |
| B0.hc_w_only | B0 | 实验 | False | 实现✓ | missing | implemented_unverified | 47 | 39 | 0 |
| B0.hc_ce_only | B0 | 实验 | False | 实现✓ | missing | implemented_unverified | 151 | 72 | 0 |
| B0.hc_w_ce | B0 | 实验 | False | 实现✓ | missing | implemented_unverified | 27 | 18 | 0 |
| A1 | A1 | 代码 | False | 注册✓ | missing | - | 1010 | 525 | 0 |
| A1.ALM_ALL | A1 | 代码 | False | 实现✓ | complete | equivalent_within_margin_low_power(ER) | 14 | 14 | 3 |
| A1.PBM_ALL | A1 | 代码 | False | 实现✓ | complete | constraint_violation_worse(ER) | 3 | 3 | 3 |
| A1.HYBRID_ALM_PBM | A1 | 实验 | False | 实现✗ | missing | inactive_under_independence_only | 0 | 0 | 0 |
| A1.STOCHASTIC_PBM | A1 | 代码 | False | 实现✓ | complete | constraint_violation_worse(ER) | 13 | 3 | 3 |
| A1.SCO_LAYER | A1 | 代码 | False | 实现✓ | complete | contradicted_low_power(ER) | 35 | 32 | 3 |
| A2 | A2 | 代码 | False | 注册✓ | missing | - | 1010 | 525 | 0 |
| A2.W_target_residual | A2 | 代码 | False | 实现✓ | complete | inconclusive(ER) | 6 | 6 | 3 |
| A2.W_mask | A2 | 代码 | False | 实现✓ | missing | implemented_unverified | 0 | 0 | 0 |
| A2.W_bias_calibration | A2 | 代码 | False | 实现✓ | missing | implemented_unverified | 0 | 0 | 0 |
| A2.balanced_batch | A2 | 代码 | True | 实现✓ | missing | implemented_unverified | 12 | 0 | 0 |
| A2.pruning | A2 | 代码 | True | 实现✓ | missing | implemented_unverified | 5 | 0 | 0 |
| A2.W_adds_to_CE | A2 | 实验 | False | 实现✓ | missing | implemented_unverified | 27 | 18 | 0 |
| A3 | A3 | 代码 | False | 注册✓ | missing | - | 1010 | 525 | 0 |
| A3.covariance | A3 | 实验 | True | 实现✗ | missing | out_of_scope(ER) | 0 | 0 | 0 |
| A3.partial_correlation | A3 | 实验 | True | 实现✓ | missing | - | 179 | 176 | 0 |
| A3.pcorr.linear_ER | A3 | 实验 | True | 实现✓ | complete | inconclusive(ER) | 179 | 176 | 3 |
| A3.pcorr.nonlinear_ER | A3 | 实验 | True | 实现✓ | complete | contradicted_low_power(ER); inconclusive(ER) | 60 | 60 | 3 |
| A3.pcorr.SF | A3 | 实验 | True | 实现✓ | complete | contradicted_low_power(SF) | 26 | 24 | 3 |
| A3.discrete_CMI | A3 | 实验 | True | 实现✓ | missing | - | 97 | 69 | 0 |
| A3.quantile_residual | A3 | 代码 | True | 实现✓ | complete | constraint_violation_worse(ER) | 0 | 0 | 3 |
| A3.window_SE | A3 | 代码 | True | 实现✓ | missing | implemented_unverified | 122 | 119 | 0 |
| A3.HAC_SE | A3 | 代码 | True | 实现✓ | missing | implemented_unverified | 0 | 0 | 0 |
| A3.sign_flip_filter | A3 | 代码 | True | 实现✓ | missing | implemented_unverified | 0 | 0 | 0 |
| A3.KCI_HSIC | A3 | 文献 | True | 假设 | missing | literature_hypothesis | 0 | 0 | 0 |
| A4 | A4 | 代码 | False | 注册✓ | missing | - | 1010 | 525 | 0 |
| A4.edge_penalty | A4 | 代码 | True | 实现✓ | missing | implemented_unverified | 0 | 0 | 0 |
| A4.parents_limit | A4 | 代码 | True | 实现✓ | missing | implemented_unverified | 0 | 0 | 0 |
| A4.clique_cap | A4 | 代码 | True | 实现✓ | missing | implemented_unverified | 0 | 0 | 0 |
| A4.MIP_time_gap | A4 | 代码 | True | 实现✓ | missing | implemented_unverified | 25 | 0 | 0 |
| A4.stability_selection | A4 | 文献 | True | 假设 | missing | literature_hypothesis | 0 | 0 | 0 |
| A4.non_gaussian_moments | A4 | 文献 | True | 假设 | missing | literature_hypothesis | 0 | 0 | 0 |
| A4.graph_quality | A4 | 代码 | True | 实现✓ | missing | implemented_unverified | 145 | 117 | 0 |
| A4.constraint_quality | A4 | 代码 | True | 实现✓ | missing | implemented_unverified | 270 | 176 | 0 |
| A4.solver_stability | A4 | 代码 | True | 实现✓ | missing | implemented_unverified | 270 | 176 | 0 |
| A5 | A5 | 代码 | False | 注册✓ | missing | - | 1010 | 525 | 0 |
| A5.tree_count | A5 | 实验 | True | 实现✓ | complete | inconclusive(ER) | 45 | 39 | 3 |
| A5.hidden_depth | A5 | 代码 | True | 实现✓ | missing | implemented_unverified | 295 | 188 | 0 |
| A5.lr_wd | A5 | 代码 | True | 实现✓ | missing | implemented_unverified | 340 | 227 | 0 |
| A5.n_outer_inner | A5 | 代码 | True | 实现✓ | missing | implemented_unverified | 322 | 212 | 0 |
| A6 | A6 | 代码 | False | 注册✓ | missing | - | 1010 | 525 | 0 |
| A6.standardization | A6 | 代码 | True | 实现✓ | missing | - | 408 | 162 | 0 |
| A6.lag | A6 | 代码 | True | 实现✓ | complete | inconclusive(FRED) | 30 | 30 | 30 |
| A6.trend | A6 | 代码 | True | 实现✓ | complete | inconclusive(FRED) | 26 | 25 | 25 |
| A6.regime | A6 | 代码 | True | 实现✓ | complete | inconclusive(FRED) | 24 | 24 | 24 |
| A6.huber | A6 | 代码 | True | 实现✓ | complete | inconclusive(FRED) | 24 | 24 | 24 |
| A7 | A7 | 代码 | False | 实现✓ | missing | - | 22 | 16 | 0 |
| A7.codiet_cmi | A7 | 实验 | True | 实现✓ | missing | implemented_unverified | 22 | 16 | 0 |
| H1 | H1 | 实验 | False | 注册✓ | missing | open | 3 | 3 | 0 |
| H1.ce_vs_nn | H1 | 实验 | False | 实现✓ | missing | open | 3 | 3 | 0 |
| A8 | A8 | 实验 | False | 注册✓ | missing | - | 571 | 347 | 0 |
| A8.smooth_er | A8 | 实验 | False | 实现✓ | missing | implemented_unverified | 0 | 0 | 0 |
| A8.compositional_er | A8 | 实验 | False | 实现✓ | missing | implemented_unverified | 0 | 0 | 0 |
| A8.highdim | A8 | 实验 | False | 实现✓ | missing | implemented_unverified | 0 | 0 | 0 |
| A8.periodic | A8 | 实验 | False | 实现✓ | missing | implemented_unverified | 0 | 0 | 0 |
| A8.temporal | A8 | 实验 | False | 实现✓ | missing | unverified | 0 | 0 | 0 |
| G0 | G0 | 代码 | False | 注册✓ | missing | - | 1010 | 525 | 0 |
| G0.1_metric_schema_valid | G0 | 代码 | False | 实现✓ | missing | checker_implemented | 525 | 525 | 0 |
| G0.2_complete_artifact | G0 | 代码 | False | 实现✓ | missing | checker_implemented | 422 | 422 | 0 |
| G0.3_exact_split_hash | G0 | 代码 | False | 实现✓ | missing | open | 0 | 0 | 0 |
| G0.4_nuisance_config_hash | G0 | 代码 | False | 实现✓ | missing | open | 0 | 0 | 0 |
| G0.5_fold_W_cache_hash | G0 | 代码 | False | 实现✓ | missing | open | 0 | 0 | 0 |
| G0.6_frozen_selection_receipt | G0 | 代码 | False | 实现✓ | missing | open | 0 | 0 | 0 |
| F0 | F0 | 代码 | False | 实现✗ | missing | open | 0 | 0 | 0 |
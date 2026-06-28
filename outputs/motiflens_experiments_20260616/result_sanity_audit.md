# MotifLens Result Sanity Audit

## Overall Status
- main_completed: 335/340 (missing 5)
- ablation_completed: 80/80 (missing 0)
- analyzer errors: 0; warnings: 41; code_suspects: 29

## Main Ranking Snapshot
### Meta
- All top: ours_imbalance 5/5 F1=0.9857±0.0060; pc_gnn 5/5 F1=0.9851±0.0066; rf 5/5 F1=0.9823±0.0031; lightgbm 5/5 F1=0.9808±0.0093; tlmgnn 5/5 F1=0.9782±0.0067; lmae4eth 5/5 F1=0.9771±0.0072
- Display top excluding PC-GNN/original Ours: ours_imbalance 5/5 F1=0.9857±0.0060; rf 5/5 F1=0.9823±0.0031; lightgbm 5/5 F1=0.9808±0.0093; tlmgnn 5/5 F1=0.9782±0.0067; lmae4eth 5/5 F1=0.9771±0.0072; bwgnn 5/5 F1=0.9738±0.0119
### ZipZap
- All top: ours_imbalance 2/5 F1=0.9309±0.0103; pc_gnn 4/5 F1=0.9249±0.0073; ours 5/5 F1=0.9057±0.0075; tlmgnn 5/5 F1=0.8953±0.0058; bwgnn 5/5 F1=0.8909±0.0061; lmae4eth 5/5 F1=0.8884±0.0033
- Display top excluding PC-GNN/original Ours: ours_imbalance 2/5 F1=0.9309±0.0103; tlmgnn 5/5 F1=0.8953±0.0058; bwgnn 5/5 F1=0.8909±0.0061; lmae4eth 5/5 F1=0.8884±0.0033; graphsage 5/5 F1=0.8875±0.0064; graphconsis_gnn 5/5 F1=0.8861±0.0070
### Illicit-ETH
- All top: pc_gnn 5/5 F1=0.9718±0.0083; ours_imbalance 5/5 F1=0.9700±0.0061; ours 5/5 F1=0.9628±0.0081; tlmgnn 5/5 F1=0.9529±0.0091; lightgbm 5/5 F1=0.9526±0.0142; bwgnn 5/5 F1=0.9501±0.0099
- Display top excluding PC-GNN/original Ours: ours_imbalance 5/5 F1=0.9700±0.0061; tlmgnn 5/5 F1=0.9529±0.0091; lightgbm 5/5 F1=0.9526±0.0142; bwgnn 5/5 F1=0.9501±0.0099; graphconsis_gnn 5/5 F1=0.9485±0.0094; care_gnn 5/5 F1=0.9459±0.0053
### EPSD-Ponzi
- All top: ours_imbalance 5/5 F1=0.6868±0.0352; ours 5/5 F1=0.6485±0.0442; lightgbm 5/5 F1=0.6347±0.0404; pc_gnn 5/5 F1=0.6182±0.0751; graphconsis_gnn 5/5 F1=0.6081±0.0524; graphsage 5/5 F1=0.6035±0.0508
- Display top excluding PC-GNN/original Ours: ours_imbalance 5/5 F1=0.6868±0.0352; lightgbm 5/5 F1=0.6347±0.0404; graphconsis_gnn 5/5 F1=0.6081±0.0524; graphsage 5/5 F1=0.6035±0.0508; tlmgnn 5/5 F1=0.5975±0.0131; care_gnn 5/5 F1=0.5969±0.0294

## Imbalance-Aware Evidence Ablation Deltas
| Dataset | Base | Rule | Qwen | Rule+Qwen | Rule-Base | Qwen-Base | Rule+Qwen-Rule | Rule+Qwen-Qwen | Done |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Meta | 0.9846 | 0.9809 | 0.9835 | 0.9857 | -0.0037 | -0.0011 | 0.0048 | 0.0022 | Base:5/5; Rule:5/5; Qwen:5/5; Rule+Qwen:5/5 |
| ZipZap | 0.9253 | 0.9289 | 0.9177 | 0.9309 | 0.0037 | -0.0076 | 0.0019 | 0.0132 | Base:4/5; Rule:3/5; Qwen:2/5; Rule+Qwen:2/5 |
| Illicit-ETH | 0.9728 | 0.9708 | 0.9752 | 0.9700 | -0.0020 | 0.0024 | -0.0008 | -0.0052 | Base:5/5; Rule:5/5; Qwen:5/5; Rule+Qwen:5/5 |
| EPSD-Ponzi | 0.6320 | 0.6819 | 0.6522 | 0.6868 | 0.0499 | 0.0202 | 0.0049 | 0.0346 | Base:5/5; Rule:5/5; Qwen:5/5; Rule+Qwen:5/5 |

## Raw Tabular Policy Audit
- logreg: metrics=20, raw_node_features_only violations=0
- mlp: metrics=20, raw_node_features_only violations=0
- rf: metrics=20, raw_node_features_only violations=0
- lightgbm: metrics=20, raw_node_features_only violations=0

## Suspicious / Uncomfortable Points
| Severity | Dataset | Point | Evidence | Likely interpretation |
|---|---|---|---|---|
| MEDIUM | Meta | Rule evidence 低于 Base | Rule-Base=-0.0037; Base=0.9846, Rule=0.9809 | deterministic rule 在该数据集可能带噪或标签定义不匹配。 |
| HIGH | Illicit-ETH | Rule+Qwen 低于 Qwen-only | Rule+Qwen-Qwen=-0.0052; Qwen=0.9752, Rule+Qwen=0.9700 | 更像 rule evidence 与 LLM motif 在该数据集上冲突/噪声叠加，尤其 Illicit-ETH。 |
| LOW | ZipZap | 主表存在未完成行 | care_gnn Done=4/5, F1=0.8835 | 未完成均值不能当最终结论。 |
| LOW | ZipZap | 主表存在未完成行 | pc_gnn Done=4/5, F1=0.9249 | 未完成均值不能当最终结论。 |
| LOW | ZipZap | 主表存在未完成行 | ours_imbalance Done=2/5, F1=0.9309 | 未完成均值不能当最终结论。 |
| MEDIUM | Meta | Raw RF 强于 GraphSAGE | RF=0.9823, GraphSAGE=0.9692 | 可能原始账户统计特征高度可分；已审计 raw-node-only，但论文中要解释。 |
| MEDIUM | EPSD-Ponzi | seed 方差偏高 | mlp std=0.0427 | split 敏感；要报告 mean±std，避免单 seed。 |
| MEDIUM | EPSD-Ponzi | seed 方差偏高 | rf std=0.0740 | split 敏感；要报告 mean±std，避免单 seed。 |
| MEDIUM | EPSD-Ponzi | seed 方差偏高 | lightgbm std=0.0404 | split 敏感；要报告 mean±std，避免单 seed。 |
| MEDIUM | EPSD-Ponzi | seed 方差偏高 | gcn std=0.0478 | split 敏感；要报告 mean±std，避免单 seed。 |
| MEDIUM | EPSD-Ponzi | seed 方差偏高 | gat std=0.0498 | split 敏感；要报告 mean±std，避免单 seed。 |
| MEDIUM | EPSD-Ponzi | seed 方差偏高 | graphsage std=0.0508 | split 敏感；要报告 mean±std，避免单 seed。 |
| MEDIUM | EPSD-Ponzi | seed 方差偏高 | graphconsis_gnn std=0.0524 | split 敏感；要报告 mean±std，避免单 seed。 |
| MEDIUM | EPSD-Ponzi | seed 方差偏高 | pc_gnn std=0.0751 | split 敏感；要报告 mean±std，避免单 seed。 |
| MEDIUM | EPSD-Ponzi | seed 方差偏高 | bwgnn std=0.0597 | split 敏感；要报告 mean±std，避免单 seed。 |
| MEDIUM | EPSD-Ponzi | seed 方差偏高 | ours std=0.0442 | split 敏感；要报告 mean±std，避免单 seed。 |
| LOW | All | LLM edge coverage 稀疏 | LLM edge coverage 约 0.07%-0.89%，低于 rule coverage。 | 这是 sparse typed evidence 的设计，但论文中要明说它不是全图逐边解释。 |

## Recommendation
- ZipZap MotifLens is promising but currently partial; wait for 5/5 before final ranking.
- Illicit-ETH Rule+Qwen is the one result that most deserves code/data audit or method adjustment, because Qwen-only > Rule+Qwen and Rule < Base.
- RF/LightGBM strong on Meta/Illicit is plausible but reviewer-sensitive; keep raw-node-only audit in appendix or experiment notes.
- EPSD-Ponzi is split-sensitive; always report mean±std and avoid single-run claims.
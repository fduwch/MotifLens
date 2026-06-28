# ResearchH Experiment Anomaly Report

Generated: 2026-06-16 13:27:58 UTC

Main: 340/340 complete; missing 0.
Ablation: 80/80 complete; missing 0.
Warnings: 41; Errors: 0; Code-suspect findings: 29.

## Missing By Method
- none

## Findings
- [warning] high_std (EPSD-Ponzi bwgnn): EPSD-Ponzi/bwgnn test_f1 std=0.0597 over 5 seeds; inspect split sensitivity before using a single-seed claim.
- [warning] high_std (EPSD-Ponzi gat): EPSD-Ponzi/gat test_f1 std=0.0498 over 5 seeds; inspect split sensitivity before using a single-seed claim.
- [warning] high_std (EPSD-Ponzi gcn): EPSD-Ponzi/gcn test_f1 std=0.0478 over 5 seeds; inspect split sensitivity before using a single-seed claim.
- [warning] high_std (EPSD-Ponzi graphconsis_gnn): EPSD-Ponzi/graphconsis_gnn test_f1 std=0.0524 over 5 seeds; inspect split sensitivity before using a single-seed claim.
- [warning] high_std (EPSD-Ponzi graphsage): EPSD-Ponzi/graphsage test_f1 std=0.0508 over 5 seeds; inspect split sensitivity before using a single-seed claim.
- [warning] high_std (EPSD-Ponzi lightgbm): EPSD-Ponzi/lightgbm test_f1 std=0.0404 over 5 seeds; inspect split sensitivity before using a single-seed claim.
- [warning] high_std (EPSD-Ponzi mlp): EPSD-Ponzi/mlp test_f1 std=0.0427 over 5 seeds; inspect split sensitivity before using a single-seed claim.
- [warning] high_std (EPSD-Ponzi ours): EPSD-Ponzi/ours test_f1 std=0.0442 over 5 seeds; inspect split sensitivity before using a single-seed claim.
- [warning] high_std (EPSD-Ponzi pc_gnn): EPSD-Ponzi/pc_gnn test_f1 std=0.0751 over 5 seeds; inspect split sensitivity before using a single-seed claim.
- [warning] high_std (EPSD-Ponzi rf): EPSD-Ponzi/rf test_f1 std=0.0740 over 5 seeds; inspect split sensitivity before using a single-seed claim.
- [warning] high_std (Illicit-ETH logreg): Illicit-ETH/logreg test_f1 std=0.0501 over 5 seeds; inspect split sensitivity before using a single-seed claim.
- [warning] high_std (Meta gcn): Meta/gcn test_f1 std=0.0248 over 5 seeds; inspect split sensitivity before using a single-seed claim.
- [ok] raw_tabular_policy: All completed tabular baseline metrics satisfy raw-node-feature-only checks.
- [warning] sequence_cache_unverified (Meta zipzap 42): outputs/paper_sota_baselines_20260615/Meta/seed_42/zipzap/metrics.json does not report sequence_cache_metadata_verified=true; cache may be correct by path but is not self-verifying.
  Likely code issue: cache_mismatch_or_unverifiable
  Suspect files: scripts/train_sota_baselines.py, llm_rationale_gnn/data.py
  Check: Sequence cache should include cache_version, data_dir, num_nodes, num_edges, seq_len, and neighbor_buckets.
  Check: Graph cache metadata should match dataset and seed.
  Check: If cache metadata is missing or mismatched, rebuild cache before accepting the run.
- [warning] sequence_cache_unverified (Meta lmae4eth 42): outputs/paper_sota_baselines_20260615/Meta/seed_42/lmae4eth/metrics.json does not report sequence_cache_metadata_verified=true; cache may be correct by path but is not self-verifying.
  Likely code issue: cache_mismatch_or_unverifiable
  Suspect files: scripts/train_sota_baselines.py, llm_rationale_gnn/data.py
  Check: Sequence cache should include cache_version, data_dir, num_nodes, num_edges, seq_len, and neighbor_buckets.
  Check: Graph cache metadata should match dataset and seed.
  Check: If cache metadata is missing or mismatched, rebuild cache before accepting the run.
- [warning] sequence_cache_unverified (Meta bert4eth 42): outputs/paper_sota_baselines_20260615/Meta/seed_42/bert4eth/metrics.json does not report sequence_cache_metadata_verified=true; cache may be correct by path but is not self-verifying.
  Likely code issue: cache_mismatch_or_unverifiable
  Suspect files: scripts/train_sota_baselines.py, llm_rationale_gnn/data.py
  Check: Sequence cache should include cache_version, data_dir, num_nodes, num_edges, seq_len, and neighbor_buckets.
  Check: Graph cache metadata should match dataset and seed.
  Check: If cache metadata is missing or mismatched, rebuild cache before accepting the run.
- [warning] sequence_cache_unverified (Meta tlmgnn 42): outputs/paper_sota_baselines_20260615/Meta/seed_42/tlmgnn/metrics.json does not report sequence_cache_metadata_verified=true; cache may be correct by path but is not self-verifying.
  Likely code issue: cache_mismatch_or_unverifiable
  Suspect files: scripts/train_sota_baselines.py, llm_rationale_gnn/data.py
  Check: Sequence cache should include cache_version, data_dir, num_nodes, num_edges, seq_len, and neighbor_buckets.
  Check: Graph cache metadata should match dataset and seed.
  Check: If cache metadata is missing or mismatched, rebuild cache before accepting the run.
- [warning] sequence_cache_unverified (Meta bwgnn 42): outputs/paper_sota_baselines_20260615/Meta/seed_42/bwgnn/metrics.json does not report sequence_cache_metadata_verified=true; cache may be correct by path but is not self-verifying.
  Likely code issue: cache_mismatch_or_unverifiable
  Suspect files: scripts/train_sota_baselines.py, llm_rationale_gnn/data.py
  Check: Sequence cache should include cache_version, data_dir, num_nodes, num_edges, seq_len, and neighbor_buckets.
  Check: Graph cache metadata should match dataset and seed.
  Check: If cache metadata is missing or mismatched, rebuild cache before accepting the run.
- [warning] sequence_cache_unverified (Meta zipzap 43): outputs/paper_sota_baselines_20260615/Meta/seed_43/zipzap/metrics.json does not report sequence_cache_metadata_verified=true; cache may be correct by path but is not self-verifying.
  Likely code issue: cache_mismatch_or_unverifiable
  Suspect files: scripts/train_sota_baselines.py, llm_rationale_gnn/data.py
  Check: Sequence cache should include cache_version, data_dir, num_nodes, num_edges, seq_len, and neighbor_buckets.
  Check: Graph cache metadata should match dataset and seed.
  Check: If cache metadata is missing or mismatched, rebuild cache before accepting the run.
- [warning] sequence_cache_unverified (Meta lmae4eth 43): outputs/paper_sota_baselines_20260615/Meta/seed_43/lmae4eth/metrics.json does not report sequence_cache_metadata_verified=true; cache may be correct by path but is not self-verifying.
  Likely code issue: cache_mismatch_or_unverifiable
  Suspect files: scripts/train_sota_baselines.py, llm_rationale_gnn/data.py
  Check: Sequence cache should include cache_version, data_dir, num_nodes, num_edges, seq_len, and neighbor_buckets.
  Check: Graph cache metadata should match dataset and seed.
  Check: If cache metadata is missing or mismatched, rebuild cache before accepting the run.
- [warning] sequence_cache_unverified (Meta bert4eth 43): outputs/paper_sota_baselines_20260615/Meta/seed_43/bert4eth/metrics.json does not report sequence_cache_metadata_verified=true; cache may be correct by path but is not self-verifying.
  Likely code issue: cache_mismatch_or_unverifiable
  Suspect files: scripts/train_sota_baselines.py, llm_rationale_gnn/data.py
  Check: Sequence cache should include cache_version, data_dir, num_nodes, num_edges, seq_len, and neighbor_buckets.
  Check: Graph cache metadata should match dataset and seed.
  Check: If cache metadata is missing or mismatched, rebuild cache before accepting the run.
- [warning] sequence_cache_unverified (Meta tlmgnn 43): outputs/paper_sota_baselines_20260615/Meta/seed_43/tlmgnn/metrics.json does not report sequence_cache_metadata_verified=true; cache may be correct by path but is not self-verifying.
  Likely code issue: cache_mismatch_or_unverifiable
  Suspect files: scripts/train_sota_baselines.py, llm_rationale_gnn/data.py
  Check: Sequence cache should include cache_version, data_dir, num_nodes, num_edges, seq_len, and neighbor_buckets.
  Check: Graph cache metadata should match dataset and seed.
  Check: If cache metadata is missing or mismatched, rebuild cache before accepting the run.
- [warning] sequence_cache_unverified (Meta bwgnn 43): outputs/paper_sota_baselines_20260615/Meta/seed_43/bwgnn/metrics.json does not report sequence_cache_metadata_verified=true; cache may be correct by path but is not self-verifying.
  Likely code issue: cache_mismatch_or_unverifiable
  Suspect files: scripts/train_sota_baselines.py, llm_rationale_gnn/data.py
  Check: Sequence cache should include cache_version, data_dir, num_nodes, num_edges, seq_len, and neighbor_buckets.
  Check: Graph cache metadata should match dataset and seed.
  Check: If cache metadata is missing or mismatched, rebuild cache before accepting the run.
- [warning] sequence_cache_unverified (Meta zipzap 44): outputs/paper_sota_baselines_20260615/Meta/seed_44/zipzap/metrics.json does not report sequence_cache_metadata_verified=true; cache may be correct by path but is not self-verifying.
  Likely code issue: cache_mismatch_or_unverifiable
  Suspect files: scripts/train_sota_baselines.py, llm_rationale_gnn/data.py
  Check: Sequence cache should include cache_version, data_dir, num_nodes, num_edges, seq_len, and neighbor_buckets.
  Check: Graph cache metadata should match dataset and seed.
  Check: If cache metadata is missing or mismatched, rebuild cache before accepting the run.
- [warning] sequence_cache_unverified (Meta lmae4eth 44): outputs/paper_sota_baselines_20260615/Meta/seed_44/lmae4eth/metrics.json does not report sequence_cache_metadata_verified=true; cache may be correct by path but is not self-verifying.
  Likely code issue: cache_mismatch_or_unverifiable
  Suspect files: scripts/train_sota_baselines.py, llm_rationale_gnn/data.py
  Check: Sequence cache should include cache_version, data_dir, num_nodes, num_edges, seq_len, and neighbor_buckets.
  Check: Graph cache metadata should match dataset and seed.
  Check: If cache metadata is missing or mismatched, rebuild cache before accepting the run.
- [warning] sequence_cache_unverified (Meta bert4eth 44): outputs/paper_sota_baselines_20260615/Meta/seed_44/bert4eth/metrics.json does not report sequence_cache_metadata_verified=true; cache may be correct by path but is not self-verifying.
  Likely code issue: cache_mismatch_or_unverifiable
  Suspect files: scripts/train_sota_baselines.py, llm_rationale_gnn/data.py
  Check: Sequence cache should include cache_version, data_dir, num_nodes, num_edges, seq_len, and neighbor_buckets.
  Check: Graph cache metadata should match dataset and seed.
  Check: If cache metadata is missing or mismatched, rebuild cache before accepting the run.
- [warning] sequence_cache_unverified (Meta tlmgnn 44): outputs/paper_sota_baselines_20260615/Meta/seed_44/tlmgnn/metrics.json does not report sequence_cache_metadata_verified=true; cache may be correct by path but is not self-verifying.
  Likely code issue: cache_mismatch_or_unverifiable
  Suspect files: scripts/train_sota_baselines.py, llm_rationale_gnn/data.py
  Check: Sequence cache should include cache_version, data_dir, num_nodes, num_edges, seq_len, and neighbor_buckets.
  Check: Graph cache metadata should match dataset and seed.
  Check: If cache metadata is missing or mismatched, rebuild cache before accepting the run.
- [warning] sequence_cache_unverified (Meta bwgnn 44): outputs/paper_sota_baselines_20260615/Meta/seed_44/bwgnn/metrics.json does not report sequence_cache_metadata_verified=true; cache may be correct by path but is not self-verifying.
  Likely code issue: cache_mismatch_or_unverifiable
  Suspect files: scripts/train_sota_baselines.py, llm_rationale_gnn/data.py
  Check: Sequence cache should include cache_version, data_dir, num_nodes, num_edges, seq_len, and neighbor_buckets.
  Check: Graph cache metadata should match dataset and seed.
  Check: If cache metadata is missing or mismatched, rebuild cache before accepting the run.
- [warning] sequence_cache_unverified (Illicit-ETH zipzap 42): outputs/paper_sota_baselines_20260615/Illicit-ETH/seed_42/zipzap/metrics.json does not report sequence_cache_metadata_verified=true; cache may be correct by path but is not self-verifying.
  Likely code issue: cache_mismatch_or_unverifiable
  Suspect files: scripts/train_sota_baselines.py, llm_rationale_gnn/data.py
  Check: Sequence cache should include cache_version, data_dir, num_nodes, num_edges, seq_len, and neighbor_buckets.
  Check: Graph cache metadata should match dataset and seed.
  Check: If cache metadata is missing or mismatched, rebuild cache before accepting the run.
- [warning] sequence_cache_unverified (Illicit-ETH lmae4eth 42): outputs/paper_sota_baselines_20260615/Illicit-ETH/seed_42/lmae4eth/metrics.json does not report sequence_cache_metadata_verified=true; cache may be correct by path but is not self-verifying.
  Likely code issue: cache_mismatch_or_unverifiable
  Suspect files: scripts/train_sota_baselines.py, llm_rationale_gnn/data.py
  Check: Sequence cache should include cache_version, data_dir, num_nodes, num_edges, seq_len, and neighbor_buckets.
  Check: Graph cache metadata should match dataset and seed.
  Check: If cache metadata is missing or mismatched, rebuild cache before accepting the run.
- [warning] sequence_cache_unverified (Illicit-ETH bert4eth 42): outputs/paper_sota_baselines_20260615/Illicit-ETH/seed_42/bert4eth/metrics.json does not report sequence_cache_metadata_verified=true; cache may be correct by path but is not self-verifying.
  Likely code issue: cache_mismatch_or_unverifiable
  Suspect files: scripts/train_sota_baselines.py, llm_rationale_gnn/data.py
  Check: Sequence cache should include cache_version, data_dir, num_nodes, num_edges, seq_len, and neighbor_buckets.
  Check: Graph cache metadata should match dataset and seed.
  Check: If cache metadata is missing or mismatched, rebuild cache before accepting the run.
- [warning] sequence_cache_unverified (Illicit-ETH tlmgnn 42): outputs/paper_sota_baselines_20260615/Illicit-ETH/seed_42/tlmgnn/metrics.json does not report sequence_cache_metadata_verified=true; cache may be correct by path but is not self-verifying.
  Likely code issue: cache_mismatch_or_unverifiable
  Suspect files: scripts/train_sota_baselines.py, llm_rationale_gnn/data.py
  Check: Sequence cache should include cache_version, data_dir, num_nodes, num_edges, seq_len, and neighbor_buckets.
  Check: Graph cache metadata should match dataset and seed.
  Check: If cache metadata is missing or mismatched, rebuild cache before accepting the run.
- [warning] sequence_cache_unverified (Illicit-ETH bwgnn 42): outputs/paper_sota_baselines_20260615/Illicit-ETH/seed_42/bwgnn/metrics.json does not report sequence_cache_metadata_verified=true; cache may be correct by path but is not self-verifying.
  Likely code issue: cache_mismatch_or_unverifiable
  Suspect files: scripts/train_sota_baselines.py, llm_rationale_gnn/data.py
  Check: Sequence cache should include cache_version, data_dir, num_nodes, num_edges, seq_len, and neighbor_buckets.
  Check: Graph cache metadata should match dataset and seed.
  Check: If cache metadata is missing or mismatched, rebuild cache before accepting the run.
- [warning] sequence_cache_unverified (Illicit-ETH zipzap 43): outputs/paper_sota_baselines_20260615/Illicit-ETH/seed_43/zipzap/metrics.json does not report sequence_cache_metadata_verified=true; cache may be correct by path but is not self-verifying.
  Likely code issue: cache_mismatch_or_unverifiable
  Suspect files: scripts/train_sota_baselines.py, llm_rationale_gnn/data.py
  Check: Sequence cache should include cache_version, data_dir, num_nodes, num_edges, seq_len, and neighbor_buckets.
  Check: Graph cache metadata should match dataset and seed.
  Check: If cache metadata is missing or mismatched, rebuild cache before accepting the run.
- [warning] sequence_cache_unverified (Illicit-ETH lmae4eth 43): outputs/paper_sota_baselines_20260615/Illicit-ETH/seed_43/lmae4eth/metrics.json does not report sequence_cache_metadata_verified=true; cache may be correct by path but is not self-verifying.
  Likely code issue: cache_mismatch_or_unverifiable
  Suspect files: scripts/train_sota_baselines.py, llm_rationale_gnn/data.py
  Check: Sequence cache should include cache_version, data_dir, num_nodes, num_edges, seq_len, and neighbor_buckets.
  Check: Graph cache metadata should match dataset and seed.
  Check: If cache metadata is missing or mismatched, rebuild cache before accepting the run.
- [warning] sequence_cache_unverified (Illicit-ETH bert4eth 43): outputs/paper_sota_baselines_20260615/Illicit-ETH/seed_43/bert4eth/metrics.json does not report sequence_cache_metadata_verified=true; cache may be correct by path but is not self-verifying.
  Likely code issue: cache_mismatch_or_unverifiable
  Suspect files: scripts/train_sota_baselines.py, llm_rationale_gnn/data.py
  Check: Sequence cache should include cache_version, data_dir, num_nodes, num_edges, seq_len, and neighbor_buckets.
  Check: Graph cache metadata should match dataset and seed.
  Check: If cache metadata is missing or mismatched, rebuild cache before accepting the run.
- [warning] sequence_cache_unverified (Illicit-ETH tlmgnn 43): outputs/paper_sota_baselines_20260615/Illicit-ETH/seed_43/tlmgnn/metrics.json does not report sequence_cache_metadata_verified=true; cache may be correct by path but is not self-verifying.
  Likely code issue: cache_mismatch_or_unverifiable
  Suspect files: scripts/train_sota_baselines.py, llm_rationale_gnn/data.py
  Check: Sequence cache should include cache_version, data_dir, num_nodes, num_edges, seq_len, and neighbor_buckets.
  Check: Graph cache metadata should match dataset and seed.
  Check: If cache metadata is missing or mismatched, rebuild cache before accepting the run.
- [warning] sequence_cache_unverified (Illicit-ETH bwgnn 43): outputs/paper_sota_baselines_20260615/Illicit-ETH/seed_43/bwgnn/metrics.json does not report sequence_cache_metadata_verified=true; cache may be correct by path but is not self-verifying.
  Likely code issue: cache_mismatch_or_unverifiable
  Suspect files: scripts/train_sota_baselines.py, llm_rationale_gnn/data.py
  Check: Sequence cache should include cache_version, data_dir, num_nodes, num_edges, seq_len, and neighbor_buckets.
  Check: Graph cache metadata should match dataset and seed.
  Check: If cache metadata is missing or mismatched, rebuild cache before accepting the run.
- [warning] sequence_cache_unverified (Illicit-ETH zipzap 44): outputs/paper_sota_baselines_20260615/Illicit-ETH/seed_44/zipzap/metrics.json does not report sequence_cache_metadata_verified=true; cache may be correct by path but is not self-verifying.
  Likely code issue: cache_mismatch_or_unverifiable
  Suspect files: scripts/train_sota_baselines.py, llm_rationale_gnn/data.py
  Check: Sequence cache should include cache_version, data_dir, num_nodes, num_edges, seq_len, and neighbor_buckets.
  Check: Graph cache metadata should match dataset and seed.
  Check: If cache metadata is missing or mismatched, rebuild cache before accepting the run.
- [warning] sequence_cache_unverified (Illicit-ETH lmae4eth 44): outputs/paper_sota_baselines_20260615/Illicit-ETH/seed_44/lmae4eth/metrics.json does not report sequence_cache_metadata_verified=true; cache may be correct by path but is not self-verifying.
  Likely code issue: cache_mismatch_or_unverifiable
  Suspect files: scripts/train_sota_baselines.py, llm_rationale_gnn/data.py
  Check: Sequence cache should include cache_version, data_dir, num_nodes, num_edges, seq_len, and neighbor_buckets.
  Check: Graph cache metadata should match dataset and seed.
  Check: If cache metadata is missing or mismatched, rebuild cache before accepting the run.
- [ok] baseline_evidence_policy: Completed non-Ours baseline metrics do not report evidence usage.
- [info] delta_ours_graphsage (Meta): Meta: Ours-GraphSAGE delta F1 = +0.0039.
- [warning] ours_below_lightgbm (Meta): Meta: Ours F1=0.9732 is below raw LightGBM F1=0.9808; inspect whether graph/evidence helps this dataset.
  Likely code issue: llm_delta_issue
  Suspect files: llm_rationale_gnn/data.py, scripts/train_sampled.py, logs/researchh_summarize_paper_5seed_qwen_counter_clean_20260615.py
  Check: Check Qwen motif evidence coverage and typed-evidence deduplication.
  Check: Check Rule+Qwen uses scalar_plus_motif evidence mode with correct evidence files.
  Check: If LLM hurts most seeds, inspect evidence quality rather than hiding the result.
- [info] delta_ours_graphsage (ZipZap): ZipZap: Ours-GraphSAGE delta F1 = +0.0182.
- [info] delta_ours_graphsage (Illicit-ETH): Illicit-ETH: Ours-GraphSAGE delta F1 = +0.0176.
- [info] delta_ours_graphsage (EPSD-Ponzi): EPSD-Ponzi: Ours-GraphSAGE delta F1 = +0.0450.
- [warning] qwen_not_improving_base (EPSD-Ponzi): EPSD-Ponzi: Qwen motif minus Base GraphSAGE delta=-0.0019.
  Likely code issue: llm_delta_issue
  Suspect files: llm_rationale_gnn/data.py, scripts/train_sampled.py, logs/researchh_summarize_paper_5seed_qwen_counter_clean_20260615.py
  Check: Check Qwen motif evidence coverage and typed-evidence deduplication.
  Check: Check Rule+Qwen uses scalar_plus_motif evidence mode with correct evidence files.
  Check: If LLM hurts most seeds, inspect evidence quality rather than hiding the result.

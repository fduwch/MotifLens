# Experiment Summary Scripts

This directory contains the scripts used to summarize MotifLens paper results and run sanity audits over completed `metrics.json` files.

The scripts assume the repository root as their base path and expect full run outputs under the original `outputs/` subdirectories, for example:

```text
outputs/paper_5seed_qwen_counter_clean_20260615/
outputs/paper_5seed_qwen_counter_clean_pctrain_20260616/
outputs/paper_main_baselines_clean_20260615/
outputs/paper_sota_baselines_20260615/
```

This review package includes aggregate CSV/Markdown outputs, not all per-run `metrics.json` files or checkpoints. These scripts are mainly useful after rerunning the experiments locally.

Typical commands after full runs are available:

```bash
python experiments/researchh_make_main_and_ablation_tables_clean_20260615.py
python experiments/researchh_result_analyzer_20260616.py
```

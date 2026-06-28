# Result Artifacts

This directory contains small derived artifacts used to verify the MotifLens paper tables during review.

Included:

- aggregate and per-seed CSV files for the main comparison and evidence ablation;
- Markdown tables used during manuscript preparation;
- evidence quality, motif distribution, interpretability, and efficiency summaries;
- anomaly/sanity audit reports.

Excluded:

- raw Ethereum datasets;
- full graph caches;
- model checkpoints;
- full LLM response dumps;
- large per-run training directories.

The included files are intended for table inspection and consistency checks. To reproduce them from scratch, place the processed datasets under `data/processed/`, run the training scripts, and regenerate summaries using the scripts in `experiments/`.

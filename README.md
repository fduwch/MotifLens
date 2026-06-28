# MotifLens Review Artifact

This repository contains the code and compact result artifacts used for the MotifLens paper, **MotifLens: Interpretable Ethereum Fraud Detection via LLM Motif Evidence**.

MotifLens uses rule-based transaction evidence and schema-constrained LLM Motif evidence as fixed numerical evidence channels for a sampled GraphSAGE classifier. The LLM is used offline to generate typed motif rows. It is not used as the final fraud classifier at inference time.

This package is prepared for paper review. It includes the implementation, table-generation scripts, small aggregate result files, and dataset-format instructions. Full raw Ethereum datasets, model checkpoints, graph caches, private RPC outputs, and full LLM response dumps are not included.

## Structure

```text
llm_rationale_gnn/          Core graph loading, evidence attachment, metrics, and GNN modules.
scripts/                    Dataset conversion, evidence generation, training, baselines, and explanation scripts.
experiments/                Scripts used to summarize paper tables and audit completed metrics.
outputs/                    Small aggregate CSV/Markdown result artifacts used for table verification.
data/                       Dataset placement and format notes.
prompts/                    LLM Motif evidence prompt documentation.
```

The package directory is named `llm_rationale_gnn` for compatibility with the experiment scripts.

## Environment

The experiments were run with Python 3.11, PyTorch, PyTorch Geometric, scikit-learn, pandas, and NumPy. Install PyTorch and PyTorch Geometric with versions matching your CUDA environment, then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

A conda environment sketch is also provided:

```bash
conda env create -f environment.yml
```

Optional packages:

- `openai`: required for OpenAI-compatible LLM evidence generation.
- `lightgbm`: required only for the LightGBM tabular baseline.

## Data Format

Raw benchmark data are not bundled. After downloading datasets from their original sources, convert or place them into:

```text
data/processed/<dataset_name>/
  nodes.csv
  edges.csv
  labels.csv        # optional if labels/splits are already in nodes.csv
```

Required edge columns:

```csv
src,dst,timestamp,amount
```

Recommended node columns:

```csv
node_id,label,split,<feature columns...>
```

The loader removes label-, split-, target-, relation-, source-, and fraud-specific metadata from model features before training.

For a quick local check, generate a synthetic dataset:

```bash
python scripts/make_synthetic_dataset.py --out data/processed/synthetic
```

## Smoke Test

```bash
python scripts/make_synthetic_dataset.py --out data/processed/synthetic

python scripts/train_sampled.py \
  --data data/processed/synthetic \
  --out outputs/smoke/synthetic_seed42/base \
  --model sage \
  --seed 42 \
  --epochs 5
```

This checks the loader, model construction, and training loop on a small synthetic graph.

## LLM Motif Evidence

MotifLens can call an OpenAI-compatible chat-completion endpoint. Example:

```bash
python scripts/generate_llm_motif_evidence.py \
  --data data/processed/<dataset_name> \
  --out outputs/evidence/<dataset_name>/llm_motifs.csv \
  --cards-out outputs/evidence/<dataset_name>/cards.jsonl \
  --responses-out outputs/evidence/<dataset_name>/responses.jsonl \
  --llm-base-url http://localhost:8000/v1 \
  --llm-api-key EMPTY \
  --llm-model Qwen/Qwen3-14B \
  --prompt-no-think
```

For development without an LLM endpoint, use `--teacher heuristic` where supported.

The prompt template, motif taxonomy, and expected JSON response schema are documented in `prompts/llm_motif_prompt.md`. The executable version is also kept inline in `scripts/generate_llm_motif_evidence.py`.

## Training MotifLens

Example sampled GraphSAGE training with evidence channels:

```bash
python scripts/train_sampled.py \
  --data data/processed/<dataset_name> \
  --evidence outputs/evidence/<dataset_name>/llm_motifs.csv \
  --out outputs/runs/<dataset_name>/seed_42/motiflens \
  --model sage \
  --seed 42 \
  --ignore-split-column \
  --balanced-train \
  --balanced-pos-repeat 16 \
  --balanced-neg-ratio 4.0 \
  --loss focal
```

The paper experiments use five stratified resplits with seeds `42,43,44,45,46`.

## Baselines

Baseline scripts are included for the comparison methods used in the paper:

```text
scripts/train_feature_baselines.py
scripts/train_fraud_gnn_baselines.py
scripts/train_sota_baselines.py
```

Raw-node-feature tabular baseline example:

```bash
python scripts/train_feature_baselines.py \
  --data data/processed/<dataset_name> \
  --out outputs/baselines/<dataset_name>/seed_42 \
  --methods rf,mlp \
  --raw-node-features-only \
  --seed 42
```

## Included Result Artifacts

Small derived files are included under `outputs/` so that reviewers can inspect the reported tables without rerunning every experiment:

```text
outputs/paper_tables_clean_20260615/
  main_comparison_aggregate.csv
  main_comparison_per_seed.csv
  ablation_aggregate.csv
  ablation_per_seed.csv
  anomaly_report.md

outputs/motiflens_experiments_20260616/
  evidence_quality_coverage_table.md
  motif_distribution_table.csv
  interpretability_case_studies.md
  efficiency_analysis.md
```

These are aggregate or compact summary artifacts. Full training outputs and checkpoints are intentionally omitted from this review package.

## Regenerating Tables

If full per-run `metrics.json` files are available under the expected `outputs/` roots, the paper tables can be regenerated with:

```bash
python experiments/researchh_make_main_and_ablation_tables_clean_20260615.py
python experiments/researchh_result_analyzer_20260616.py
```

The included aggregate CSV/Markdown files can also be inspected directly.

## Notes

This review artifact excludes credentials, private endpoints, raw third-party benchmark archives, model checkpoints, large graph caches, and full LLM response dumps. Before uploading to a review system or a temporary repository, it is still worth running a final scan:

```bash
git grep -n -i "api_key\|secret\|password\|cookie\|BEGIN .*PRIVATE KEY"
```

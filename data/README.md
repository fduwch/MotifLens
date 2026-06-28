# Data Directory

Raw benchmark data are not included in this repository.

Place downloaded datasets under:

```text
data/raw/
```

Convert each dataset into the MotifLens processed graph format:

```text
data/processed/<dataset_name>/
  nodes.csv
  edges.csv
  labels.csv        # optional
```

Required edge columns:

```text
src,dst,timestamp,amount
```

Recommended node columns:

```text
node_id,label,split,<feature columns...>
```

The training loader removes label-, split-, target-, relation-, source-, and fraud-specific metadata columns before model training.

For smoke tests, run:

```bash
python scripts/make_synthetic_dataset.py --out data/processed/synthetic
```

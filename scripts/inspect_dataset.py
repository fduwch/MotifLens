#!/usr/bin/env python
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def describe_file(path: Path) -> None:
    print(f"\n== {path} ==")
    print(f"size={path.stat().st_size}")
    suffix = path.suffix.lower()
    try:
        if suffix in {".csv", ".tsv"}:
            sep = "\t" if suffix == ".tsv" else ","
            df = pd.read_csv(path, sep=sep, nrows=5)
            print("columns=", list(df.columns))
            print(df.head().to_string(index=False))
        elif suffix == ".npz":
            arr = np.load(path, allow_pickle=True)
            print("keys=", list(arr.keys()))
            for key in arr.keys():
                value = arr[key]
                print(f"{key}: shape={getattr(value, 'shape', None)} dtype={getattr(value, 'dtype', None)}")
        elif suffix in {".pt", ".pth"}:
            obj = torch.load(path, map_location="cpu")
            print(type(obj))
            if isinstance(obj, dict):
                print("keys=", list(obj.keys()))
                for key, value in obj.items():
                    print(f"{key}: type={type(value)} shape={getattr(value, 'shape', None)}")
        elif suffix in {".pkl", ".pickle"}:
            with path.open("rb") as fh:
                obj = pickle.load(fh)
            print(type(obj))
            if isinstance(obj, dict):
                print("keys=", list(obj.keys()))
                for key, value in obj.items():
                    print(f"{key}: type={type(value)} shape={getattr(value, 'shape', None)}")
        else:
            print("unsupported preview")
    except Exception as exc:
        print("preview_error=", type(exc).__name__, str(exc)[:300])


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect an unknown graph dataset directory.")
    parser.add_argument("--path", required=True)
    parser.add_argument("--max-files", type=int, default=40)
    args = parser.parse_args()

    root = Path(args.path)
    files = []
    for suffix in ("*.csv", "*.tsv", "*.npz", "*.pt", "*.pth", "*.pkl", "*.pickle"):
        files.extend(root.rglob(suffix))
    for path in sorted(files)[: args.max_files]:
        describe_file(path)
    print(f"\ninspected={min(len(files), args.max_files)} total_candidates={len(files)}")


if __name__ == "__main__":
    main()

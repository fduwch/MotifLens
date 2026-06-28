#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import random
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


FORTA_URLS = {
    "malicious_smart_contracts": "https://raw.githubusercontent.com/forta-network/labelled-datasets/main/labels/1/malicious_smart_contracts.csv",
    "phishing_scams": "https://raw.githubusercontent.com/forta-network/labelled-datasets/main/labels/1/phishing_scams.csv",
    "etherscan_malicious_labels": "https://raw.githubusercontent.com/forta-network/labelled-datasets/main/labels/1/etherscan_malicious_labels.csv",
}
ILLICIT_ETH_URL = "https://raw.githubusercontent.com/sfarrugia15/Ethereum_Fraud_Detection/master/Account_Stats/Complete.csv"


def norm_addr(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("0x") and len(text) == 42:
        return text
    return ""


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except Exception:
        return default


def json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)


def download_if_needed(path: Path, url: str, download: bool) -> None:
    if path.exists():
        return
    if not download:
        raise FileNotFoundError(f"Missing {path}; pass --download to fetch {url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    print(json.dumps({"phase": "download", "url": url, "out": str(path)}, ensure_ascii=False), flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "ResearchH2-dataset-prep/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    tmp.replace(path)


def split_labels(addresses: list[str], labels: dict[str, int], train_ratio: float, val_ratio: float, seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    by_label: dict[int, list[str]] = defaultdict(list)
    for address in addresses:
        by_label[int(labels[address])].append(address)
    split: dict[str, str] = {}
    for label, values in by_label.items():
        rng.shuffle(values)
        n = len(values)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        for address in values[:n_train]:
            split[address] = "train"
        for address in values[n_train : n_train + n_val]:
            split[address] = "val"
        for address in values[n_train + n_val :]:
            split[address] = "test"
    return split


def read_standard_labels(data_dir: Path, source_name: str, progress_every: int) -> tuple[set[str], set[str], Counter]:
    path = data_dir / "nodes.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    positives: set[str] = set()
    normals: set[str] = set()
    stats: Counter = Counter()
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            address = norm_addr(row.get("node_id") or row.get("address") or row.get("id"))
            label = safe_int(row.get("label"), -1)
            if not address or label not in (0, 1):
                stats["ignored"] += 1
                continue
            is_target = str(row.get("is_target", "1")).strip().lower()
            if is_target in {"0", "false", "no"}:
                stats["context_labeled"] += 1
                continue
            if label == 1:
                positives.add(address)
                stats["positive"] += 1
            else:
                normals.add(address)
                stats["normal"] += 1
            if progress_every and (stats["positive"] + stats["normal"]) % progress_every == 0:
                print(
                    json.dumps(
                        {
                            "phase": "read_standard_labels",
                            "source": source_name,
                            "rows": int(stats["positive"] + stats["normal"]),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    return positives, normals, stats


def parse_source_arg(items: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected NAME=PATH, got {item}")
        name, value = item.split("=", 1)
        out[name.strip()] = Path(value).expanduser()
    return out


def prepare_illicit_eth(raw_dir: Path, out_dir: Path, download: bool, train_ratio: float, val_ratio: float, seed: int) -> dict[str, Any]:
    raw_path = raw_dir / "illicit_eth_complete.csv"
    download_if_needed(raw_path, ILLICIT_ETH_URL, download)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels: dict[str, int] = {}
    raw_counts: Counter = Counter()
    fields: list[str] = []
    rows_by_address: dict[str, dict[str, Any]] = {}
    conflict_addresses: set[str] = set()
    with raw_path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = list(reader.fieldnames or [])
        for row in reader:
            address = norm_addr(row.get("Address"))
            if not address:
                raw_counts["bad_address"] += 1
                continue
            label = safe_int(row.get("FLAG"), -1)
            if label not in (0, 1):
                raw_counts["bad_label"] += 1
                continue
            raw_counts[f"raw_label_{label}"] += 1
            item = {"node_id": address, "label": label, "is_target": 1, "label_source": "illicit_eth_complete"}
            for key, value in row.items():
                if key not in {"Index", "Address", "FLAG"}:
                    item[key] = value
            if address in labels:
                raw_counts["duplicate_rows"] += 1
                if labels[address] != label:
                    raw_counts["duplicate_label_conflicts"] += 1
                    conflict_addresses.add(address)
                continue
            labels[address] = label
            rows_by_address[address] = item

    for address in conflict_addresses:
        labels.pop(address, None)
        rows_by_address.pop(address, None)

    counts = Counter({f"label_{label}": count for label, count in Counter(labels.values()).items()})
    counts.update({k: v for k, v in raw_counts.items() if k in {"bad_address", "bad_label", "duplicate_rows", "duplicate_label_conflicts"}})

    split = split_labels(sorted(labels), labels, train_ratio, val_ratio, seed)
    fieldnames = ["node_id", "label", "split", "is_target", "label_source"] + [
        f for f in fields if f not in {"Index", "Address", "FLAG"}
    ]
    with (out_dir / "labels.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows_by_address.values():
            row["split"] = split[row["node_id"]]
            writer.writerow(row)

    summary = {
        "dataset": "Illicit-ETH",
        "source_url": ILLICIT_ETH_URL,
        "raw_path": str(raw_path),
        "labels_path": str(out_dir / "labels.csv"),
        "counts": {str(k): int(v) for k, v in counts.items()},
        "raw_counts": {str(k): int(v) for k, v in raw_counts.items()},
        "excluded_conflict_addresses": len(conflict_addresses),
        "addresses": len(labels),
    }
    json_dump(out_dir / "summary.json", summary)
    return {"summary": summary, "labels": labels}


def add_positive(
    positives: dict[str, dict[str, Any]],
    address: str,
    source: str,
    tag: str = "",
    is_contract: str = "",
) -> None:
    address = norm_addr(address)
    if not address:
        return
    item = positives.setdefault(
        address,
        {"node_id": address, "label": 1, "is_target": 1, "sources": set(), "tags": set(), "is_contract_values": set()},
    )
    item["sources"].add(source)
    if tag:
        item["tags"].add(tag)
    if is_contract not in ("", None):
        item["is_contract_values"].add(str(is_contract))


def prepare_forta_positives(
    raw_dir: Path,
    out_dir: Path,
    download: bool,
    include_creators_as_positive: bool,
) -> tuple[dict[str, int], dict[str, Any]]:
    raw_paths = {name: raw_dir / f"forta_{name}.csv" for name in FORTA_URLS}
    for name, url in FORTA_URLS.items():
        download_if_needed(raw_paths[name], url, download)

    positives: dict[str, dict[str, Any]] = {}
    creators: dict[str, dict[str, Any]] = {}
    counts: Counter = Counter()

    with raw_paths["malicious_smart_contracts"].open("r", encoding="utf-8", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh):
            contract = norm_addr(row.get("contract_address"))
            creator = norm_addr(row.get("contract_creator"))
            tag = row.get("contract_tag") or row.get("contract_creator_etherscan_label") or ""
            if contract:
                add_positive(positives, contract, "forta_malicious_smart_contract", tag=tag, is_contract="True")
                counts["malicious_contract_addresses"] += 1
            if creator:
                item = creators.setdefault(
                    creator,
                    {"node_id": creator, "label": 1, "is_target": 1, "sources": set(), "tags": set(), "is_contract_values": set()},
                )
                item["sources"].add("forta_contract_creator")
                if tag:
                    item["tags"].add(tag)
                counts["contract_creator_addresses"] += 1
                if include_creators_as_positive:
                    add_positive(positives, creator, "forta_contract_creator", tag=tag, is_contract="False")

    with raw_paths["phishing_scams"].open("r", encoding="utf-8", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh):
            address = norm_addr(row.get("address"))
            tag = row.get("etherscan_tag") or row.get("etherscan_labels") or ""
            if address:
                add_positive(positives, address, "forta_phishing_scams", tag=tag, is_contract=str(row.get("is_contract", "")))
                counts["phishing_scams_addresses"] += 1

    with raw_paths["etherscan_malicious_labels"].open("r", encoding="utf-8", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh):
            address = norm_addr(row.get("banned_address"))
            tag = row.get("wallet_tag") or row.get("data_source") or ""
            if address:
                add_positive(positives, address, "forta_etherscan_malicious_labels", tag=tag)
                counts["etherscan_malicious_addresses"] += 1

    out_dir.mkdir(parents=True, exist_ok=True)

    def write_address_rows(path: Path, rows: dict[str, dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["node_id", "label", "split", "is_target", "label_source", "tags", "is_contract"],
            )
            writer.writeheader()
            labels = {addr: 1 for addr in rows}
            split = split_labels(sorted(labels), labels, 0.70, 0.15, 42) if rows else {}
            for address in sorted(rows):
                item = rows[address]
                writer.writerow(
                    {
                        "node_id": address,
                        "label": 1,
                        "split": split.get(address, ""),
                        "is_target": 1,
                        "label_source": "|".join(sorted(item["sources"])),
                        "tags": "|".join(sorted(item["tags"])),
                        "is_contract": "|".join(sorted(item["is_contract_values"])),
                    }
                )

    write_address_rows(out_dir / "positive_addresses.csv", positives)
    write_address_rows(out_dir / "contract_creator_auxiliary.csv", creators)
    labels = {addr: 1 for addr in positives}
    summary = {
        "dataset": "Forta-Malicious",
        "raw_paths": {k: str(v) for k, v in raw_paths.items()},
        "include_creators_as_positive": include_creators_as_positive,
        "counts": {str(k): int(v) for k, v in counts.items()},
        "positive_addresses": len(positives),
        "creator_auxiliary_addresses": len(creators),
        "positive_path": str(out_dir / "positive_addresses.csv"),
        "creator_auxiliary_path": str(out_dir / "contract_creator_auxiliary.csv"),
    }
    json_dump(out_dir / "positive_summary.json", summary)
    return labels, summary


def write_global_positives(path: Path, source_map: dict[str, set[str]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["node_id", "sources"])
        writer.writeheader()
        for address in sorted(source_map):
            writer.writerow({"node_id": address, "sources": "|".join(sorted(source_map[address]))})
    return {"path": str(path), "addresses": len(source_map)}


def write_normal_candidates(
    path: Path,
    normal_source_map: dict[str, set[str]],
    global_positive: set[str],
    progress_every: int,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    removed = 0
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["node_id", "label", "is_target", "label_source"])
        writer.writeheader()
        for address in sorted(normal_source_map):
            if address in global_positive:
                removed += 1
                continue
            writer.writerow(
                {
                    "node_id": address,
                    "label": 0,
                    "is_target": 1,
                    "label_source": "reused_normal:" + "|".join(sorted(normal_source_map[address])),
                }
            )
            kept += 1
            if progress_every and kept % progress_every == 0:
                print(json.dumps({"phase": "write_normal_candidates", "kept": kept, "removed": removed}, ensure_ascii=False), flush=True)
    return {"path": str(path), "kept": kept, "removed_positive_overlap": removed}


def read_labels_file(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        yield from csv.DictReader(fh)


def write_combined_labels(forta_dir: Path, train_ratio: float, val_ratio: float, seed: int, progress_every: int) -> dict[str, Any]:
    rows: list[dict[str, str]] = []
    labels: dict[str, int] = {}
    for path in [forta_dir / "positive_addresses.csv", forta_dir / "normal_candidates.csv"]:
        for row in read_labels_file(path):
            address = norm_addr(row.get("node_id"))
            label = safe_int(row.get("label"), -1)
            if not address or label not in (0, 1):
                continue
            labels[address] = label
            rows.append(
                {
                    "node_id": address,
                    "label": str(label),
                    "is_target": "1",
                    "label_source": row.get("label_source", ""),
                    "tags": row.get("tags", ""),
                    "is_contract": row.get("is_contract", ""),
                }
            )
            if progress_every and len(rows) % progress_every == 0:
                print(json.dumps({"phase": "combine_forta_labels", "rows": len(rows)}, ensure_ascii=False), flush=True)
    split = split_labels(sorted(labels), labels, train_ratio, val_ratio, seed)
    out = forta_dir / "labels.csv"
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["node_id", "label", "split", "is_target", "label_source", "tags", "is_contract"])
        writer.writeheader()
        for row in rows:
            row["split"] = split[row["node_id"]]
            writer.writerow(row)
    counts = Counter(labels.values())
    return {"path": str(out), "rows": len(rows), "label_counts": {str(k): int(v) for k, v in counts.items()}}


def write_fetch_tasks(path: Path, dataset: str, labels_path: Path, to_block: int | None, progress_every: int) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    counts: Counter = Counter()
    with labels_path.open("r", encoding="utf-8", errors="replace", newline="") as in_fh, path.open("w", encoding="utf-8", newline="") as out_fh:
        reader = csv.DictReader(in_fh)
        writer = csv.DictWriter(
            out_fh,
            fieldnames=["dataset", "chain", "address", "label", "role", "from_block", "to_block", "api_path"],
        )
        writer.writeheader()
        for row in reader:
            address = norm_addr(row.get("node_id"))
            label = safe_int(row.get("label"), -1)
            if not address or label not in (0, 1):
                continue
            role = "positive" if label == 1 else "reused_normal"
            writer.writerow(
                {
                    "dataset": dataset,
                    "chain": "eth",
                    "address": address,
                    "label": label,
                    "role": role,
                    "from_block": "",
                    "to_block": "" if to_block is None else str(to_block),
                    "api_path": "/api/public/v1/address/transactions",
                }
            )
            rows += 1
            counts[str(label)] += 1
            if progress_every and rows % progress_every == 0:
                print(json.dumps({"phase": "write_fetch_tasks", "dataset": dataset, "rows": rows}, ensure_ascii=False), flush=True)
    return {"path": str(path), "rows": rows, "label_counts": {str(k): int(v) for k, v in counts.items()}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare label pools and platform API fetch tasks for Ethereum fraud datasets.")
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--progress-every", type=int, default=500000)
    parser.add_argument("--to-block", type=int, default=None)
    parser.add_argument("--include-forta-creators-as-positive", action="store_true")
    parser.add_argument(
        "--normal-source",
        action="append",
        default=[],
        help="Reusable normal source as NAME=/path/to/standard_dataset_dir. Defaults to PhishTrans and MulDiGraph.",
    )
    parser.add_argument(
        "--positive-source",
        action="append",
        default=[],
        help="Extra positive source as NAME=/path/to/standard_dataset_dir. Defaults to PhishTrans and MulDiGraph.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    project_root = Path(args.project_root).resolve()
    out_root = Path(args.out_dir).resolve() if args.out_dir else project_root / "data" / "prepared"
    raw_dir = out_root / "_raw"
    out_root.mkdir(parents=True, exist_ok=True)

    default_sources = {
        "PhishTrans": project_root / "data" / "researchh" / "PhishTrans",
        "MulDiGraph": project_root / "data" / "researchh" / "MulDiGraph",
    }
    normal_sources = parse_source_arg(args.normal_source) if args.normal_source else default_sources
    positive_sources = parse_source_arg(args.positive_source) if args.positive_source else default_sources

    print(
        json.dumps(
            {
                "phase": "start",
                "project_root": str(project_root),
                "out_root": str(out_root),
                "normal_sources": {k: str(v) for k, v in normal_sources.items()},
                "positive_sources": {k: str(v) for k, v in positive_sources.items()},
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    global_positive_sources: dict[str, set[str]] = defaultdict(set)
    normal_source_map: dict[str, set[str]] = defaultdict(set)
    source_summaries: dict[str, Any] = {}

    all_standard_names = sorted(set(normal_sources) | set(positive_sources))
    for name in all_standard_names:
        data_dir = positive_sources.get(name) or normal_sources.get(name)
        assert data_dir is not None
        positives, normals, stats = read_standard_labels(data_dir, name, args.progress_every)
        source_summaries[name] = {"path": str(data_dir), "stats": {str(k): int(v) for k, v in stats.items()}}
        if name in positive_sources:
            for address in positives:
                global_positive_sources[address].add(name)
        if name in normal_sources:
            for address in normals:
                normal_source_map[address].add(name)

    illicit = prepare_illicit_eth(raw_dir, out_root / "illicit_eth", args.download, args.train_ratio, args.val_ratio, args.seed)
    for address, label in illicit["labels"].items():
        if label == 1:
            global_positive_sources[address].add("Illicit-ETH")
        elif "Illicit-ETH" in normal_sources:
            normal_source_map[address].add("Illicit-ETH")

    forta_labels, forta_summary = prepare_forta_positives(
        raw_dir,
        out_root / "forta_malicious",
        args.download,
        args.include_forta_creators_as_positive,
    )
    for address in forta_labels:
        global_positive_sources[address].add("Forta-Malicious")

    global_positive_summary = write_global_positives(out_root / "global_positive_addresses.csv", global_positive_sources)
    normal_summary = write_normal_candidates(
        out_root / "forta_malicious" / "normal_candidates.csv",
        normal_source_map,
        set(global_positive_sources),
        args.progress_every,
    )
    forta_labels_summary = write_combined_labels(
        out_root / "forta_malicious",
        args.train_ratio,
        args.val_ratio,
        args.seed,
        args.progress_every,
    )

    illicit_tasks = write_fetch_tasks(
        out_root / "illicit_eth" / "address_fetch_tasks.csv",
        "Illicit-ETH",
        out_root / "illicit_eth" / "labels.csv",
        args.to_block,
        args.progress_every,
    )
    forta_tasks = write_fetch_tasks(
        out_root / "forta_malicious" / "address_fetch_tasks.csv",
        "Forta-Malicious",
        out_root / "forta_malicious" / "labels.csv",
        args.to_block,
        args.progress_every,
    )

    manifest = {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "project_root": str(project_root),
        "out_root": str(out_root),
        "to_block": args.to_block,
        "normal_reuse_rule": "Reuse label=0 target addresses from normal_sources after removing the global positive blacklist.",
        "global_positive_rule": "Union of positive labels from standard sources, Illicit-ETH, and Forta-Malicious.",
        "source_summaries": source_summaries,
        "illicit_eth": illicit["summary"],
        "forta_malicious": {
            "positive_summary": forta_summary,
            "normal_candidates": normal_summary,
            "labels": forta_labels_summary,
        },
        "global_positive_addresses": global_positive_summary,
        "fetch_tasks": {
            "Illicit-ETH": illicit_tasks,
            "Forta-Malicious": forta_tasks,
        },
        "elapsed_seconds": round(time.time() - started, 2),
    }
    json_dump(out_root / "manifest.json", manifest)
    print(json.dumps({"phase": "done", "manifest": str(out_root / "manifest.json"), **manifest}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

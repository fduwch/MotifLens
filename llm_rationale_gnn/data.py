from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data


ID_COLUMNS = {"node_id", "id", "address", "account", "addr"}
SRC_COLUMNS = ("src", "source", "from", "from_address", "sender")
DST_COLUMNS = ("dst", "target", "to", "to_address", "receiver")
LABEL_COLUMNS = ("label", "y", "is_fraud", "is_phisher", "class")
SPLIT_COLUMNS = ("split", "mask")
TIME_COLUMNS = ("timestamp", "time", "datetime", "block_timestamp")
AMOUNT_COLUMNS = ("amount", "value", "eth_value", "value_eth", "ether")
CONTROL_NODE_COLUMNS = {"is_target"}
CONTROL_NODE_NAME_PATTERNS = (
    "label",
    "target",
    "split",
    "class",
    "fraud",
    "phish",
    "ponzi",
    "illicit",
    "source",
)
CONTROL_EDGE_NAME_PATTERNS = (
    "label",
    "root",
    "relation",
    "hop",
    "split",
    "target",
    "class",
    "fraud",
    "phish",
    "ponzi",
    "illicit",
)
DERIVED_NODE_FEATURE_COLUMNS = [
    "log_in_degree",
    "log_out_degree",
    "log_total_degree",
    "log_in_amount",
    "log_out_amount",
    "log_total_amount",
    "in_amount_ratio",
    "out_amount_ratio",
    "log_mean_in_amount",
    "log_mean_out_amount",
    "log_max_in_amount",
    "log_max_out_amount",
    "log_active_span",
    "log_in_span",
    "log_out_span",
    "log_in_rate",
    "log_out_rate",
]


@dataclass
class LoadedGraph:
    data: Data
    node_ids: np.ndarray
    edge_frame: pd.DataFrame
    feature_columns: list[str]
    edge_feature_columns: list[str]


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format: {path}")


def find_column(df: pd.DataFrame, candidates: Iterable[str], required: bool = True) -> Optional[str]:
    lowered = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    if required:
        raise ValueError(f"Could not find any column among {list(candidates)} in {list(df.columns)}")
    return None


def first_existing(data_dir: Path, names: Iterable[str]) -> Optional[Path]:
    for name in names:
        path = data_dir / name
        if path.exists():
            return path
    return None


def parse_time_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").astype("float64")
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    return (parsed.astype("int64") / 1e9).where(parsed.notna(), np.nan)


def choose_amount_column(edge_df: pd.DataFrame) -> Optional[str]:
    return find_column(edge_df, AMOUNT_COLUMNS, required=False)


def choose_time_column(edge_df: pd.DataFrame) -> Optional[str]:
    return find_column(edge_df, TIME_COLUMNS, required=False)


def make_node_index(nodes: pd.DataFrame, node_col: str) -> Tuple[Dict[str, int], np.ndarray]:
    node_ids = nodes[node_col].astype(str).to_numpy()
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    return node_to_idx, node_ids


def infer_nodes(edge_df: pd.DataFrame, src_col: str, dst_col: str) -> pd.DataFrame:
    ids = pd.concat([edge_df[src_col], edge_df[dst_col]], ignore_index=True).astype(str).drop_duplicates()
    return pd.DataFrame({"node_id": ids.to_numpy()})


def merge_labels(nodes: pd.DataFrame, data_dir: Path, node_col: str) -> pd.DataFrame:
    label_path = first_existing(data_dir, ("labels.csv", "labels.tsv", "label.csv", "label.tsv"))
    if label_path is None:
        return nodes
    labels = read_table(label_path)
    label_node_col = find_column(labels, ID_COLUMNS)
    label_col = find_column(labels, LABEL_COLUMNS)
    keep = [label_node_col, label_col]
    split_col = find_column(labels, SPLIT_COLUMNS, required=False)
    if split_col:
        keep.append(split_col)
    labels = labels[keep].rename(columns={label_node_col: node_col})
    if label_col in nodes.columns:
        return nodes
    return nodes.merge(labels, on=node_col, how="left")


def is_control_node_feature_column(column: str) -> bool:
    name = str(column).strip().lower()
    return any(pattern in name for pattern in CONTROL_NODE_NAME_PATTERNS)


def is_control_edge_feature_column(column: str) -> bool:
    name = str(column).strip().lower()
    return any(pattern in name for pattern in CONTROL_EDGE_NAME_PATTERNS)


def numeric_feature_columns(df: pd.DataFrame, excluded: set[str]) -> list[str]:
    cols: list[str] = []
    for col in df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def numeric_node_feature_columns(df: pd.DataFrame, excluded: set[str]) -> list[str]:
    cols: list[str] = []
    for col in numeric_feature_columns(df, excluded):
        if not is_control_node_feature_column(col):
            cols.append(col)
    return cols


def numeric_edge_feature_columns(df: pd.DataFrame, excluded: set[str]) -> list[str]:
    cols: list[str] = []
    for col in numeric_feature_columns(df, excluded):
        if not is_control_edge_feature_column(col):
            cols.append(col)
    return cols


def should_log_scale_edge_column(column: str) -> bool:
    name = str(column).lower()
    keywords = (
        "amount",
        "value",
        "gas",
        "price",
        "timestamp",
        "time",
        "block",
    )
    return any(keyword in name for keyword in keywords)


def numeric_matrix(df: pd.DataFrame, columns: list[str], log_scale_edges: bool = False) -> np.ndarray:
    if not columns:
        return np.empty((len(df), 0), dtype=np.float32)
    arrays: list[np.ndarray] = []
    for column in columns:
        values = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        arr = values.to_numpy(dtype=np.float64)
        if log_scale_edges and should_log_scale_edge_column(column):
            arr = np.log1p(np.clip(arr, 0.0, None))
        arrays.append(arr)
    return np.column_stack(arrays).astype(np.float32)


def derive_node_features(edge_df: pd.DataFrame, num_nodes: int) -> np.ndarray:
    src = edge_df["src_idx"].to_numpy()
    dst = edge_df["dst_idx"].to_numpy()
    amount = edge_df.get("amount_num", pd.Series(np.ones(len(edge_df)))).fillna(0).to_numpy(dtype=np.float64)
    amount = np.maximum(amount, 0)

    out_degree = np.bincount(src, minlength=num_nodes).astype(np.float64)
    in_degree = np.bincount(dst, minlength=num_nodes).astype(np.float64)
    out_amount = np.bincount(src, weights=amount, minlength=num_nodes).astype(np.float64)
    in_amount = np.bincount(dst, weights=amount, minlength=num_nodes).astype(np.float64)
    total_degree = in_degree + out_degree
    total_amount = in_amount + out_amount
    mean_in_amount = np.divide(in_amount, in_degree + 1e-9)
    mean_out_amount = np.divide(out_amount, out_degree + 1e-9)

    max_in_amount = np.zeros(num_nodes, dtype=np.float64)
    max_out_amount = np.zeros(num_nodes, dtype=np.float64)
    np.maximum.at(max_in_amount, dst, amount)
    np.maximum.at(max_out_amount, src, amount)

    active_span = np.zeros(num_nodes, dtype=np.float64)
    in_span = np.zeros(num_nodes, dtype=np.float64)
    out_span = np.zeros(num_nodes, dtype=np.float64)
    if "timestamp_num" in edge_df.columns:
        ts = edge_df["timestamp_num"].fillna(0).to_numpy(dtype=np.float64)
        valid = ts > 0
        if valid.any():
            big = np.float64(np.nanmax(ts[valid]) + 1.0)

            first_in = np.full(num_nodes, big, dtype=np.float64)
            last_in = np.zeros(num_nodes, dtype=np.float64)
            first_out = np.full(num_nodes, big, dtype=np.float64)
            last_out = np.zeros(num_nodes, dtype=np.float64)

            np.minimum.at(first_in, dst[valid], ts[valid])
            np.maximum.at(last_in, dst[valid], ts[valid])
            np.minimum.at(first_out, src[valid], ts[valid])
            np.maximum.at(last_out, src[valid], ts[valid])

            has_in = first_in < big
            has_out = first_out < big
            in_span[has_in] = np.maximum(last_in[has_in] - first_in[has_in], 0.0)
            out_span[has_out] = np.maximum(last_out[has_out] - first_out[has_out], 0.0)

            first_seen = np.minimum(first_in, first_out)
            last_seen = np.maximum(last_in, last_out)
            has_any = first_seen < big
            active_span[has_any] = np.maximum(last_seen[has_any] - first_seen[has_any], 0.0)

    day = 86400.0
    in_rate = np.divide(in_degree, in_span / day + 1.0)
    out_rate = np.divide(out_degree, out_span / day + 1.0)

    features = np.stack(
        [
            np.log1p(in_degree),
            np.log1p(out_degree),
            np.log1p(total_degree),
            np.log1p(in_amount),
            np.log1p(out_amount),
            np.log1p(total_amount),
            np.divide(in_amount, total_amount + 1e-9),
            np.divide(out_amount, total_amount + 1e-9),
            np.log1p(mean_in_amount),
            np.log1p(mean_out_amount),
            np.log1p(max_in_amount),
            np.log1p(max_out_amount),
            np.log1p(active_span / day),
            np.log1p(in_span / day),
            np.log1p(out_span / day),
            np.log1p(in_rate),
            np.log1p(out_rate),
        ],
        axis=1,
    )
    return features


def standardize(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    values[~np.isfinite(values)] = 0
    if values.size == 0:
        return values
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (values - mean) / std


def build_masks(labels: np.ndarray, split_values: Optional[np.ndarray], seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid = np.flatnonzero(labels >= 0)
    train_mask = np.zeros(len(labels), dtype=bool)
    val_mask = np.zeros(len(labels), dtype=bool)
    test_mask = np.zeros(len(labels), dtype=bool)

    if split_values is not None:
        split = np.array([str(x).lower() for x in split_values])
        train_mask = split == "train"
        val_mask = np.isin(split, ["val", "valid", "validation"])
        test_mask = split == "test"
        return torch.tensor(train_mask), torch.tensor(val_mask), torch.tensor(test_mask)

    if len(valid) == 0:
        raise ValueError("No labels found. Provide labels in nodes.csv or labels.csv.")

    y_valid = labels[valid]
    stratify = y_valid if len(np.unique(y_valid)) > 1 and min(np.bincount(y_valid.astype(int))) >= 2 else None
    train_idx, temp_idx = train_test_split(valid, test_size=0.30, random_state=seed, stratify=stratify)
    temp_y = labels[temp_idx]
    stratify_temp = temp_y if len(np.unique(temp_y)) > 1 and min(np.bincount(temp_y.astype(int))) >= 2 else None
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, random_state=seed, stratify=stratify_temp)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    return torch.tensor(train_mask), torch.tensor(val_mask), torch.tensor(test_mask)


def load_graph(data_dir: str | Path, seed: int = 42, build_edge_attr: bool = True, use_split_column: bool = True) -> LoadedGraph:
    data_dir = Path(data_dir)
    edge_path = first_existing(data_dir, ("edges.csv", "edges.tsv", "edge.csv", "edge.tsv"))
    if edge_path is None:
        raise FileNotFoundError(f"No edges.csv/edges.tsv found under {data_dir}")

    edge_df = read_table(edge_path)
    src_col = find_column(edge_df, SRC_COLUMNS)
    dst_col = find_column(edge_df, DST_COLUMNS)

    node_path = first_existing(data_dir, ("nodes.csv", "nodes.tsv", "node.csv", "node.tsv"))
    if node_path is not None:
        nodes = read_table(node_path)
        node_col = find_column(nodes, ID_COLUMNS)
    else:
        nodes = infer_nodes(edge_df, src_col, dst_col)
        node_col = "node_id"

    nodes[node_col] = nodes[node_col].astype(str)
    nodes = merge_labels(nodes, data_dir, node_col)
    node_to_idx, node_ids = make_node_index(nodes, node_col)

    edge_df = edge_df.copy()
    edge_df[src_col] = edge_df[src_col].astype(str)
    edge_df[dst_col] = edge_df[dst_col].astype(str)
    edge_df = edge_df[edge_df[src_col].isin(node_to_idx) & edge_df[dst_col].isin(node_to_idx)].reset_index(drop=True)
    edge_df["edge_pos"] = np.arange(len(edge_df), dtype=np.int64)
    edge_df["src_idx"] = edge_df[src_col].map(node_to_idx).astype(np.int64)
    edge_df["dst_idx"] = edge_df[dst_col].map(node_to_idx).astype(np.int64)

    amount_col = choose_amount_column(edge_df)
    if amount_col is not None:
        edge_df["amount_num"] = pd.to_numeric(edge_df[amount_col], errors="coerce").fillna(0.0)
    else:
        edge_df["amount_num"] = 1.0

    time_col = choose_time_column(edge_df)
    if time_col is not None:
        edge_df["timestamp_num"] = parse_time_series(edge_df[time_col]).fillna(0.0)

    label_col = find_column(nodes, LABEL_COLUMNS, required=False)
    if label_col is None:
        labels = np.full(len(nodes), -1, dtype=np.int64)
    else:
        labels = pd.to_numeric(nodes[label_col], errors="coerce").fillna(-1).astype(np.int64).to_numpy()

    split_col = find_column(nodes, SPLIT_COLUMNS, required=False)
    split_values = nodes[split_col].to_numpy() if split_col and use_split_column else None

    excluded_node = {node_col, *CONTROL_NODE_COLUMNS}
    if label_col:
        excluded_node.add(label_col)
    if split_col:
        excluded_node.add(split_col)
    feature_columns = numeric_node_feature_columns(nodes, excluded_node)
    derived = derive_node_features(edge_df, len(nodes))
    if feature_columns:
        x = np.concatenate([numeric_matrix(nodes, feature_columns), derived], axis=1)
        feature_columns = feature_columns + DERIVED_NODE_FEATURE_COLUMNS
    else:
        x = derived
        feature_columns = DERIVED_NODE_FEATURE_COLUMNS.copy()
    x = standardize(x)

    excluded_edge = {src_col, dst_col}
    edge_feature_columns = numeric_edge_feature_columns(edge_df, excluded_edge)
    edge_feature_columns = [
        c for c in edge_feature_columns if c not in {"src_idx", "dst_idx", "edge_pos", "raw_value"}
    ]
    edge_attr = None
    if build_edge_attr and edge_feature_columns:
        edge_attr_arr = numeric_matrix(edge_df, edge_feature_columns, log_scale_edges=True)
        edge_attr = torch.tensor(standardize(edge_attr_arr), dtype=torch.float32)

    edge_index = torch.tensor(edge_df[["src_idx", "dst_idx"]].to_numpy().T, dtype=torch.long)
    y = torch.tensor(labels, dtype=torch.long)
    train_mask, val_mask, test_mask = build_masks(labels, split_values, seed)

    data = Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
    )
    data.node_ids = node_ids
    data.edge_pos = torch.tensor(edge_df["edge_pos"].to_numpy(), dtype=torch.long)
    return LoadedGraph(data=data, node_ids=node_ids, edge_frame=edge_df, feature_columns=feature_columns, edge_feature_columns=edge_feature_columns)


def load_evidence(evidence_path: str | Path, num_edges: int) -> tuple[torch.Tensor, list[str]]:
    path = Path(evidence_path)
    df = read_table(path)
    if "edge_pos" not in df.columns:
        raise ValueError("Evidence file must contain edge_pos.")
    if "evidence_score" not in df.columns:
        raise ValueError("Evidence file must contain evidence_score.")

    score = torch.zeros(num_edges, dtype=torch.float32)
    pos = pd.to_numeric(df["edge_pos"], errors="coerce").fillna(-1).astype(int).to_numpy()
    valid = (pos >= 0) & (pos < num_edges)
    values = pd.to_numeric(df.loc[valid, "evidence_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0).to_numpy()
    score[torch.tensor(pos[valid], dtype=torch.long)] = torch.tensor(values, dtype=torch.float32)
    evidence_types = sorted(str(x) for x in df.get("evidence_type", pd.Series(dtype=str)).dropna().unique())
    return score, evidence_types


def append_node_evidence_features(data: Data, edge_scores: torch.Tensor) -> Data:
    num_nodes = data.num_nodes
    edge_index = data.edge_index
    src = edge_index[0]
    dst = edge_index[1]
    device = data.x.device
    scores = edge_scores.to(device)

    def scatter_sum(index: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(num_nodes, device=device, dtype=torch.float32)
        out.scatter_add_(0, index, values)
        return out

    in_sum = scatter_sum(dst, scores)
    out_sum = scatter_sum(src, scores)
    in_count = scatter_sum(dst, (scores > 0).float())
    out_count = scatter_sum(src, (scores > 0).float())

    in_max = torch.zeros(num_nodes, device=device, dtype=torch.float32)
    out_max = torch.zeros(num_nodes, device=device, dtype=torch.float32)
    if hasattr(in_max, "scatter_reduce_"):
        in_max.scatter_reduce_(0, dst, scores, reduce="amax", include_self=False)
        out_max.scatter_reduce_(0, src, scores, reduce="amax", include_self=False)

    extra = torch.stack(
        [
            torch.log1p(in_sum),
            torch.log1p(out_sum),
            torch.log1p(in_count),
            torch.log1p(out_count),
            in_max,
            out_max,
        ],
        dim=1,
    )
    data.x = torch.cat([data.x, extra], dim=1)
    return data


def _read_evidence_chunks(path: Path, chunksize: int) -> Iterable[pd.DataFrame]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        yield from pd.read_csv(path, chunksize=chunksize)
        return
    if suffix == ".tsv":
        yield from pd.read_csv(path, sep="\t", chunksize=chunksize)
        return
    yield read_table(path)


def append_typed_node_evidence_features(
    data: Data,
    evidence_path: str | Path,
    evidence_types: Sequence[str],
    polarities: Sequence[str] = ("support", "counter"),
    chunksize: int = 1_000_000,
) -> tuple[Data, list[str], dict[str, int]]:
    """Append motif/polarity-aware node evidence features.

    For each evidence type and polarity, this adds the same six auditable
    aggregates used by the scalar evidence path: in/out sum, count, and max.
    """
    path = Path(evidence_path)
    if not evidence_types:
        raise ValueError("evidence_types cannot be empty for typed evidence features.")

    type_names = [str(x) for x in evidence_types]
    polarity_names = [str(x).lower() for x in polarities]
    channel_keys = [(typ, polarity) for typ in type_names for polarity in polarity_names]
    channel_to_idx = {key: idx for idx, key in enumerate(channel_keys)}
    channels = len(channel_keys)

    num_nodes = int(data.num_nodes)
    edge_index = data.edge_index.cpu()
    src_all = edge_index[0]
    dst_all = edge_index[1]

    in_sum = torch.zeros((num_nodes, channels), dtype=torch.float32)
    out_sum = torch.zeros((num_nodes, channels), dtype=torch.float32)
    in_count = torch.zeros((num_nodes, channels), dtype=torch.float32)
    out_count = torch.zeros((num_nodes, channels), dtype=torch.float32)
    in_max = torch.zeros((num_nodes, channels), dtype=torch.float32)
    out_max = torch.zeros((num_nodes, channels), dtype=torch.float32)

    total_rows = 0
    # Deduplicate by edge and channel before node aggregation. The same on-chain
    # edge may be observed from multiple target-address prompts; counting that
    # multiplicity would leak target-context density rather than motif evidence.
    dedup_scores: dict[tuple[int, int], float] = {}

    for chunk in _read_evidence_chunks(path, max(1, int(chunksize))):
        total_rows += int(len(chunk))
        required = {"edge_pos", "evidence_score", "evidence_type"}
        missing = required.difference(chunk.columns)
        if missing:
            raise ValueError(f"Evidence file missing columns for typed features: {sorted(missing)}")

        if "polarity" not in chunk.columns:
            chunk = chunk.assign(polarity="support")

        pos = pd.to_numeric(chunk["edge_pos"], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
        scores = pd.to_numeric(chunk["evidence_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0).to_numpy(dtype=np.float32)
        types = chunk["evidence_type"].astype(str).to_numpy()
        pols = chunk["polarity"].astype(str).str.lower().to_numpy()

        valid = (pos >= 0) & (pos < int(data.num_edges)) & (scores > 0)
        if not valid.any():
            continue

        for edge_pos, score, typ, polarity in zip(pos[valid], scores[valid], types[valid], pols[valid]):
            channel = channel_to_idx.get((str(typ), str(polarity).lower()))
            if channel is None:
                continue
            key = (int(edge_pos), int(channel))
            old_score = dedup_scores.get(key)
            if old_score is None or float(score) > old_score:
                dedup_scores[key] = float(score)

    used_rows = len(dedup_scores)
    positive_edges = {edge_pos for edge_pos, _ in dedup_scores}
    type_counts = {typ: 0 for typ in type_names}
    polarity_counts = {polarity: 0 for polarity in polarity_names}

    by_channel: dict[int, list[tuple[int, float]]] = {idx: [] for idx in range(channels)}
    for (edge_pos, channel), score in dedup_scores.items():
        by_channel[channel].append((edge_pos, score))

    for (typ, polarity), channel in channel_to_idx.items():
        entries = by_channel.get(channel, [])
        if not entries:
            continue
        edge_pos = torch.tensor([edge for edge, _ in entries], dtype=torch.long)
        values = torch.tensor([score for _, score in entries], dtype=torch.float32)
        src = src_all[edge_pos]
        dst = dst_all[edge_pos]

        in_sum[:, channel].scatter_add_(0, dst, values)
        out_sum[:, channel].scatter_add_(0, src, values)
        ones = torch.ones_like(values)
        in_count[:, channel].scatter_add_(0, dst, ones)
        out_count[:, channel].scatter_add_(0, src, ones)
        if hasattr(in_max[:, channel], "scatter_reduce_"):
            in_max[:, channel].scatter_reduce_(0, dst, values, reduce="amax", include_self=True)
            out_max[:, channel].scatter_reduce_(0, src, values, reduce="amax", include_self=True)

        count = int(len(entries))
        type_counts[typ] += count
        polarity_counts[polarity] += count

    feature_blocks = [
        ("in_sum", torch.log1p(in_sum)),
        ("out_sum", torch.log1p(out_sum)),
        ("in_count", torch.log1p(in_count)),
        ("out_count", torch.log1p(out_count)),
        ("in_max", in_max),
        ("out_max", out_max),
    ]
    extra = torch.cat([block for _, block in feature_blocks], dim=1).to(data.x.device)
    feature_names = [
        f"typed_evidence_{name}_{typ}_{polarity}"
        for name, _ in feature_blocks
        for typ, polarity in channel_keys
    ]
    data.x = torch.cat([data.x, extra], dim=1)
    stats = {
        "typed_evidence_rows_scanned": int(total_rows),
        "typed_evidence_rows_used": int(used_rows),
        "typed_evidence_edges": int(len(positive_edges)),
        "typed_evidence_channels": int(channels),
        "typed_evidence_features": int(extra.size(1)),
        "typed_evidence_type_counts": type_counts,
        "typed_evidence_polarity_counts": polarity_counts,
    }
    return data, feature_names, stats


def augment_training_edges(
    data: Data,
    edge_scores: torch.Tensor,
    add_reverse_edges: bool = False,
    add_self_loops: bool = False,
) -> tuple[Data, torch.Tensor]:
    """Add training-only reverse edges and explicit self loops.

    This keeps the converted dataset unchanged while allowing node classifiers to
    aggregate outgoing behavior and own-node features. Evidence scores are
    duplicated for reverse edges and set to zero for self loops.
    """
    if not add_reverse_edges and not add_self_loops:
        return data, edge_scores

    edge_indices = [data.edge_index]
    score_parts = [edge_scores]
    edge_attr = data.edge_attr
    attr_parts: list[torch.Tensor] | None = [] if edge_attr is not None or add_reverse_edges else None

    if attr_parts is not None:
        if edge_attr is None:
            base_attr = torch.zeros((data.num_edges, 1), dtype=torch.float32)
            attr_parts.append(torch.zeros_like(base_attr))
        elif add_reverse_edges:
            reverse_flag = torch.zeros((edge_attr.size(0), 1), dtype=edge_attr.dtype)
            attr_parts.append(torch.cat([edge_attr, reverse_flag], dim=1))
        else:
            attr_parts.append(edge_attr)

    if add_reverse_edges:
        edge_indices.append(data.edge_index.flip(0))
        score_parts.append(edge_scores.clone())
        if attr_parts is not None:
            if edge_attr is None:
                attr_parts.append(torch.ones((data.num_edges, 1), dtype=torch.float32))
            else:
                reverse_flag = torch.ones((edge_attr.size(0), 1), dtype=edge_attr.dtype)
                attr_parts.append(torch.cat([edge_attr, reverse_flag], dim=1))

    if add_self_loops:
        loops = torch.arange(data.num_nodes, dtype=torch.long)
        edge_indices.append(torch.stack([loops, loops], dim=0))
        score_parts.append(torch.zeros(data.num_nodes, dtype=edge_scores.dtype))
        if attr_parts is not None:
            attr_dim = attr_parts[0].size(1)
            attr_parts.append(torch.zeros((data.num_nodes, attr_dim), dtype=attr_parts[0].dtype))

    data.edge_index = torch.cat(edge_indices, dim=1)
    if attr_parts is not None:
        data.edge_attr = torch.cat(attr_parts, dim=0)
    data.edge_pos = torch.arange(data.edge_index.size(1), dtype=torch.long)
    return data, torch.cat(score_parts, dim=0)

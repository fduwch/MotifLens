from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GraphSAGE, GCNConv


class RationaleGAT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        hidden_channels: int = 128,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.30,
        edge_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be >= 1")
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(
            GATConv(
                in_channels,
                hidden_channels,
                heads=heads,
                dropout=dropout,
                edge_dim=edge_dim,
                add_self_loops=False,
                concat=True,
            )
        )
        for _ in range(layers - 1):
            self.convs.append(
                GATConv(
                    hidden_channels * heads,
                    hidden_channels,
                    heads=heads,
                    dropout=dropout,
                    edge_dim=edge_dim,
                    add_self_loops=False,
                    concat=True,
                )
            )
        self.classifier = nn.Linear(hidden_channels * heads, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None, return_attention: bool = False):
        last_alpha = None
        for conv in self.convs:
            x = F.dropout(x, p=self.dropout, training=self.training)
            if return_attention:
                x, (_, alpha) = conv(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
                last_alpha = alpha
            else:
                x = conv(x, edge_index, edge_attr=edge_attr)
            x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        logits = self.classifier(x)
        if return_attention:
            return logits, last_alpha
        return logits


class BaselineGCN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, hidden_channels: int = 128, layers: int = 2, dropout: float = 0.30) -> None:
        super().__init__()
        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None, return_attention: bool = False):
        del edge_attr
        for conv in self.convs:
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = F.relu(conv(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        logits = self.classifier(x)
        if return_attention:
            return logits, None
        return logits


class BaselineSAGE(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, hidden_channels: int = 128, layers: int = 2, dropout: float = 0.30) -> None:
        super().__init__()
        self.dropout = dropout
        self.encoder = GraphSAGE(in_channels, hidden_channels, num_layers=layers, out_channels=hidden_channels, dropout=dropout)
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None, return_attention: bool = False):
        del edge_attr
        x = self.encoder(x, edge_index)
        logits = self.classifier(x)
        if return_attention:
            return logits, None
        return logits


class EvidenceGatedSAGE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        evidence_start: int,
        evidence_dim: int,
        hidden_channels: int = 128,
        layers: int = 2,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        if evidence_start <= 0:
            raise ValueError("evidence_start must split base and evidence features.")
        if evidence_dim <= 0:
            raise ValueError("evidence_dim must be positive for EvidenceGatedSAGE.")
        if evidence_start + evidence_dim > in_channels:
            raise ValueError("evidence feature slice exceeds input dimension.")

        self.dropout = dropout
        self.evidence_start = evidence_start
        self.evidence_dim = evidence_dim
        self.base_encoder = GraphSAGE(evidence_start, hidden_channels, num_layers=layers, out_channels=hidden_channels, dropout=dropout)
        self.evidence_encoder = nn.Sequential(
            nn.Linear(evidence_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.Sigmoid(),
        )
        self.classifier = nn.Linear(hidden_channels, num_classes)
        self.last_gate: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None, return_attention: bool = False):
        del edge_attr
        x_base = x[:, : self.evidence_start]
        x_evidence = x[:, self.evidence_start : self.evidence_start + self.evidence_dim]
        h_base = self.base_encoder(x_base, edge_index)
        h_evidence = self.evidence_encoder(x_evidence)
        gate = self.gate(torch.cat([h_base, h_evidence], dim=-1))
        self.last_gate = gate.detach()
        h = h_base + gate * h_evidence
        h = F.dropout(h, p=self.dropout, training=self.training)
        logits = self.classifier(h)
        if return_attention:
            return logits, gate.mean(dim=1)
        return logits


def scatter_sum(values: torch.Tensor, index: torch.Tensor, size: int) -> torch.Tensor:
    out = torch.zeros(size, dtype=values.dtype, device=values.device)
    out.scatter_add_(0, index, values)
    return out


def edge_rationale_kl_loss(alpha: Optional[torch.Tensor], edge_index: torch.Tensor, evidence_score: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if alpha is None:
        return torch.tensor(0.0, device=edge_index.device)
    if alpha.dim() == 2:
        attention = alpha.mean(dim=1)
    else:
        attention = alpha

    score = evidence_score.to(edge_index.device).float().clamp(0.0, 1.0)
    dst = edge_index[1]
    score_sum = scatter_sum(score, dst, num_nodes)
    valid_edge = score_sum[dst] > 0
    if valid_edge.sum() == 0:
        return torch.tensor(0.0, device=edge_index.device)

    target = score[valid_edge] / (score_sum[dst[valid_edge]] + 1e-12)
    pred = attention[valid_edge].clamp_min(1e-12)
    loss = -(target * torch.log(pred)).sum()
    valid_nodes = (score_sum > 0).sum().clamp_min(1)
    return loss / valid_nodes


def build_model(
    name: str,
    in_channels: int,
    num_classes: int,
    hidden_channels: int,
    layers: int,
    heads: int,
    dropout: float,
    edge_dim: Optional[int],
    evidence_start: Optional[int] = None,
    evidence_dim: Optional[int] = None,
):
    name = name.lower()
    if name == "rationale_gat":
        return RationaleGAT(in_channels, num_classes, hidden_channels, layers, heads, dropout, edge_dim)
    if name == "gcn":
        return BaselineGCN(in_channels, num_classes, hidden_channels, layers, dropout)
    if name == "sage":
        return BaselineSAGE(in_channels, num_classes, hidden_channels, layers, dropout)
    if name == "evidence_gated_sage":
        if evidence_start is None or evidence_dim is None:
            raise ValueError("evidence_gated_sage requires evidence_start and evidence_dim.")
        return EvidenceGatedSAGE(in_channels, num_classes, evidence_start, evidence_dim, hidden_channels, layers, dropout)
    raise ValueError(f"Unknown model: {name}")

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv, GCNConv


class StationGNN(nn.Module):
    """
    GCNConv (слой 1) + GATv2Conv с edge_attr (слой 2) + residual linear skip.
    """

    def __init__(
        self,
        node_in_dim: int,
        gnn_hidden: int = 64,
        emb_dim: int = 32,
        dropout: float = 0.1,
        heads: int = 4,
    ):
        super().__init__()
        self.conv1 = GCNConv(node_in_dim, gnn_hidden)
        self.conv2 = GATv2Conv(
            in_channels=gnn_hidden,
            out_channels=emb_dim,
            heads=heads,
            concat=False,
            dropout=dropout,
            edge_dim=1,
        )
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Linear(node_in_dim, emb_dim) if node_in_dim != emb_dim else nn.Identity()
        self.skip_gate = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x:           [N, node_in_dim]
        edge_index:  [2, E]
        edge_weight: [E]  — расстояния или веса (обязательно)
        Returns:     [N, emb_dim]
        """
        if edge_weight is None or edge_weight.numel() == 0:
            raise ValueError("edge_weight (distance/weight) обязателен для GATv2 edge_attr.")

        edge_attr = edge_weight.view(-1, 1).to(x.dtype)

        h = self.conv1(x, edge_index, edge_weight=edge_weight)
        h = F.relu(h)
        h = self.drop(h)
        z = self.conv2(h, edge_index, edge_attr=edge_attr)
        return z + self.skip_gate * self.skip(x)

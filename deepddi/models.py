from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch
from torch import nn
import torch.nn.functional as F

class MLPBackbone(nn.Module):

    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float, batch_norm: bool=True) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = input_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(current, hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden))
            current = hidden
        self.network = nn.Sequential(*layers)
        self.output_dim = current

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class MultiLabelMLP(nn.Module):

    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float, output_dim: int) -> None:
        super().__init__()
        self.backbone = MLPBackbone(input_dim, hidden_dims, dropout, batch_norm=True)
        self.head = nn.Linear(self.backbone.output_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

class OrdinalMLP(nn.Module):

    def __init__(self, input_dim: int, hidden_dims: list[int], dropout: float, n_classes: int) -> None:
        super().__init__()
        if n_classes < 3:
            raise ValueError('OrdinalMLP requires at least three classes.')
        self.n_classes = n_classes
        self.backbone = MLPBackbone(input_dim, hidden_dims, dropout, batch_norm=True)
        self.score = nn.Linear(self.backbone.output_dim, 1)
        self.threshold_base = nn.Parameter(torch.tensor(0.0))
        self.threshold_deltas = nn.Parameter(torch.zeros(n_classes - 2))

    def ordered_thresholds(self) -> torch.Tensor:
        if self.n_classes == 2:
            return self.threshold_base.view(1)
        deltas = F.softplus(self.threshold_deltas) + 0.0001
        return torch.cat([self.threshold_base.view(1), self.threshold_base.view(1) + torch.cumsum(deltas, dim=0)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        score = self.score(self.backbone(x))
        thresholds = self.ordered_thresholds().view(1, -1)
        return score - thresholds

def cumulative_logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    q = torch.sigmoid(logits)
    pieces = [1.0 - q[:, :1]]
    if q.shape[1] > 1:
        pieces.append(q[:, :-1] - q[:, 1:])
    pieces.append(q[:, -1:])
    probs = torch.cat(pieces, dim=1)
    probs = torch.clamp(probs, min=0.0)
    return probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)

def build_model(config: dict[str, Any], input_dim: int, output_dim: int) -> nn.Module:
    model_cfg = config['model']
    kind = model_cfg['kind']
    hidden_dims = [int(value) for value in model_cfg['hidden_dims']]
    dropout = float(model_cfg['dropout'])
    if kind == 'multilabel_mlp':
        return MultiLabelMLP(input_dim, hidden_dims, dropout, output_dim)
    if kind == 'ordinal_mlp':
        return OrdinalMLP(input_dim, hidden_dims, dropout, output_dim)
    raise ValueError(f'Unknown model kind: {kind}')

def model_metadata(model: nn.Module) -> dict[str, Any]:
    return {'parameter_count': sum((parameter.numel() for parameter in model.parameters())), 'trainable_parameter_count': sum((parameter.numel() for parameter in model.parameters() if parameter.requires_grad))}

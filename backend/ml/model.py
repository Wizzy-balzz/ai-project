"""
model.py
--------
GraphSAGE-based node-level binary classifier for attack involvement prediction.

=============================================================================
ARCHITECTURE DECISION RECORD (ADR)
=============================================================================

Choice: GraphSAGE (node features only) with edge_attr PRESERVED but NOT
        consumed by message passing.

Two options were evaluated:

Option A: Vanilla GraphSAGE (SAGEConv)
  - Uses node features x ∈ ℝ^{N×43} and edge_index for message passing.
  - Does NOT natively incorporate edge_attr in the aggregation step.
  - PyTorch Geometric SAGEConv does not accept edge_attr.

Option B: Edge-aware GNN (e.g., GATv2Conv, GeneralConv, NNConv)
  - GATv2Conv: attention-weighted aggregation; attention computed per edge,
    but attention coefficients do not directly use edge_attr values.
  - NNConv: passes edge_attr through a neural network to parameterise the
    aggregation kernel — directly uses edge features. However, it requires
    a fixed-size edge MLP which adds parameters and training complexity for
    only 9 edge features.
  - GeneralConv: supports edge_attr in a general message-passing framework.

DECISION: GraphSAGE (Option A)

Technical justification:
1. Node features (43 dim) already ENCODE edge-derived signals:
   - `attack_weighted_in_degree` (idx 40): sum of attack_weight on in-edges
   - `anomaly_signal` (idx 41): sum of anomaly_count on in-edges
   - `historical_attack_exposure` (idx 42): sum of hist_attack_count on in-edges
   These three features aggregate the most discriminative edge signals into
   node-level representations BEFORE the GNN, making a separate edge-MLP
   redundant for this phase.
2. With small synthetic graphs (10–60 nodes per scenario), a simpler model
   is less prone to overfitting than an edge-parameterised network.
3. GraphSAGE with mean aggregation is well-studied for inductive learning,
   which is exactly what we need: generalization to unseen topologies.
4. Edge features are PRESERVED in data.edge_attr for future phases
   (Phase 4 attack path scoring) where edge-level traversal probability
   is needed. Nothing is lost.

WHAT IS DOCUMENTED:
- edge_attr ∈ ℝ^{E×9} is stored on every Data object.
- GraphSAGE message passing uses only node features and graph structure.
- Edge features are NOT learned by the GNN in Phase 3.
- Phase 4 will use edge_attr directly for path-level scoring.

=============================================================================
MODEL ARCHITECTURE
=============================================================================

Name:           AttackGNN (GraphSAGE backbone)
Type:           Node-level binary classifier
Task:           Predict y[i] ∈ {0,1} (attack involvement)

Input:
  x             Node feature matrix  [N, 43]
  edge_index    COO directed edges   [2, E]

Output:
  logits        Raw class logits     [N, 2]
                (softmax not applied inside model)

Architecture:
  Layer 0:  Linear projection     43 → hidden_channels
            + BatchNorm1d(hidden_channels)
            + ReLU
            + Dropout(dropout)

  Layer 1:  SAGEConv              hidden → hidden  (mean aggregator)
            + BatchNorm1d(hidden)
            + ReLU
            + Dropout

  Layer 2:  SAGEConv              hidden → hidden  (mean aggregator)
            + BatchNorm1d(hidden)
            + ReLU
            + Dropout

  (optional) Layer 3: SAGEConv    hidden → hidden  (if num_layers=3)

  Output:   Linear                hidden → 2

Activation:   ReLU (between layers)
Dropout:      Applied after each SAGEConv (default: 0.3)
Aggregator:   Mean (GraphSAGE default; normalises by neighbour count)
BatchNorm:    Applied after each hidden layer (stabilizes small-graph training)

Design choices:
- Input projection layer lets us decouple feature dimensionality from
  hidden_channels without touching SAGEConv.
- BatchNorm before message passing stabilizes training across graphs of
  different sizes.
- Residual connections are NOT used at this phase; the graphs are small
  enough that residuals don't add value.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import SAGEConv

logger = logging.getLogger(__name__)


class AttackGNN(nn.Module):
    """
    GraphSAGE-based node-level binary classifier.

    Predicts attack involvement probability for each node in a network graph.

    Architecture:
      - Linear input projection (43 → hidden_channels)
      - num_layers SAGEConv message-passing layers
      - BatchNorm1d + ReLU + Dropout after each hidden layer
      - Linear output head (hidden_channels → 2)

    Outputs raw logits [N, 2]. Apply softmax externally for probabilities.

    Args:
        in_channels:     Input feature dimension. Must be 43 (FEATURE_DIM).
        hidden_channels: Hidden representation size. Default 64.
        num_layers:      Number of SAGEConv layers (1, 2, or 3). Default 2.
        dropout:         Dropout probability (0.0–1.0). Default 0.3.
        num_classes:     Output classes. Must be 2 for binary classification.

    Usage:
        model = AttackGNN(in_channels=43)
        logits = model(data.x, data.edge_index)  # [N, 2]
        probs  = F.softmax(logits, dim=-1)        # [N, 2], columns = P(cls0), P(cls1)
        preds  = logits.argmax(dim=-1)            # [N]
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 2,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError(f"num_layers must be ≥ 1, got {num_layers}.")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0.0, 1.0), got {dropout}.")
        if num_classes != 2:
            raise ValueError(
                f"AttackGNN is a binary classifier; num_classes must be 2, got {num_classes}."
            )

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_classes = num_classes

        # Input projection: maps 43-dim raw features into hidden space.
        # Separate from SAGEConv so that feature dimensionality change is
        # explicit and doesn't couple to message-passing hyper-parameters.
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        self.input_bn = nn.BatchNorm1d(hidden_channels)

        # SAGEConv layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))

        # Output head: raw logits for 2 classes
        self.output_head = nn.Linear(hidden_channels, num_classes)

        self._init_weights()

        logger.info(
            "AttackGNN initialised: in=%d, hidden=%d, layers=%d, dropout=%.2f",
            in_channels, hidden_channels, num_layers, dropout,
        )

    def _init_weights(self) -> None:
        """Kaiming uniform initialisation for linear layers."""
        nn.init.kaiming_uniform_(self.input_proj.weight, nonlinearity="relu")
        nn.init.zeros_(self.input_proj.bias)
        nn.init.kaiming_uniform_(self.output_head.weight, nonlinearity="relu")
        nn.init.zeros_(self.output_head.bias)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x:          Node feature matrix, shape [N, in_channels].
            edge_index: COO edge index, shape [2, E], dtype=torch.long.

        Returns:
            logits: Raw class logits, shape [N, num_classes=2].
                    DO NOT apply softmax here; call F.softmax externally.
        """
        # Input projection
        h = self.input_proj(x)
        h = self.input_bn(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        # GraphSAGE message-passing layers
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index)
            # BatchNorm requires ≥2 nodes when training; handle single-node edge cases
            if h.shape[0] > 1 or not self.training:
                h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        # Output logits
        logits = self.output_head(h)
        return logits

    def predict_proba(self, x: Tensor, edge_index: Tensor) -> Tensor:
        """
        Compute softmax probabilities.

        Args:
            x:          Node features [N, in_channels].
            edge_index: COO edge index [2, E].

        Returns:
            probs: Softmax probabilities [N, 2].
                   Column 0 = P(not involved), Column 1 = P(involved).
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, edge_index)
            return F.softmax(logits, dim=-1)

    def count_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def architecture_summary(self) -> dict:
        """Return a human-readable architecture description dict."""
        return {
            "model_class": "AttackGNN",
            "backbone": "GraphSAGE (SAGEConv, mean aggregator)",
            "in_channels": self.in_channels,
            "hidden_channels": self.hidden_channels,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "num_classes": self.num_classes,
            "output": "raw logits [N, 2]",
            "activation": "ReLU",
            "normalization": "BatchNorm1d",
            "edge_attr_used": False,
            "edge_attr_note": (
                "edge_attr is preserved in Data objects for Phase 4 path scoring "
                "but is NOT consumed by SAGEConv message passing in Phase 3."
            ),
            "trainable_parameters": self.count_parameters(),
        }

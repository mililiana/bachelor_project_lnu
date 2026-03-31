"""
XNet: Cauchy-based Neural Network for Efficient High-Precision Reranking

Uses trainable Cauchy activation function for high-order accuracy
with very few parameters. Designed to replace deep MLPs with
shallow (1-2 layer) networks while achieving 10x lower error.

Cauchy activation: φ(x) = (λ₁·x + λ₂) / (x² + d²)
Where:
    λ₁: captures odd parts of data distribution
    λ₂: captures even parts
    d: controls smoothness and localization
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Tuple
import time


class CauchyActivation(nn.Module):
    """
    Trainable Cauchy activation function.

    Formula: φ(x) = (λ₁·x) / (x² + d²) + λ₂ / (x² + d²)
           = (λ₁·x + λ₂) / (x² + d²)

    Parameters (all trainable):
        λ₁ (lambda1): Captures odd parts of data distribution
        λ₂ (lambda2): Captures even parts of data distribution
        d: Controls smoothness and localization (focus)

    Properties:
        - Approaches 0 as x → ±∞ (localized response)
        - λ₁ term creates anti-symmetric component
        - λ₂ term creates symmetric component
        - Small d → sharp, focused response
        - Large d → broad, smooth response
    """

    def __init__(self, features: int = 1, init_d: float = 1.0):
        """
        Args:
            features: Number of features (for per-feature parameters)
            init_d: Initial value for d parameter
        """
        super(CauchyActivation, self).__init__()

        # Trainable parameters
        self.lambda1 = nn.Parameter(torch.ones(features))
        self.lambda2 = nn.Parameter(torch.zeros(features))
        self.d = nn.Parameter(torch.ones(features) * init_d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Cauchy activation.

        Args:
            x: Input tensor of shape (batch, features)

        Returns:
            Activated tensor of same shape
        """
        # Ensure d is positive for numerical stability
        d_sq = self.d**2 + 1e-8

        # Denominator: x² + d²
        denom = x**2 + d_sq

        # Numerator: λ₁·x + λ₂
        numer = self.lambda1 * x + self.lambda2

        return numer / denom

    def get_params(self) -> Dict[str, torch.Tensor]:
        """Return current parameter values."""
        return {
            "lambda1": self.lambda1.detach().cpu(),
            "lambda2": self.lambda2.detach().cpu(),
            "d": self.d.detach().cpu(),
        }


class XNet(nn.Module):
    """
    XNet: Shallow Cauchy-based network for efficient scoring.

    Key advantages over MLP:
    - Fewer layers needed (typically 1-2)
    - Higher precision with same parameters
    - Faster inference
    - Trainable activation shape

    Architecture:
        Input → Linear → CauchyActivation → Linear → Sigmoid
    """

    def __init__(
        self,
        input_dim: int = 6,
        hidden_dim: int = 16,
        output_dim: int = 1,
        num_layers: int = 1,
    ):
        """
        Args:
            input_dim: Number of input features
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (1 for relevance score)
            num_layers: Number of Cauchy layers (1 or 2 recommended)
        """
        super(XNet, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        layers = []

        # First layer: input → hidden
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(CauchyActivation(features=hidden_dim, init_d=1.0))

        # Additional Cauchy layers (if any)
        for i in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(CauchyActivation(features=hidden_dim, init_d=0.5))

        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Initialize linear layer weights."""
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.net(x)

    def get_cauchy_params(self) -> List[Dict]:
        """Get parameters from all Cauchy layers."""
        params = []
        for module in self.net:
            if isinstance(module, CauchyActivation):
                params.append(module.get_params())
        return params

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class XNetEfficient(nn.Module):
    """
    Ultra-efficient single-layer XNet.

    For maximum speed: just input → Cauchy → output.
    Useful for benchmarking against deeper architectures.
    """

    def __init__(self, input_dim: int = 6, output_dim: int = 1):
        super(XNetEfficient, self).__init__()

        self.linear = nn.Linear(input_dim, output_dim)
        self.cauchy = CauchyActivation(features=output_dim, init_d=1.0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.cauchy(x)
        return self.sigmoid(x + 0.5)  # Shift to avoid negative outputs


def xnet_rerank(
    model: XNet,
    features: torch.Tensor,
    doc_ids: Optional[List[str]] = None,
) -> Tuple[torch.Tensor, Optional[List[Tuple[str, float]]]]:
    """
    Rerank documents using trained XNet model.

    Args:
        model: Trained XNet model
        features: Feature tensor of shape (num_docs, num_features)
        doc_ids: Optional list of document IDs

    Returns:
        Tuple of (scores tensor, ranked list if doc_ids provided)
    """
    model.eval()

    with torch.no_grad():
        scores = model(features)

    if doc_ids is not None:
        scores_list = scores.squeeze().tolist()
        if isinstance(scores_list, float):
            scores_list = [scores_list]

        ranked = list(zip(doc_ids, scores_list))
        ranked.sort(key=lambda x: x[1], reverse=True)
        return scores, ranked

    return scores, None

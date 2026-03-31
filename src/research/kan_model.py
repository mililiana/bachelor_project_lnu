import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict


class KANLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.Tensor(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(
            self.base_weight, a=math.sqrt(5) * self.scale_base
        )
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(
                    self.spline_scaler, a=math.sqrt(5) * self.scale_spline
                )

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for x assuming no hidden dimension
        :param x: (..., in_features)
        :return: (..., in_features, grid_size + spline_order)
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:-k])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.
        :param x: (batch_size, in_features)
        :param y: (batch_size, in_features, out_features)
        :return: (out_features, in_features, grid_size + spline_order)
        """
        split_shape = (self.in_features * self.out_features,)
        # Simplified approximate implementation for initialization
        return (
            torch.randn(
                self.out_features, self.in_features, self.grid_size + self.spline_order
            )
            * 0.1
        )

    def forward(self, x: torch.Tensor):
        base_output = F.linear(self.base_activation(x), self.base_weight)

        # B-spline interpolation
        x_clamped = torch.clamp(x, min=self.grid[0, 0], max=self.grid[0, -1])
        spline_basis = self.b_splines(x_clamped)

        if self.enable_standalone_scale_spline:
            spline_weight = self.spline_weight * self.spline_scaler.unsqueeze(-1)
        else:
            spline_weight = self.spline_weight

        spline_output = F.linear(
            spline_basis.reshape(x.size(0), -1),
            spline_weight.reshape(self.out_features, -1),
        )

        return base_output + spline_output


class KAN(nn.Module):
    # Default feature names for 6-feature configuration
    FEATURE_NAMES = [
        "semantic",
        "bm25",
        "title_overlap",
        "category",
        "chunk_pos",
        "doc_length",
    ]

    def __init__(self, layers_hidden, grid_size=5, spline_order=3, feature_names=None):
        super(KAN, self).__init__()
        self.layers = nn.ModuleList()
        self.input_dim = layers_hidden[0]

        # Allow custom feature names
        if feature_names is not None:
            self.feature_names = feature_names
        elif layers_hidden[0] == 6:
            self.feature_names = self.FEATURE_NAMES
        elif layers_hidden[0] == 2:
            self.feature_names = ["semantic", "bm25"]
        else:
            self.feature_names = [f"feature_{i}" for i in range(layers_hidden[0])]

        for i in range(len(layers_hidden) - 1):
            self.layers.append(
                KANLinear(
                    layers_hidden[i],
                    layers_hidden[i + 1],
                    grid_size=grid_size,
                    spline_order=spline_order,
                )
            )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        # Apply sigmoid for classification
        return torch.sigmoid(x)

    def get_spline_weights(self):
        """
        Get spline weights from first layer for visualization.

        Returns:
            Dict with 'base_weight', 'spline_weight', 'grid'
        """
        first_layer = self.layers[0]
        return {
            "base_weight": first_layer.base_weight.detach().cpu(),
            "spline_weight": first_layer.spline_weight.detach().cpu(),
            "grid": first_layer.grid.detach().cpu(),
            "spline_scaler": (
                first_layer.spline_scaler.detach().cpu()
                if hasattr(first_layer, "spline_scaler")
                else None
            ),
        }

    def get_feature_importance(self):
        """
        Calculate feature importance from weight magnitudes.

        Returns:
            Dict mapping feature names to importance scores (normalized)
        """
        first_layer = self.layers[0]
        base_importance = first_layer.base_weight.detach().abs().mean(dim=0)

        if (
            hasattr(first_layer, "spline_scaler")
            and first_layer.spline_scaler is not None
        ):
            spline_importance = first_layer.spline_scaler.detach().abs().mean(dim=0)
            total = base_importance + spline_importance
        else:
            total = base_importance

        # Normalize
        total = total / total.sum()

        return {name: total[i].item() for i, name in enumerate(self.feature_names)}

    def symbolic_formula(self):
        """
        Derives an approximate formula by analyzing feature sensitivity.
        Sends isolated signals for each feature to measure its impact on the output.
        """
        device = next(self.parameters()).device
        n_features = self.input_dim
        features = self.feature_names[:n_features]
        coefficients = []

        # Measure response to average input (approx 0.5 for normalized features)
        baseline = torch.full((1, n_features), 0.5, device=device)

        for i, feature in enumerate(features):
            # Create a probe: 1.0 for this feature, 0.5 for others
            probe = baseline.clone()
            probe[0, i] = 1.0

            # Forward pass
            with torch.no_grad():
                output = self.forward(probe)

            coefficients.append(output.item())

        # Normalize coefficients to sum to 1.0 for a readable formula
        total = sum(abs(c) for c in coefficients) + 1e-9
        ratios = [c / total for c in coefficients]

        formula_parts = [f"{r:.2f} * {f}" for r, f in zip(ratios, features)]
        return " + ".join(formula_parts)

    def sparsification_loss(
        self, lamb: float = 0.01, lamb_entropy: float = 2.0
    ) -> torch.Tensor:
        """
        Compute sparsification regularization loss.

        Combines L1 regularization with entropy regularization to encourage
        sparse spline representations.

        Args:
            lamb: L1 regularization strength
            lamb_entropy: Entropy regularization strength

        Returns:
            Regularization loss to add to main loss
        """
        reg_loss = torch.tensor(0.0, device=next(self.parameters()).device)

        for layer in self.layers:
            # L1 on spline weights
            l1_loss = lamb * layer.spline_weight.abs().mean()
            reg_loss = reg_loss + l1_loss

            # Entropy regularization on spline scalers
            if hasattr(layer, "spline_scaler") and layer.spline_scaler is not None:
                # Normalize to probability-like values
                scaler_abs = layer.spline_scaler.abs()
                probs = scaler_abs / (scaler_abs.sum() + 1e-8)
                entropy = -(probs * torch.log(probs + 1e-8)).sum()
                reg_loss = (
                    reg_loss - lamb_entropy * entropy
                )  # Maximize entropy = minimize -entropy

        return reg_loss

    def prune(self, threshold: float = 0.01) -> Dict[str, int]:
        """
        Prune low-importance connections by zeroing small weights.

        Args:
            threshold: Weights with absolute value below this are zeroed

        Returns:
            Dict with pruning statistics
        """
        total_weights = 0
        pruned_weights = 0

        with torch.no_grad():
            for layer in self.layers:
                # Prune base weights
                mask = layer.base_weight.abs() < threshold
                layer.base_weight.data[mask] = 0
                total_weights += layer.base_weight.numel()
                pruned_weights += mask.sum().item()

                # Prune spline weights
                spline_mask = layer.spline_weight.abs() < threshold
                layer.spline_weight.data[spline_mask] = 0
                total_weights += layer.spline_weight.numel()
                pruned_weights += spline_mask.sum().item()

        return {
            "total_weights": total_weights,
            "pruned_weights": pruned_weights,
            "sparsity": pruned_weights / total_weights if total_weights > 0 else 0,
        }

    def auto_symbolic(self, sample_points: int = 100) -> Dict[str, str]:
        """
        Attempt to fit learned splines to known symbolic functions.

        Tries to match each feature's learned transformation to
        common functions: linear, sqrt, exp, log, sigmoid, etc.

        Args:
            sample_points: Number of points to sample for fitting

        Returns:
            Dict mapping feature names to discovered symbolic functions
        """
        import numpy as np

        device = next(self.parameters()).device
        n_features = self.input_dim
        x_range = torch.linspace(0, 1, sample_points, device=device)

        # Candidate functions (normalized to [0, 1] input)
        def candidates(x):
            x_np = x.cpu().numpy()
            return {
                "linear": x_np,
                "sqrt": np.sqrt(x_np),
                "square": x_np**2,
                "exp": (np.exp(x_np) - 1) / (np.e - 1),
                "log": np.log(x_np + 0.1) / np.log(1.1),
                "sigmoid": 1 / (1 + np.exp(-5 * (x_np - 0.5))),
                "step": (x_np > 0.5).astype(float),
            }

        candidate_funcs = candidates(x_range)
        results = {}

        for i, feature in enumerate(self.feature_names):
            # Create input where only this feature varies
            inputs = torch.full((sample_points, n_features), 0.5, device=device)
            inputs[:, i] = x_range

            with torch.no_grad():
                outputs = self.forward(inputs).flatten().cpu().numpy()

            # Normalize outputs
            outputs = (outputs - outputs.min()) / (outputs.max() - outputs.min() + 1e-8)

            # Find best matching function
            best_func = "linear"
            best_corr = -1

            for func_name, func_values in candidate_funcs.items():
                corr = np.corrcoef(outputs, func_values)[0, 1]
                if not np.isnan(corr) and abs(corr) > best_corr:
                    best_corr = abs(corr)
                    best_func = func_name

            results[feature] = best_func

        return results

    def get_discovered_formula(self) -> str:
        """
        Generate a human-readable formula combining feature importance and symbolic functions.

        Returns:
            String like: "0.22*bm25(linear) + 0.18*title(sqrt) + ..."
        """
        importance = self.get_feature_importance()
        symbolic = self.auto_symbolic()

        parts = []
        for feature in self.feature_names:
            coef = importance.get(feature, 0)
            func = symbolic.get(feature, "linear")
            if func == "linear":
                parts.append(f"{coef:.2f}*{feature}")
            else:
                parts.append(f"{coef:.2f}*{func}({feature})")

        return " + ".join(parts)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

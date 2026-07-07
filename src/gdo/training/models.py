"""
models.py
=========
PyTorch model architectures used across all training experiments.

Models
------
MLP   — Multilayer Perceptron for MNIST digit classification
CNN   — Convolutional Neural Network for CIFAR-10 image classification

Design decisions
----------------
- Both models accept config objects (MlpConfig, CnnConfig) from
  ``gdo.config`` so architecture hyperparameters are never hardcoded.
- Both expose ``get_grad_norms()`` so the Trainer can log gradient
  norm per layer to MLflow — critical for Notebook 3's gradient
  clipping demonstration.
- Input shapes:
    MLP:  (B, 784) — flattened 28x28 MNIST image
    CNN:  (B, 1, 32, 32) for MNIST or (B, 3, 32, 32) for CIFAR-10
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from gdo.config import CnnConfig, MlpConfig

logger = logging.getLogger(__name__)


class MLP(nn.Module):
    """
    Multilayer Perceptron for tabular / flattened image classification.

    Architecture::

        Input (784) → [Linear → BN → Activation → Dropout] × N → Linear (10)

    Parameters
    ----------
    config:
        ``MlpConfig`` from ``gdo.config``.

    Notes
    -----
    BatchNorm is applied BEFORE the activation function (pre-activation).
    This follows the convention in modern deep learning and prevents
    the saturation issue with post-activation BatchNorm.
    """

    def __init__(self, config: MlpConfig) -> None:
        super().__init__()
        self.config = config

        activation_map = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}
        act_cls = activation_map[config.activation]

        layers: list[nn.Module] = []
        in_dim = config.input_dim

        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if config.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(act_cls())
            if config.dropout > 0.0:
                layers.append(nn.Dropout(p=config.dropout))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, config.output_dim))
        self.net = nn.Sequential(*layers)

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info("[MLP] Initialized | params=%d | architecture=%s", n_params, config.hidden_dims)

    def _init_weights(self) -> None:
        """He initialization for ReLU/GELU, Xavier for Tanh."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.config.activation in ("relu", "gelu"):
                    nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                else:
                    nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x:
            Input tensor of shape (B, input_dim) or (B, C, H, W) which
            will be flattened automatically.

        Returns
        -------
        torch.Tensor
            Logits of shape (B, output_dim).
        """
        if x.dim() > 2:
            x = x.flatten(start_dim=1)
        return self.net(x)

    def get_grad_norms(self) -> dict[str, float]:
        """
        Return the L2 gradient norm per named parameter.

        Called by the Trainer after ``loss.backward()`` for gradient
        norm monitoring and logging.

        Returns
        -------
        dict[str, float]
            ``{param_name: grad_norm}``. Only includes parameters
            with non-None gradients.
        """
        norms: dict[str, float] = {}
        for name, param in self.named_parameters():
            if param.grad is not None:
                norms[name] = float(param.grad.norm(2).item())
        return norms

    def get_total_grad_norm(self) -> float:
        """Return the global gradient norm (all parameters combined)."""
        total_norm = 0.0
        for param in self.parameters():
            if param.grad is not None:
                total_norm += float(param.grad.norm(2).item()) ** 2
        return total_norm**0.5


class CNN(nn.Module):
    """
    Small Convolutional Neural Network for CIFAR-10 classification.

    Architecture::

        Conv1 (32 filters, 3×3) → BN → ReLU → MaxPool(2×2)
        Conv2 (64 filters, 3×3) → BN → ReLU → MaxPool(2×2)
        Conv3 (128 filters, 3×3) → BN → ReLU
        AdaptiveAvgPool → Flatten
        FC(512) → BN → ReLU → Dropout → FC(num_classes)

    This architecture is deep enough to show meaningful scheduler effects
    (Notebook 3) but fast enough to train on a CPU in a few minutes.

    Parameters
    ----------
    config:
        ``CnnConfig`` from ``gdo.config``.
    """

    def __init__(self, config: CnnConfig) -> None:
        super().__init__()
        self.config = config

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(config.in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((2, 2))  # output: (B, 128, 2, 2) = 512 features
        self.classifier = nn.Sequential(
            nn.Linear(128 * 2 * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=config.dropout),
            nn.Linear(512, config.num_classes),
        )

        self._init_weights()
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info("[CNN] Initialized | params=%d", n_params)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d | nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x:
            Input tensor of shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            Logits of shape (B, num_classes).
        """
        x = self.features(x)
        x = self.pool(x)
        x = x.flatten(start_dim=1)
        return self.classifier(x)

    def get_grad_norms(self) -> dict[str, float]:
        """Return L2 gradient norm per named parameter."""
        norms: dict[str, float] = {}
        for name, param in self.named_parameters():
            if param.grad is not None:
                norms[name] = float(param.grad.norm(2).item())
        return norms

    def get_total_grad_norm(self) -> float:
        """Return global gradient norm (all parameters combined)."""
        total_norm = 0.0
        for param in self.parameters():
            if param.grad is not None:
                total_norm += float(param.grad.norm(2).item()) ** 2
        return total_norm**0.5


def build_model(config: MlpConfig | CnnConfig) -> nn.Module:
    """
    Factory function: build the correct model from config type.

    Parameters
    ----------
    config:
        Either ``MlpConfig`` or ``CnnConfig``.

    Returns
    -------
    nn.Module
        Instantiated and initialized model.
    """
    from gdo.config import CnnConfig, MlpConfig

    if isinstance(config, MlpConfig):
        return MLP(config)
    elif isinstance(config, CnnConfig):
        return CNN(config)
    else:
        raise TypeError(f"Unknown config type: {type(config)}")

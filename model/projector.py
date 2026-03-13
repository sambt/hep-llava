"""MLP Projector: Maps physics encoder outputs to LLM embedding space.

Follows the LLaVA 1.5 architecture: Linear -> GELU -> Linear.
"""

import torch
import torch.nn as nn


class MLPProjector(nn.Module):
    """Two-layer MLP projector (LLaVA 1.5 style).

    Projects from the physics encoder's hidden dimension to the LLM's
    embedding dimension.

    Architecture:
        Linear(input_dim, output_dim) -> GELU -> Linear(output_dim, output_dim)
    """

    def __init__(self, input_dim: int = 512, output_dim: int = 4096):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project physics encoder outputs to LLM space.

        Args:
            x: [B, N, input_dim] from physics encoder.

        Returns:
            [B, N, output_dim] projected embeddings.
        """
        return self.projector(x)

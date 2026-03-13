"""Physics Token Encoder: Transformer over VQ-VAE discrete tokens.

Takes a sequence of codebook indices from OmniJet-alpha and produces
contextualized hidden states for projection into the LLM embedding space.
"""

import math

import torch
import torch.nn as nn


class PhysicsTokenEncoder(nn.Module):
    """Transformer encoder over discrete VQ-VAE jet constituent tokens.

    Architecture:
        Token embedding (codebook_size -> hidden_dim)
        + Learned positional embedding (max_seq_len -> hidden_dim)
        -> Transformer encoder layers
        -> Sequence of hidden states [B, N, hidden_dim]
    """

    def __init__(
        self,
        vocab_size: int = 8192,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 128,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len

        # Token embedding: codebook index -> hidden_dim
        # +1 for padding token (index 0)
        self.token_embedding = nn.Embedding(vocab_size + 1, hidden_dim, padding_idx=0)

        # Learned positional encoding (particles are pT-ordered)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-norm for better training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        # Layer norm on output
        self.output_norm = nn.LayerNorm(hidden_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights following common transformer practices."""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        # Zero out padding embedding
        with torch.no_grad():
            self.token_embedding.weight[0].zero_()

    def forward(
        self,
        token_indices: torch.LongTensor,
        attention_mask: torch.BoolTensor | None = None,
    ) -> torch.Tensor:
        """Encode a sequence of VQ-VAE token indices.

        Args:
            token_indices: [B, N] codebook indices (0 = padding).
            attention_mask: [B, N] boolean mask (True = valid, False = padding).
                If None, non-zero tokens are treated as valid.

        Returns:
            [B, N, hidden_dim] contextualized hidden states.
        """
        B, N = token_indices.shape
        assert N <= self.max_seq_len, f"Sequence length {N} exceeds max {self.max_seq_len}"

        # Token embeddings
        tok_emb = self.token_embedding(token_indices)  # [B, N, D]

        # Position embeddings
        positions = torch.arange(N, device=token_indices.device).unsqueeze(0)  # [1, N]
        pos_emb = self.position_embedding(positions)  # [1, N, D]

        # Combine
        x = tok_emb + pos_emb  # [B, N, D]

        # Create attention mask for transformer
        # PyTorch TransformerEncoder expects src_key_padding_mask where True = IGNORE
        if attention_mask is None:
            padding_mask = token_indices == 0  # True where padding
        else:
            padding_mask = ~attention_mask  # Invert: True where we should mask

        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # Output normalization
        x = self.output_norm(x)

        return x

    @property
    def output_dim(self) -> int:
        return self.hidden_dim

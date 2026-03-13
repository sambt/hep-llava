"""OmniJet Foundation Encoder: frozen pretrained OmniJet-alpha generative model.

Uses the pretrained OmniJet-alpha generative transformer (``BackboneModel``) as a
frozen feature extractor.  The generative model was pretrained on millions of jets and
learned rich, contextualised constituent-level representations via next-token prediction.

We extract the final-layer hidden states (shape ``[B, N, hidden_dim]``) rather than the
classification logits or generated tokens, and pass them to the MLP projector.

Architecture summary (from the 8192-token checkpoint):
  - embedding_dim = 256
  - n_GPT_blocks  = 3
  - n_heads       = 8
  - vocab_size    = 8194  (8192 codebook + 2 special tokens)
  - max_sequence_len = 128
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------


def find_generative_checkpoint(omnijet_dir: str | Path) -> Path:
    """Search the standard location for the generative model checkpoint.

    Looks inside ``{omnijet_dir}/checkpoints/generative_8192_tokens/`` for any
    ``*.ckpt`` file.

    Args:
        omnijet_dir: Root directory of the cloned OmniJet-alpha repository.

    Returns:
        Path to the first ``.ckpt`` file found.

    Raises:
        FileNotFoundError: If no checkpoint is found at the expected location.
    """
    search_dir = Path(omnijet_dir) / "checkpoints" / "generative_8192_tokens"
    if not search_dir.exists():
        raise FileNotFoundError(
            f"Expected generative checkpoint directory not found: {search_dir}\n"
            "Make sure the OmniJet-alpha repository has been set up correctly and "
            "the generative checkpoint has been downloaded."
        )

    ckpts = list(search_dir.glob("*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(
            f"No .ckpt files found in {search_dir}.\n"
            "Download the generative model checkpoint following the OmniJet-alpha README."
        )

    return ckpts[0]


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class OmniJetFoundationEncoder(nn.Module):
    """Frozen OmniJet-alpha generative model used as a physics feature extractor.

    The generative model was pretrained on millions of jets and learned
    contextualised constituent-level representations via autoregressive
    next-token prediction.  We extract its final-layer hidden states rather
    than training a physics encoder from scratch.

    The underlying architecture is ``BackboneModel`` (a stack of GPT decoder
    blocks) from ``gabbro.models.gpt_model``.  It lives inside a
    ``BackboneNextTokenPredictionLightning`` PyTorch-Lightning wrapper in the
    checkpoint — we extract only the ``module`` sub-module so we can call it
    directly without Lightning.

    Args:
        omnijet_dir: Path to the cloned OmniJet-alpha repository.
        checkpoint_path: Path to the generative model ``.ckpt`` file.
            If ``None``, auto-detected via :func:`find_generative_checkpoint`.
        freeze: If ``True`` (default), all parameters are frozen immediately
            after loading.  Should almost always be ``True``.

    Input:
        token_indices: ``[B, N]`` int64 VQ-VAE codebook indices.
        attention_mask: ``[B, N]`` bool mask — ``True`` = valid particle,
            ``False`` = padding.

    Output:
        ``[B, N, hidden_dim]`` float32 tensor of final-layer hidden states.
        ``hidden_dim`` is auto-detected from the checkpoint (256 for the
        standard 8192-token model).
    """

    def __init__(
        self,
        omnijet_dir: str | Path,
        checkpoint_path: str | Path | None = None,
        freeze: bool = True,
    ) -> None:
        super().__init__()

        omnijet_dir = Path(omnijet_dir)

        # ---- Locate checkpoint -------------------------------------------------
        if checkpoint_path is None:
            checkpoint_path = find_generative_checkpoint(omnijet_dir)
        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Generative checkpoint not found: {checkpoint_path}\n"
                "Provide a valid path via the ``checkpoint_path`` argument or "
                "ensure the file exists at the default location "
                f"({omnijet_dir}/checkpoints/generative_8192_tokens/*.ckpt)."
            )

        # ---- Add OmniJet-alpha to sys.path so gabbro imports work -------------
        omnijet_str = str(omnijet_dir)
        if omnijet_str not in sys.path:
            sys.path.insert(0, omnijet_str)

        try:
            from gabbro.models.gpt_model import BackboneModel  # noqa: F401 – verify import
        except ImportError as exc:
            raise ImportError(
                "Could not import gabbro.models.gpt_model from the OmniJet-alpha "
                f"repository at {omnijet_dir}.\n"
                "Make sure the repository is cloned correctly and its dependencies "
                "are installed (pip install -e <omnijet_dir>).\n"
                f"Original error: {exc}"
            ) from exc

        # ---- Load checkpoint ---------------------------------------------------
        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

        # Extract hyper-parameters from the Lightning checkpoint
        model_kwargs: dict = ckpt["hyper_parameters"]["model_kwargs"]

        # Instantiate BackboneModel directly (without the Lightning wrapper)
        from gabbro.models.gpt_model import BackboneModel

        self.backbone = BackboneModel(**model_kwargs)

        # The state dict keys use the "module." prefix (Lightning wraps in self.module)
        state_dict = ckpt["state_dict"]
        backbone_state = {
            k[len("module."):]: v
            for k, v in state_dict.items()
            if k.startswith("module.")
        }

        # "tril" buffers are causal masks; their size is fixed to max_sequence_len.
        # The BackboneModel registers them so they are auto-created on __init__.
        # We skip them here to avoid shape mismatches when seq_len differs.
        backbone_state_no_tril = {k: v for k, v in backbone_state.items() if "tril" not in k}

        missing, unexpected = self.backbone.load_state_dict(
            backbone_state_no_tril, strict=False
        )
        # Only "tril" buffers should be missing — anything else is unexpected.
        non_tril_missing = [k for k in missing if "tril" not in k]
        if non_tril_missing:
            raise RuntimeError(
                f"Unexpected missing keys when loading OmniJet backbone: {non_tril_missing}"
            )

        self._hidden_dim: int = model_kwargs["embedding_dim"]

        self.backbone.eval()

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    # ---- Properties --------------------------------------------------------

    @property
    def hidden_dim(self) -> int:
        """Output feature dimension (``embedding_dim``) of the backbone."""
        return self._hidden_dim

    # ---- Forward -----------------------------------------------------------

    def forward(
        self,
        token_indices: torch.LongTensor,
        attention_mask: torch.BoolTensor | None = None,
    ) -> torch.Tensor:
        """Extract final-layer hidden states from the generative backbone.

        The backbone is a causal GPT-style transformer.  We pass the token
        sequence through all GPT blocks and return the final hidden states.
        Padding positions are masked out in the attention computation via
        the ``padding_mask`` argument of ``BackboneModel.forward``.

        Args:
            token_indices: ``[B, N]`` int64 VQ-VAE codebook indices.
            attention_mask: ``[B, N]`` bool mask.  ``True`` = valid particle.
                If ``None``, all positions are treated as valid.

        Returns:
            ``[B, N, hidden_dim]`` hidden states from the final transformer layer.
        """
        # BackboneModel.forward expects padding_mask with 1=valid, 0=padding
        # (used as a multiplicative mask in ClassificationHead and as an additive
        # mask in MultiHeadAttention — see gpt_model.py lines 66-73).
        padding_mask: torch.Tensor | None = None
        if attention_mask is not None:
            padding_mask = attention_mask.float()  # True → 1.0, False → 0.0

        hidden_states: torch.Tensor = self.backbone(token_indices, padding_mask=padding_mask)
        return hidden_states  # [B, N, hidden_dim]

"""Shared model-loading helper for inference scripts (demo, chat).

Loads a PhysLLaVA model with optional Stage 2 checkpoint and LoRA adapter.
Handles both the new run-namespaced layout and the legacy flat layout so the
scripts work regardless of which training pipeline produced the checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import torch

from model.physllava import PhysLLaVA


def _find_stage2_checkpoint(data_dir: Path, config: dict) -> tuple[Path | None, Path | None]:
    """Search for a Stage 2 checkpoint, trying new and legacy layouts.

    Returns:
        (checkpoint_pt, lora_dir) — either may be None if not found.
    """
    # New namespaced layout: runs/{run_name}/checkpoints/stage2/
    run_name = config.get("run_name", "default")
    new_pt = data_dir / "runs" / run_name / "checkpoints" / "stage2" / "final.pt"
    new_lora = data_dir / "runs" / run_name / "checkpoints" / "stage2" / "final_lora"

    if new_pt.exists():
        return new_pt, (new_lora if new_lora.exists() else None)

    # Legacy flat layout: checkpoints/stage2/
    legacy_pt = data_dir / "checkpoints" / "stage2" / "final.pt"
    legacy_lora = data_dir / "checkpoints" / "stage2" / "final_lora"

    if legacy_pt.exists():
        return legacy_pt, (legacy_lora if legacy_lora.exists() else None)

    return None, None


def _find_tokenized_dir(data_dir: Path, config: dict) -> Path | None:
    """Search for the tokenized jets directory.

    Returns:
        Path to directory containing tokenized_jets.json, or None.
    """
    from scripts.config import derive_token_set_name

    # New layout: tokenized/{token_set_name}/
    classes = config.get("dataset", {}).get("classes", [])
    tokenizer_type = config.get("tokenizer", {}).get("type", "omnijet_vqvae")
    token_set_name = config.get("token_set_name") or derive_token_set_name(classes, tokenizer_type)
    new_dir = data_dir / "tokenized" / token_set_name

    if (new_dir / "tokenized_jets.json").exists():
        return new_dir

    # Legacy flat layout: tokenized_jets/
    legacy_dir = data_dir / "tokenized_jets"
    if (legacy_dir / "tokenized_jets.json").exists():
        return legacy_dir

    return None


def load_model_for_inference(
    config: dict,
    checkpoint_path: str | Path | None = None,
    lora_dir: str | Path | None = None,
    device: str = "cuda",
) -> PhysLLaVA:
    """Build and load a PhysLLaVA model ready for inference.

    Args:
        config: Full config dict (from ``load_config``).
        checkpoint_path: Explicit path to a Stage 2 ``final.pt``.  If
            ``None``, auto-detected from the config data_dir.
        lora_dir: Explicit path to a LoRA adapter directory.  If ``None``,
            auto-detected alongside the checkpoint.
        device: Torch device string.

    Returns:
        Model moved to *device* and set to ``eval()`` mode.
    """
    data_dir = Path(config["data_dir"])

    # Resolve checkpoint
    if checkpoint_path is None:
        checkpoint_path, auto_lora = _find_stage2_checkpoint(data_dir, config)
        if lora_dir is None:
            lora_dir = auto_lora
    else:
        checkpoint_path = Path(checkpoint_path)
        if lora_dir is None:
            # Try conventional sibling directory
            lora_candidate = checkpoint_path.parent / (checkpoint_path.stem + "_lora")
            if lora_candidate.exists():
                lora_dir = lora_candidate

    print("Building model...")
    model = PhysLLaVA(
        physics_encoder_config=config["physics_encoder"],
        projector_config=config["projector"],
        llm_name=config["llm"]["model_name"],
        torch_dtype=config["llm"]["torch_dtype"],
        use_flash_attention=config["llm"].get("use_flash_attention", False),
        data_dir=str(data_dir),
    )

    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
        model.physics_encoder.load_state_dict(ckpt["physics_encoder"])
        model.projector.load_state_dict(ckpt["projector"])
    else:
        print(f"WARNING: No Stage 2 checkpoint found — running with random weights.")

    if lora_dir and Path(lora_dir).exists():
        print(f"Loading LoRA adapter: {lora_dir}")
        from peft import PeftModel
        model.llm = PeftModel.from_pretrained(model.llm, str(lora_dir))
    else:
        print("No LoRA adapter found — using base LLM weights.")

    model = model.to(device)
    model.eval()
    return model

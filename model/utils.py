"""Utility functions for PhysLLaVA model."""

from pathlib import Path

import torch
import yaml


def load_config(config_path: str = "configs/default.yaml") -> dict:
    """Load YAML config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_model_from_checkpoint(
    config: dict,
    checkpoint_path: str,
    device: str = "cuda",
    load_lora: bool = True,
):
    """Load a fully trained PhysLLaVA model from checkpoint.

    Args:
        config: Full config dict.
        checkpoint_path: Path to .pt checkpoint file.
        device: Torch device.
        load_lora: Whether to load LoRA adapter (Stage 2).

    Returns:
        PhysLLaVA model ready for inference.
    """
    from model.physllava import PhysLLaVA

    model = PhysLLaVA(
        physics_encoder_config=config["physics_encoder"],
        projector_config=config["projector"],
        llm_name=config["llm"]["model_name"],
        torch_dtype=config["llm"]["torch_dtype"],
        use_flash_attention=config["llm"].get("use_flash_attention", True),
    )

    # Load encoder + projector weights
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.physics_encoder.load_state_dict(ckpt["physics_encoder"])
    model.projector.load_state_dict(ckpt["projector"])

    # Load LoRA adapter if available
    if load_lora:
        lora_dir = str(checkpoint_path).replace(".pt", "_lora")
        if Path(lora_dir).exists():
            from peft import PeftModel
            model.llm = PeftModel.from_pretrained(model.llm, lora_dir)

    model = model.to(device)
    model.eval()
    return model


def count_parameters(model: torch.nn.Module) -> dict:
    """Count parameters by component."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}

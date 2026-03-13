"""Stage 1 Training: Feature Alignment (Projector Pretraining).

Freezes the LLM and trains only the physics encoder + MLP projector
on caption data. This aligns the physics token representations with
the LLM's embedding space.
"""

import argparse
import os
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.physllava import PhysLLaVA
from training.dataset import build_stage1_dataset


def train_stage1(config: dict, data_dir: str, device: str = "cuda"):
    """Run Stage 1 training.

    Args:
        config: Full config dict.
        data_dir: Root data directory.
        device: Torch device.
    """
    from scripts.config import get_paths, save_effective_config

    stage1_cfg = config["stage1"]
    paths = get_paths(config)
    checkpoint_dir = paths["stage1_checkpoint_dir"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save effective config snapshot to run directory
    save_effective_config(config, paths)

    # Optional wandb logging
    use_wandb = config["logging"].get("use_wandb", False)
    if use_wandb:
        import wandb
        wandb.init(
            project=config["logging"]["wandb_project"],
            name="stage1_alignment",
            config=config,
        )

    print("=" * 60)
    print("Stage 1: Feature Alignment Training")
    print("=" * 60)

    # Build model
    print("Building model...")
    model = PhysLLaVA(
        physics_encoder_config=config["physics_encoder"],
        projector_config=config["projector"],
        llm_name=config["llm"]["model_name"],
        torch_dtype=config["llm"]["torch_dtype"],
        use_flash_attention=config["llm"].get("use_flash_attention", True),
    )

    # Freeze LLM for Stage 1
    model.freeze_llm()

    # Enable gradient checkpointing for the LLM to reduce memory (even when frozen,
    # activations still need memory during the forward pass when computing loss)
    if hasattr(model.llm, 'gradient_checkpointing_enable'):
        model.llm.gradient_checkpointing_enable()

    param_info = model.get_trainable_params()
    print(f"Trainable parameters: {param_info['trainable']:,} ({param_info['trainable_pct']:.2f}%)")
    print(f"Frozen parameters: {param_info['frozen']:,}")
    print(f"By component: {param_info['by_component']}")

    model = model.to(device)

    # Build dataset
    print("Building dataset...")
    dataset = build_stage1_dataset(data_dir, model.tokenizer, paths=paths)
    print(f"Dataset size: {len(dataset)} samples")

    dataloader = DataLoader(
        dataset,
        batch_size=stage1_cfg["batch_size"],
        shuffle=True,
        num_workers=stage1_cfg.get("dataloader_num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )

    # Optimizer — only trainable params
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=stage1_cfg["learning_rate"],
        weight_decay=stage1_cfg.get("weight_decay", 0.0),
    )

    # Scheduler
    total_steps = len(dataloader) * stage1_cfg["num_epochs"]
    warmup_steps = int(total_steps * stage1_cfg.get("warmup_ratio", 0.03))
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)

    # Training loop
    global_step = 0
    for epoch in range(stage1_cfg["num_epochs"]):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{stage1_cfg['num_epochs']}")
        for batch in pbar:
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                jet_token_indices=batch["jet_token_indices"],
                jet_attention_mask=batch["jet_attention_mask"],
                labels=batch["labels"],
            )

            loss = outputs.loss

            # Backward
            loss.backward()

            # Gradient clipping
            max_grad_norm = stage1_cfg.get("max_grad_norm", 1.0)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)

            optimizer.step()
            if global_step >= warmup_steps:
                scheduler.step()
            optimizer.zero_grad()

            # Logging
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{optimizer.param_groups[0]['lr']:.2e}"})

            if use_wandb and global_step % config["logging"].get("log_every_n_steps", 10) == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/epoch": epoch + 1,
                    "train/global_step": global_step,
                })

            # Save checkpoint periodically
            save_every = config["logging"].get("save_every_n_steps", 500)
            if global_step % save_every == 0:
                ckpt_path = checkpoint_dir / f"step_{global_step}.pt"
                torch.save({
                    "physics_encoder": model.physics_encoder.state_dict(),
                    "projector": model.projector.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "global_step": global_step,
                    "epoch": epoch,
                    "config": config,
                }, ckpt_path)
                print(f"\nSaved checkpoint to {ckpt_path}")

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")

    # Save final checkpoint
    final_path = checkpoint_dir / "final.pt"
    torch.save({
        "physics_encoder": model.physics_encoder.state_dict(),
        "projector": model.projector.state_dict(),
        "global_step": global_step,
        "epoch": stage1_cfg["num_epochs"],
        "config": config,
    }, final_path)
    print(f"Saved final Stage 1 checkpoint to {final_path}")

    if use_wandb:
        wandb.finish()

    return final_path


def main():
    parser = argparse.ArgumentParser(description="PhysLLaVA Stage 1 Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--override", type=str, default=None, help="Path to an override YAML config")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    from scripts.config import load_config

    config = load_config(args.config, args.override)
    if args.data_dir is not None:
        config["data_dir"] = args.data_dir

    data_dir = config["data_dir"]
    train_stage1(config, data_dir, args.device)


if __name__ == "__main__":
    main()

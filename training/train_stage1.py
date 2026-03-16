"""Stage 1 Training: Feature Alignment (Projector Pretraining).

Freezes the LLM and trains only the physics encoder + MLP projector
on caption data. This aligns the physics token representations with
the LLM's embedding space.
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.physllava import PhysLLaVA
from training.dataset import build_stage1_dataset
from training.early_stopping import EarlyStopper


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """Supervised NT-Xent contrastive loss (SupCon).

    Pulls together embeddings from the same class and pushes apart
    embeddings from different classes.  Jets of the same class are
    positive pairs; all others are negatives.

    Args:
        features: [B, D] feature vectors (will be L2-normalised internally).
        labels:   [B] integer class indices.
        temperature: Temperature scaling factor (lower = sharper separation).

    Returns:
        Scalar loss.  Returns zero (differentiable) if the batch contains no
        positive pairs (e.g., all samples are from distinct classes).
    """
    features = F.normalize(features, dim=1)
    B = features.shape[0]

    # Similarity matrix
    sim = torch.matmul(features, features.T) / temperature  # [B, B]

    # Mask out diagonal (self-similarity)
    eye = torch.eye(B, device=features.device, dtype=torch.bool)
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~eye  # [B, B]

    if not pos_mask.any():
        # No positive pairs in this batch — return differentiable zero
        return (features * 0).sum()

    # Log-softmax over all non-self entries (numerically stable)
    sim_masked = sim.masked_fill(eye, float("-inf"))
    log_denom = torch.logsumexp(sim_masked, dim=1, keepdim=True)  # [B, 1]
    log_prob = sim - log_denom  # [B, B]

    # Average log-prob at positive positions per anchor
    n_pos = pos_mask.float().sum(dim=1).clamp(min=1)
    loss_per_anchor = -(log_prob * pos_mask.float()).sum(dim=1) / n_pos

    # Only include anchors that have at least one positive
    has_pos = pos_mask.any(dim=1)
    return loss_per_anchor[has_pos].mean()


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
        from scripts.config import get_wandb_run_id
        run_id = get_wandb_run_id(config)
        wandb.init(
            project=config["logging"]["wandb_project"],
            entity=config["logging"].get("wandb_entity") or None,
            id=run_id,
            name=config.get("run_name", "default"),
            resume="allow",
            config=config,
            tags=["stage1"],
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
        data_dir=data_dir,
    )

    # Optionally unfreeze last N layers of OmniJet backbone
    encoder_cfg = config.get("physics_encoder", {})
    if encoder_cfg.get("type") == "omnijet_foundation":
        n_unfreeze = encoder_cfg.get("unfreeze_last_n_layers", 0)
        if n_unfreeze > 0:
            model.physics_encoder.unfreeze_last_n_layers(n_unfreeze)

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

    # Contrastive alignment setup (supervised SupCon on pooled encoder features)
    contrast_cfg = stage1_cfg.get("contrastive", {})
    use_contrastive = contrast_cfg.get("enabled", False)
    contrast_weight = contrast_cfg.get("weight", 0.1)
    contrast_temp = contrast_cfg.get("temperature", 0.07)
    contrast_proj_dim = contrast_cfg.get("proj_dim", 128)

    contrast_head: nn.Module | None = None
    if use_contrastive:
        encoder_dim = model.physics_encoder.hidden_dim
        contrast_head = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.ReLU(),
            nn.Linear(encoder_dim, contrast_proj_dim),
        ).to(device)
        print(
            f"Contrastive alignment enabled: weight={contrast_weight}, "
            f"temperature={contrast_temp}, proj_dim={contrast_proj_dim}"
        )

    # Optimizer — only trainable params (including contrastive head if enabled)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if contrast_head is not None:
        trainable_params = trainable_params + list(contrast_head.parameters())
    optimizer = AdamW(
        trainable_params,
        lr=stage1_cfg["learning_rate"],
        weight_decay=stage1_cfg.get("weight_decay", 0.0),
    )

    # Scheduler
    total_steps = len(dataloader) * stage1_cfg["num_epochs"]
    warmup_steps = int(total_steps * stage1_cfg.get("warmup_ratio", 0.03))
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)

    # Early stopping
    stopper = EarlyStopper.from_config(stage1_cfg)
    if stopper.enabled:
        es = stage1_cfg.get("early_stopping", {})
        print(
            f"Early stopping enabled: patience={stopper.patience} checks, "
            f"check_every={stopper.check_every_n_steps} steps, "
            f"min_delta={stopper.min_delta}, min_steps={stopper.min_steps}"
        )

    # Training loop
    global_step = 0
    stopped_early = False
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

            # Supervised contrastive loss on mean-pooled encoder features
            if use_contrastive and contrast_head is not None:
                # Get encoder hidden states [B, N, D]
                enc_out = model.physics_encoder(
                    batch["jet_token_indices"],
                    batch["jet_attention_mask"],
                )
                # Mean pool over valid tokens [B, D]
                mask_f = batch["jet_attention_mask"].float().unsqueeze(-1)
                pooled = (enc_out * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
                proj = contrast_head(pooled)  # [B, proj_dim]
                loss_c = supervised_contrastive_loss(
                    proj, batch["class_idx"], temperature=contrast_temp
                )
                loss = loss + contrast_weight * loss_c
            else:
                loss_c = None

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

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "ema": f"{stopper.ema:.4f}" if stopper.ema is not None else "n/a",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            })

            if use_wandb and global_step % config["logging"].get("log_every_n_steps", 10) == 0:
                log_dict = {
                    "stage1/loss": loss.item(),
                    "stage1/lr": optimizer.param_groups[0]["lr"],
                    "stage1/epoch": epoch + 1,
                    "stage1/global_step": global_step,
                }
                if stopper.ema is not None:
                    log_dict["stage1/loss_ema"] = stopper.ema
                if loss_c is not None:
                    log_dict["stage1/loss_contrastive"] = loss_c.item()
                wandb.log(log_dict)

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

            # Early stopping check
            if stopper.update(loss.item(), global_step):
                print(f"\nEarly stopping at step {global_step}. {stopper.status()}")
                stopped_early = True
                break

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")
        if use_wandb:
            wandb.log({
                "stage1/epoch_avg_loss": avg_loss,
                "stage1/epoch": epoch + 1,
            })

        if stopped_early:
            break

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

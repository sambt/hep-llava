"""Stage 2 Training: Visual Instruction Tuning.

Trains the physics encoder + projector + LLM (with LoRA) on the full
mixture of captions and QA data at all difficulty levels.
"""

import argparse
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.physllava import PhysLLaVA
from training.dataset import build_stage2_dataset


def train_stage2(
    config: dict,
    data_dir: str,
    stage1_checkpoint: str | None = None,
    device: str = "cuda",
):
    """Run Stage 2 training.

    Args:
        config: Full config dict.
        data_dir: Root data directory.
        stage1_checkpoint: Path to Stage 1 final checkpoint.
        device: Torch device.
    """
    stage2_cfg = config["stage2"]
    checkpoint_dir = Path(data_dir) / "checkpoints" / "stage2"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Optional wandb logging
    use_wandb = config["logging"].get("use_wandb", False)
    if use_wandb:
        import wandb
        wandb.init(
            project=config["logging"]["wandb_project"],
            name="stage2_instruction_tuning",
            config=config,
        )

    print("=" * 60)
    print("Stage 2: Instruction Tuning")
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

    # Load Stage 1 checkpoint (physics encoder + projector weights)
    if stage1_checkpoint is None:
        stage1_checkpoint = Path(data_dir) / "checkpoints" / "stage1" / "final.pt"

    if Path(stage1_checkpoint).exists():
        print(f"Loading Stage 1 checkpoint from {stage1_checkpoint}")
        ckpt = torch.load(stage1_checkpoint, map_location="cpu", weights_only=True)
        model.physics_encoder.load_state_dict(ckpt["physics_encoder"])
        model.projector.load_state_dict(ckpt["projector"])
    else:
        print(f"WARNING: Stage 1 checkpoint not found at {stage1_checkpoint}")
        print("Training from scratch (physics encoder + projector not pretrained)")

    # Apply LoRA to LLM
    lora_cfg = stage2_cfg["lora"]
    lora_config = LoraConfig(
        r=lora_cfg["rank"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Enable gradient checkpointing if configured
    if stage2_cfg.get("gradient_checkpointing", True):
        model.llm.gradient_checkpointing_enable()

    model.llm = get_peft_model(model.llm, lora_config)
    model.llm.print_trainable_parameters()

    # All components trainable in Stage 2
    # (physics_encoder + projector + LoRA adapters on LLM)
    model = model.to(device)

    param_info = model.get_trainable_params()
    print(f"Trainable parameters: {param_info['trainable']:,} ({param_info['trainable_pct']:.2f}%)")
    print(f"By component: {param_info['by_component']}")

    # Build dataset
    print("Building dataset...")
    dataset = build_stage2_dataset(data_dir, model.tokenizer)
    print(f"Dataset size: {len(dataset)} samples")

    dataloader = DataLoader(
        dataset,
        batch_size=stage2_cfg["batch_size"],
        shuffle=True,
        num_workers=stage2_cfg.get("dataloader_num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )

    # Optimizer — all trainable params (encoder + projector + LoRA)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        trainable_params,
        lr=stage2_cfg["learning_rate"],
        weight_decay=stage2_cfg.get("weight_decay", 0.0),
    )

    # Scheduler
    grad_accum = stage2_cfg.get("gradient_accumulation_steps", 1)
    effective_batch_size = stage2_cfg["batch_size"] * grad_accum
    steps_per_epoch = len(dataloader) // grad_accum
    total_steps = steps_per_epoch * stage2_cfg["num_epochs"]
    warmup_steps = int(total_steps * stage2_cfg.get("warmup_ratio", 0.03))
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)

    # Training loop
    global_step = 0
    optimizer.zero_grad()

    for epoch in range(stage2_cfg["num_epochs"]):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{stage2_cfg['num_epochs']}")
        for batch_idx, batch in enumerate(pbar):
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

            loss = outputs.loss / grad_accum
            loss.backward()

            if (batch_idx + 1) % grad_accum == 0:
                # Gradient clipping
                max_grad_norm = stage2_cfg.get("max_grad_norm", 1.0)
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)

                optimizer.step()
                if global_step >= warmup_steps:
                    scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                actual_loss = loss.item() * grad_accum
                epoch_loss += actual_loss
                num_batches += 1

                pbar.set_postfix({
                    "loss": f"{actual_loss:.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                })

                if use_wandb and global_step % config["logging"].get("log_every_n_steps", 10) == 0:
                    import wandb
                    wandb.log({
                        "train/loss": actual_loss,
                        "train/lr": optimizer.param_groups[0]["lr"],
                        "train/epoch": epoch + 1,
                        "train/global_step": global_step,
                    })

                # Save checkpoint periodically
                save_every = config["logging"].get("save_every_n_steps", 500)
                if global_step % save_every == 0:
                    _save_stage2_checkpoint(
                        model, optimizer, global_step, epoch, config, checkpoint_dir
                    )

        avg_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")

    # Save final checkpoint
    _save_stage2_checkpoint(
        model, optimizer, global_step, stage2_cfg["num_epochs"], config,
        checkpoint_dir, name="final",
    )

    if use_wandb:
        import wandb
        wandb.finish()

    return checkpoint_dir / "final.pt"


def _save_stage2_checkpoint(
    model, optimizer, global_step, epoch, config, checkpoint_dir, name=None,
):
    """Save a Stage 2 checkpoint."""
    ckpt_name = name or f"step_{global_step}"
    ckpt_path = checkpoint_dir / f"{ckpt_name}.pt"

    # Save physics encoder + projector + LoRA weights
    save_dict = {
        "physics_encoder": model.physics_encoder.state_dict(),
        "projector": model.projector.state_dict(),
        "global_step": global_step,
        "epoch": epoch,
        "config": config,
    }

    # Save LoRA adapter separately (compatible with peft loading)
    lora_dir = checkpoint_dir / f"{ckpt_name}_lora"
    model.llm.save_pretrained(str(lora_dir))

    torch.save(save_dict, ckpt_path)
    print(f"Saved Stage 2 checkpoint to {ckpt_path}")
    print(f"Saved LoRA adapter to {lora_dir}")


def main():
    parser = argparse.ArgumentParser(description="PhysLLaVA Stage 2 Training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--stage1-checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_dir = args.data_dir or config["data_dir"]
    train_stage2(config, data_dir, args.stage1_checkpoint, args.device)


if __name__ == "__main__":
    main()

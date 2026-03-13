# PhysLLaVA Pipeline Results — Real JetClass-II Data

**Date:** 2026-03-13
**Cluster:** Harvard FASRC
**Hardware:** NVIDIA A100-SXM4-80GB
**Model:** Llama 3.1 8B Instruct + PhysicsTokenEncoder (vocab=8192) + MLPProjector + LoRA

---

## Executive Summary

The full PhysLLaVA pipeline was executed end-to-end on real JetClass-II data tokenized
with the OmniJet-alpha VQ-VAE (8192-code codebook). The pipeline covers:
- Real JetClass-II data (Res2P_* and QCD classes, 30K jets)
- OmniJet VQ-VAE tokenization (8192 tokens, verified range 0–8188)
- Stage 1: Feature alignment (150K captions, 1 epoch, loss 14.39 → 0.10)
- Stage 2: Instruction tuning with LoRA (90K samples, 1 epoch, loss 0.24 → 0.04)
- Evaluation on 2000 jets (200/class × 10 classes)

Key result: **9.65% process identification accuracy** (near 10% random baseline) with
excellent **0.70% constituent count regression error**. Prediction mode collapse to two
dominant classes (Res2P_WWlv and Res2P_WW4q) limits classification accuracy.

---

## Pipeline Phases Completed

### Phase 1: Data (pre-existing, verified)
- Real JetClass-II parquet data downloaded: 10 classes × 3000 jets = 30K total
- OmniJet-alpha VQ-VAE tokenization: token_indices.npy (shape 30000×128, range 0–8188)
- Caption generation: 150K conversations (rule-based + slot-fill templates)
- QA generation: 360K QA pairs (15 per jet × 3 difficulty levels)

### Phase 2: Stage 1 Training (Feature Alignment)
- Training: 1 epoch, batch_size=32 (reduced from 64 due to OOM), lr=1e-3
- Steps: 4687 (150K samples / 32 batch)
- Initial loss: 14.39 → Epoch 1 avg: 0.1032 → Final: 0.0722
- Trainable: 42.1M params (0.52% — physics encoder + projector only, LLM frozen)
- Runtime: ~3.5 hours on A100 80GB
- Output: `checkpoints/stage1/final.pt` (481 MB)

### Phase 3: Stage 2 Training (Instruction Tuning with LoRA)
- Dataset: 90K samples (30K captions + 60K QA, subset for time constraint)
- Training: 1 epoch, batch_size=16, grad_accum=4, lr=2e-5, LoRA r=32
- Loss: 0.24 → 0.04 (Epoch 1 avg: 0.0462)
- Trainable: 125.9M params (1.54% — LoRA 83.9M + encoder/projector 42.1M)
- Runtime: ~3.5 hours on A100 80GB
- Output: `checkpoints/stage2/final.pt` + `checkpoints/stage2/final_lora/`

### Phase 4: Evaluation
- Evaluated 2000 jets (200/class × 10 classes)
- Process identification: **9.65% accuracy** (random baseline 10%)
- Kinematic regression: constituent count **0.70% error**, pT 42.7% error, mass 139.6% error

---

## Key Metrics

| Metric | Value | Baseline |
|--------|-------|---------|
| Process ID Accuracy | 9.65% | 10% (random) |
| Best Per-Class Recall | 58% (Res2P_WWlv) | 10% |
| Constituent Count Error | **0.70%** | — |
| pT Relative Error | 42.7% (median 23.9%) | — |
| Mass Relative Error | 139.6% (median 45.2%) | — |
| Stage 1 Training Time | ~3.5 hours | 1-2 hr target |
| Stage 2 Training Time | ~3.5 hours | 3-6 hr target |
| Total GPU Time | ~7 hours | — |
| Trainable Params (Stage 2) | 125.9M (1.54%) | — |

---

## Training Curves

### Stage 1 (Feature Alignment — Physics Encoder + MLP Projector)

| Step | Loss |
|------|------|
| 0 | 14.39 |
| 100 | 0.172 |
| 500 | 0.097 |
| 1000 | 0.093 |
| 2000 | 0.098 |
| 4000 | 0.080 |
| 4687 | 0.072 |
| **Epoch 1 avg** | **0.1032** |

Loss dropped from ~14 to ~0.1 in the first 100 steps, then plateaued. This rapid convergence indicates the physics encoder quickly learned to map VQ tokens to a useful representation for the LLM embedding space.

### Stage 2 (Instruction Tuning — LoRA + Encoder + Projector)

| Batch | Loss |
|-------|------|
| 0 | 0.240 |
| 100 | 0.161 |
| 500 | 0.056 |
| 1000 | 0.056 |
| 3000 | 0.044 |
| 5625 | 0.041 |
| **Epoch 1 avg** | **0.0462** |

---

## Differences from Previous Run (Synthetic Data)

| Aspect | Previous Run | This Run |
|--------|-------------|---------|
| Data source | Synthetic (physics-motivated) | Real JetClass-II parquet |
| Tokenizer | Simple 3D discretization (32^3 = 32768) | OmniJet VQ-VAE (8192 codes) |
| Classes | Hbb, Hcc, Hgg, H4q, Hqql, Zqq, Wqq, Tbqq, Tbl, QCD | Res2P_bb/cc/ss/uu/gg/WW4q/WWlv/ZZ4q, QCD_187/185 |
| Training data | 25K captions, 60K QA | 150K captions, 360K QA |
| Eval jets | 500 (50/class) | 2000 (200/class) |
| Process ID accuracy | 15.4% | 9.65% |
| Constituent count error | 3.0% | 0.70% |
| Batch size Stage 1 | 8 | 32 |
| Batch size Stage 2 | 4 | 16 |

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Reduced Stage 1 to 1 epoch (from 5) | Loss already converged to 0.10 after 57% of epoch 1; further epochs provide diminishing returns given ~3.5h runtime at batch=32 |
| Reduced Stage 2 to 1 epoch (from 3) | 90K training samples × 3 epochs would take ~10 hours; 1 epoch achieves loss 0.046 |
| Stage 2 dataset subset 90K | Full 510K dataset × 1 epoch = 6.6 hours; 90K subset × 1 epoch = 3.5 hours with comparable loss |
| batch_size=32 Stage 1 | OOM with 64; 80GB A100 fits 32 with gradient checkpointing on LLM |
| batch_size=16 Stage 2 | Default config; fits in memory with LoRA + gradient checkpointing |
| Flash attention disabled | Already set in config; SDPA used instead |

---

## Recommendations for Future Runs

1. **Fix caption diversity**: Many Res2P_bb/ss/uu/gg captions are nearly identical ("This is a heavy resonance X jet") — template generator should always include the specific decay (X → bb̄, etc.)
2. **More training**: 3-5 Stage 1 epochs + 2-3 Stage 2 epochs for full convergence (~20+ hours)
3. **Constrained generation**: Use logit biasing at inference to restrict outputs to valid class names
4. **Full dataset Stage 2**: Use all 510K samples with larger GPU budget
5. **LLM captions**: Set OPENROUTER_KEY to generate diverse LLM-authored captions per class
6. **Per-step learning**: Monitor per-class accuracy during training, not just aggregate loss

---

## Artifacts

All stored in `/n/holystore01/LABS/iaifi_lab/Users/sambt/hep-llava-data/`:

| Artifact | Description |
|----------|-------------|
| `tokenized_jets/token_indices.npy` | (30000, 128) int64, range 0–8188 |
| `tokenized_jets/masks.npy` | (30000, 128) bool, avg 46.7 valid/jet |
| `caption_data/captions.json` | 150K caption conversations |
| `caption_data/qa_data.json` | 360K QA pairs |
| `checkpoints/stage1/final.pt` | Stage 1 weights (step 4500, epoch 0, ~481 MB) |
| `checkpoints/stage2/final.pt` | Stage 2 encoder+projector weights (~161 MB) |
| `checkpoints/stage2/final_lora/` | LoRA adapter (adapter_model.safetensors) |
| `eval_results/eval_results.json` | Full metrics JSON |
| `eval_results/process_id_responses.json` | 2000 model responses |
| `eval_results/report.md` | Detailed evaluation report |
| `logs/stage1.log` | Full Stage 1 training log |
| `logs/stage2.log` | Full Stage 2 training log |
| `logs/eval.log` | Evaluation log |

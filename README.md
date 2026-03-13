# PhysLLaVA

A LLaVA-style multimodal model connecting tokenized particle physics jets to natural language. PhysLLaVA lets you ask questions about jets — their origin process, kinematics, and substructure — in plain English.

## Architecture

```
Jet constituents (pT, η, φ)
        │
        ▼
OmniJet-alpha VQ-VAE          ← pretrained, frozen
(VQVAENormFormer, 8192-token codebook)
        │  discrete token indices [B, N]
        ▼
PhysicsTokenEncoder            ← trained
(Transformer, 6 layers, 512-dim)
        │  hidden states [B, N, 512]
        ▼
MLPProjector                   ← trained
(Linear→GELU→Linear, 512→4096)
        │  physics tokens in LLM space [B, N, 4096]
        ▼
Llama 3.1 8B Instruct          ← frozen (Stage 1) / LoRA (Stage 2)
        │
        ▼
Natural language response
```

The `<jet>` placeholder in the text prompt is replaced with the projected jet embeddings before the LLM forward pass — identical to how LLaVA handles image tokens.

## Dataset

[JetClass-II](https://huggingface.co/datasets/jet-universe/jetclass2) (arXiv:2405.12972) — 10 classes sampled from the `Res2P` and `QCD` splits:

| Class | Physics | Label |
|---|---|---|
| `Res2P_bb` | X → bb̄ (bottom quark pair) | 0 |
| `Res2P_cc` | X → cc̄ (charm quark pair) | 1 |
| `Res2P_ss` | X → ss̄ (strange quark pair) | 2 |
| `Res2P_uu` | X → uū (up quark pair) | 4 |
| `Res2P_gg` | X → gg (gluon pair) | 5 |
| `Res2P_WW4q` | X → WW → qqqq (4-prong hadronic) | 6 |
| `Res2P_WWlv` | X → WW → qqℓν (semi-leptonic) | 7 |
| `Res2P_ZZ4q` | X → ZZ → qqqq (4-prong hadronic) | 9 |
| `QCD_187` | QCD multijet background | 187 |
| `QCD_185` | QCD multijet background (sub-type) | 185 |

3,000 jets per class (30,000 total). Constituents are stored as variable-length `part_deta`, `part_dphi`, `part_px`, `part_py` arrays per jet.

## Training

Two-stage training following LLaVA:

**Stage 1 — Feature Alignment:** Freeze the LLM. Train only the `PhysicsTokenEncoder` and `MLPProjector` on caption data (150K conversations). Aligns the physics representation with the LLM's embedding space.

**Stage 2 — Instruction Tuning:** Unfreeze the LLM with LoRA (rank 32). Continue training encoder + projector. Fine-tune on QA data (360K pairs: factual, kinematic, reasoning).

## Training Data

Generated from JetClass-II truth-level labels and jet kinematics:

- **150K captions** — rule-based and template-filled descriptions like:
  > *"A 2-prong jet from X → bb̄, containing 43 constituents with pT = 872 GeV."*

- **360K QA pairs** across three difficulty levels:
  - *Factual:* "What process produced this jet?" → "X → bb̄"
  - *Kinematic:* "What is the transverse momentum of this jet?" → "872 GeV"
  - *Reasoning:* "Why does this jet have a high τ₂/τ₁ ratio?" → "..."

## Quickstart

### Setup

```bash
conda create -n physllava python=3.11 -y
conda activate physllava
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Run the full pipeline

```bash
# Set required environment variables
export HF_TOKEN=<your_huggingface_token>       # for Llama 3.1 8B
export OPENROUTER_KEY=<your_key>               # optional, for LLM-generated captions

# Edit configs/default.yaml: set data_dir to your storage path

# 1. Download JetClass-II subset
python -m data.download_jetclass --config configs/default.yaml

# 2. Tokenize with OmniJet-alpha VQ-VAE
python -m data.tokenize_jets --config configs/default.yaml --device cuda

# 3. Generate captions and QA
python -m data.generate_captions --config configs/default.yaml --skip-llm
python -m data.generate_qa --config configs/default.yaml

# 4. Train Stage 1
python -m training.train_stage1 --config configs/default.yaml --device cuda

# 5. Train Stage 2
python -m training.train_stage2 --config configs/default.yaml --device cuda

# 6. Evaluate
python -m eval.evaluate --config configs/default.yaml --device cuda
```

### Smoke test

```bash
python -c "from model.physllava import PhysLLaVA; print('OK')"
```

### Submitting to Slurm

`submit_experiment.sh` is a Slurm batch script for the `iaifi_gpu_priority` partition. It wraps the full pipeline and accepts arguments to select which stage(s) to run and which experiment configuration to use.

```bash
# Create the log directory first (only needed once)
mkdir -p slurm_logs

# Full pipeline with defaults
sbatch submit_experiment.sh --stage all

# Full pipeline for a named experiment
sbatch submit_experiment.sh --stage all \
  --override configs/experiments/heavy_flavor.yaml

# Training only (reuse existing tokenized data and captions)
sbatch submit_experiment.sh --stage train \
  --override configs/experiments/omnijet_foundation.yaml

# Re-run evaluation for an existing checkpoint
sbatch submit_experiment.sh --stage eval \
  --override configs/experiments/heavy_flavor.yaml

# Ad-hoc class selection without a separate experiment file
sbatch submit_experiment.sh --stage all \
  --classes Res2P_bb,Res2P_cc,Res2P_gg,QCD_187
```

**`--stage` options:**

| Value | Runs |
|---|---|
| `all` | download → tokenize → captions → qa → stage1 → stage2 → eval |
| `data` | download → tokenize → captions → qa |
| `train` | stage1 → stage2 |
| `stage1` | feature alignment only |
| `stage2` | instruction tuning only (requires existing stage1 checkpoint) |
| `eval` | evaluation only (requires existing stage2 checkpoint) |

**All flags:**

| Flag | Default | Description |
|---|---|---|
| `--stage` | `all` | Pipeline stage(s) to run (see above) |
| `--config` | `configs/default.yaml` | Base config file |
| `--override` | *(none)* | Experiment override YAML, deep-merged on top of base |
| `--classes` | *(from config)* | Comma-separated class list, overrides `dataset.classes` |
| `--skip-llm` / `--no-skip-llm` | `--skip-llm` | Whether to call OpenRouter for LLM-generated captions |

Logs are written to `slurm_logs/output-<jobid>.out`.

## Configuration

All settings live in `configs/default.yaml`. Key fields:

```yaml
data_dir: "/path/to/your/storage"   # where datasets and checkpoints are saved

tokenizer:
  codebook_size: 8192                # must match OmniJet checkpoint
  checkpoint_path: null              # auto-detected under data_dir/omnijet_alpha/

physics_encoder:
  vocab_size: 8192
  hidden_dim: 512
  num_layers: 6

llm:
  model_name: "meta-llama/Llama-3.1-8B-Instruct"
  torch_dtype: "bfloat16"
```

## Repository Layout

```
├── configs/
│   └── default.yaml           # all hyperparameters and paths
├── data/
│   ├── download_jetclass.py   # download JetClass-II subset from HuggingFace
│   ├── tokenize_jets.py       # OmniJet-alpha VQ-VAE tokenization (+ simple fallback)
│   ├── generate_captions.py   # rule-based + template + LLM captions
│   ├── generate_qa.py         # multi-level QA generation
│   └── llm_client.py          # OpenRouter client for LLM-generated captions
├── model/
│   ├── physics_encoder.py     # Transformer over VQ-token sequences
│   ├── projector.py           # MLP projector (LLaVA 1.5 style)
│   └── physllava.py           # full model: encoder + projector + LLM
├── training/
│   ├── dataset.py             # PyTorch dataset for Stage 1 and Stage 2
│   ├── train_stage1.py        # feature alignment training
│   └── train_stage2.py        # LoRA instruction tuning
├── eval/
│   └── evaluate.py            # evaluation: process ID + kinematic regression
├── scripts/
│   └── setup.sh               # environment setup
└── cluster_instructions.md    # orchestration instructions for autonomous runs
```

## Requirements

- Python 3.11, PyTorch ≥ 2.1
- CUDA GPU (tested on A100 80GB; ~40–60GB needed for Stage 2 with LoRA)
- HuggingFace token for `meta-llama/Llama-3.1-8B-Instruct`
- OmniJet-alpha cloned to `{data_dir}/omnijet_alpha/` (auto-cloned by `tokenize_jets.py`)

## References

- **LLaVA:** Liu et al., "Visual Instruction Tuning" (2023) — the multimodal architecture this follows
- **OmniJet-alpha:** Birk et al., "OmniJet-α: The first cross-task foundation model for particle physics" (2024) — the VQ-VAE tokenizer
- **JetClass-II:** Qu et al., "JetClass-II: A Large-Scale Dataset for Deep Learning in Jet Physics" (2024, arXiv:2405.12972) — the training dataset

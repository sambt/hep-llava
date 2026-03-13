# PhysLLaVA — LLaVA-Style Adapter for Particle Physics Jets

## Project overview

PhysLLaVA is a multimodal model that connects tokenized particle physics jet
data to natural language, using the LLaVA architecture.  The goal is to train
a model that can answer physics questions about a jet given only its discrete
VQ-VAE token representation.

**Architecture:**
OmniJet-alpha VQ-VAE tokenizer → Physics Encoder → MLP Projector → Llama 3.1 8B Instruct

The `<jet>` placeholder token in a text prompt is replaced at forward-pass time
by the projected jet embeddings, so the LLM sees the jet as a sequence of soft
embeddings interleaved with the text.

**Training strategy (two-stage LLaVA-style):**
1. **Stage 1 — Feature alignment:** freeze the LLM, train only the Physics
   Encoder + MLP Projector on ~150K caption pairs.
2. **Stage 2 — Instruction tuning:** train Encoder + Projector + LoRA adapters
   on the LLM jointly on a mixture of captions and physics QA pairs.

## Conventions

- Python 3.10+, PyTorch 2.x
- `black` for formatting, `ruff` for linting
- Type hints on all function signatures
- All configs in YAML under `configs/`; all paths via config or CLI — no hardcoded paths
- Conda environment: `physllava`

## Key files

```
data/
  download_jetclass.py    # fetch JetClass-II parquet subset from HuggingFace
  load_jetclass1.py       # read JetClass (v1) ROOT files directly with uproot
  jetclass_labels.py      # official JetClass-II label ↔ integer mapping (188 classes)
  jetclass1_labels.py     # JetClass (v1) class list + physics descriptions (10 classes)
  tokenize_jets.py        # run OmniJet-alpha VQ-VAE → token_indices.npy / masks.npy
  generate_captions.py    # rule-based + LLM captions for tokenized jets
  generate_qa.py          # physics QA pairs for instruction tuning
  llm_client.py           # OpenRouter client for LLM caption generation

model/
  physics_encoder.py      # 6-layer causal Transformer over VQ token sequences
  omnijet_encoder.py      # frozen OmniJet-alpha generative backbone (alternative)
  projector.py            # MLP projector: encoder_dim → LLM hidden dim
  physllava.py            # top-level PhysLLaVA model; handles <jet> token injection

training/
  dataset.py              # PhysLLaVADataset; build_stage1_dataset / build_stage2_dataset
  train_stage1.py         # Stage 1 trainer (encoder + projector only)
  train_stage2.py         # Stage 2 trainer (encoder + projector + LoRA)
  early_stopping.py       # EarlyStopper: EMA-based batch-wise early stopping

scripts/
  config.py               # load_config, get_paths, get_wandb_run_id, save_effective_config
  load_model.py           # shared inference helper: builds model + loads Stage 2 checkpoint
  demo.py                 # batch demo doc generator (Markdown output)
  chat.py                 # interactive REPL for chatting with the model about a jet

eval/
  evaluate.py             # process ID accuracy + kinematic QA metrics

configs/
  default.yaml                         # base config (edit data_dir here)
  experiments/heavy_flavor.yaml        # JetClass-II: X_bb, X_cc, QCD_ss, QCD_light
  experiments/full_res2p.yaml          # JetClass-II: all 8 Res2P + 2 QCD classes
  experiments/omnijet_foundation.yaml  # frozen OmniJet generative backbone
  experiments/jetclass1_higgs.yaml     # JetClass (v1): Higgs + W/Z classes
  experiments/jetclass1_full.yaml      # JetClass (v1): all 10 classes
```

## Dataset sources

Set `dataset.source` in the config:

- **`"jetclass2"` (default):** Downloads from HuggingFace (`jet-universe/jetclass2`).
  Requires `HF_TOKEN` env var.  Class names are official JetClass-II strings like
  `X_bb`, `X_cc`, `QCD_light` — defined in `data/jetclass_labels.py`.
  Run `python -m data.download_jetclass` before tokenizing.

- **`"jetclass1"`:** Reads local ROOT files (already downloaded at
  `/n/holystore01/LABS/iaifi_lab/Lab/sambt/JetClass`).
  No download step needed.  Class names match ROOT file prefixes:
  `HToBB`, `HToCC`, `HToGG`, `HToWW4Q`, `HToWW2Q1L`, `TTBar`, `TTBarLep`,
  `WToQQ`, `ZToQQ`, `ZJetsToNuNu`.
  Set `dataset.jetclass1_path` and optionally `dataset.jetclass1_split`.

## Directory layout (runtime artifacts)

```
{data_dir}/
├── jetclass2_subset/          # raw JetClass-II parquet files (one per class)
├── tokenized/
│   └── {token_set_name}/      # shared across runs with the same classes + tokenizer
│       ├── token_indices.npy
│       ├── masks.npy
│       ├── tokenized_jets.json
│       └── tokenizer_meta.json
├── llm_captions/
│   └── {token_set_name}/      # LLM-generated captions (expensive — shared)
└── runs/
    └── {run_name}/
        ├── config.yaml         # effective config snapshot
        ├── caption_data/       # rule-based captions + QA (cheap — per run)
        ├── checkpoints/
        │   ├── stage1/final.pt
        │   └── stage2/final.pt + final_lora/
        └── eval_results/
```

`token_set_name` is auto-derived from sorted class list + tokenizer type unless
explicitly set.  Multiple runs with the same classes share tokenized data.

## Config keys to know

```yaml
data_dir: "/path/to/data"          # all artifacts go here
run_name: "my_experiment"          # identifies the run dir under {data_dir}/runs/
token_set_name: null               # auto-derived if null

dataset:
  source: "jetclass2"              # or "jetclass1"
  jetclass1_path: null             # required when source=jetclass1
  jetclass1_split: "train"
  num_jets_per_class: 3000
  classes: [...]

physics_encoder:
  type: "custom"                   # or "omnijet_foundation" (frozen pretrained backbone)

logging:
  use_wandb: true                  # W&B enabled by default
  wandb_project: "physllava"
  wandb_entity: null               # set to your W&B team if needed

stage1:
  early_stopping:
    enabled: true
    patience: 5                    # checks without improvement before stopping
    check_every_n_steps: 200
    min_delta: 0.001
    ema_alpha: 0.05                # smoothing; each step = 5% of EMA

stage2:
  early_stopping:
    enabled: true
    check_every_n_steps: 100       # fewer steps per epoch due to grad accumulation
```

## W&B logging design

Stage 1 and Stage 2 share a **single W&B run** (same `id`, `resume="allow"`),
keyed by `run_name`.  Metrics are namespaced by stage:

- `stage1/loss`, `stage1/loss_ema`, `stage1/lr`, `stage1/epoch_avg_loss`
- `stage2/loss`, `stage2/loss_ema`, `stage2/lr`, `stage2/epoch_avg_loss`

The run ID is derived by `scripts/config.get_wandb_run_id(config)`.  Override
via `logging.wandb_run_id` if needed.

## Physics encoder backends

- **`type: "custom"`** — 6-layer causal Transformer trained from scratch during
  Stage 1.  `hidden_dim=512` → projector output `4096` (matches Llama 3.1 8B).
- **`type: "omnijet_foundation"`** — frozen OmniJet-alpha generative model
  (pretrained on millions of jets).  Hidden dim is 256 (fixed by checkpoint).
  `projector.input_dim` is overridden automatically.  Only projector trains in Stage 1.

## Running the pipeline

```bash
conda activate physllava

# JetClass-II (default)
python -m data.download_jetclass  --config configs/default.yaml --override configs/experiments/heavy_flavor.yaml
python -m data.tokenize_jets      --config configs/default.yaml --override configs/experiments/heavy_flavor.yaml --device cuda
python -m data.generate_captions  --config configs/default.yaml --override configs/experiments/heavy_flavor.yaml
python -m data.generate_qa        --config configs/default.yaml --override configs/experiments/heavy_flavor.yaml
python -m training.train_stage1   --config configs/default.yaml --override configs/experiments/heavy_flavor.yaml --device cuda
python -m training.train_stage2   --config configs/default.yaml --override configs/experiments/heavy_flavor.yaml --device cuda
python -m eval.evaluate           --config configs/default.yaml --override configs/experiments/heavy_flavor.yaml --device cuda

# JetClass (v1) — skip download step
python -m data.tokenize_jets      --config configs/default.yaml --override configs/experiments/jetclass1_higgs.yaml --device cuda
# ... then same caption/QA/training/eval steps

# Or run everything at once
python -m scripts.run_pipeline --config configs/default.yaml --override configs/experiments/jetclass1_higgs.yaml
```

## Inference

```bash
# Interactive chat
python -m scripts.chat --config configs/default.yaml --device cuda

# Batch demo document
python -m scripts.demo --config configs/default.yaml --device cuda --n-per-class 2
```

`scripts/load_model.py` auto-detects Stage 2 checkpoints from both the new
namespaced layout (`runs/{run_name}/checkpoints/stage2/`) and the legacy flat
layout (`checkpoints/stage2/`).

## Known issues / active development areas

- **Process ID accuracy is near-random (~10%)** after the first real-data run.
  Root cause: caption templates for most classes produce identical generic text
  ("This is a heavy resonance X jet"), giving the model no signal to distinguish
  between classes.  Fix needed in `data/generate_captions.py`: add class-specific
  templates so each class has discriminative caption content.

- **Caption generation** currently uses rule-based templates + optional LLM
  generation via OpenRouter.  The LLM-generated captions require `OPENROUTER_KEY`
  env var.  The rule-based path works without any API key.

## Testing

```bash
# Smoke test
python -c "from model.physllava import PhysLLaVA; print('OK')"

# Unit tests (when available)
pytest tests/
```

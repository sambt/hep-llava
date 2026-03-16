# PhysLLaVA

PhysLLaVA is a LLaVA-style multimodal model that connects tokenized particle
physics jets to natural language.

Architecture: **OmniJet-alpha VQ-VAE → Physics Encoder → MLP Projector → Llama 3.1 8B**.

## Quick start

```bash
# Set up conda environment
conda env create -f environment.yml   # or see scripts/setup.sh
conda activate physllava

# Edit configs/default.yaml: set data_dir and API tokens
# Then run the full pipeline:
python -m scripts.run_pipeline --config configs/default.yaml
```

See `cluster_instructions.md` for the full orchestrated multi-agent pipeline.

---

## Experiment Runs

### What `run_name` and `token_set_name` do

**`run_name`** (default: `"default"`) is the logical identifier for a training
experiment.  All run-specific artifacts (checkpoints, caption data, eval results,
the effective config snapshot) live under:

```
{data_dir}/runs/{run_name}/
```

**`token_set_name`** identifies the shared tokenized jet data directory.  Because
tokenization is expensive (VQ-VAE encoding of the whole dataset), different runs
that use the same jet classes and tokenizer share the same tokenized data.  The
tokenized data lives under:

```
{data_dir}/tokenized/{token_set_name}/
```

### Directory layout

```
{data_dir}/
├── jetclass2_subset/              <- raw downloads (always shared)
├── tokenized/
│   └── {token_set_name}/          <- shared tokenized data
│       ├── token_indices.npy
│       ├── masks.npy
│       ├── tokenized_jets.json
│       └── tokenizer_meta.json
├── llm_captions/
│   └── {token_set_name}/          <- shared LLM captions (API cost!)
│       └── llm_captions.json
└── runs/
    └── {run_name}/
        ├── config.yaml            <- effective config snapshot saved at run start
        ├── caption_data/          <- rule-based captions + QA (cheap, per-run)
        │   ├── captions.json
        │   └── qa_data.json
        ├── checkpoints/
        │   ├── stage1/
        │   └── stage2/
        └── eval_results/
```

### `token_set_name` auto-derivation

If `token_set_name` is `null` in the config (the default), it is automatically
derived from the sorted class list and the tokenizer type:

```
X_bb_X_cc  +  "omnijet_vqvae"  →  "X_bb_X_cc__omnijet_vqvae"
```

Classes are sorted for stability so that the same physical dataset always maps
to the same directory regardless of list order in the config.

If the resulting name exceeds 80 characters, a stable 8-character SHA-256 prefix
is used instead (`cls_<8hex>`), and the full description is written to
`{data_dir}/tokenized/token_set_index.json` for human inspection.

You can also set `token_set_name` explicitly in your config to force sharing
across runs that happen to use different class orderings, or to give a
human-readable name:

```yaml
token_set_name: "jetclass2_10class"
```

### Writing an override YAML

All scripts accept `--override path/to/override.yaml`.  The override is
**deep-merged** on top of the base config:

- Scalars: override value wins.
- Dicts: merged recursively.
- Lists: override replaces entirely (e.g. `dataset.classes`).

```yaml
# configs/experiments/my_experiment.yaml
run_name: "my_experiment"
dataset:
  classes:         # list → replaces the default 10-class list entirely
    - X_bb
    - X_cc
    - QCD_light
stage1:
  num_epochs: 10   # scalar → overrides default
```

### The `--override` flag

Every script supports `--override`:

```bash
python -m data.download_jetclass --config configs/default.yaml \
    --override configs/experiments/heavy_flavor.yaml

python -m data.tokenize_jets     --config configs/default.yaml \
    --override configs/experiments/heavy_flavor.yaml

python -m data.generate_captions --config configs/default.yaml \
    --override configs/experiments/heavy_flavor.yaml

python -m data.generate_qa       --config configs/default.yaml \
    --override configs/experiments/heavy_flavor.yaml

python -m training.train_stage1  --config configs/default.yaml \
    --override configs/experiments/heavy_flavor.yaml

python -m training.train_stage2  --config configs/default.yaml \
    --override configs/experiments/heavy_flavor.yaml

python -m eval.evaluate          --config configs/default.yaml \
    --override configs/experiments/heavy_flavor.yaml
```

### The `--classes` flag

`data/download_jetclass.py` and `data/tokenize_jets.py` also accept `--classes`
as a shorthand for overriding `dataset.classes` without writing an override file:

```bash
python -m data.download_jetclass --config configs/default.yaml \
    --classes X_bb,X_cc,QCD_light

python -m data.tokenize_jets --config configs/default.yaml \
    --classes X_bb,X_cc,QCD_light
```

The `token_set_name` is re-derived from the new class list automatically.

### Running two variants that share tokenized data

```bash
# Run A: standard 10-class default
python -m training.train_stage1 --config configs/default.yaml

# Run B: heavy-flavor subset — downloads/tokenizes its own class subset,
# but if the token_set_name happens to match an existing tokenized dir,
# that data is reused automatically
python -m training.train_stage1 --config configs/default.yaml \
    --override configs/experiments/heavy_flavor.yaml

# Run C: share the same tokenized data as Run A but use a different model
# (same default classes → same token_set_name → same tokenized_dir)
python -m training.train_stage1 --config configs/default.yaml \
    --override configs/experiments/omnijet_foundation.yaml
```

### Provided experiment configs

| File | run_name | Dataset | Description |
|------|----------|---------|-------------|
| `configs/experiments/heavy_flavor.yaml` | `heavy_flavor` | JetClass-II | X_bb, X_cc, QCD_ss, QCD_light |
| `configs/experiments/full_res2p.yaml` | `full_res2p` | JetClass-II | All 8 Res2P classes + 2 QCD |
| `configs/experiments/omnijet_foundation.yaml` | `omnijet_foundation` | JetClass-II | Frozen OmniJet backbone |
| `configs/experiments/jetclass1_higgs.yaml` | `jetclass1_higgs` | JetClass (v1) | 5 Higgs + W + Z classes |
| `configs/experiments/jetclass1_full.yaml` | `jetclass1_full` | JetClass (v1) | All 10 JetClass-I classes |

---

## Dataset Sources

PhysLLaVA supports two jet datasets.  Select between them with `dataset.source`
in your config or override YAML.

### JetClass-II (`source: "jetclass2"`, default)

**JetClass-II** (arxiv:2405.12972) is downloaded on demand from HuggingFace
(`jet-universe/jetclass2`).  It uses a new class taxonomy with 188 fine-grained
labels, identified by official string names such as `X_bb`, `X_cc`, `QCD_light`.

- Requires a HuggingFace token (set `HF_TOKEN` env var).
- Run `python -m data.download_jetclass` before tokenizing.
- Class names come from `data/jetclass_labels.py`.

Default 10-class selection (set in `configs/default.yaml`):

| Class | Label | Process |
|-------|-------|---------|
| `X_bb` | 0 | X → bb̄ |
| `X_cc` | 1 | X → cc̄ |
| `X_ss` | 2 | X → ss̄ |
| `X_bc` | 4 | X → bc̄ |
| `X_cs` | 5 | X → cs̄ |
| `X_bq` | 6 | X → bq̄ |
| `X_cq` | 7 | X → cq̄ |
| `X_gg` | 9 | X → gg |
| `QCD_ss` | 185 | QCD (ss̄) |
| `QCD_light` | 187 | QCD multijet |

### JetClass (`source: "jetclass1"`)

**JetClass** (arxiv:2202.03772) is the original 10-class dataset, stored locally
as ROOT files.  No download step is needed — point `jetclass1_path` at the
directory that contains `train_100M/`, `val_5M/`, and `test_20M/`.

- **No HuggingFace token required.**
- **No download step** — data is read directly from the ROOT files.
- Class names match the ROOT file prefixes exactly.

Available classes:

| Class | Process | Prongs |
|-------|---------|--------|
| `HToBB` | H → bb̄ | 2 |
| `HToCC` | H → cc̄ | 2 |
| `HToGG` | H → gg | 2 |
| `HToWW2Q1L` | H → WW → qqℓν | 3 |
| `HToWW4Q` | H → WW → qqqq | 4 |
| `TTBar` | tt̄ (fully hadronic) | 5 |
| `TTBarLep` | tt̄ (semi-leptonic) | 4 |
| `WToQQ` | W → qq̄ | 2 |
| `ZJetsToNuNu` | Z → νν̄ + jets | 1 |
| `ZToQQ` | Z → qq̄ | 2 |

### Switching between datasets

The key config fields to set when using JetClass (v1):

```yaml
# Required
dataset:
  source: "jetclass1"
  jetclass1_path: "/path/to/JetClass"   # dir containing train_100M/, val_5M/, test_20M/
  jetclass1_split: "train"              # "train", "val", or "test"
  num_jets_per_class: 5000
  classes:
    - HToBB
    - HToCC
    - WToQQ
    - ZToQQ
```

Because the data is already local, the download step is skipped entirely —
go straight to tokenization:

```bash
# JetClass-II workflow (download first)
python -m data.download_jetclass --config configs/default.yaml \
    --override configs/experiments/heavy_flavor.yaml
python -m data.tokenize_jets     --config configs/default.yaml \
    --override configs/experiments/heavy_flavor.yaml --device cuda

# JetClass (v1) workflow (no download needed)
python -m data.tokenize_jets     --config configs/default.yaml \
    --override configs/experiments/jetclass1_higgs.yaml --device cuda
```

The rest of the pipeline (caption generation, training, evaluation) is
identical for both sources.

### Full JetClass (v1) run

Use the provided experiment configs as starting points:

```bash
OVERRIDE=configs/experiments/jetclass1_higgs.yaml

python -m data.tokenize_jets    --config configs/default.yaml --override $OVERRIDE --device cuda
python -m data.generate_captions --config configs/default.yaml --override $OVERRIDE
python -m data.generate_qa      --config configs/default.yaml --override $OVERRIDE
python -m training.train_stage1 --config configs/default.yaml --override $OVERRIDE --device cuda
python -m training.train_stage2 --config configs/default.yaml --override $OVERRIDE --device cuda
python -m eval.evaluate         --config configs/default.yaml --override $OVERRIDE --device cuda
```

Or run everything in one shot:

```bash
python -m scripts.run_pipeline --config configs/default.yaml \
    --override configs/experiments/jetclass1_higgs.yaml
```

---

## Inference: Demo and Interactive Chat

Once training is complete, two scripts let you interact with the fine-tuned
model directly.  Both load the Stage 2 checkpoint and LoRA adapter
automatically from `{data_dir}` (or you can supply explicit paths).

### Batch demo document (`scripts/demo.py`)

Generates a Markdown file showing the model's answers to a curated set of
physics questions for a sample of jets.  The output includes each jet's
ground-truth kinematics alongside the model responses.

```bash
# One jet per class → writes {data_dir}/demo_output.md
python -m scripts.demo --config configs/default.yaml --device cuda

# Three jets from selected classes, custom output path
python -m scripts.demo --config configs/default.yaml \
    --classes X_bb,X_cc,QCD_light --n-per-class 3 \
    --output results/demo.md

# Use a named experiment config
python -m scripts.demo --config configs/default.yaml \
    --override configs/experiments/heavy_flavor.yaml \
    --n-per-class 2
```

**All flags:**

| Flag | Default | Description |
|---|---|---|
| `--config` | `configs/default.yaml` | Base config file |
| `--override` | *(none)* | Experiment override YAML |
| `--output` | `{data_dir}/demo_output.md` | Output Markdown path |
| `--n-per-class` | `1` | Jets to sample per class |
| `--classes` | *(from config)* | Comma-separated class subset |
| `--checkpoint` | *(auto-detected)* | Explicit Stage 2 `final.pt` path |
| `--lora-dir` | *(auto-detected)* | Explicit LoRA adapter directory |
| `--device` | `cuda` | Torch device |
| `--max-new-tokens` | `256` | Max tokens per model response |
| `--temperature` | `0.1` | Sampling temperature |
| `--seed` | `42` | Random seed for jet selection |

The output looks like:

```markdown
## Jet 1/10 — `X_bb` (X → bb̄)
| Property | Value |
|---|---|
| pT | 872.3 GeV |
| Mass (soft-drop) | 124.1 GeV |
| Constituents | 43 |
...

**Q:** What physics process produced this jet?
**A:** This jet was produced by a heavy resonance decaying into a
bottom quark-antiquark pair (X → bb̄). The two b-quarks each
hadronise into a subjet, giving the characteristic 2-prong structure...
```

### Interactive chat (`scripts/chat.py`)

A terminal REPL for asking freeform questions about a single jet.
The user picks a jet by class or ID; the model sees only its VQ-VAE tokens.

```bash
# Random jet
python -m scripts.chat --config configs/default.yaml --device cuda

# Start on a random jet from a specific class
python -m scripts.chat --config configs/default.yaml --jet-class QCD_light

# Start on a specific jet by ID
python -m scripts.chat --config configs/default.yaml --jet-id X_bb_00042

# With an experiment override
python -m scripts.chat --config configs/default.yaml \
    --override configs/experiments/heavy_flavor.yaml
```

The session looks like:

```
═══════════════════════════════════════════════════
  PhysLLaVA Interactive Chat
  Type a question about the jet shown below.
  Commands: /new  /new <class|jet_id>  /info  /suggest  /quit
═══════════════════════════════════════════════════

┌─────────────────────────────────────────────────┐
│  Jet: HToWW4Q_01337                             │
│  Class: HToWW4Q                                 │
│  Physics: H → WW → qqqq (4-prong hadronic)     │
├─────────────────────────────────────────────────┤
│  pT                    643.2 GeV                │
│  Mass (soft-drop)      81.4 GeV                 │
│  Constituents          67                       │
│  τ₂/τ₁                0.412                    │
└─────────────────────────────────────────────────┘

You: What makes this jet unusual compared to a QCD jet?
PhysLLaVA: This jet shows a characteristic 4-prong structure from
the X → WW → qqqq decay chain...

You: /new X_bb
[loads a new X_bb jet and displays its properties]

You: /suggest
[prints a list of example questions]

You: /quit
```

**In-session commands:**

| Command | Description |
|---|---|
| `/new` | Pick a new random jet |
| `/new <class>` | Pick a random jet from a class (e.g. `/new QCD_light`) |
| `/new <jet_id>` | Load a specific jet (e.g. `/new X_bb_00042`) |
| `/info` | Re-display the current jet's kinematics |
| `/suggest` | Print example questions to try |
| `/quit` or `/exit` | Exit |

**All flags:**

| Flag | Default | Description |
|---|---|---|
| `--config` | `configs/default.yaml` | Base config file |
| `--override` | *(none)* | Experiment override YAML |
| `--jet-id` | *(random)* | Start on a specific jet ID |
| `--jet-class` | *(random)* | Start on a random jet from this class |
| `--checkpoint` | *(auto-detected)* | Explicit Stage 2 `final.pt` path |
| `--lora-dir` | *(auto-detected)* | Explicit LoRA adapter directory |
| `--device` | `cuda` | Torch device |
| `--max-new-tokens` | `256` | Max tokens per response |
| `--temperature` | `0.1` | Sampling temperature |

---

## Physics Encoder Backends

PhysLLaVA supports two physics encoder backends, selected via `physics_encoder.type`
in the config.

### `type: "custom"` (default)

Trains a 6-layer causal transformer from scratch on the VQ-VAE token sequences.
The encoder is jointly trained with the MLP projector during Stage 1 and Stage 2.

**When to use:** Default choice.  Faster setup, no external checkpoint dependency.
Works well when you have enough Stage 1 caption data to align the encoder.

```yaml
physics_encoder:
  type: "custom"
  vocab_size: 8192      # must match tokenizer codebook_size
  hidden_dim: 512
  num_layers: 6
  num_heads: 8
  dropout: 0.1
  max_seq_len: 128
```

### `type: "omnijet_foundation"`

Uses the pretrained **OmniJet-alpha generative model** as a frozen backbone.
The generative model was pretrained on millions of jets via autoregressive
next-token prediction and has learned rich, contextualised constituent-level
representations.

We extract the final-layer hidden states (`[B, N, 256]`) from the frozen backbone
and pass them directly to the MLP projector.  No physics encoder training is needed
during Stage 1 — only the projector is trained.

**When to use:** When you want richer physics representations without training an
encoder from scratch.  Especially useful when Stage 1 data is limited.

```yaml
physics_encoder:
  type: "omnijet_foundation"
  freeze: true             # default; should almost always be true
  # omnijet_dir: null      # auto-detected as {data_dir}/omnijet_alpha/
  # generative_checkpoint: null  # auto-detected from omnijet_dir
```

**`projector.input_dim` is auto-set** to `encoder.hidden_dim` (256 for the
standard 8192-token checkpoint) when using `omnijet_foundation`.  You do not need
to set it manually; any value in the config is overridden.

**Prerequisites:** The OmniJet-alpha repository must be cloned and the generative
checkpoint must be present at:

```
{data_dir}/omnijet_alpha/checkpoints/generative_8192_tokens/*.ckpt
```

See `data/tokenize_jets.py` → `setup_omnijet()` for the clone step.

### Summary

| | `custom` | `omnijet_foundation` |
|-|----------|----------------------|
| Encoder trained? | Yes (Stage 1 + 2) | No (frozen) |
| hidden_dim | 512 (configurable) | 256 (fixed by checkpoint) |
| Requires OmniJet checkpoint? | No | Yes |
| Setup complexity | Low | Medium |
| Physics representations | Learned from captions | Pretrained on millions of jets |

### Partial unfreezing (`type: "omnijet_foundation"` only)

The OmniJet backbone can be partially unfrozen to let the last few transformer
layers adapt to the downstream task.  Set `physics_encoder.unfreeze_last_n_layers`
to the number of GPT blocks (out of 3) to unfreeze.  This is applied at the
start of both Stage 1 and Stage 2.

```yaml
physics_encoder:
  type: "omnijet_foundation"
  freeze: true
  unfreeze_last_n_layers: 2   # unfreeze last 2/3 GPT blocks
```

Use a lower Stage 1 learning rate (~5e-4) when unfreezing to avoid
destabilising the pretrained representations.

---

## Training Stages — Detailed Description

PhysLLaVA uses a two-stage LLaVA-style training procedure.  This section
describes exactly what each stage does.

### Stage 1: Feature Alignment

**Goal:** Teach the MLP projector (and optionally the physics encoder) to map
jet token representations into the LLM's embedding space, so that the LLM
can "read" a jet as if it were a sequence of soft text tokens.

**What is frozen:**
- The entire LLM (`meta-llama/Llama-3.1-8B-Instruct`): all 8B parameters are
  frozen and do not receive gradients.
- When `physics_encoder.type = "omnijet_foundation"` with `freeze: true` (and
  `unfreeze_last_n_layers: 0`): the OmniJet backbone is also frozen.

**What is trained:**
- The **MLP projector** (2-layer: `Linear → GELU → Linear`, ~18M parameters).
- When `type = "custom"`: the **6-layer causal transformer physics encoder**
  (~23M parameters) is also trained from scratch.
- When `unfreeze_last_n_layers > 0`: the last N GPT blocks of the OmniJet
  backbone are also unfrozen.
- When `contrastive.enabled: true`: a small **contrastive projection head**
  (~0.3M parameters) is trained alongside, then discarded after Stage 1.

**Loss function:**
The primary loss is the standard **causal language modelling (LM) cross-entropy
loss** on the assistant's response tokens only.  The input format is:

```
User: <jet>\nDescribe this jet.\nAssistant: <response>
```

The `<jet>` placeholder is replaced at forward-pass time by the projected jet
embeddings.  Loss is computed only on the `<response>` tokens (not the
question or the jet embeddings).

When `contrastive.enabled: true`, a **supervised contrastive loss (SupCon)**
is added:

```
L_total = L_LM + λ * L_SupCon
```

The SupCon loss mean-pools the encoder's hidden states over the jet token
sequence, projects them to a 128-dimensional space, and applies the NT-Xent
loss using jet class labels as supervision.  Jets of the same class are pulled
together; jets of different classes are pushed apart.  This directly forces the
encoder to produce class-discriminative representations before Stage 2.

**Training data:**
Caption conversations generated by `data/generate_captions.py` from
`{data_dir}/runs/{run_name}/caption_data/captions.json`.  Each jet gets:
- 3 class-specific **physics reasoning captions** (paired with reasoning
  prompts like "Reason from the observables: what process produced this jet?")
- 3 class-specific **descriptive captions** (paired with descriptive prompts)
- 2 **slot-fill kinematic captions** (observational, varied templates)

All captions are written to teach the model to reason from jet mass,
N-subjettiness ratios, and constituent count to a physics conclusion.

**Typical scale:** ~160K–200K caption conversations for 10 classes × 20K jets.

---

### Stage 2: Instruction Tuning

**Goal:** Fine-tune the model to follow natural-language instructions about jets
and produce accurate, reasoning-based physics responses.

**What is frozen:**
- Nothing: all components receive gradients (though the LLM is fine-tuned only
  through LoRA adapters, not full fine-tuning).

**What is trained:**
- The **physics encoder** (same as Stage 1 — continues training).
- The **MLP projector** (continues training, initialized from Stage 1).
- **LoRA (or DoRA) adapters** inserted into the LLM's linear layers.

**LoRA adapters** are low-rank weight perturbations added to the attention and
MLP projection matrices.  They allow the LLM to acquire jet physics knowledge
without catastrophic forgetting of its general language ability.

**DoRA** (Weight-Decomposed LoRA, `use_dora: true`) decomposes each weight
matrix into magnitude and direction components and applies LoRA only to the
direction.  This consistently outperforms standard LoRA on language-vision
tasks.  It is a drop-in replacement with no inference overhead.

**Target modules:** `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj,
down_proj` — all linear layers in the LLM's attention and MLP blocks.  This is
important: physics domain knowledge lives in the MLP layers as well as
attention, so targeting only attention (q, v) under-adapts the model.

**Loss function:**
The same causal LM cross-entropy loss, computed on assistant response tokens
only.  No contrastive loss in Stage 2 (the encoder is already aligned from
Stage 1).

**Training data:**
A mixture of:
- Stage 1 caption data (same captions)
- QA data generated by `data/generate_qa.py` at three difficulty levels:
  - **Factual** (level 1): process identification, constituent count, prong count,
    b-quark presence.
  - **Kinematic** (level 2): numerical questions about pT, mass, η, τ₂/τ₁.
  - **Reasoning** (level 3): "Given this jet's mass and substructure, identify
    the parent particle"; comparison to QCD; decay topology description.

**Gradient accumulation** is used to simulate a larger effective batch size
(`gradient_accumulation_steps × batch_size`).

**Checkpointing:** Physics encoder + projector weights are saved in `final.pt`.
LoRA adapters are saved separately in `final_lora/` (compatible with PEFT
`PeftModel.from_pretrained`).

---

## Advanced Options

### DoRA (Weight-Decomposed LoRA)

Enable DoRA instead of standard LoRA in Stage 2:

```yaml
stage2:
  lora:
    use_dora: true
    rank: 64     # higher rank recommended with DoRA
    alpha: 128
```

DoRA decomposes weights into magnitude + direction and adapts only the
direction via LoRA.  It consistently outperforms standard LoRA on LLaVA-style
tasks (ICML 2024 Oral paper, arXiv:2402.09353).

### Contrastive Alignment in Stage 1

Enable the supervised contrastive loss on pooled encoder features:

```yaml
stage1:
  contrastive:
    enabled: true
    weight: 0.1      # trade-off with LM loss
    temperature: 0.07
    proj_dim: 128
```

This adds a SupCon loss that pulls together encoder embeddings of same-class
jets and pushes apart different-class jets.  Inspired by Prot2Text-V2
(arXiv:2505.11194), which uses a similar H-SCALE contrastive alignment for
protein-to-language model training.

### Recommended config for best performance

See `configs/experiments/jetclass1_full_omnijet_unfreeze_dora.yaml`:

- OmniJet backbone with last 2/3 GPT blocks unfrozen
- DoRA with rank 64 (vs standard LoRA rank 32)
- Contrastive alignment in Stage 1
- 20K jets per class

---

## Caption Design Philosophy

Captions are the primary training signal for the model to learn *why* a jet
comes from a given physics process.  The caption templates are designed to
teach reasoning chains, not just class labels.

**Bad caption (old):** `"This is a Higgs boson jet."`

This is not useful — any of the 5 Higgs decay modes would produce this caption,
so the model can't distinguish them.

**Good caption (new):**
> "The soft-drop mass of 127.3 GeV falls within the Higgs boson mass window
> (~125 GeV), and τ₂/τ₁ = 0.23 strongly supports a two-prong substructure, as
> expected for a two-body decay.  This jet originates from H → bb̄: the Higgs
> decays to two bottom quarks, each of which hadronizes into a distinct subjet.
> B-hadrons from the b quarks travel a few mm before decaying, leaving displaced
> secondary vertices."

Key principles:
1. **Connect observable → conclusion**: mass window, τ₂/τ₁, τ₃/τ₂ are cited
   explicitly with their implications.
2. **Class-discriminative**: each class has unique templates that cannot
   describe any other class.
3. **Physics reasoning**: the caption explains *why* the observable implies the
   conclusion, building the model's intuition.
4. **Diverse prompts**: captions are paired with varied question prompts
   (descriptive, reasoning, comparative) so the model learns to answer
   many question types.

### Per-class discriminative features used

| Class | Key discriminants |
|-------|------------------|
| `HToBB` | Mass ~125 GeV, τ₂/τ₁ < 0.4, b-hadron displaced vertices |
| `HToCC` | Mass ~125 GeV, τ₂/τ₁ < 0.4, softer displaced vertices (D mesons) |
| `HToGG` | Mass ~125 GeV, τ₂/τ₁ moderate, broader gluon showers, no heavy flavor |
| `HToWW2Q1L` | Mass ~125 GeV, τ₂/τ₁ elevated (3-prong), semi-leptonic topology |
| `HToWW4Q` | Mass ~125 GeV, τ₂/τ₁ high (4-prong), both W hadronic |
| `TTBar` | Mass ~173 GeV, very complex substructure (≥5 prong), two b quarks |
| `TTBarLep` | Mass ~173 GeV, 4-prong, one leptonic W |
| `WToQQ` | Mass ~80 GeV (W mass), τ₂/τ₁ low (2-prong), no heavy flavor |
| `ZJetsToNuNu` | Low soft-drop mass (QCD-like ISR jet), invisible Z |
| `ZToQQ` | Mass ~91 GeV (Z mass), τ₂/τ₁ low (2-prong), no heavy flavor |


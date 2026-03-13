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
QCD_Hbb_Hcc  +  "omnijet_vqvae"  →  "Hbb_Hcc_QCD__omnijet_vqvae"
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
    - Hbb
    - Hcc
    - QCD
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
    --classes Hbb,Hcc,QCD

python -m data.tokenize_jets --config configs/default.yaml \
    --classes Hbb,Hcc,QCD
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

| File | run_name | Description |
|------|----------|-------------|
| `configs/experiments/heavy_flavor.yaml` | `heavy_flavor` | Res2P_bb/cc + QCD_187/185 |
| `configs/experiments/omnijet_foundation.yaml` | `omnijet_foundation` | Frozen OmniJet backbone |
| `configs/experiments/full_res2p.yaml` | `full_res2p` | All Res2P classes + QCD |

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
    --classes Res2P_bb,Res2P_cc,QCD_187 --n-per-class 3 \
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
## Jet 1/10 — `Res2P_bb` (X → bb̄ (bottom quark pair))
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
python -m scripts.chat --config configs/default.yaml --jet-class QCD_187

# Start on a specific jet by ID
python -m scripts.chat --config configs/default.yaml --jet-id Res2P_bb_00042

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
│  Jet: Res2P_WW4q_01337                          │
│  Class: Res2P_WW4q                              │
│  Physics: X → WW → qqqq (4-prong hadronic)     │
├─────────────────────────────────────────────────┤
│  pT                    643.2 GeV                │
│  Mass (soft-drop)      81.4 GeV                 │
│  Constituents          67                       │
│  τ₂/τ₁                0.412                    │
└─────────────────────────────────────────────────┘

You: What makes this jet unusual compared to a QCD jet?
PhysLLaVA: This jet shows a characteristic 4-prong structure from
the X → WW → qqqq decay chain...

You: /new Res2P_bb
[loads a new Res2P_bb jet and displays its properties]

You: /suggest
[prints a list of example questions]

You: /quit
```

**In-session commands:**

| Command | Description |
|---|---|
| `/new` | Pick a new random jet |
| `/new <class>` | Pick a random jet from a class (e.g. `/new QCD_187`) |
| `/new <jet_id>` | Load a specific jet (e.g. `/new Res2P_bb_00042`) |
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

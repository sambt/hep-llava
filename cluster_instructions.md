# PhysLLaVA Cluster Orchestration Instructions

This file contains instructions for a Claude Code instance running on a GPU cluster
to autonomously execute the PhysLLaVA project pipeline. The orchestrator should use
agent teams to parallelize work and manage the multi-stage pipeline.

## Configuration

**Before starting, set these paths in `configs/default.yaml`:**

```yaml
# REQUIRED: Set this to a directory on the cluster with sufficient storage (~100GB)
# This is where datasets, model weights, checkpoints, and artifacts will be stored.
data_dir: "__DATA_DIR__"  # <-- SET THIS (e.g., /scratch/user/physllava_data)
```

Also verify:
- `llm.model_name`: Set to a Llama 3.1 8B model you have access to (may need HuggingFace token)
- `logging.use_wandb`: Set to `true` if wandb is configured, `false` otherwise

**Environment variables (must be set in the shell):**
- `HF_TOKEN` — HuggingFace token for gated model access (Llama 3.1)
- `OPENROUTER_API_KEY` — OpenRouter API key for LLM-generated captions (Strategy 2)

The env var names are configurable via `env.hf_token_var` and `env.openrouter_token_var`
in the config. All enterprise LLM inference (caption generation) goes through OpenRouter
via `data/llm_client.py`. If `OPENROUTER_API_KEY` is not set, LLM captions are skipped
gracefully — Strategies 1 and 3 still produce plenty of training data.

---

## Orchestration Architecture

The orchestrator (main Claude Code instance) should manage the pipeline using
specialized agents for each phase. Use the **Agent tool** with parallel execution
where tasks are independent.

### Agent Team Structure

```
Orchestrator (main Claude Code)
├── Agent 1: Environment Setup
├── Agent 2: Data Pipeline (sequential sub-steps)
│   ├── 2a: Download JetClass-II
│   ├── 2b: Setup OmniJet-alpha + tokenize jets
│   └── 2c: Generate captions + QA data (can parallelize captions & QA)
├── Agent 3: Training Stage 1
├── Agent 4: Training Stage 2
├── Agent 5: Evaluation
└── Agent 6: Results Analysis & Reporting
```

---

## Phase-by-Phase Instructions

### Phase 0: Environment Setup

**Agent task:** Set up the Python environment and verify GPU access.

```bash
# Run the setup script
bash scripts/setup.sh

# Verify the installation
python -c "from model.physllava import PhysLLaVA; print('Model import OK')"
python -c "import torch; assert torch.cuda.is_available(), 'No GPU!'; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

**If conda is not available**, use `pip install -r requirements.txt` in a virtualenv.

**HuggingFace login** (if needed for gated models like Llama 3.1):
```bash
huggingface-cli login --token $HF_TOKEN
```

### Phase 1: Data Pipeline

#### Step 1a: Download JetClass-II

```bash
python -m data.download_jetclass --config configs/default.yaml
```

**Expected output:** `{data_dir}/jetclass2_subset/` with 10 parquet files + metadata.
**Verify:** Check that each class has ~3000 jets.

**Troubleshooting:**
- If the HuggingFace download is slow or fails, try setting `HF_DATASETS_CACHE`.
- If JetClass-II (`jet-universe/jetclass2`) is not available, fall back to JetClass v1
  (`jet-universe/jetclass`). NOTE: v1 has limited truth info — captions will be
  class-label-only. Adjust `data/download_jetclass.py` accordingly.
- The download script uses streaming mode. If memory is an issue, reduce `num_jets_per_class`.

#### Step 1b: Setup OmniJet-alpha & Tokenize Jets

**IMPORTANT:** This step requires careful integration with OmniJet-alpha's codebase.
The agent should:

1. Clone OmniJet-alpha:
   ```bash
   cd {data_dir}
   git clone https://github.com/uhh-pd-ml/omnijet_alpha.git
   cd omnijet_alpha
   pip install -e .  # or install dependencies as specified in their README
   ```

2. **Inspect the repo** to understand:
   - How to load the VQ-VAE model (import paths, class names)
   - Where checkpoints are stored or how to download them
   - What input format the encoder expects (normalization, feature ordering)
   - How the `encode` method works (what it returns)

3. **CRITICAL: Adapt `data/tokenize_jets.py`** based on what you find:
   - Fix import paths for the VQ-VAE model class
   - Fix the model loading code (may use PyTorch Lightning, plain PyTorch, etc.)
   - Fix the encode call signature
   - Ensure input preprocessing matches what OmniJet-alpha expects
   - The example notebook `example_tokenize_and_reconstruct_jets.ipynb` in the repo
     is the best reference for the correct API

4. Run tokenization:
   ```bash
   python -m data.tokenize_jets --config configs/default.yaml --device cuda
   ```

**Expected output:** `{data_dir}/tokenized_jets/` with:
- `tokenized_jets.json` (metadata + token indices)
- `token_indices.npy` (array of VQ-VAE indices)
- `masks.npy` (boolean masks)

**Fallback if OmniJet-alpha doesn't work:**
If the pretrained VQ-VAE can't be loaded, implement a simple discretization fallback:
- Bin each feature (pT_rel, eta_rel, phi_rel) into N bins
- Create composite token index = bin_pt * N^2 + bin_eta * N + bin_phi
- This is less expressive but allows the pipeline to proceed

#### Step 1c: Generate Captions & QA Data

These can run in parallel:

```bash
# Captions (rule-based + template)
python -m data.generate_captions --config configs/default.yaml &

# QA pairs
python -m data.generate_qa --config configs/default.yaml &

wait
```

**Expected output:** `{data_dir}/caption_data/` with:
- `captions.json` (~150K caption conversations)
- `qa_data.json` (~360K QA conversations)
- `qa_factual.json`, `qa_kinematic.json`, `qa_reasoning.json` (split by level)

**LLM-generated captions (Strategy 2):**
If `OPENROUTER_API_KEY` is set, `generate_captions.py` automatically calls OpenRouter
(via `data/llm_client.py`) to generate ~30 rich LLM captions per class using the model
specified in `captions.llm_caption_model`. This runs sequentially (one API call per caption)
so it takes a few minutes. To skip it: `python -m data.generate_captions --skip-llm`.
All LLM inference for data generation goes through OpenRouter — never direct provider APIs.

### Phase 2: Training Stage 1 (Feature Alignment)

```bash
python -m training.train_stage1 --config configs/default.yaml --device cuda
```

**Expected runtime:** 1-2 hours on A100 80GB.

**What to monitor:**
- Loss should decrease steadily
- If loss is stuck or NaN, reduce learning rate (try 5e-4 or 1e-4)
- Check GPU memory usage — if OOM, reduce batch_size to 32 or 16

**Expected output:** `{data_dir}/checkpoints/stage1/final.pt`

**Sanity check after Stage 1:**
Write a quick script to verify the model can produce coherent text:
```python
import torch
from model.physllava import PhysLLaVA
import yaml, numpy as np

with open("configs/default.yaml") as f:
    config = yaml.safe_load(f)

model = PhysLLaVA(
    physics_encoder_config=config["physics_encoder"],
    projector_config=config["projector"],
    llm_name=config["llm"]["model_name"],
)
# Load stage1 checkpoint
ckpt = torch.load(f"{config['data_dir']}/checkpoints/stage1/final.pt", weights_only=True)
model.physics_encoder.load_state_dict(ckpt["physics_encoder"])
model.projector.load_state_dict(ckpt["projector"])
model = model.cuda().eval()

# Test with a random jet
tokens = np.load(f"{config['data_dir']}/tokenized_jets/token_indices.npy")
masks = np.load(f"{config['data_dir']}/tokenized_jets/masks.npy")

jet_tokens = torch.from_numpy(tokens[0:1].astype(np.int64)).cuda()
jet_mask = torch.from_numpy(masks[0:1].astype(bool)).cuda()

prompt = "User: <jet>\nDescribe this jet.\nAssistant:"
enc = model.tokenizer(prompt, return_tensors="pt").to("cuda")

out = model.generate(
    input_ids=enc["input_ids"],
    attention_mask=enc["attention_mask"],
    jet_token_indices=jet_tokens,
    jet_attention_mask=jet_mask,
    max_new_tokens=128,
)
print(model.tokenizer.decode(out[0], skip_special_tokens=True))
```

### Phase 3: Training Stage 2 (Instruction Tuning)

```bash
python -m training.train_stage2 --config configs/default.yaml --device cuda
```

**Expected runtime:** 3-6 hours on A100 80GB.

**What to monitor:**
- Same as Stage 1, but loss should start lower (from pretrained encoder/projector)
- LoRA training is memory-efficient; if OOM, reduce batch_size or increase gradient_accumulation_steps
- With bf16 + LoRA rank 32 + gradient checkpointing, memory usage should be ~40-60GB

**Expected output:**
- `{data_dir}/checkpoints/stage2/final.pt` (encoder + projector weights)
- `{data_dir}/checkpoints/stage2/final_lora/` (LoRA adapter for LLM)

### Phase 4: Evaluation

```bash
python -m eval.evaluate --config configs/default.yaml --device cuda
```

**Expected output:** `{data_dir}/eval_results/` with:
- `eval_results.json` (metrics summary)
- `process_id_responses.json` (detailed model responses)

**Key metrics to report:**
1. **10-class accuracy** for process identification (baseline: 10% random)
2. **Mean relative error** for kinematic QA (pT, mass, constituent count)
3. **Qualitative examples** of reasoning responses

### Phase 5: Results Analysis

After evaluation, the agent should:
1. Parse `eval_results.json` and create a summary
2. Identify failure modes (which classes are confused, which questions fail)
3. Generate confusion matrix visualization
4. Save a human-readable report to `{data_dir}/eval_results/report.md`

---

## Troubleshooting & Adaptation Notes

### Common Issues

1. **OmniJet-alpha import errors:**
   The repo structure may differ from what `tokenize_jets.py` expects.
   The agent should inspect the repo and fix imports. Look at their
   example notebooks for the correct API.

2. **JetClass-II data format:**
   Column names may differ between parquet and ROOT formats.
   If `part_deta`/`part_dphi` are not available, compute them from `part_eta - jet_eta` etc.

3. **Memory issues during training:**
   - Reduce batch_size
   - Increase gradient_accumulation_steps proportionally
   - Enable gradient_checkpointing (already default)
   - Use 4-bit quantization via bitsandbytes if needed

4. **Model generates gibberish after Stage 1:**
   - This is somewhat expected — Stage 1 only trains the projector
   - The quality should improve significantly after Stage 2
   - If it's completely incoherent, check that <jet> token replacement works correctly

5. **Low process identification accuracy:**
   - Check that the keyword extraction in evaluate.py matches model output style
   - Try more permissive matching or switch to LLM-based evaluation
   - May need more training data or more epochs

### Adaptation Checklist

When the cluster agent starts, it should:

- [ ] Update `data_dir` in `configs/default.yaml`
- [ ] Verify GPU access and CUDA version
- [ ] Install all dependencies
- [ ] Clone and inspect OmniJet-alpha, adapt `tokenize_jets.py` as needed
- [ ] Download JetClass-II data
- [ ] Run tokenization
- [ ] Generate caption + QA data
- [ ] Train Stage 1
- [ ] Sanity check Stage 1 outputs
- [ ] Train Stage 2
- [ ] Run evaluation
- [ ] Generate report

---

## File Manifest

```
llava_hep/
├── CLAUDE.md                        # Project conventions
├── cluster_instructions.md          # THIS FILE — orchestration instructions
├── requirements.txt                 # Python dependencies
├── configs/
│   └── default.yaml                 # Main config (SET data_dir!)
├── data/
│   ├── __init__.py
│   ├── download_jetclass.py         # Download JetClass-II subset
│   ├── tokenize_jets.py             # Tokenize with OmniJet-alpha VQ-VAE
│   ├── generate_captions.py         # Rule-based + template + LLM captions
│   ├── generate_qa.py              # Multi-level QA generation
│   └── llm_client.py               # OpenRouter LLM client for caption generation
├── model/
│   ├── __init__.py
│   ├── physics_encoder.py           # Transformer over VQ tokens
│   ├── projector.py                 # MLP projector (LLaVA 1.5 style)
│   └── physllava.py                 # Full model: encoder + projector + LLM
├── training/
│   ├── __init__.py
│   ├── dataset.py                   # PyTorch dataset classes
│   ├── train_stage1.py              # Feature alignment training
│   └── train_stage2.py              # Instruction tuning with LoRA
├── eval/
│   ├── __init__.py
│   └── evaluate.py                  # Evaluation pipeline
└── scripts/
    ├── __init__.py
    └── setup.sh                     # Environment setup script
```

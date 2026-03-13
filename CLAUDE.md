# PhysLLaVA - LLaVA-Style Adapter for Particle Physics Jets

## Project Overview
Multimodal model connecting tokenized particle physics jet data to natural language.
Architecture: OmniJet-alpha VQ-VAE → Physics Encoder → MLP Projector → Llama 3.1 8B.

## Conventions
- Python 3.10+, PyTorch 2.x
- Use `black` for formatting, `ruff` for linting
- Type hints on all function signatures
- Configs in YAML under `configs/`
- All paths configurable via config or CLI args — no hardcoded absolute paths
- Data artifacts go in the directory specified by `data_dir` in configs
- Model checkpoints go in `{data_dir}/checkpoints/`

## Key Components
- `data/` — Data pipeline: download, tokenize, caption/QA generation
- `model/` — PhysicsEncoder, MLPProjector, PhysLLaVA model
- `training/` — Stage 1 (alignment) and Stage 2 (instruction tuning) trainers
- `eval/` — Evaluation pipeline and metrics
- `configs/` — YAML configuration files

## Running
See `cluster_instructions.md` for the full orchestrated pipeline.
Quick manual run: `python -m scripts.run_pipeline --config configs/default.yaml`

## Testing
Run `pytest tests/` for unit tests (when available).
Smoke test: `python -c "from model.physllava import PhysLLaVA; print('OK')"`

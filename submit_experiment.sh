#!/bin/bash
#SBATCH --partition=iaifi_gpu_priority
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/output-%j.out

# Usage:
#   sbatch submit_experiment.sh --stage all [--config configs/default.yaml] [--override configs/experiments/heavy_flavor.yaml] [--classes Res2P_bb,Res2P_cc,QCD_187]
#
# --stage: which pipeline stage(s) to run
#     all        run the full pipeline: download → tokenize → captions → qa → stage1 → stage2 → eval
#     data       download + tokenize + captions + qa
#     train      stage1 + stage2
#     stage1     feature alignment only
#     stage2     instruction tuning only (requires stage1 checkpoint)
#     eval       evaluation only (requires stage2 checkpoint)
#
# --config:    base config file (default: configs/default.yaml)
# --override:  experiment override YAML (optional, e.g. configs/experiments/heavy_flavor.yaml)
# --classes:   comma-separated class list override (optional, e.g. Res2P_bb,Res2P_cc,QCD_187)
# --skip-llm:  skip LLM-generated captions (default: true; set --no-skip-llm to enable)
#
# Examples:
#   # Full run with defaults
#   sbatch submit_experiment.sh --stage all
#
#   # Heavy-flavor experiment
#   sbatch submit_experiment.sh --stage all --override configs/experiments/heavy_flavor.yaml
#
#   # OmniJet foundation encoder, training only
#   sbatch submit_experiment.sh --stage train --override configs/experiments/omnijet_foundation.yaml
#
#   # Quick eval of an existing checkpoint
#   sbatch submit_experiment.sh --stage eval --override configs/experiments/heavy_flavor.yaml

set -euo pipefail

# ── Parse arguments ────────────────────────────────────────────────────────────
STAGE="all"
CONFIG="configs/default.yaml"
OVERRIDE=""
CLASSES=""
SKIP_LLM="--skip-llm"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)    STAGE="$2";    shift 2 ;;
        --config)   CONFIG="$2";   shift 2 ;;
        --override) OVERRIDE="$2"; shift 2 ;;
        --classes)  CLASSES="$2";  shift 2 ;;
        --skip-llm)    SKIP_LLM="--skip-llm";  shift ;;
        --no-skip-llm) SKIP_LLM="";            shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ── Environment ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p slurm_logs

source /n/sw/Miniforge3-25.3.1-0/etc/profile.d/conda.sh
conda activate physllava

# Build the common flags passed to every Python script
COMMON_FLAGS="--config $CONFIG"
[[ -n "$OVERRIDE" ]] && COMMON_FLAGS="$COMMON_FLAGS --override $OVERRIDE"
CLASSES_FLAG=""
[[ -n "$CLASSES" ]] && CLASSES_FLAG="--classes $CLASSES"

echo "========================================"
echo "PhysLLaVA Experiment"
echo "  Stage:    $STAGE"
echo "  Config:   $CONFIG"
echo "  Override: ${OVERRIDE:-<none>}"
echo "  Classes:  ${CLASSES:-<from config>}"
echo "  Job ID:   ${SLURM_JOB_ID:-local}"
echo "========================================"

# ── Pipeline stages ────────────────────────────────────────────────────────────

run_download() {
    echo "--- [$(date +%H:%M:%S)] Downloading JetClass-II ---"
    python -m data.download_jetclass $COMMON_FLAGS $CLASSES_FLAG
}

run_tokenize() {
    echo "--- [$(date +%H:%M:%S)] Tokenizing jets ---"
    python -m data.tokenize_jets $COMMON_FLAGS $CLASSES_FLAG --device cuda
}

run_captions() {
    echo "--- [$(date +%H:%M:%S)] Generating captions ---"
    python -m data.generate_captions $COMMON_FLAGS $SKIP_LLM
}

run_qa() {
    echo "--- [$(date +%H:%M:%S)] Generating QA pairs ---"
    python -m data.generate_qa $COMMON_FLAGS
}

run_stage1() {
    echo "--- [$(date +%H:%M:%S)] Stage 1: feature alignment ---"
    python -m training.train_stage1 $COMMON_FLAGS --device cuda
}

run_stage2() {
    echo "--- [$(date +%H:%M:%S)] Stage 2: instruction tuning ---"
    python -m training.train_stage2 $COMMON_FLAGS --device cuda
}

run_eval() {
    echo "--- [$(date +%H:%M:%S)] Evaluation ---"
    python -m eval.evaluate $COMMON_FLAGS --device cuda
}

# ── Dispatch ───────────────────────────────────────────────────────────────────

case "$STAGE" in
    all)
        run_download
        run_tokenize
        run_captions
        run_qa
        run_stage1
        run_stage2
        run_eval
        ;;
    data)
        run_download
        run_tokenize
        run_captions
        run_qa
        ;;
    train)
        run_stage1
        run_stage2
        ;;
    stage1)
        run_stage1
        ;;
    stage2)
        run_stage2
        ;;
    eval)
        run_eval
        ;;
    *)
        echo "Unknown stage: $STAGE"
        echo "Valid stages: all, data, train, stage1, stage2, eval"
        exit 1
        ;;
esac

echo "--- [$(date +%H:%M:%S)] Done (stage=$STAGE) ---"

"""Evaluation pipeline for PhysLLaVA.

Evaluates:
1. Process identification accuracy (10-class classification)
2. Kinematic QA accuracy (numerical answer extraction)
3. Reasoning quality (qualitative + automated metrics)
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.download_jetclass import CLASS_INFO
from model.physllava import PhysLLaVA
from training.dataset import PhysLLaVADataset


# =============================================================================
# Process identification evaluation
# =============================================================================

PROCESS_ID_PROMPTS = [
    "What type of particle produced this jet?",
    "What physics process generated this jet?",
    "Identify the origin of this jet.",
]

CLASS_KEYWORDS = {
    "Hbb": ["higgs", "h →", "h->", "hbb", "h → bb", "bb̄", "bottom"],
    "Hcc": ["higgs", "hcc", "h → cc", "cc̄", "charm"],
    "Hgg": ["higgs", "hgg", "h → gg", "gluon", "h to gg"],
    "H4q": ["higgs", "h4q", "h → 4q", "ww*", "zz*", "four quark"],
    "Hqql": ["higgs", "hqql", "h → qq", "lepton", "neutrino", "qqℓν"],
    "Zqq": ["z boson", "zqq", "z →", "z->"],
    "Wqq": ["w boson", "wqq", "w →", "w->"],
    "Tbqq": ["top", "tbqq", "t →", "t->", "hadronic", "bqq"],
    "Tbl": ["top", "tbl", "leptonic", "bℓν", "blv"],
    "QCD": ["qcd", "light quark", "gluon jet", "q/g"],
}


def extract_predicted_class(response: str) -> str | None:
    """Extract predicted class from model response using keyword matching."""
    response_lower = response.lower()

    # Score each class by keyword matches
    scores = {}
    for cls, keywords in CLASS_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw.lower() in response_lower)
        if score > 0:
            scores[cls] = score

    if not scores:
        return None

    # Handle ambiguities (e.g., "Higgs" appears in multiple classes)
    # Prefer more specific matches
    return max(scores, key=scores.get)


def evaluate_process_identification(
    model: PhysLLaVA,
    eval_data: list[dict],
    token_indices: np.ndarray,
    masks: np.ndarray,
    jet_id_to_idx: dict,
    device: str = "cuda",
    max_new_tokens: int = 256,
) -> dict:
    """Evaluate process identification accuracy.

    Args:
        model: PhysLLaVA model.
        eval_data: List of jet metadata dicts.
        token_indices: [N, seq_len] array of VQ-VAE tokens.
        masks: [N, seq_len] boolean masks.
        jet_id_to_idx: Mapping from jet_id to index.
        device: Torch device.
        max_new_tokens: Max tokens to generate.

    Returns:
        Dict with accuracy metrics.
    """
    model.eval()

    true_classes = []
    pred_classes = []
    responses = []

    prompt = f"<jet>\n{PROCESS_ID_PROMPTS[0]}"

    for jet_meta in tqdm(eval_data, desc="Process identification"):
        jet_id = jet_meta["jet_id"]
        true_cls = jet_meta["class"]
        idx = jet_id_to_idx[jet_id]

        # Prepare inputs
        jet_tokens = torch.from_numpy(token_indices[idx:idx+1].astype(np.int64)).to(device)
        jet_mask = torch.from_numpy(masks[idx:idx+1].astype(bool)).to(device)

        text_encoding = model.tokenizer(
            f"User: {prompt}\nAssistant:",
            return_tensors="pt",
            padding=True,
        ).to(device)

        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=text_encoding["input_ids"],
                attention_mask=text_encoding["attention_mask"],
                jet_token_indices=jet_tokens,
                jet_attention_mask=jet_mask,
                max_new_tokens=max_new_tokens,
                temperature=0.1,
            )

        response = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Extract only the generated part (after "Assistant:")
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()

        pred_cls = extract_predicted_class(response)

        true_classes.append(true_cls)
        pred_classes.append(pred_cls or "unknown")
        responses.append({"jet_id": jet_id, "true": true_cls, "predicted": pred_cls, "response": response})

    # Compute metrics
    valid_mask = [p != "unknown" for p in pred_classes]
    valid_true = [t for t, v in zip(true_classes, valid_mask) if v]
    valid_pred = [p for p, v in zip(pred_classes, valid_mask) if v]

    results = {
        "total_samples": len(true_classes),
        "valid_predictions": sum(valid_mask),
        "unknown_predictions": len(true_classes) - sum(valid_mask),
        "accuracy": accuracy_score(valid_true, valid_pred) if valid_true else 0.0,
        "per_class_report": classification_report(valid_true, valid_pred, output_dict=True, zero_division=0) if valid_true else {},
        "responses": responses,
    }

    return results


# =============================================================================
# Kinematic QA evaluation
# =============================================================================

def extract_number(text: str) -> float | None:
    """Extract the first number from a text response."""
    # Look for patterns like "123.4 GeV", "approximately 500", etc.
    patterns = [
        r"(\d+\.?\d*)\s*GeV",
        r"(?:approximately|about|roughly|~)\s*(\d+\.?\d*)",
        r"pT\s*(?:=|is|of)\s*(\d+\.?\d*)",
        r"mass\s*(?:=|is|of)\s*(\d+\.?\d*)",
        r"(\d+\.?\d*)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return float(match.group(1))
    return None


def evaluate_kinematic_qa(
    model: PhysLLaVA,
    eval_data: list[dict],
    token_indices: np.ndarray,
    masks: np.ndarray,
    jet_id_to_idx: dict,
    device: str = "cuda",
) -> dict:
    """Evaluate kinematic QA performance."""
    model.eval()

    # Sample some kinematic questions
    questions = {
        "jet_pt": "What is the transverse momentum of this jet?",
        "jet_mass": "What is the mass of this jet?",
        "constituent_count": "How many constituents does this jet have?",
    }

    results_by_question = {}

    for q_key, question in questions.items():
        errors = []
        prompt = f"<jet>\n{question}"

        for jet_meta in tqdm(eval_data[:100], desc=f"Kinematic QA: {q_key}"):
            jet_id = jet_meta["jet_id"]
            idx = jet_id_to_idx[jet_id]

            # Get ground truth
            if q_key == "jet_pt":
                true_val = jet_meta.get("jet_pt", 0)
            elif q_key == "jet_mass":
                true_val = jet_meta.get("jet_sdmass", 0)
            elif q_key == "constituent_count":
                true_val = jet_meta.get("jet_nparticles", 0)
            else:
                continue

            # Prepare inputs
            jet_tokens = torch.from_numpy(token_indices[idx:idx+1].astype(np.int64)).to(device)
            jet_mask = torch.from_numpy(masks[idx:idx+1].astype(bool)).to(device)

            text_encoding = model.tokenizer(
                f"User: {prompt}\nAssistant:",
                return_tensors="pt",
                padding=True,
            ).to(device)

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=text_encoding["input_ids"],
                    attention_mask=text_encoding["attention_mask"],
                    jet_token_indices=jet_tokens,
                    jet_attention_mask=jet_mask,
                    max_new_tokens=128,
                    temperature=0.1,
                )

            response = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            if "Assistant:" in response:
                response = response.split("Assistant:")[-1].strip()

            pred_val = extract_number(response)
            if pred_val is not None and true_val > 0:
                rel_error = abs(pred_val - true_val) / true_val
                errors.append(rel_error)

        results_by_question[q_key] = {
            "num_valid": len(errors),
            "mean_relative_error": float(np.mean(errors)) if errors else None,
            "median_relative_error": float(np.median(errors)) if errors else None,
        }

    return results_by_question


# =============================================================================
# Main evaluation
# =============================================================================

def run_evaluation(
    config: dict,
    data_dir: str,
    checkpoint_path: str | None = None,
    device: str = "cuda",
):
    """Run full evaluation pipeline."""
    eval_cfg = config["eval"]
    output_dir = Path(data_dir) / "eval_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PhysLLaVA Evaluation")
    print("=" * 60)

    # Build model and load checkpoint
    print("Building model...")
    model = PhysLLaVA(
        physics_encoder_config=config["physics_encoder"],
        projector_config=config["projector"],
        llm_name=config["llm"]["model_name"],
        torch_dtype=config["llm"]["torch_dtype"],
        use_flash_attention=config["llm"].get("use_flash_attention", True),
    )

    # Load Stage 2 checkpoint
    if checkpoint_path is None:
        checkpoint_path = Path(data_dir) / "checkpoints" / "stage2" / "final.pt"

    if Path(checkpoint_path).exists():
        print(f"Loading checkpoint from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.physics_encoder.load_state_dict(ckpt["physics_encoder"])
        model.projector.load_state_dict(ckpt["projector"])

        # Load LoRA adapter
        lora_dir = str(checkpoint_path).replace(".pt", "_lora")
        if Path(lora_dir).exists():
            from peft import PeftModel
            model.llm = PeftModel.from_pretrained(model.llm, lora_dir)
    else:
        print(f"WARNING: Checkpoint not found at {checkpoint_path}")

    model = model.to(device)
    model.eval()

    # Load eval data
    tokenized_path = Path(data_dir) / "tokenized_jets" / "tokenized_jets.json"
    with open(tokenized_path) as f:
        all_jets = json.load(f)

    token_indices = np.load(Path(data_dir) / "tokenized_jets" / "token_indices.npy")
    jet_masks = np.load(Path(data_dir) / "tokenized_jets" / "masks.npy")
    jet_id_to_idx = {j["jet_id"]: i for i, j in enumerate(all_jets)}

    # Sample eval jets (balanced across classes)
    n_per_class = eval_cfg.get("num_eval_samples_per_class", 200)
    eval_jets = []
    by_class = defaultdict(list)
    for j in all_jets:
        by_class[j["class"]].append(j)
    for cls, jets in by_class.items():
        eval_jets.extend(jets[:n_per_class])

    print(f"Evaluating on {len(eval_jets)} jets")

    # 1. Process identification
    print("\n--- Process Identification ---")
    process_results = evaluate_process_identification(
        model, eval_jets, token_indices, jet_masks, jet_id_to_idx, device
    )
    print(f"Accuracy: {process_results['accuracy']:.4f}")
    print(f"Valid predictions: {process_results['valid_predictions']}/{process_results['total_samples']}")

    # 2. Kinematic QA
    print("\n--- Kinematic QA ---")
    kinematic_results = evaluate_kinematic_qa(
        model, eval_jets, token_indices, jet_masks, jet_id_to_idx, device
    )
    for q_key, metrics in kinematic_results.items():
        print(f"  {q_key}: mean_rel_error = {metrics['mean_relative_error']}")

    # Save all results
    all_results = {
        "process_identification": {
            k: v for k, v in process_results.items() if k != "responses"
        },
        "kinematic_qa": kinematic_results,
        "config": config,
    }

    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Save detailed responses separately
    responses_path = output_dir / "process_id_responses.json"
    with open(responses_path, "w") as f:
        json.dump(process_results["responses"], f, indent=2)

    print(f"\nResults saved to {output_dir}")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate PhysLLaVA")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_dir = args.data_dir or config["data_dir"]
    run_evaluation(config, data_dir, args.checkpoint, args.device)


if __name__ == "__main__":
    main()

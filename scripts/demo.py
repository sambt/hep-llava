"""Generate a human-readable Markdown demo document for PhysLLaVA.

Loads the fine-tuned model, selects a sample of jets (one per class by
default), asks a curated set of questions about each jet, and writes the
resulting Q&A pairs — along with the jet's ground-truth kinematics — to a
Markdown file.

Usage
-----
Basic (one jet per class, writes demo_output.md):

    python -m scripts.demo --config configs/default.yaml --device cuda

Select specific classes and more jets per class:

    python -m scripts.demo --config configs/default.yaml \\
        --classes Res2P_bb,Res2P_cc,QCD_187 --n-per-class 3

Use a named experiment config:

    python -m scripts.demo --config configs/default.yaml \\
        --override configs/experiments/heavy_flavor.yaml

Specify an explicit checkpoint:

    python -m scripts.demo --config configs/default.yaml \\
        --checkpoint /path/to/final.pt --lora-dir /path/to/final_lora
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from scripts.config import load_config
from scripts.load_model import _find_tokenized_dir, load_model_for_inference


# ---------------------------------------------------------------------------
# Curated questions
# ---------------------------------------------------------------------------

QUESTIONS = [
    "What physics process produced this jet?",
    "How many constituents does this jet have?",
    "What is the transverse momentum of this jet?",
    "Describe the substructure of this jet.",
    "Is this jet consistent with QCD background or a heavy resonance decay? Explain your reasoning.",
]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

CLASS_DESCRIPTIONS = {
    "Res2P_bb": "X → bb̄ (bottom quark pair)",
    "Res2P_cc": "X → cc̄ (charm quark pair)",
    "Res2P_ss": "X → ss̄ (strange quark pair)",
    "Res2P_uu": "X → uū (up quark pair)",
    "Res2P_gg": "X → gg (gluon pair)",
    "Res2P_WW4q": "X → WW → qqqq (4-prong hadronic)",
    "Res2P_WWlv": "X → WW → qqℓν (semi-leptonic)",
    "Res2P_ZZ4q": "X → ZZ → qqqq (4-prong hadronic)",
    "QCD_187": "QCD multijet background",
    "QCD_185": "QCD multijet background (sub-type)",
}


def _jet_header(jet_meta: dict, idx: int, total: int) -> str:
    cls = jet_meta["class"]
    desc = CLASS_DESCRIPTIONS.get(cls, cls)
    lines = [
        f"## Jet {idx}/{total} — `{cls}` ({desc})",
        "",
        "| Property | Value |",
        "|---|---|",
        f"| Jet ID | `{jet_meta['jet_id']}` |",
        f"| pT | {jet_meta.get('jet_pt', 'N/A'):.1f} GeV |" if isinstance(jet_meta.get("jet_pt"), float) else f"| pT | {jet_meta.get('jet_pt', 'N/A')} GeV |",
        f"| Mass (soft-drop) | {jet_meta.get('jet_sdmass', 'N/A'):.1f} GeV |" if isinstance(jet_meta.get("jet_sdmass"), float) else f"| Mass (soft-drop) | {jet_meta.get('jet_sdmass', 'N/A')} GeV |",
        f"| η | {jet_meta.get('jet_eta', 'N/A'):.3f} |" if isinstance(jet_meta.get("jet_eta"), float) else f"| η | {jet_meta.get('jet_eta', 'N/A')} |",
        f"| Constituents | {jet_meta.get('jet_nparticles', 'N/A')} |",
        f"| τ₁ | {jet_meta.get('jet_tau1', 'N/A'):.3f} |" if isinstance(jet_meta.get("jet_tau1"), float) else f"| τ₁ | {jet_meta.get('jet_tau1', 'N/A')} |",
        f"| τ₂ | {jet_meta.get('jet_tau2', 'N/A'):.3f} |" if isinstance(jet_meta.get("jet_tau2"), float) else f"| τ₂ | {jet_meta.get('jet_tau2', 'N/A')} |",
        f"| τ₂/τ₁ | {jet_meta['jet_tau2']/jet_meta['jet_tau1']:.3f} |" if jet_meta.get("jet_tau1") and jet_meta.get("jet_tau2") and jet_meta["jet_tau1"] > 0 else "",
        "",
    ]
    return "\n".join(l for l in lines if l is not None)


def _ask_one(
    model,
    question: str,
    jet_tokens: torch.Tensor,
    jet_mask: torch.Tensor,
    device: str,
    max_new_tokens: int,
    temperature: float,
) -> str:
    prompt = f"User: <jet>\n{question}\nAssistant:"
    encoding = model.tokenizer(prompt, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            jet_token_indices=jet_tokens,
            jet_attention_mask=jet_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
    full_text = model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Extract only the generated portion after "Assistant:"
    if "Assistant:" in full_text:
        return full_text.split("Assistant:")[-1].strip()
    return full_text.strip()


# ---------------------------------------------------------------------------
# Main demo routine
# ---------------------------------------------------------------------------

def run_demo(
    config: dict,
    output_path: str | Path,
    n_per_class: int = 1,
    classes: list[str] | None = None,
    checkpoint_path: str | None = None,
    lora_dir: str | None = None,
    device: str = "cuda",
    max_new_tokens: int = 256,
    temperature: float = 0.1,
    seed: int = 42,
) -> None:
    """Generate the demo Markdown document.

    Args:
        config: Loaded config dict.
        output_path: Path to write the Markdown output.
        n_per_class: Number of jets to sample per class.
        classes: Optional list of classes to include.  Defaults to all in
            config.
        checkpoint_path: Explicit Stage 2 checkpoint path.
        lora_dir: Explicit LoRA adapter directory.
        device: Torch device.
        max_new_tokens: Max tokens per response.
        temperature: Sampling temperature.
        seed: Random seed for jet selection.
    """
    random.seed(seed)
    np.random.seed(seed)
    output_path = Path(output_path)

    # Load tokenized data
    data_dir = Path(config["data_dir"])
    tok_dir = _find_tokenized_dir(data_dir, config)
    if tok_dir is None:
        raise FileNotFoundError(
            f"Could not find tokenized jets under {data_dir}. "
            "Run data.tokenize_jets first."
        )

    print(f"Loading tokenized jets from {tok_dir}")
    with open(tok_dir / "tokenized_jets.json") as f:
        all_jets = json.load(f)
    token_indices = np.load(tok_dir / "token_indices.npy")
    jet_masks = np.load(tok_dir / "masks.npy")
    jet_id_to_idx = {j["jet_id"]: i for i, j in enumerate(all_jets)}

    # Group by class
    by_class: dict[str, list[dict]] = defaultdict(list)
    for j in all_jets:
        by_class[j["class"]].append(j)

    selected_classes = classes or list(by_class.keys())
    selected_jets: list[dict] = []
    for cls in selected_classes:
        pool = by_class.get(cls, [])
        if not pool:
            print(f"WARNING: no jets found for class {cls!r}, skipping.")
            continue
        chosen = random.sample(pool, min(n_per_class, len(pool)))
        selected_jets.extend(chosen)

    print(f"Selected {len(selected_jets)} jets across {len(selected_classes)} classes")

    # Load model
    model = load_model_for_inference(config, checkpoint_path, lora_dir, device)

    # Build Markdown
    lines: list[str] = [
        "# PhysLLaVA — Example Model Interactions",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ",
        f"**Model:** PhysLLaVA (Llama 3.1 8B + PhysicsTokenEncoder + MLPProjector)  ",
        f"**Dataset:** JetClass-II ({', '.join(selected_classes)})  ",
        f"**Checkpoint:** `{checkpoint_path or 'auto-detected'}`  ",
        f"**Device:** {device}",
        "",
        "---",
        "",
        "Each section shows a jet's ground-truth kinematics followed by the "
        "model's answers to a set of physics questions.  The model sees only "
        "the VQ-VAE token sequence — no kinematic labels are given at "
        "inference time.",
        "",
        "---",
        "",
    ]

    total = len(selected_jets)
    for jet_idx, jet_meta in enumerate(selected_jets, start=1):
        print(f"[{jet_idx}/{total}] {jet_meta['jet_id']}")
        lines.append(_jet_header(jet_meta, jet_idx, total))

        idx = jet_id_to_idx[jet_meta["jet_id"]]
        jet_tokens = torch.from_numpy(
            token_indices[idx : idx + 1].astype(np.int64)
        ).to(device)
        jet_mask = torch.from_numpy(jet_masks[idx : idx + 1].astype(bool)).to(device)

        for q in QUESTIONS:
            print(f"  Q: {q[:60]}...")
            answer = _ask_one(
                model, q, jet_tokens, jet_mask, device, max_new_tokens, temperature
            )
            lines.append(f"**Q:** {q}")
            lines.append("")
            lines.append(f"**A:** {answer}")
            lines.append("")

        lines.append("---")
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"\nDemo saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a Markdown demo document with PhysLLaVA Q&A examples."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--override", default=None,
        help="Override YAML config (experiment file)."
    )
    parser.add_argument(
        "--output", default=None,
        help="Output Markdown file path.  Defaults to "
             "{data_dir}/demo_output.md or demo_output.md."
    )
    parser.add_argument(
        "--n-per-class", type=int, default=1,
        help="Number of jets to sample per class (default: 1)."
    )
    parser.add_argument(
        "--classes", default=None,
        help="Comma-separated list of classes to include.  "
             "Defaults to all classes in config."
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--lora-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config, args.override)

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = Path(config["data_dir"]) / "demo_output.md"

    classes = args.classes.split(",") if args.classes else None

    run_demo(
        config=config,
        output_path=output_path,
        n_per_class=args.n_per_class,
        classes=classes,
        checkpoint_path=args.checkpoint,
        lora_dir=args.lora_dir,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

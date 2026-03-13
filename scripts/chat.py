"""Interactive chat with PhysLLaVA about a particle physics jet.

The user selects a jet (by ID, class, or at random), sees its ground-truth
kinematics, and then types freeform physics questions.  The model sees only
the VQ-VAE token sequence and generates natural-language answers.

Usage
-----
Random jet:

    python -m scripts.chat --config configs/default.yaml --device cuda

A specific jet by ID:

    python -m scripts.chat --config configs/default.yaml \\
        --jet-id Res2P_bb_00042

Random jet from a specific class:

    python -m scripts.chat --config configs/default.yaml \\
        --jet-class QCD_187

Interactive commands
--------------------
  /new              Pick a new random jet
  /new <class>      Pick a new jet from a specific class
  /new <jet_id>     Load a specific jet by ID
  /info             Re-display the current jet's properties
  /suggest          Print some example questions to try
  /quit  or  /exit  Exit
"""

from __future__ import annotations

import argparse
import json
import random
import readline  # noqa: F401 — enables arrow-key history in input()
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from scripts.config import load_config
from scripts.load_model import _find_tokenized_dir, load_model_for_inference


# ---------------------------------------------------------------------------
# Suggested questions
# ---------------------------------------------------------------------------

SUGGESTED_QUESTIONS = [
    "What physics process produced this jet?",
    "How many constituents does this jet have?",
    "What is the transverse momentum of this jet?",
    "Describe the substructure of this jet.",
    "What is the τ₂/τ₁ N-subjettiness ratio telling us about this jet?",
    "Is this jet consistent with QCD background or a heavy resonance?",
    "What would be the best discriminating variable to separate this jet from QCD?",
    "How does the mass of this jet compare to the W/Z/Higgs boson mass?",
]

def _class_description(class_name: str) -> str:
    """Return a human-readable description for a JetClass-II label string."""
    from data.download_jetclass import _physics_for_label
    info = _physics_for_label(class_name)
    return f"{info['process']} ({info['decay']})"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_jet_info(jet_meta: dict) -> None:
    cls = jet_meta["class"]
    desc = _class_description(cls)
    tau1 = jet_meta.get("jet_tau1")
    tau2 = jet_meta.get("jet_tau2")
    ratio_str = f"{tau2/tau1:.3f}" if tau1 and tau2 and tau1 > 0 else "N/A"

    print()
    print("┌─────────────────────────────────────────────────┐")
    print(f"│  Jet: {jet_meta['jet_id']:<43}│")
    print(f"│  Class: {cls:<41}│")
    print(f"│  Physics: {desc:<39}│")
    print("├─────────────────────────────────────────────────┤")

    def row(label: str, value: str) -> None:
        print(f"│  {label:<20}  {value:<24}│")

    row("pT", f"{jet_meta.get('jet_pt', 'N/A'):.1f} GeV" if isinstance(jet_meta.get("jet_pt"), float) else str(jet_meta.get("jet_pt", "N/A")))
    row("Mass (soft-drop)", f"{jet_meta.get('jet_sdmass', 'N/A'):.1f} GeV" if isinstance(jet_meta.get("jet_sdmass"), float) else str(jet_meta.get("jet_sdmass", "N/A")))
    row("η", f"{jet_meta.get('jet_eta', 'N/A'):.3f}" if isinstance(jet_meta.get("jet_eta"), float) else str(jet_meta.get("jet_eta", "N/A")))
    row("Constituents", str(jet_meta.get("jet_nparticles", "N/A")))
    row("τ₁", f"{tau1:.3f}" if tau1 is not None else "N/A")
    row("τ₂", f"{tau2:.3f}" if tau2 is not None else "N/A")
    row("τ₂/τ₁", ratio_str)
    print("└─────────────────────────────────────────────────┘")
    print()


def _ask(
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
    if "Assistant:" in full_text:
        return full_text.split("Assistant:")[-1].strip()
    return full_text.strip()


def _select_jet(
    by_class: dict[str, list[dict]],
    jet_id_map: dict[str, dict],
    spec: str | None = None,
) -> dict | None:
    """Select a jet by ID, class name, or randomly.

    Args:
        by_class: Jets grouped by class.
        jet_id_map: jet_id → jet_meta dict.
        spec: ``None`` for random, a jet_id, or a class name.

    Returns:
        A jet_meta dict, or ``None`` if not found.
    """
    if spec is None:
        all_jets = [j for jets in by_class.values() for j in jets]
        return random.choice(all_jets) if all_jets else None

    # Try exact jet_id match first
    if spec in jet_id_map:
        return jet_id_map[spec]

    # Try class name
    if spec in by_class:
        pool = by_class[spec]
        return random.choice(pool) if pool else None

    return None


# ---------------------------------------------------------------------------
# Main interactive loop
# ---------------------------------------------------------------------------

def run_chat(
    config: dict,
    initial_jet_spec: str | None = None,
    checkpoint_path: str | None = None,
    lora_dir: str | None = None,
    device: str = "cuda",
    max_new_tokens: int = 256,
    temperature: float = 0.1,
) -> None:
    """Start the interactive chat session.

    Args:
        config: Loaded config dict.
        initial_jet_spec: Jet ID, class name, or ``None`` for random.
        checkpoint_path: Explicit Stage 2 checkpoint.
        lora_dir: Explicit LoRA adapter directory.
        device: Torch device.
        max_new_tokens: Max tokens per model response.
        temperature: Sampling temperature.
    """
    data_dir = Path(config["data_dir"])

    # Load tokenized data
    tok_dir = _find_tokenized_dir(data_dir, config)
    if tok_dir is None:
        raise FileNotFoundError(
            f"Could not find tokenized jets under {data_dir}. "
            "Run data.tokenize_jets first."
        )

    print(f"Loading tokenized jets from {tok_dir} ...")
    with open(tok_dir / "tokenized_jets.json") as f:
        all_jets = json.load(f)
    token_indices = np.load(tok_dir / "token_indices.npy")
    jet_masks = np.load(tok_dir / "masks.npy")

    jet_id_to_idx = {j["jet_id"]: i for i, j in enumerate(all_jets)}
    jet_id_to_meta = {j["jet_id"]: j for j in all_jets}
    by_class: dict[str, list[dict]] = defaultdict(list)
    for j in all_jets:
        by_class[j["class"]].append(j)

    # Load model
    model = load_model_for_inference(config, checkpoint_path, lora_dir, device)

    print()
    print("═" * 51)
    print("  PhysLLaVA Interactive Chat")
    print("  Type a question about the jet shown below.")
    print("  Commands: /new  /new <class|jet_id>  /info  /suggest  /quit")
    print("═" * 51)

    # Select initial jet
    current_jet = _select_jet(by_class, jet_id_to_meta, initial_jet_spec)
    if current_jet is None:
        print("No jets found. Check that data has been tokenized.")
        return

    _print_jet_info(current_jet)

    available_classes = sorted(by_class.keys())

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # ── Commands ──────────────────────────────────────────────────
        if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
            print("Goodbye!")
            break

        if user_input.lower() == "/info":
            _print_jet_info(current_jet)
            continue

        if user_input.lower() == "/suggest":
            print("\nSome questions to try:")
            for q in SUGGESTED_QUESTIONS:
                print(f"  • {q}")
            print()
            continue

        if user_input.lower().startswith("/new"):
            parts = user_input.split(maxsplit=1)
            spec = parts[1].strip() if len(parts) > 1 else None
            new_jet = _select_jet(by_class, jet_id_to_meta, spec)
            if new_jet is None:
                print(
                    f"  Could not find jet for {spec!r}. "
                    f"Available classes: {', '.join(available_classes)}"
                )
            else:
                current_jet = new_jet
                _print_jet_info(current_jet)
            continue

        if user_input.startswith("/"):
            print(f"  Unknown command: {user_input}")
            print("  Commands: /new  /new <class|jet_id>  /info  /suggest  /quit")
            continue

        # ── Model inference ──────────────────────────────────────────
        idx = jet_id_to_idx[current_jet["jet_id"]]
        jet_tokens = torch.from_numpy(
            token_indices[idx : idx + 1].astype(np.int64)
        ).to(device)
        jet_mask = torch.from_numpy(jet_masks[idx : idx + 1].astype(bool)).to(device)

        print("PhysLLaVA: ", end="", flush=True)
        answer = _ask(
            model, user_input, jet_tokens, jet_mask, device, max_new_tokens, temperature
        )
        print(answer)
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactive chat with PhysLLaVA about particle jets."
    )
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--override", default=None)
    parser.add_argument(
        "--jet-id", default=None,
        help="Start with a specific jet ID (e.g. Res2P_bb_00042)."
    )
    parser.add_argument(
        "--jet-class", default=None,
        help="Start with a random jet from this class (e.g. QCD_187)."
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--lora-dir", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.1)
    args = parser.parse_args()

    config = load_config(args.config, args.override)

    initial_spec = args.jet_id or args.jet_class  # jet_id takes priority

    run_chat(
        config=config,
        initial_jet_spec=initial_spec,
        checkpoint_path=args.checkpoint,
        lora_dir=args.lora_dir,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()

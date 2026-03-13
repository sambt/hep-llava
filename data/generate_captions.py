"""Generate captions for tokenized jets using multiple strategies.

Strategy 1: Rule-based factual captions from truth-level info
Strategy 2: LLM-generated rich captions (via OpenRouter)
Strategy 3: Template-based with slot-filling
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import yaml

from data.download_jetclass import CLASS_INFO
from data.llm_client import chat_completion


# =============================================================================
# Strategy 1: Rule-based factual captions
# =============================================================================

RULE_BASED_TEMPLATES = [
    # Basic identification
    "This is a jet originating from a {particle} {decay_clause}.",
    "This jet was produced by the decay of a {particle} ({process}).",
    "The jet corresponds to a {particle} decaying via {process}.",

    # With jet kinematics
    "This is a {particle} jet with transverse momentum {jet_pt:.0f} GeV and mass {jet_sdmass:.1f} GeV.",
    "A {particle} ({process}) produced this jet. It has pT = {jet_pt:.0f} GeV, η = {jet_eta:.2f}, and soft-drop mass = {jet_sdmass:.1f} GeV.",

    # With constituent info
    "This jet contains {jet_nparticles} reconstructed particles and originates from {process}. The jet pT is {jet_pt:.0f} GeV.",
    "A {n_prong}-prong jet from {process}, containing {jet_nparticles} constituents with pT = {jet_pt:.0f} GeV.",

    # With substructure
    "This {particle} jet ({process}) has N-subjettiness ratio τ₂/τ₁ = {tau21:.3f}, consistent with a {n_prong}-prong structure.",
    "Jet from {process}: pT = {jet_pt:.0f} GeV, mass = {jet_sdmass:.1f} GeV, {jet_nparticles} constituents, τ₂/τ₁ = {tau21:.3f}.",

    # Concise
    "{process} jet, pT = {jet_pt:.0f} GeV.",
    "This is a {particle} jet.",

    # Detailed
    (
        "This jet originates from a {particle} undergoing {process}. "
        "It has a transverse momentum of {jet_pt:.0f} GeV, pseudorapidity η = {jet_eta:.2f}, "
        "soft-drop mass of {jet_sdmass:.1f} GeV, and contains {jet_nparticles} reconstructed particles. "
        "The N-subjettiness values are τ₁ = {jet_tau1:.3f}, τ₂ = {jet_tau2:.3f}, τ₃ = {jet_tau3:.3f}."
    ),

    # QCD-specific
    "This is a QCD jet initiated by a light quark or gluon, with pT = {jet_pt:.0f} GeV and {jet_nparticles} constituents.",

    # Physics context
    (
        "This boosted jet is the result of {process}. "
        "At this energy scale (pT ≈ {jet_pt:.0f} GeV), the decay products are collimated into a single fat jet."
    ),

    # Energy fraction description
    "A {process} jet carrying {jet_energy:.0f} GeV of energy, reconstructed with {jet_nparticles} particles.",
]


def generate_rule_based_caption(jet_meta: dict) -> str:
    """Generate a single rule-based caption for a jet."""
    cls = jet_meta["class"]
    info = CLASS_INFO[cls]

    # Build template variables
    tau21 = (
        jet_meta.get("jet_tau2", 0) / max(jet_meta.get("jet_tau1", 1e-8), 1e-8)
    )

    template_vars = {
        "particle": info["particle"],
        "decay": info["decay"],
        "process": info["process"],
        "decay_clause": f"to {info['decay']}" if cls != "QCD" else "",
        "n_prong": info["n_prongs"],
        "jet_pt": jet_meta.get("jet_pt", 0),
        "jet_eta": jet_meta.get("jet_eta", 0),
        "jet_phi": jet_meta.get("jet_phi", 0),
        "jet_energy": jet_meta.get("jet_energy", 0),
        "jet_sdmass": jet_meta.get("jet_sdmass", 0),
        "jet_nparticles": jet_meta.get("jet_nparticles", 0),
        "jet_tau1": jet_meta.get("jet_tau1", 0),
        "jet_tau2": jet_meta.get("jet_tau2", 0),
        "jet_tau3": jet_meta.get("jet_tau3", 0),
        "tau21": tau21,
    }

    # Filter templates appropriate for this class
    templates = RULE_BASED_TEMPLATES.copy()
    if cls == "QCD":
        # Use QCD-specific template
        templates = [t for t in templates if "QCD" in t or "{process}" in t]
    else:
        # Skip QCD-specific templates
        templates = [t for t in templates if "QCD" not in t]

    template = random.choice(templates)
    try:
        return template.format(**template_vars)
    except (KeyError, ValueError):
        # Fallback to simplest template
        return f"This is a {info['particle']} jet ({info['process']})."


# =============================================================================
# Strategy 3: Template-based with slot-filling (diverse linguistic variants)
# =============================================================================

SLOT_FILL_TEMPLATES = [
    # Observational / descriptive
    "Looking at this jet, we see {jet_nparticles} reconstructed particles with a combined transverse momentum of {jet_pt:.0f} GeV.",
    "The jet exhibits a soft-drop mass of {jet_sdmass:.1f} GeV and contains {jet_nparticles} constituents.",
    "This is a wide-angle jet at η = {jet_eta:.2f} with {jet_nparticles} particles and pT = {jet_pt:.0f} GeV.",
    "A highly collimated jet carrying {jet_energy:.0f} GeV, composed of {jet_nparticles} reconstructed objects.",
    "The reconstructed jet has {jet_nparticles} constituents, transverse momentum {jet_pt:.0f} GeV, and invariant mass {jet_sdmass:.1f} GeV.",

    # Substructure-focused
    "The jet's substructure reveals τ₂/τ₁ = {tau21:.3f}, suggesting a {substructure_description}.",
    "With N-subjettiness values τ₁ = {jet_tau1:.3f} and τ₂ = {jet_tau2:.3f}, this jet's internal structure is {substructure_quality}.",
    "Substructure analysis: τ₁ = {jet_tau1:.3f}, τ₂ = {jet_tau2:.3f}, τ₃ = {jet_tau3:.3f}. The jet mass is {jet_sdmass:.1f} GeV.",

    # Comparative / contextual
    "At a transverse momentum of {jet_pt:.0f} GeV, this jet is in the boosted regime where heavy particle decays are collimated.",
    "This {jet_pt:.0f} GeV jet with mass {jet_sdmass:.1f} GeV {mass_context}.",
    "The {jet_nparticles}-particle jet has a mass-to-pT ratio of {mass_pt_ratio:.3f}.",

    # Active voice, varied sentence structure
    "We observe a jet with pT = {jet_pt:.0f} GeV containing {jet_nparticles} reconstructed particles.",
    "The detector recorded {jet_nparticles} particles forming a jet of {jet_pt:.0f} GeV transverse momentum.",
    "Reconstructed from {jet_nparticles} particles, this jet carries a transverse momentum of {jet_pt:.0f} GeV.",
    "A jet of mass {jet_sdmass:.1f} GeV and {jet_nparticles} constituents was reconstructed at η = {jet_eta:.2f}.",

    # Passive voice variants
    "This jet was reconstructed with a transverse momentum of {jet_pt:.0f} GeV and a soft-drop mass of {jet_sdmass:.1f} GeV.",
    "{jet_nparticles} particles were clustered into this jet, which carries {jet_energy:.0f} GeV of energy.",

    # Technical detail
    "Jet kinematics: pT = {jet_pt:.0f} GeV, η = {jet_eta:.2f}, φ = {jet_phi:.2f}, E = {jet_energy:.0f} GeV, m_SD = {jet_sdmass:.1f} GeV.",
    "The anti-kT R=0.8 jet has pT = {jet_pt:.0f} GeV, {jet_nparticles} constituents, and groomed mass {jet_sdmass:.1f} GeV.",

    # Informal / accessible
    "This jet has about {jet_nparticles} particles and weighs in at {jet_sdmass:.1f} GeV with a pT of {jet_pt:.0f} GeV.",
    "Here we have a {jet_pt:.0f} GeV jet with {jet_nparticles} tracks and a mass of {jet_sdmass:.1f} GeV.",

    # Combined with truth (when class is revealed)
    "This {process} jet was reconstructed with pT = {jet_pt:.0f} GeV, mass = {jet_sdmass:.1f} GeV, and {jet_nparticles} constituents.",
    "The {particle} produced a jet with {jet_nparticles} particles, pT = {jet_pt:.0f} GeV, and τ₂/τ₁ = {tau21:.3f}.",
    "A {process} decay yielded this {jet_pt:.0f} GeV jet containing {jet_nparticles} reconstructed particles.",
    "Originating from {process}, this jet has a mass of {jet_sdmass:.1f} GeV and pT of {jet_pt:.0f} GeV.",

    # Substructure + truth
    (
        "This {particle} jet ({process}) shows {substructure_description} substructure "
        "with τ₂/τ₁ = {tau21:.3f}. It contains {jet_nparticles} particles and has pT = {jet_pt:.0f} GeV."
    ),
    (
        "The jet from {process} has mass {jet_sdmass:.1f} GeV, {jet_nparticles} constituents, "
        "and substructure consistent with a {n_prong}-prong decay (τ₂/τ₁ = {tau21:.3f})."
    ),

    # Energy/momentum flow
    "The jet's energy of {jet_energy:.0f} GeV is distributed among {jet_nparticles} reconstructed particles.",
    "With pT = {jet_pt:.0f} GeV and η = {jet_eta:.2f}, this jet lies within the detector's central region." if abs(0) < 2.0 else "placeholder",

    # Multi-sentence detailed
    (
        "This jet was reconstructed from {jet_nparticles} particles with a total transverse momentum of {jet_pt:.0f} GeV. "
        "The soft-drop grooming algorithm yields a mass of {jet_sdmass:.1f} GeV. "
        "The N-subjettiness ratio τ₂/τ₁ = {tau21:.3f} provides information about the jet's prong structure."
    ),
    (
        "A large-radius jet with {jet_nparticles} constituents. "
        "Its kinematics are: pT = {jet_pt:.0f} GeV, η = {jet_eta:.2f}, mass = {jet_sdmass:.1f} GeV. "
        "It is consistent with a {substructure_description} topology."
    ),

    # Quantitative summary
    "Summary: {jet_nparticles} particles, pT = {jet_pt:.0f} GeV, m = {jet_sdmass:.1f} GeV, τ₂₁ = {tau21:.3f}.",

    # Physics interpretation
    "The mass of {jet_sdmass:.1f} GeV {mass_interpretation}.",
    "This jet's pT of {jet_pt:.0f} GeV places it in the {pt_regime} regime.",

    # Constituent-level (generic)
    "The leading constituent carries a significant fraction of the jet's {jet_pt:.0f} GeV transverse momentum.",
    "Among the {jet_nparticles} constituents, the energy is distributed across charged and neutral particles.",

    # Truth-level decay products (when aux_genpart available)
    "The truth-level decay products include particles with a combined invariant mass near {jet_sdmass:.1f} GeV.",

    # Diverse closings
    "This boosted {particle} jet from {process} is a textbook example of a collimated heavy-particle decay.",
    "The jet properties — mass {jet_sdmass:.1f} GeV, {n_prong} prongs, pT {jet_pt:.0f} GeV — are characteristic of {process}.",

    # Questions as captions (metacognitive)
    "Given {jet_nparticles} particles and mass {jet_sdmass:.1f} GeV, this jet is consistent with a boosted heavy resonance decay.",

    # Counting particles by type (placeholder — will be filled in based on data)
    "The jet contains a mixture of charged hadrons, neutral hadrons, and photons among its {jet_nparticles} constituents.",

    # Mass window context
    "The jet mass of {jet_sdmass:.1f} GeV is {mass_window_comment}.",

    # Two-sentence physics summary
    (
        "This {process} jet has pT = {jet_pt:.0f} GeV and mass = {jet_sdmass:.1f} GeV. "
        "Its {n_prong}-prong substructure is captured by τ₂/τ₁ = {tau21:.3f}."
    ),
]


def _substructure_description(tau21: float, n_prongs: int) -> str:
    """Describe substructure qualitatively."""
    if tau21 < 0.3:
        return "a clear two-prong"
    elif tau21 < 0.5:
        return "a moderately two-prong"
    elif tau21 < 0.7:
        return "a mixed"
    else:
        return "a single-prong or diffuse"


def _substructure_quality(tau21: float) -> str:
    if tau21 < 0.3:
        return "highly two-pronged"
    elif tau21 < 0.5:
        return "moderately substructured"
    else:
        return "relatively diffuse"


def _mass_context(sdmass: float) -> str:
    if 70 < sdmass < 100:
        return "is in the W/Z mass window"
    elif 110 < sdmass < 140:
        return "is consistent with the Higgs boson mass"
    elif 150 < sdmass < 200:
        return "is in the top quark mass range"
    elif sdmass < 30:
        return "is consistent with a light QCD jet"
    else:
        return f"of {sdmass:.1f} GeV suggests a heavy resonance decay"


def _mass_interpretation(sdmass: float) -> str:
    if 75 < sdmass < 95:
        return f"suggests a W or Z boson origin"
    elif 115 < sdmass < 135:
        return "is near the Higgs boson mass of 125 GeV"
    elif 160 < sdmass < 185:
        return "is near the top quark mass"
    elif sdmass < 20:
        return "indicates a light-quark or gluon jet with no heavy resonance"
    else:
        return f"places this jet in an intermediate mass region"


def _mass_window_comment(sdmass: float) -> str:
    if 75 < sdmass < 95:
        return "within the W/Z boson mass window (80-91 GeV)"
    elif 115 < sdmass < 135:
        return "within the Higgs boson mass window (~125 GeV)"
    elif 160 < sdmass < 185:
        return "consistent with a top quark decay (~173 GeV)"
    elif sdmass < 20:
        return "very low, typical of QCD jets"
    else:
        return f"not within standard resonance mass windows"


def _pt_regime(pt: float) -> str:
    if pt > 800:
        return "highly boosted"
    elif pt > 500:
        return "moderately boosted"
    else:
        return "mildly boosted"


def generate_slot_fill_caption(jet_meta: dict) -> str:
    """Generate a template-based caption with slot-filling."""
    cls = jet_meta["class"]
    info = CLASS_INFO[cls]

    tau21 = jet_meta.get("jet_tau2", 0) / max(jet_meta.get("jet_tau1", 1e-8), 1e-8)
    sdmass = jet_meta.get("jet_sdmass", 0)
    pt = jet_meta.get("jet_pt", 0)

    template_vars = {
        "particle": info["particle"],
        "decay": info["decay"],
        "process": info["process"],
        "n_prong": info["n_prongs"],
        "jet_pt": pt,
        "jet_eta": jet_meta.get("jet_eta", 0),
        "jet_phi": jet_meta.get("jet_phi", 0),
        "jet_energy": jet_meta.get("jet_energy", 0),
        "jet_sdmass": sdmass,
        "jet_nparticles": jet_meta.get("jet_nparticles", 0),
        "jet_tau1": jet_meta.get("jet_tau1", 0),
        "jet_tau2": jet_meta.get("jet_tau2", 0),
        "jet_tau3": jet_meta.get("jet_tau3", 0),
        "tau21": tau21,
        "substructure_description": _substructure_description(tau21, info["n_prongs"]),
        "substructure_quality": _substructure_quality(tau21),
        "mass_context": _mass_context(sdmass),
        "mass_interpretation": _mass_interpretation(sdmass),
        "mass_window_comment": _mass_window_comment(sdmass),
        "mass_pt_ratio": sdmass / max(pt, 1e-8),
        "pt_regime": _pt_regime(pt),
    }

    template = random.choice(SLOT_FILL_TEMPLATES)
    try:
        return template.format(**template_vars)
    except (KeyError, ValueError, IndexError):
        return f"This jet has pT = {pt:.0f} GeV, mass = {sdmass:.1f} GeV, and {jet_meta.get('jet_nparticles', 0)} constituents."


# =============================================================================
# Strategy 2: LLM-generated rich captions (via OpenRouter)
# =============================================================================

LLM_CAPTION_SYSTEM_PROMPT = """\
You are a particle physics expert writing natural-language descriptions of \
particle jets from LHC collisions. Given structured metadata about a jet, \
write a single paragraph (2-5 sentences) describing it. Vary your language \
and style — sometimes be technical, sometimes more accessible. Include both \
qualitative physics interpretation and quantitative details from the metadata. \
Do NOT use bullet points or lists. Just output the caption text, nothing else."""


def _format_jet_for_llm(jet_meta: dict) -> str:
    """Format jet metadata as a structured prompt for the LLM."""
    cls = jet_meta["class"]
    info = CLASS_INFO[cls]
    tau21 = jet_meta.get("jet_tau2", 0) / max(jet_meta.get("jet_tau1", 1e-8), 1e-8)

    return (
        f"Jet class: {cls}\n"
        f"Parent particle: {info['particle']}\n"
        f"Decay process: {info['process']} ({info['decay']})\n"
        f"Expected prong structure: {info['n_prongs']}-prong\n"
        f"Jet pT: {jet_meta.get('jet_pt', 0):.1f} GeV\n"
        f"Jet eta: {jet_meta.get('jet_eta', 0):.2f}\n"
        f"Jet energy: {jet_meta.get('jet_energy', 0):.1f} GeV\n"
        f"Soft-drop mass: {jet_meta.get('jet_sdmass', 0):.1f} GeV\n"
        f"Number of constituents: {jet_meta.get('jet_nparticles', 0)}\n"
        f"tau_1: {jet_meta.get('jet_tau1', 0):.4f}\n"
        f"tau_2: {jet_meta.get('jet_tau2', 0):.4f}\n"
        f"tau_3: {jet_meta.get('jet_tau3', 0):.4f}\n"
        f"tau21 ratio: {tau21:.4f}"
    )


def generate_llm_caption(
    jet_meta: dict,
    config: dict,
) -> str | None:
    """Generate a single LLM caption via OpenRouter.

    Returns None if the API call fails (allows graceful degradation).
    """
    caption_cfg = config.get("captions", {})
    model = caption_cfg.get("llm_caption_model", "anthropic/claude-sonnet-4")
    max_tokens = caption_cfg.get("llm_caption_max_tokens", 300)

    prompt = _format_jet_for_llm(jet_meta)

    try:
        return chat_completion(
            messages=[
                {"role": "system", "content": LLM_CAPTION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            model=model,
            max_tokens=max_tokens,
            temperature=0.8,
            config=config,
        )
    except Exception as e:
        print(f"  LLM caption failed for {jet_meta.get('jet_id', '?')}: {e}")
        return None


def generate_llm_captions_for_class(
    jets: list[dict],
    config: dict,
    num_per_class: int = 30,
) -> list[dict]:
    """Generate LLM captions for a sample of jets from one class.

    Args:
        jets: List of jet metadata dicts (all same class).
        config: Full config dict.
        num_per_class: How many LLM captions to generate for this class.

    Returns:
        List of conversation dicts in LLaVA format.
    """
    sampled = random.sample(jets, min(num_per_class, len(jets)))
    conversations = []

    prompts = [
        "<jet>\nDescribe this jet in detail.",
        "<jet>\nWhat can you tell me about this jet?",
        "<jet>\nProvide a rich description of this particle physics jet.",
        "<jet>\nAnalyze this jet and describe what you observe.",
        "<jet>\nDescribe the physics of this jet.",
    ]

    for jet_meta in sampled:
        caption = generate_llm_caption(jet_meta, config)
        if caption:
            conversations.append({
                "id": f"{jet_meta['jet_id']}_llm_{len(conversations)}",
                "jet_id": jet_meta["jet_id"],
                "conversations": [
                    {"from": "human", "value": random.choice(prompts)},
                    {"from": "gpt", "value": caption},
                ],
                "caption_type": "llm_generated",
            })

    return conversations


# =============================================================================
# Caption generation main function
# =============================================================================

def generate_captions_for_jet(
    jet_meta: dict,
    num_rule_based: int = 2,
    num_slot_fill: int = 3,
) -> list[dict]:
    """Generate multiple captions for a single jet.

    Returns a list of conversation dicts in LLaVA format.
    """
    conversations = []

    # Rule-based captions
    for _ in range(num_rule_based):
        caption = generate_rule_based_caption(jet_meta)
        conversations.append({
            "id": f"{jet_meta['jet_id']}_rb_{len(conversations)}",
            "jet_id": jet_meta["jet_id"],
            "conversations": [
                {"from": "human", "value": "<jet>\nDescribe this jet."},
                {"from": "gpt", "value": caption},
            ],
            "caption_type": "rule_based",
        })

    # Slot-fill captions
    for _ in range(num_slot_fill):
        caption = generate_slot_fill_caption(jet_meta)
        # Vary the prompts too
        prompts = [
            "<jet>\nDescribe this jet.",
            "<jet>\nWhat can you tell me about this jet?",
            "<jet>\nProvide a description of this particle physics jet.",
            "<jet>\nAnalyze this jet and describe its properties.",
            "<jet>\nWhat are the characteristics of this jet?",
            "<jet>\nSummarize the key features of this jet.",
        ]
        conversations.append({
            "id": f"{jet_meta['jet_id']}_sf_{len(conversations)}",
            "jet_id": jet_meta["jet_id"],
            "conversations": [
                {"from": "human", "value": random.choice(prompts)},
                {"from": "gpt", "value": caption},
            ],
            "caption_type": "slot_fill",
        })

    return conversations


def generate_all_captions(
    data_dir: str,
    config: dict,
    seed: int = 42,
    skip_llm: bool = False,
) -> Path:
    """Generate captions for all tokenized jets.

    Args:
        data_dir: Root data directory.
        config: Full config dict.
        seed: Random seed.
        skip_llm: If True, skip Strategy 2 (LLM-generated captions).

    Returns:
        Path to output captions JSON.
    """
    random.seed(seed)

    from scripts.config import get_paths
    _paths = get_paths(config)
    tokenized_path = _paths["tokenized_dir"] / "tokenized_jets.json"
    with open(tokenized_path) as f:
        all_jets = json.load(f)

    print(f"Generating captions for {len(all_jets)} jets...")

    # Strategy 1 + 3: Rule-based and slot-fill captions
    all_conversations = []
    for jet_meta in all_jets:
        convos = generate_captions_for_jet(
            jet_meta,
            num_rule_based=2,
            num_slot_fill=3,
        )
        all_conversations.extend(convos)

    print(f"  Rule-based + slot-fill: {len(all_conversations)} conversations")

    # Strategy 2: LLM-generated captions via OpenRouter
    if not skip_llm:
        openrouter_var = config.get("env", {}).get("openrouter_token_var", "OPENROUTER_API_KEY")
        if os.environ.get(openrouter_var):
            print("Generating LLM captions via OpenRouter...")
            num_per_class = config.get("captions", {}).get("num_llm_generated_per_class", 30)

            # Group jets by class
            jets_by_class: dict[str, list] = {}
            for j in all_jets:
                jets_by_class.setdefault(j["class"], []).append(j)

            llm_count = 0
            for cls, jets in jets_by_class.items():
                print(f"  Generating LLM captions for {cls}...")
                llm_convos = generate_llm_captions_for_class(jets, config, num_per_class)
                all_conversations.extend(llm_convos)
                llm_count += len(llm_convos)

            print(f"  LLM-generated: {llm_count} conversations")
        else:
            print(f"  Skipping LLM captions ({openrouter_var} not set)")
    else:
        print("  Skipping LLM captions (--skip-llm flag)")

    # Save
    output_dir = _paths["caption_data_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "captions.json"
    with open(output_path, "w") as f:
        json.dump(all_conversations, f, indent=2)

    print(f"Generated {len(all_conversations)} total caption conversations")
    print(f"Saved to {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate captions for tokenized jets")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--override", type=str, default=None, help="Path to an override YAML config")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM-generated captions (Strategy 2)")
    args = parser.parse_args()

    from scripts.config import load_config

    config = load_config(args.config, args.override)
    if args.data_dir is not None:
        config["data_dir"] = args.data_dir

    data_dir = config["data_dir"]
    generate_all_captions(data_dir, config, args.seed, skip_llm=args.skip_llm)


if __name__ == "__main__":
    main()

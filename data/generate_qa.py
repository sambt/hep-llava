"""Generate question-answering data for tokenized jets.

Three difficulty levels:
- Level 1 (factual): Simple identification and counting
- Level 2 (kinematic): Kinematic quantities and properties
- Level 3 (reasoning): Physics reasoning about the jet
"""

import argparse
import json
import random
from pathlib import Path

import yaml

from data.download_jetclass import CLASS_INFO


# =============================================================================
# Level 1: Factual QA templates
# =============================================================================

FACTUAL_QA = [
    {
        "questions": [
            "What type of particle produced this jet?",
            "What physics process generated this jet?",
            "Identify the origin of this jet.",
            "What particle is this jet from?",
            "Can you identify this jet's parent particle?",
        ],
        "answer_fn": "process_identification",
    },
    {
        "questions": [
            "How many constituents does this jet have?",
            "How many particles make up this jet?",
            "What is the particle multiplicity of this jet?",
            "Count the number of reconstructed particles in this jet.",
        ],
        "answer_fn": "constituent_count",
    },
    {
        "questions": [
            "Is this a QCD jet or a heavy resonance decay?",
            "Does this jet originate from a heavy particle decay?",
            "Is this jet from a light quark/gluon or a massive particle?",
        ],
        "answer_fn": "is_qcd",
    },
    {
        "questions": [
            "How many prongs does this jet have?",
            "What is the expected prong structure of this jet?",
            "Is this a one-prong, two-prong, or multi-prong jet?",
        ],
        "answer_fn": "prong_count",
    },
    {
        "questions": [
            "Does this jet contain b quarks?",
            "Are there bottom quarks in this jet's decay chain?",
        ],
        "answer_fn": "has_b_quarks",
    },
]


# =============================================================================
# Level 2: Kinematic QA templates
# =============================================================================

KINEMATIC_QA = [
    {
        "questions": [
            "What is the transverse momentum of this jet?",
            "What is the jet pT?",
            "How much transverse momentum does this jet carry?",
            "Report the pT of this jet in GeV.",
        ],
        "answer_fn": "jet_pt",
    },
    {
        "questions": [
            "What is the mass of this jet?",
            "What is the soft-drop mass?",
            "What is the groomed jet mass?",
            "Report the invariant mass of this jet.",
        ],
        "answer_fn": "jet_mass",
    },
    {
        "questions": [
            "What is the pseudorapidity of this jet?",
            "Where in the detector is this jet located (η)?",
            "What is the η value of this jet?",
        ],
        "answer_fn": "jet_eta",
    },
    {
        "questions": [
            "What is the jet energy?",
            "How much total energy does this jet carry?",
            "Report the energy of this jet in GeV.",
        ],
        "answer_fn": "jet_energy",
    },
    {
        "questions": [
            "What is the N-subjettiness ratio τ₂/τ₁ for this jet?",
            "What is the τ₂₁ value of this jet?",
            "Report the ratio of τ₂ to τ₁ for this jet.",
        ],
        "answer_fn": "tau21",
    },
    {
        "questions": [
            "What is the mass-to-pT ratio of this jet?",
            "What fraction of the jet pT is its mass?",
        ],
        "answer_fn": "mass_pt_ratio",
    },
]


# =============================================================================
# Level 3: Reasoning QA templates
# =============================================================================

REASONING_QA = [
    {
        "questions": [
            "Based on the jet substructure, what particle likely produced this jet?",
            "Given the jet properties, what is your best guess for the originating particle?",
            "What can you infer about this jet's origin from its kinematic properties?",
            "Use the jet mass and substructure to reason about what created this jet.",
        ],
        "answer_fn": "reason_about_origin",
    },
    {
        "questions": [
            "Is this jet consistent with a two-prong decay? Explain.",
            "Does the substructure suggest a two-body decay?",
            "Based on τ₂/τ₁, is this jet likely from a two-body resonance decay?",
        ],
        "answer_fn": "reason_two_prong",
    },
    {
        "questions": [
            "What does the jet mass tell you about its origin?",
            "Can you identify the parent particle from the jet mass?",
            "Based on the mass alone, what are the candidate parent particles?",
        ],
        "answer_fn": "reason_from_mass",
    },
    {
        "questions": [
            "Compare this jet's properties to a typical QCD jet. What differences do you see?",
            "How does this jet differ from a standard QCD background jet?",
            "Is this jet more signal-like or background-like? Why?",
        ],
        "answer_fn": "compare_to_qcd",
    },
    {
        "questions": [
            "If you were designing a tagger for this type of jet, what features would be most discriminating?",
            "What properties of this jet would help distinguish it from background?",
        ],
        "answer_fn": "tagging_features",
    },
    {
        "questions": [
            "Describe the expected decay topology for this jet's parent particle.",
            "What does the decay chain look like for the particle that produced this jet?",
            "Explain the physics process that led to this jet.",
        ],
        "answer_fn": "describe_topology",
    },
]


# =============================================================================
# Answer generation functions
# =============================================================================

def _answer_process_identification(jet_meta: dict) -> str:
    info = CLASS_INFO[jet_meta["class"]]
    return (
        f"This jet originates from a {info['particle']}. "
        f"The specific process is {info['process']}, where the {info['particle']} "
        f"decays to {info['decay']}."
    )


def _answer_constituent_count(jet_meta: dict) -> str:
    n = jet_meta.get("jet_nparticles", 0)
    return f"This jet contains {n} reconstructed particles (constituents)."


def _answer_is_qcd(jet_meta: dict) -> str:
    cls = jet_meta["class"]
    if cls == "QCD":
        return (
            "This is a QCD jet, initiated by a light quark or gluon. "
            "It does not originate from the decay of a heavy resonance."
        )
    else:
        info = CLASS_INFO[cls]
        return (
            f"This jet originates from a heavy particle decay — specifically, "
            f"a {info['particle']} ({info['process']}). It is not a QCD jet."
        )


def _answer_prong_count(jet_meta: dict) -> str:
    info = CLASS_INFO[jet_meta["class"]]
    n = info["n_prongs"]
    if n == 1:
        return "This is a single-prong (1-prong) jet, typical of QCD jets."
    elif n == 2:
        return f"This is a two-prong jet, consistent with a two-body decay ({info['process']})."
    elif n == 3:
        return f"This is a three-prong jet, consistent with {info['process']}."
    else:
        return f"This jet has a {n}-prong structure from {info['process']}."


def _answer_has_b_quarks(jet_meta: dict) -> str:
    cls = jet_meta["class"]
    b_classes = {"Hbb", "Tbqq", "Tbl"}
    if cls in b_classes:
        return f"Yes, this jet contains b quarks in its decay chain ({CLASS_INFO[cls]['process']})."
    else:
        return f"No, this jet ({CLASS_INFO[cls]['process']}) does not primarily contain b quarks."


def _answer_jet_pt(jet_meta: dict) -> str:
    pt = jet_meta.get("jet_pt", 0)
    return f"The transverse momentum (pT) of this jet is approximately {pt:.0f} GeV."


def _answer_jet_mass(jet_meta: dict) -> str:
    mass = jet_meta.get("jet_sdmass", 0)
    return f"The soft-drop groomed mass of this jet is approximately {mass:.1f} GeV."


def _answer_jet_eta(jet_meta: dict) -> str:
    eta = jet_meta.get("jet_eta", 0)
    region = "central" if abs(eta) < 1.5 else "forward"
    return f"The pseudorapidity η of this jet is {eta:.2f}, placing it in the {region} region of the detector."


def _answer_jet_energy(jet_meta: dict) -> str:
    energy = jet_meta.get("jet_energy", 0)
    return f"The total energy of this jet is approximately {energy:.0f} GeV."


def _answer_tau21(jet_meta: dict) -> str:
    tau1 = jet_meta.get("jet_tau1", 1e-8)
    tau2 = jet_meta.get("jet_tau2", 0)
    ratio = tau2 / max(tau1, 1e-8)
    if ratio < 0.3:
        interp = "strongly suggests a two-prong substructure"
    elif ratio < 0.5:
        interp = "indicates moderate two-prong substructure"
    else:
        interp = "suggests a more single-prong or diffuse structure"
    return f"The N-subjettiness ratio τ₂/τ₁ = {ratio:.3f}, which {interp}."


def _answer_mass_pt_ratio(jet_meta: dict) -> str:
    mass = jet_meta.get("jet_sdmass", 0)
    pt = jet_meta.get("jet_pt", 1)
    ratio = mass / max(pt, 1e-8)
    return f"The mass-to-pT ratio is {ratio:.3f} ({mass:.1f} GeV / {pt:.0f} GeV)."


def _answer_reason_about_origin(jet_meta: dict) -> str:
    info = CLASS_INFO[jet_meta["class"]]
    mass = jet_meta.get("jet_sdmass", 0)
    tau1 = jet_meta.get("jet_tau1", 1e-8)
    tau2 = jet_meta.get("jet_tau2", 0)
    tau21 = tau2 / max(tau1, 1e-8)
    n_particles = jet_meta.get("jet_nparticles", 0)

    reasoning = f"Let me analyze the jet properties. "
    reasoning += f"The jet mass is {mass:.1f} GeV. "

    if 75 < mass < 95:
        reasoning += "This mass is consistent with a W boson (~80 GeV) or Z boson (~91 GeV). "
    elif 115 < mass < 135:
        reasoning += "This mass is near the Higgs boson mass (~125 GeV). "
    elif 160 < mass < 185:
        reasoning += "This mass is near the top quark mass (~173 GeV). "
    elif mass < 20:
        reasoning += "This low mass is typical of QCD jets. "

    if tau21 < 0.35:
        reasoning += f"The τ₂/τ₁ = {tau21:.3f} indicates clear two-prong substructure. "
    elif tau21 > 0.6:
        reasoning += f"The τ₂/τ₁ = {tau21:.3f} suggests single-prong or diffuse structure. "

    reasoning += f"Based on these properties, this jet is most consistent with {info['process']}."
    return reasoning


def _answer_reason_two_prong(jet_meta: dict) -> str:
    info = CLASS_INFO[jet_meta["class"]]
    tau1 = jet_meta.get("jet_tau1", 1e-8)
    tau2 = jet_meta.get("jet_tau2", 0)
    tau21 = tau2 / max(tau1, 1e-8)

    if tau21 < 0.4:
        consistency = "yes, strongly consistent"
        explanation = "The low τ₂/τ₁ ratio indicates that the jet energy is distributed into two distinct subjets."
    elif tau21 < 0.6:
        consistency = "moderately consistent"
        explanation = "The τ₂/τ₁ ratio is in an intermediate range, suggesting possible but not conclusive two-prong structure."
    else:
        consistency = "not strongly consistent"
        explanation = "The high τ₂/τ₁ ratio suggests the jet is more consistent with a single-prong topology."

    expected = f"The true process is {info['process']}, which has a {info['n_prongs']}-prong topology."
    return f"With τ₂/τ₁ = {tau21:.3f}, this jet is {consistency} with a two-prong decay. {explanation} {expected}"


def _answer_reason_from_mass(jet_meta: dict) -> str:
    mass = jet_meta.get("jet_sdmass", 0)
    info = CLASS_INFO[jet_meta["class"]]

    candidates = []
    if 70 < mass < 100:
        candidates.append("W boson (80.4 GeV) or Z boson (91.2 GeV)")
    if 110 < mass < 140:
        candidates.append("Higgs boson (125 GeV)")
    if 155 < mass < 195:
        candidates.append("top quark (173 GeV)")
    if mass < 30:
        candidates.append("light-quark/gluon QCD jet")

    if candidates:
        candidate_str = ", or ".join(candidates)
        return (
            f"The jet mass of {mass:.1f} GeV is consistent with: {candidate_str}. "
            f"The actual origin is {info['process']}."
        )
    else:
        return (
            f"The jet mass of {mass:.1f} GeV doesn't fall cleanly in a standard resonance window. "
            f"The actual origin is {info['process']}."
        )


def _answer_compare_to_qcd(jet_meta: dict) -> str:
    cls = jet_meta["class"]
    info = CLASS_INFO[cls]
    mass = jet_meta.get("jet_sdmass", 0)
    tau1 = jet_meta.get("jet_tau1", 1e-8)
    tau2 = jet_meta.get("jet_tau2", 0)
    tau21 = tau2 / max(tau1, 1e-8)

    if cls == "QCD":
        return (
            f"This is actually a QCD jet. It has a mass of {mass:.1f} GeV and τ₂/τ₁ = {tau21:.3f}. "
            "QCD jets typically have low mass and high τ₂/τ₁ (close to 1), indicating no clear substructure."
        )
    else:
        differences = []
        if mass > 50:
            differences.append(f"its high mass ({mass:.1f} GeV) compared to typical QCD jets (~10-20 GeV)")
        if tau21 < 0.5:
            differences.append(f"its low τ₂/τ₁ ({tau21:.3f}), indicating multi-prong substructure")
        if not differences:
            differences.append("its overall kinematic properties")

        diff_str = " and ".join(differences)
        return (
            f"Compared to a typical QCD jet, this {info['process']} jet differs in {diff_str}. "
            f"These features make it distinguishable from the QCD background."
        )


def _answer_tagging_features(jet_meta: dict) -> str:
    info = CLASS_INFO[jet_meta["class"]]
    cls = jet_meta["class"]

    features = ["Jet mass (soft-drop groomed)", "N-subjettiness ratios (τ₂/τ₁, τ₃/τ₂)"]

    if cls in {"Hbb", "Tbqq", "Tbl"}:
        features.append("b-tagging information (displaced vertices, impact parameters)")
    if cls in {"H4q", "Tbqq"}:
        features.append("τ₃/τ₂ for three-prong vs two-prong discrimination")
    if cls in {"Hqql", "Tbl"}:
        features.append("Lepton identification within the jet")
    if cls == "Hcc":
        features.append("c-tagging (charm hadron identification)")

    features_str = "\n".join(f"- {f}" for f in features)
    return (
        f"For tagging {info['process']} jets, the most discriminating features would be:\n"
        f"{features_str}\n"
        f"Modern approaches use all constituent-level information via graph neural networks or transformers."
    )


def _answer_describe_topology(jet_meta: dict) -> str:
    cls = jet_meta["class"]
    info = CLASS_INFO[cls]

    topologies = {
        "Hbb": "The Higgs boson decays to a bb̄ pair. Each b quark hadronizes, producing two distinct subjets within the fat jet. The b-hadrons travel a few mm before decaying, creating displaced vertices.",
        "Hcc": "The Higgs boson decays to a cc̄ pair. Similar to H→bb̄ but with charm quarks, which produce shorter-lived hadrons and softer displaced vertices.",
        "Hgg": "The Higgs boson decays to two gluons. Each gluon produces a broad shower of hadrons, making this channel harder to distinguish from QCD.",
        "H4q": "The Higgs boson decays to WW* or ZZ*, each of which further decays to a quark pair, yielding four quarks total. This creates a complex four-prong substructure.",
        "Hqql": "The Higgs boson decays to WW* or ZZ*, with one boson decaying to quarks and the other to a lepton-neutrino pair. The jet contains both hadronic and leptonic activity.",
        "Zqq": "The Z boson decays to a quark-antiquark pair, creating a clean two-prong substructure with jet mass near 91 GeV.",
        "Wqq": "The W boson decays to a quark pair (different flavors), creating a two-prong substructure with jet mass near 80 GeV.",
        "Tbqq": "The top quark decays to a b quark and a W boson, which then decays hadronically (W→qq'). This creates a three-prong structure: b-jet + two light jets.",
        "Tbl": "The top quark decays to a b quark and a W boson, which then decays leptonically (W→ℓν). The jet contains a b-subjet plus leptonic activity.",
        "QCD": "This is a QCD jet from a light quark or gluon. The parton undergoes a cascade of gluon emissions and hadronization, producing a roughly conical spray of particles with no distinct prong structure.",
    }

    return topologies.get(cls, f"This jet originates from {info['process']}.")


# Answer function dispatch
ANSWER_FUNCTIONS = {
    "process_identification": _answer_process_identification,
    "constituent_count": _answer_constituent_count,
    "is_qcd": _answer_is_qcd,
    "prong_count": _answer_prong_count,
    "has_b_quarks": _answer_has_b_quarks,
    "jet_pt": _answer_jet_pt,
    "jet_mass": _answer_jet_mass,
    "jet_eta": _answer_jet_eta,
    "jet_energy": _answer_jet_energy,
    "tau21": _answer_tau21,
    "mass_pt_ratio": _answer_mass_pt_ratio,
    "reason_about_origin": _answer_reason_about_origin,
    "reason_two_prong": _answer_reason_two_prong,
    "reason_from_mass": _answer_reason_from_mass,
    "compare_to_qcd": _answer_compare_to_qcd,
    "tagging_features": _answer_tagging_features,
    "describe_topology": _answer_describe_topology,
}


def generate_qa_for_jet(
    jet_meta: dict,
    num_factual: int = 4,
    num_kinematic: int = 4,
    num_reasoning: int = 4,
) -> list[dict]:
    """Generate QA pairs for a single jet across all difficulty levels.

    Returns list of conversation dicts in LLaVA format.
    """
    conversations = []

    def _sample_qa(templates, n, level):
        sampled = random.sample(templates, min(n, len(templates)))
        for qa_template in sampled:
            question = random.choice(qa_template["questions"])
            answer_fn = ANSWER_FUNCTIONS[qa_template["answer_fn"]]
            answer = answer_fn(jet_meta)
            conversations.append({
                "id": f"{jet_meta['jet_id']}_qa_{level}_{len(conversations)}",
                "jet_id": jet_meta["jet_id"],
                "conversations": [
                    {"from": "human", "value": f"<jet>\n{question}"},
                    {"from": "gpt", "value": answer},
                ],
                "qa_level": level,
            })

    _sample_qa(FACTUAL_QA, num_factual, "factual")
    _sample_qa(KINEMATIC_QA, num_kinematic, "kinematic")
    _sample_qa(REASONING_QA, num_reasoning, "reasoning")

    return conversations


def generate_all_qa(
    data_dir: str,
    config: dict,
    seed: int = 42,
) -> Path:
    """Generate QA data for all tokenized jets."""
    random.seed(seed)

    from scripts.config import get_paths
    _paths = get_paths(config)
    tokenized_path = _paths["tokenized_dir"] / "tokenized_jets.json"
    with open(tokenized_path) as f:
        all_jets = json.load(f)

    print(f"Generating QA pairs for {len(all_jets)} jets...")

    all_qa = []
    for jet_meta in all_jets:
        qa_pairs = generate_qa_for_jet(
            jet_meta,
            num_factual=4,
            num_kinematic=4,
            num_reasoning=4,
        )
        all_qa.extend(qa_pairs)

    # Save
    output_dir = _paths["caption_data_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "qa_data.json"
    with open(output_path, "w") as f:
        json.dump(all_qa, f, indent=2)

    # Also save split by difficulty
    for level in ["factual", "kinematic", "reasoning"]:
        level_data = [qa for qa in all_qa if qa["qa_level"] == level]
        level_path = output_dir / f"qa_{level}.json"
        with open(level_path, "w") as f:
            json.dump(level_data, f, indent=2)
        print(f"  {level}: {len(level_data)} QA pairs")

    print(f"Total QA pairs: {len(all_qa)}")
    print(f"Saved to {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate QA data for tokenized jets")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--override", type=str, default=None, help="Path to an override YAML config")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from scripts.config import load_config

    config = load_config(args.config, args.override)
    if args.data_dir is not None:
        config["data_dir"] = args.data_dir

    data_dir = config["data_dir"]
    generate_all_qa(data_dir, config, args.seed)


if __name__ == "__main__":
    main()

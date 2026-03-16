"""Generate captions for tokenized jets using multiple strategies.

Strategy 1: Class-specific physics reasoning captions (rule-based, highly discriminative)
Strategy 2: LLM-generated rich captions (via OpenRouter)
Strategy 3: Template-based with slot-filling (observational / kinematic variety)

The key design principle for Strategy 1 is that every caption should model a
reasoning chain from observable jet properties (mass, N-subjettiness, constituent
count) to a conclusion about the jet's origin.  This teaches the LLM to infer
class identity from physics, not just memorize class names.
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import yaml

from data.download_jetclass import CLASS_INFO, build_class_info
from data.llm_client import chat_completion


# =============================================================================
# Helper functions for observable interpretation
# =============================================================================

def _tau21(jet_meta: dict) -> float:
    return jet_meta.get("jet_tau2", 0) / max(jet_meta.get("jet_tau1", 1e-8), 1e-8)


def _tau32(jet_meta: dict) -> float:
    return jet_meta.get("jet_tau3", 0) / max(jet_meta.get("jet_tau2", 1e-8), 1e-8)


def _substructure_desc(t21: float) -> str:
    if t21 < 0.25:
        return "very clean two-prong"
    elif t21 < 0.4:
        return "clear two-prong"
    elif t21 < 0.55:
        return "moderate two-prong"
    elif t21 < 0.7:
        return "weakly two-prong"
    else:
        return "single-prong or diffuse"


def _mass_window(sdmass: float) -> str:
    if 70 < sdmass < 95:
        return "W/Z boson mass window (80–91 GeV)"
    elif 110 < sdmass < 140:
        return "Higgs boson mass window (~125 GeV)"
    elif 155 < sdmass < 195:
        return "top quark mass region (~173 GeV)"
    elif sdmass < 30:
        return "low-mass QCD-like regime"
    else:
        return f"intermediate mass region ({sdmass:.0f} GeV)"


def _pt_regime(pt: float) -> str:
    if pt > 1000:
        return "highly boosted"
    elif pt > 600:
        return "moderately boosted"
    else:
        return "mildly boosted"


def _n_subjettiness_interp(t21: float, n_prongs: int) -> str:
    """Interpret τ₂/τ₁ given expected prong count."""
    if n_prongs == 1:
        return f"τ₂/τ₁ = {t21:.3f} is consistent with single-prong (QCD-like) substructure"
    elif n_prongs == 2:
        if t21 < 0.4:
            return f"τ₂/τ₁ = {t21:.3f} strongly supports a two-prong substructure, as expected for a two-body decay"
        else:
            return f"τ₂/τ₁ = {t21:.3f} shows moderate two-prong substructure"
    elif n_prongs == 3:
        return f"τ₂/τ₁ = {t21:.3f} is elevated, consistent with a three-prong (or more) substructure"
    else:
        return f"τ₂/τ₁ = {t21:.3f} is high, reflecting the complex multi-prong substructure expected for this decay"


# =============================================================================
# Strategy 1: Class-specific physics reasoning captions
# =============================================================================

# Generic human question prompts (varied so the model learns question diversity)
_DESCRIBE_PROMPTS = [
    "<jet>\nDescribe this jet.",
    "<jet>\nWhat can you tell me about this jet?",
    "<jet>\nAnalyze this jet.",
    "<jet>\nProvide a description of this particle physics jet.",
    "<jet>\nDescribe the physics of this jet.",
    "<jet>\nSummarize this jet.",
    "<jet>\nWhat are the key features of this jet?",
    "<jet>\nCharacterize this jet.",
]

_REASONING_PROMPTS = [
    "<jet>\nWhat physics process most likely produced this jet? Reason from the observables.",
    "<jet>\nBased on the jet's properties, what particle produced it?",
    "<jet>\nUse the jet mass and substructure to identify this jet's origin.",
    "<jet>\nWhat can the N-subjettiness and mass tell you about this jet?",
    "<jet>\nInfer the parent particle from the jet's kinematic properties.",
    "<jet>\nWhat decay process is suggested by this jet's substructure?",
    "<jet>\nGiven the jet observables, what is the most likely production mechanism?",
    "<jet>\nHow do the substructure observables constrain the origin of this jet?",
]

_COMPARISON_PROMPTS = [
    "<jet>\nHow does this jet compare to a typical QCD jet?",
    "<jet>\nWhat distinguishes this jet from QCD background?",
    "<jet>\nWould this jet pass a standard boosted-object tagger? Why or why not?",
]


# Per-class template generator functions.
# Each returns a list of caption strings for a given jet.
# Templates emphasize: mass windows, N-subjettiness reasoning, decay topology.

def _captions_htobb(jet_meta: dict) -> list[str]:
    m = jet_meta.get("jet_sdmass", 0)
    pt = jet_meta.get("jet_pt", 0)
    n = jet_meta.get("jet_nparticles", 0)
    t1 = jet_meta.get("jet_tau1", 1e-8)
    t2 = jet_meta.get("jet_tau2", 0)
    t3 = jet_meta.get("jet_tau3", 0)
    t21 = t2 / max(t1, 1e-8)
    t32 = t3 / max(t2, 1e-8)

    captions = [
        # Mass window + 2-prong reasoning
        f"The soft-drop mass of {m:.1f} GeV falls within the Higgs boson mass window (~125 GeV), "
        f"and τ₂/τ₁ = {t21:.3f} indicates a {_substructure_desc(t21)} structure. "
        f"Together, these features are characteristic of H → bb̄: a boosted Higgs decaying to two bottom quarks, "
        f"each of which hadronizes into a distinct subjet.",

        # b-quark reasoning
        f"This is a H → bb̄ jet: the Higgs boson decays to a bottom quark–antiquark pair at "
        f"pT = {pt:.0f} GeV. The two b quarks hadronize into two collimated subjets, "
        f"producing the observed two-prong substructure (τ₂/τ₁ = {t21:.3f}). "
        f"The soft-drop mass of {m:.1f} GeV reconstructs near the Higgs mass of 125 GeV. "
        f"B-hadrons from the b quarks travel a few millimeters before decaying, leaving displaced tracks.",

        # Discriminative vs HToCC
        f"The jet mass is {m:.1f} GeV, placing it in the Higgs mass window. "
        f"The {_substructure_desc(t21)} substructure (τ₂/τ₁ = {t21:.3f}) is consistent with a two-body decay. "
        f"This jet originates from H → bb̄. Unlike H → cc̄, the b quarks produce longer-lived B-hadrons "
        f"with larger displacement, making double b-tagging an effective discriminant. "
        f"It contains {n} reconstructed particles with pT = {pt:.0f} GeV.",

        # Discriminative vs HToGG
        f"With τ₂/τ₁ = {t21:.3f}, this jet shows {_substructure_desc(t21)} substructure, "
        f"consistent with a two-body b-quark decay. The mass of {m:.1f} GeV points to the Higgs. "
        f"Compared to H → gg, the b-quark subjets are sharper and carry heavier-flavor signatures. "
        f"This is H → bb̄, the dominant Higgs decay mode.",

        # Simple identification
        f"This Higgs boson jet (H → bb̄) has pT = {pt:.0f} GeV, soft-drop mass = {m:.1f} GeV, "
        f"and {n} constituents. The τ₂/τ₁ = {t21:.3f} reflects the two b-quark subjets. "
        f"B-tagging algorithms exploit the displaced decay vertices of the B-hadrons.",

        # τ₃/τ₂ reasoning
        f"The N-subjettiness ratios τ₂/τ₁ = {t21:.3f} and τ₃/τ₂ = {t32:.3f} together confirm a "
        f"two-prong topology: the jet energy is split between two subjets (two b quarks) with "
        f"no significant third prong. The jet mass of {m:.1f} GeV is consistent with the Higgs boson. "
        f"This is H → bb̄.",
    ]
    return captions


def _captions_htocc(jet_meta: dict) -> list[str]:
    m = jet_meta.get("jet_sdmass", 0)
    pt = jet_meta.get("jet_pt", 0)
    n = jet_meta.get("jet_nparticles", 0)
    t1 = jet_meta.get("jet_tau1", 1e-8)
    t2 = jet_meta.get("jet_tau2", 0)
    t21 = t2 / max(t1, 1e-8)

    captions = [
        # Mass + 2-prong + charm reasoning
        f"The soft-drop mass of {m:.1f} GeV is consistent with the Higgs boson (~125 GeV), "
        f"and τ₂/τ₁ = {t21:.3f} indicates {_substructure_desc(t21)} substructure. "
        f"This jet comes from H → cc̄: the Higgs decays to a charm quark–antiquark pair. "
        f"Charm quarks produce D mesons with shorter lifetimes (~10⁻¹³ s) than B-mesons, "
        f"yielding softer displaced tracks compared to H → bb̄.",

        # Discriminative vs HToBB
        f"Like H → bb̄, this H → cc̄ jet has a mass of {m:.1f} GeV near the Higgs mass and "
        f"a {_substructure_desc(t21)} topology (τ₂/τ₁ = {t21:.3f}). "
        f"The key difference: charm quarks produce D mesons with shorter track displacement "
        f"than b quarks, making H → cc̄ harder to tag than H → bb̄ and requiring dedicated "
        f"charm-jet (c-tagging) algorithms.",

        # Physics description
        f"This is a boosted H → cc̄ jet at pT = {pt:.0f} GeV. "
        f"The Higgs decays to two charm quarks, each forming a subjet within the fat jet. "
        f"Soft-drop mass = {m:.1f} GeV, τ₂/τ₁ = {t21:.3f}, {n} constituents. "
        f"The two-prong substructure is visible in the low τ₂/τ₁ value.",

        # Observable chain
        f"Mass = {m:.1f} GeV → Higgs mass window. "
        f"τ₂/τ₁ = {t21:.3f} → {_substructure_desc(t21)} structure → two-body decay. "
        f"No displaced b-vertices → not H → bb̄. "
        f"Conclusion: H → cc̄, where the Higgs decays to two charm quarks. "
        f"pT = {pt:.0f} GeV, {n} constituents.",

        f"This Higgs → charm jet (H → cc̄) has soft-drop mass {m:.1f} GeV and "
        f"τ₂/τ₁ = {t21:.3f}, consistent with a two-prong decay from the Higgs. "
        f"The two charm quarks form D mesons that decay within the tracker, producing "
        f"displaced secondary vertices but with shorter lifetime than b quarks.",
    ]
    return captions


def _captions_htogg(jet_meta: dict) -> list[str]:
    m = jet_meta.get("jet_sdmass", 0)
    pt = jet_meta.get("jet_pt", 0)
    n = jet_meta.get("jet_nparticles", 0)
    t1 = jet_meta.get("jet_tau1", 1e-8)
    t2 = jet_meta.get("jet_tau2", 0)
    t21 = t2 / max(t1, 1e-8)

    captions = [
        # Gluon shower reasoning
        f"The jet mass of {m:.1f} GeV falls in the Higgs mass window, and τ₂/τ₁ = {t21:.3f} "
        f"shows {_substructure_desc(t21)} substructure. "
        f"This is H → gg: the Higgs decays to two gluons via a top-quark loop. "
        f"Each gluon undergoes a broad QCD shower, making this channel broader and harder to tag "
        f"than H → bb̄ or H → cc̄. No heavy-flavor signatures are present.",

        # Discriminative vs bb/cc
        f"While the Higgs mass ({m:.1f} GeV) and two-prong topology (τ₂/τ₁ = {t21:.3f}) are "
        f"shared with H → bb̄ and H → cc̄, this jet has no heavy-flavor content. "
        f"This is H → gg, where the Higgs decays to two gluons. "
        f"Gluon jets are broader and have higher color charge than quark jets, "
        f"resulting in more soft radiation and higher constituent counts ({n} particles).",

        # QCD shower description
        f"H → gg jet: pT = {pt:.0f} GeV, mass = {m:.1f} GeV, {n} constituents. "
        f"The Higgs decays to two gluons (via a top-quark loop), each initiating a broad QCD shower. "
        f"τ₂/τ₁ = {t21:.3f}. The absence of b or c quarks means no displaced tracks; "
        f"discrimination from QCD relies on the jet mass being near 125 GeV.",

        # Broad shower note
        f"The two gluons from H → gg each develop a wide-angle shower, making this jet "
        f"broader than quark jets of the same pT. The soft-drop mass of {m:.1f} GeV "
        f"places it near the Higgs. τ₂/τ₁ = {t21:.3f} reflects the two-prong structure, "
        f"though gluon showers tend to wash out the prong definition relative to quark jets. "
        f"This is at pT = {pt:.0f} GeV with {n} reconstructed particles.",
    ]
    return captions


def _captions_htoww2q1l(jet_meta: dict) -> list[str]:
    m = jet_meta.get("jet_sdmass", 0)
    pt = jet_meta.get("jet_pt", 0)
    n = jet_meta.get("jet_nparticles", 0)
    t1 = jet_meta.get("jet_tau1", 1e-8)
    t2 = jet_meta.get("jet_tau2", 0)
    t3 = jet_meta.get("jet_tau3", 0)
    t21 = t2 / max(t1, 1e-8)
    t32 = t3 / max(t2, 1e-8)

    captions = [
        # 3-prong reasoning
        f"The elevated τ₂/τ₁ = {t21:.3f} and moderate τ₃/τ₂ = {t32:.3f} suggest a "
        f"three-prong substructure, consistent with H → WW* → qqℓν. "
        f"Here, one W decays to two quarks (hadronic) and the other to a lepton + neutrino (leptonic). "
        f"The jet mass of {m:.1f} GeV is near the Higgs; the lepton may fall inside the fat jet, "
        f"while the neutrino is invisible.",

        # Semi-leptonic topology
        f"This is a semi-leptonic Higgs jet: H → WW* → qqℓν. "
        f"One off-shell W decays hadronically (producing a qq̄ pair), the other leptonically (ℓν). "
        f"The hadronic decay and the lepton from the leptonic W both fall within the fat jet, "
        f"creating a three-prong topology (τ₂/τ₁ = {t21:.3f}, τ₃/τ₂ = {t32:.3f}). "
        f"Mass = {m:.1f} GeV, pT = {pt:.0f} GeV, {n} constituents.",

        # Comparison to HToWW4Q
        f"Unlike H → WW → qqqq (fully hadronic, 4-prong), this H → WW → qqℓν jet has "
        f"three prongs: two from the hadronic W and one from the lepton. "
        f"τ₂/τ₁ = {t21:.3f} is higher than for two-prong jets, and τ₃/τ₂ = {t32:.3f} "
        f"reflects the three-body topology. Soft-drop mass ≈ {m:.1f} GeV near the Higgs. "
        f"pT = {pt:.0f} GeV, {n} particles.",

        # Observable chain reasoning
        f"Mass {m:.1f} GeV → Higgs mass window. "
        f"τ₂/τ₁ = {t21:.3f} → not a simple two-prong. "
        f"τ₃/τ₂ = {t32:.3f} → three-prong structure present. "
        f"Semi-leptonic topology → H → WW* → qqℓν. "
        f"The lepton and two quarks each form a subjet, with the neutrino escaping undetected. "
        f"pT = {pt:.0f} GeV, {n} constituents.",
    ]
    return captions


def _captions_htoww4q(jet_meta: dict) -> list[str]:
    m = jet_meta.get("jet_sdmass", 0)
    pt = jet_meta.get("jet_pt", 0)
    n = jet_meta.get("jet_nparticles", 0)
    t1 = jet_meta.get("jet_tau1", 1e-8)
    t2 = jet_meta.get("jet_tau2", 0)
    t3 = jet_meta.get("jet_tau3", 0)
    t21 = t2 / max(t1, 1e-8)
    t32 = t3 / max(t2, 1e-8)

    captions = [
        # 4-prong reasoning
        f"The high τ₂/τ₁ = {t21:.3f} and non-negligible τ₃/τ₂ = {t32:.3f} indicate a complex "
        f"multi-prong substructure, consistent with H → WW* → qqqq. "
        f"Both W bosons decay hadronically, producing four quarks in the fat jet. "
        f"The jet mass of {m:.1f} GeV is near the Higgs. "
        f"With {n} constituents and pT = {pt:.0f} GeV, this is a highly structured boosted jet.",

        # Fully hadronic W decay
        f"This fully hadronic Higgs jet (H → WW → qqqq) has four quarks from two W decays "
        f"all captured in the fat jet. τ₂/τ₁ = {t21:.3f} is elevated (more prongs than 2), "
        f"and τ₃/τ₂ = {t32:.3f} shows structure beyond three prongs. "
        f"The soft-drop mass of {m:.1f} GeV is consistent with the Higgs. "
        f"pT = {pt:.0f} GeV, {n} particles.",

        # Comparison
        f"With four quarks (from H → W⁺W⁻ → qq'qq̄'), this jet has more constituents and "
        f"more complex substructure than two-prong Higgs decays. "
        f"τ₂/τ₁ = {t21:.3f} is higher than for H → bb̄/cc̄/gg, reflecting the 4-prong topology. "
        f"The jet mass of {m:.1f} GeV reconstructs near 125 GeV. "
        f"pT = {pt:.0f} GeV, {n} constituents.",

        # Observable reasoning chain
        f"Mass {m:.1f} GeV → Higgs mass window. "
        f"τ₂/τ₁ = {t21:.3f} is high → not two-prong. "
        f"τ₃/τ₂ = {t32:.3f} → four-prong structure. "
        f"No heavy flavor → hadronic W decays, not bb̄/cc̄. "
        f"Conclusion: H → WW* → qqqq, both W bosons decaying to light quarks. "
        f"This is one of the most challenging Higgs decay channels to tag at pT = {pt:.0f} GeV.",
    ]
    return captions


def _captions_ttbar(jet_meta: dict) -> list[str]:
    m = jet_meta.get("jet_sdmass", 0)
    pt = jet_meta.get("jet_pt", 0)
    n = jet_meta.get("jet_nparticles", 0)
    t1 = jet_meta.get("jet_tau1", 1e-8)
    t2 = jet_meta.get("jet_tau2", 0)
    t3 = jet_meta.get("jet_tau3", 0)
    t21 = t2 / max(t1, 1e-8)
    t32 = t3 / max(t2, 1e-8)

    captions = [
        # Top quark mass reasoning
        f"The jet mass of {m:.1f} GeV and highly complex substructure (τ₂/τ₁ = {t21:.3f}, "
        f"τ₃/τ₂ = {t32:.3f}) are consistent with a fully hadronic tt̄ event. "
        f"Both top quarks decay (t → bW, W → qq'), producing six partons total. "
        f"This fat jet at pT = {pt:.0f} GeV captures some or all of these decay products, "
        f"resulting in the highest constituent count ({n}) and most complex substructure "
        f"of any jet class.",

        # Multi-prong topology
        f"This is a jet from pp → tt̄ (fully hadronic). "
        f"Each top quark decays as t → b + W → b + qq', giving six partons (two b quarks + four light quarks). "
        f"The high τ₂/τ₁ = {t21:.3f} and τ₃/τ₂ = {t32:.3f} reflect this multi-prong topology. "
        f"Soft-drop mass = {m:.1f} GeV. The b quarks provide b-tagging handles. "
        f"pT = {pt:.0f} GeV, {n} constituents.",

        # Mass window reasoning
        f"With soft-drop mass {m:.1f} GeV and a very complex substructure "
        f"(τ₂/τ₁ = {t21:.3f} is high, reflecting ≥5 prongs), this is a top quark jet. "
        f"The top quark (mt ≈ 173 GeV) decays to b + W, and the W then decays to qq'. "
        f"In the fully hadronic mode, both W bosons decay hadronically, giving the maximum "
        f"number of subjets. The {n}-particle jet carries pT = {pt:.0f} GeV.",

        # Comparison to W/Z
        f"Unlike W or Z jets (mass ~80–91 GeV, two-prong), this jet has a higher mass "
        f"of {m:.1f} GeV and far more complex substructure: τ₂/τ₁ = {t21:.3f}, "
        f"τ₃/τ₂ = {t32:.3f}. This complexity is the signature of pp → tt̄ (fully hadronic): "
        f"six quarks from two top decays captured in the fat jet. "
        f"The two b quarks are taggable with b-tagging algorithms.",
    ]
    return captions


def _captions_ttbarlep(jet_meta: dict) -> list[str]:
    m = jet_meta.get("jet_sdmass", 0)
    pt = jet_meta.get("jet_pt", 0)
    n = jet_meta.get("jet_nparticles", 0)
    t1 = jet_meta.get("jet_tau1", 1e-8)
    t2 = jet_meta.get("jet_tau2", 0)
    t3 = jet_meta.get("jet_tau3", 0)
    t21 = t2 / max(t1, 1e-8)
    t32 = t3 / max(t2, 1e-8)

    captions = [
        # Semi-leptonic top
        f"This is a semi-leptonic tt̄ jet: one top decays to bqq' (hadronic W) and the other "
        f"to bℓν (leptonic W). The fat jet at pT = {pt:.0f} GeV captures the hadronic side "
        f"and possibly the b from the leptonic side. "
        f"Soft-drop mass = {m:.1f} GeV. τ₂/τ₁ = {t21:.3f}, τ₃/τ₂ = {t32:.3f}, {n} constituents.",

        # Mass + substructure reasoning
        f"The jet mass of {m:.1f} GeV and complex substructure (τ₂/τ₁ = {t21:.3f}, "
        f"τ₃/τ₂ = {t32:.3f}) are consistent with pp → tt̄ (semi-leptonic). "
        f"One W decays to ℓν, so the jet has 3–4 identifiable prongs (b + qq' + possibly a lepton) "
        f"rather than the full 5–6 of fully hadronic tt̄. "
        f"Two b quarks are present; pT = {pt:.0f} GeV, {n} particles.",

        # Comparison to TTBar
        f"Like fully hadronic tt̄, this semi-leptonic tt̄ jet has high mass ({m:.1f} GeV), "
        f"two b quarks, and complex substructure. However, one W decays leptonically (W → ℓν), "
        f"so this jet has slightly fewer purely hadronic prongs. "
        f"τ₂/τ₁ = {t21:.3f}, τ₃/τ₂ = {t32:.3f}. "
        f"A lepton may be found within the fat jet. pT = {pt:.0f} GeV, {n} constituents.",

        # Observable chain
        f"Mass {m:.1f} GeV → top quark mass region. "
        f"τ₂/τ₁ = {t21:.3f}, τ₃/τ₂ = {t32:.3f} → multi-prong structure, less complex than fully hadronic tt̄. "
        f"Two b quarks present → b-tagging active. "
        f"Semi-leptonic tt̄ (pp → tt̄ → bqq' + b̄ℓν). "
        f"pT = {pt:.0f} GeV, {n} constituents.",
    ]
    return captions


def _captions_wtoquarks(jet_meta: dict) -> list[str]:
    m = jet_meta.get("jet_sdmass", 0)
    pt = jet_meta.get("jet_pt", 0)
    n = jet_meta.get("jet_nparticles", 0)
    t1 = jet_meta.get("jet_tau1", 1e-8)
    t2 = jet_meta.get("jet_tau2", 0)
    t21 = t2 / max(t1, 1e-8)

    # W mass discrimination text
    if 72 < m < 92:
        mass_interp = f"The mass {m:.1f} GeV is within the W boson mass window (~80 GeV), directly confirming a W origin."
    else:
        mass_interp = f"The mass {m:.1f} GeV is in the vicinity of the W boson mass (~80 GeV)."

    captions = [
        # Mass window + 2-prong
        f"{mass_interp} "
        f"τ₂/τ₁ = {t21:.3f} indicates a {_substructure_desc(t21)} structure, "
        f"consistent with W → qq̄: the W boson decays to two light quarks, each forming a subjet. "
        f"No heavy flavor means b-tagging is not effective here. pT = {pt:.0f} GeV, {n} constituents.",

        # W vs Z discrimination
        f"This W → qq̄ jet has soft-drop mass {m:.1f} GeV (near the W boson mass of 80.4 GeV) "
        f"and τ₂/τ₁ = {t21:.3f} indicating {_substructure_desc(t21)} two-body substructure. "
        f"The W decays to two light quarks (e.g., ud̄ or cs̄), giving a clean two-prong jet. "
        f"Discriminating W from Z uses the ~11 GeV mass difference between them. "
        f"pT = {pt:.0f} GeV, {n} particles.",

        # Clean 2-prong description
        f"A clean two-prong jet from W → qq̄ at pT = {pt:.0f} GeV. "
        f"The W boson mass of ~80 GeV is reconstructed in the soft-drop mass: {m:.1f} GeV. "
        f"τ₂/τ₁ = {t21:.3f} reflects the two-quark decay. No b quarks are present. "
        f"This is one of the simplest boosted jet topologies: a single massive particle "
        f"decaying to exactly two light quarks.",

        # Observable chain
        f"Mass {m:.1f} GeV → W boson mass window (80 GeV). "
        f"τ₂/τ₁ = {t21:.3f} → {_substructure_desc(t21)} two-prong. "
        f"No b or c quarks → light quark decay W → qq̄. "
        f"This is a hadronic W boson decay at pT = {pt:.0f} GeV, {n} constituents.",
    ]
    return captions


def _captions_zjets_to_nunu(jet_meta: dict) -> list[str]:
    m = jet_meta.get("jet_sdmass", 0)
    pt = jet_meta.get("jet_pt", 0)
    n = jet_meta.get("jet_nparticles", 0)
    t1 = jet_meta.get("jet_tau1", 1e-8)
    t2 = jet_meta.get("jet_tau2", 0)
    t21 = t2 / max(t1, 1e-8)

    captions = [
        # ISR jet explanation (most distinctive feature)
        f"This jet is not a decay product of the Z boson — the Z decays invisibly to a neutrino pair (Z → νν̄). "
        f"Instead, this is an initial-state radiation (ISR) jet recoiling against the invisible Z. "
        f"The jet has QCD-like substructure: soft-drop mass = {m:.1f} GeV is low, "
        f"τ₂/τ₁ = {t21:.3f} is high, and the substructure looks like a standard QCD jet. "
        f"pT = {pt:.0f} GeV, {n} constituents.",

        # Z→νν + ISR context
        f"In Z → νν̄ + jets events, the Z boson escapes the detector invisibly. "
        f"This fat jet is the hadronic recoil — an ISR or FSR jet, not a Z decay product. "
        f"Its substructure is QCD-like: mass {m:.1f} GeV (low, no resonance peak), "
        f"τ₂/τ₁ = {t21:.3f} suggests {_substructure_desc(t21)} structure. "
        f"pT = {pt:.0f} GeV, {n} particles.",

        # Comparison to other classes
        f"Unlike W or Z jets (which have a mass peak at ~80–91 GeV) or Higgs jets (~125 GeV), "
        f"this ISR jet from Z → νν̄ has a soft-drop mass of only {m:.1f} GeV — QCD-like, with no "
        f"resonance structure. τ₂/τ₁ = {t21:.3f}. The true signature of this event is the "
        f"large missing transverse energy from the invisible neutrinos, not the jet itself. "
        f"pT = {pt:.0f} GeV, {n} constituents.",

        # Concise discriminative
        f"Z → νν̄ + ISR jet: the fat jet has QCD-like properties (mass = {m:.1f} GeV, "
        f"τ₂/τ₁ = {t21:.3f}), since it is a radiation jet and not a Z decay product. "
        f"The Z boson decays invisibly, leaving no hadronic activity from the Z itself. "
        f"pT = {pt:.0f} GeV, {n} constituents.",
    ]
    return captions


def _captions_ztoquarks(jet_meta: dict) -> list[str]:
    m = jet_meta.get("jet_sdmass", 0)
    pt = jet_meta.get("jet_pt", 0)
    n = jet_meta.get("jet_nparticles", 0)
    t1 = jet_meta.get("jet_tau1", 1e-8)
    t2 = jet_meta.get("jet_tau2", 0)
    t21 = t2 / max(t1, 1e-8)

    if 82 < m < 100:
        mass_interp = f"The mass {m:.1f} GeV is within the Z boson mass window (~91 GeV), strongly supporting a Z origin."
    else:
        mass_interp = f"The mass {m:.1f} GeV is near the Z boson mass (~91 GeV)."

    captions = [
        # Z mass window + 2-prong
        f"{mass_interp} "
        f"τ₂/τ₁ = {t21:.3f} indicates {_substructure_desc(t21)} substructure, "
        f"consistent with Z → qq̄: the Z boson decays to two quarks, each forming a subjet. "
        f"No heavy flavor. pT = {pt:.0f} GeV, {n} constituents.",

        # Z vs W discrimination
        f"This Z → qq̄ jet has soft-drop mass {m:.1f} GeV (near the Z mass of 91.2 GeV), "
        f"τ₂/τ₁ = {t21:.3f} confirming a {_substructure_desc(t21)} two-body decay. "
        f"The 11 GeV mass difference between the W (80.4 GeV) and Z (91.2 GeV) is the primary "
        f"discriminant between W → qq̄ and Z → qq̄ jets at the same pT. "
        f"pT = {pt:.0f} GeV, {n} particles.",

        # Clean description
        f"A clean Z → qq̄ jet at pT = {pt:.0f} GeV: the Z boson decays hadronically to two quarks, "
        f"which are collimated into a two-prong fat jet. "
        f"The soft-drop mass of {m:.1f} GeV reconstructs the Z mass. "
        f"τ₂/τ₁ = {t21:.3f}, {n} constituents.",

        # Observable chain
        f"Mass {m:.1f} GeV → Z boson mass window (91 GeV). "
        f"τ₂/τ₁ = {t21:.3f} → {_substructure_desc(t21)} two-prong. "
        f"No b or c quarks → Z → qq̄ (hadronic). "
        f"pT = {pt:.0f} GeV, {n} constituents. "
        f"The Z mass (91 GeV) is ~11 GeV higher than the W mass (80 GeV), which is the key "
        f"discriminator against W → qq̄ jets.",
    ]
    return captions


# JetClass-2 class-specific captions (shorter, covers the Res2P classes)
def _captions_res2p_bb(jet_meta: dict) -> list[str]:
    m = jet_meta.get("jet_sdmass", 0)
    pt = jet_meta.get("jet_pt", 0)
    n = jet_meta.get("jet_nparticles", 0)
    t1 = jet_meta.get("jet_tau1", 1e-8)
    t2 = jet_meta.get("jet_tau2", 0)
    t21 = t2 / max(t1, 1e-8)
    captions = [
        f"A heavy resonance X decaying to bb̄: soft-drop mass {m:.1f} GeV, τ₂/τ₁ = {t21:.3f} "
        f"({_substructure_desc(t21)} two-prong), {n} constituents, pT = {pt:.0f} GeV. "
        f"Two b quarks form distinct subjets with B-hadron displaced vertices.",
        f"X → bb̄ jet at pT = {pt:.0f} GeV. Mass {m:.1f} GeV, τ₂/τ₁ = {t21:.3f}. "
        f"The two b quarks leave secondary vertex signatures enabling b-tagging.",
        f"This two-prong jet (τ₂/τ₁ = {t21:.3f}) from a generic heavy resonance X → bb̄ "
        f"has soft-drop mass {m:.1f} GeV. The bottom quarks produce long-lived B mesons "
        f"detectable as displaced tracks. pT = {pt:.0f} GeV, {n} particles.",
    ]
    return captions


def _captions_res2p_cc(jet_meta: dict) -> list[str]:
    m = jet_meta.get("jet_sdmass", 0)
    pt = jet_meta.get("jet_pt", 0)
    n = jet_meta.get("jet_nparticles", 0)
    t1 = jet_meta.get("jet_tau1", 1e-8)
    t2 = jet_meta.get("jet_tau2", 0)
    t21 = t2 / max(t1, 1e-8)
    captions = [
        f"X → cc̄ jet: mass {m:.1f} GeV, τ₂/τ₁ = {t21:.3f} ({_substructure_desc(t21)}), "
        f"{n} constituents, pT = {pt:.0f} GeV. Charm quarks form D-meson secondary vertices "
        f"with shorter displacement than b quarks.",
        f"Heavy resonance X → cc̄: two charm quarks create two subjets. "
        f"Soft-drop mass {m:.1f} GeV, τ₂/τ₁ = {t21:.3f}, pT = {pt:.0f} GeV. "
        f"C-tagging is needed to distinguish from X → bb̄ (softer displaced vertices).",
    ]
    return captions


def _captions_res2p_ss(jet_meta: dict) -> list[str]:
    m = jet_meta.get("jet_sdmass", 0)
    pt = jet_meta.get("jet_pt", 0)
    n = jet_meta.get("jet_nparticles", 0)
    t1 = jet_meta.get("jet_tau1", 1e-8)
    t2 = jet_meta.get("jet_tau2", 0)
    t21 = t2 / max(t1, 1e-8)
    captions = [
        f"X → ss̄ jet: mass {m:.1f} GeV, τ₂/τ₁ = {t21:.3f} ({_substructure_desc(t21)}), "
        f"pT = {pt:.0f} GeV. Strange quarks produce a two-prong structure but without "
        f"heavy-flavor tagging handles. {n} constituents.",
        f"A generic heavy resonance X decaying to a strange quark pair (ss̄). "
        f"Two-prong jet with mass {m:.1f} GeV, τ₂/τ₁ = {t21:.3f}. "
        f"Harder to distinguish from QCD than b or c jets due to absence of displaced vertices.",
    ]
    return captions


def _captions_res2p_uu(jet_meta: dict) -> list[str]:
    m = jet_meta.get("jet_sdmass", 0)
    pt = jet_meta.get("jet_pt", 0)
    n = jet_meta.get("jet_nparticles", 0)
    t1 = jet_meta.get("jet_tau1", 1e-8)
    t2 = jet_meta.get("jet_tau2", 0)
    t21 = t2 / max(t1, 1e-8)
    captions = [
        f"X → uū (light quark) jet: mass {m:.1f} GeV, τ₂/τ₁ = {t21:.3f}, pT = {pt:.0f} GeV. "
        f"Up quarks form a clean two-prong structure with no displaced-vertex signatures. "
        f"The jet mass is the primary discriminant against QCD. {n} constituents.",
        f"Heavy resonance decaying to uū: two light quarks form a two-prong jet. "
        f"Mass {m:.1f} GeV (resonance peak), τ₂/τ₁ = {t21:.3f}, pT = {pt:.0f} GeV.",
    ]
    return captions


def _captions_res2p_gg(jet_meta: dict) -> list[str]:
    m = jet_meta.get("jet_sdmass", 0)
    pt = jet_meta.get("jet_pt", 0)
    n = jet_meta.get("jet_nparticles", 0)
    t1 = jet_meta.get("jet_tau1", 1e-8)
    t2 = jet_meta.get("jet_tau2", 0)
    t21 = t2 / max(t1, 1e-8)
    captions = [
        f"X → gg jet: mass {m:.1f} GeV, τ₂/τ₁ = {t21:.3f}. "
        f"Two gluons produce broad QCD showers, making this jet broader ({n} constituents) "
        f"than quark jets of the same pT ({pt:.0f} GeV). No heavy-flavor tags available.",
        f"Generic resonance X decaying to two gluons. Gluon jets are broader than quark jets "
        f"due to higher color charge. Mass {m:.1f} GeV, τ₂/τ₁ = {t21:.3f}, "
        f"pT = {pt:.0f} GeV, {n} particles.",
    ]
    return captions


def _captions_res2p_ww4q(jet_meta: dict) -> list[str]:
    m = jet_meta.get("jet_sdmass", 0)
    pt = jet_meta.get("jet_pt", 0)
    n = jet_meta.get("jet_nparticles", 0)
    t1 = jet_meta.get("jet_tau1", 1e-8)
    t2 = jet_meta.get("jet_tau2", 0)
    t3 = jet_meta.get("jet_tau3", 0)
    t21 = t2 / max(t1, 1e-8)
    t32 = t3 / max(t2, 1e-8)
    captions = [
        f"X → WW → qqqq: four quarks from two hadronic W decays in the fat jet. "
        f"Mass {m:.1f} GeV, τ₂/τ₁ = {t21:.3f} (elevated, reflecting multiple prongs), "
        f"τ₃/τ₂ = {t32:.3f}, pT = {pt:.0f} GeV, {n} constituents.",
        f"Four-prong jet from X → WW → qqqq: complex substructure (τ₂/τ₁ = {t21:.3f}, "
        f"τ₃/τ₂ = {t32:.3f}) from four quarks. Mass {m:.1f} GeV, pT = {pt:.0f} GeV.",
    ]
    return captions


def _captions_res2p_wwlv(jet_meta: dict) -> list[str]:
    m = jet_meta.get("jet_sdmass", 0)
    pt = jet_meta.get("jet_pt", 0)
    n = jet_meta.get("jet_nparticles", 0)
    t1 = jet_meta.get("jet_tau1", 1e-8)
    t2 = jet_meta.get("jet_tau2", 0)
    t3 = jet_meta.get("jet_tau3", 0)
    t21 = t2 / max(t1, 1e-8)
    t32 = t3 / max(t2, 1e-8)
    captions = [
        f"X → WW → qqℓν (semi-leptonic): one W decays hadronically, one leptonically. "
        f"Three-prong structure (τ₂/τ₁ = {t21:.3f}, τ₃/τ₂ = {t32:.3f}). "
        f"Mass {m:.1f} GeV, pT = {pt:.0f} GeV, {n} constituents. "
        f"A lepton may be found in the fat jet.",
        f"Semi-leptonic WW decay from heavy resonance X: qq and ℓν prongs. "
        f"Mass {m:.1f} GeV, τ₂/τ₁ = {t21:.3f}, τ₃/τ₂ = {t32:.3f}, pT = {pt:.0f} GeV.",
    ]
    return captions


def _captions_res2p_zz4q(jet_meta: dict) -> list[str]:
    m = jet_meta.get("jet_sdmass", 0)
    pt = jet_meta.get("jet_pt", 0)
    n = jet_meta.get("jet_nparticles", 0)
    t1 = jet_meta.get("jet_tau1", 1e-8)
    t2 = jet_meta.get("jet_tau2", 0)
    t3 = jet_meta.get("jet_tau3", 0)
    t21 = t2 / max(t1, 1e-8)
    t32 = t3 / max(t2, 1e-8)
    captions = [
        f"X → ZZ → qqqq: four quarks from two hadronic Z decays in the fat jet. "
        f"Mass {m:.1f} GeV, τ₂/τ₁ = {t21:.3f}, τ₃/τ₂ = {t32:.3f}, pT = {pt:.0f} GeV, {n} constituents.",
        f"Four-prong jet from X → ZZ → qqqq: τ₂/τ₁ = {t21:.3f} (multi-prong), "
        f"mass {m:.1f} GeV. Similar to X → WW → qqqq but with Z-boson kinematics.",
    ]
    return captions


def _captions_qcd(jet_meta: dict) -> list[str]:
    m = jet_meta.get("jet_sdmass", 0)
    pt = jet_meta.get("jet_pt", 0)
    n = jet_meta.get("jet_nparticles", 0)
    t1 = jet_meta.get("jet_tau1", 1e-8)
    t2 = jet_meta.get("jet_tau2", 0)
    t21 = t2 / max(t1, 1e-8)
    captions = [
        f"This is a QCD background jet from a light quark or gluon. "
        f"Soft-drop mass = {m:.1f} GeV is low (no heavy resonance). "
        f"τ₂/τ₁ = {t21:.3f} indicates {_substructure_desc(t21)} structure — "
        f"QCD jets lack distinct prong structure from a two-body decay. "
        f"pT = {pt:.0f} GeV, {n} constituents.",

        f"QCD jet at pT = {pt:.0f} GeV: soft-drop mass {m:.1f} GeV (QCD-like, low), "
        f"τ₂/τ₁ = {t21:.3f} (no distinct prong structure from resonance decay). "
        f"This is the dominant background to all boosted resonance searches. {n} particles.",

        f"A QCD jet from a light quark or gluon. Unlike jets from massive resonance decays, "
        f"QCD jets have low soft-drop mass ({m:.1f} GeV) and no characteristic mass peak. "
        f"τ₂/τ₁ = {t21:.3f}; any substructure arises from QCD radiation, not a two-body decay. "
        f"pT = {pt:.0f} GeV, {n} particles.",

        f"Background QCD jet: mass {m:.1f} GeV (no mass peak), τ₂/τ₁ = {t21:.3f}. "
        f"This jet lacks the clear substructure (low τ₂/τ₁) or high mass that would indicate "
        f"a boosted W, Z, Higgs, or top quark. pT = {pt:.0f} GeV, {n} constituents.",
    ]
    return captions


# Map class name → caption generator
CLASS_CAPTION_GENERATORS = {
    # JetClass-1
    "HToBB":        _captions_htobb,
    "HToCC":        _captions_htocc,
    "HToGG":        _captions_htogg,
    "HToWW2Q1L":    _captions_htoww2q1l,
    "HToWW4Q":      _captions_htoww4q,
    "TTBar":        _captions_ttbar,
    "TTBarLep":     _captions_ttbarlep,
    "WToQQ":        _captions_wtoquarks,
    "ZJetsToNuNu":  _captions_zjets_to_nunu,
    "ZToQQ":        _captions_ztoquarks,
    # JetClass-2 Res2P
    "Res2P_bb":     _captions_res2p_bb,
    "Res2P_cc":     _captions_res2p_cc,
    "Res2P_ss":     _captions_res2p_ss,
    "Res2P_uu":     _captions_res2p_uu,
    "Res2P_gg":     _captions_res2p_gg,
    "Res2P_WW4q":   _captions_res2p_ww4q,
    "Res2P_WWlv":   _captions_res2p_wwlv,
    "Res2P_ZZ4q":   _captions_res2p_zz4q,
    # JetClass-2 QCD
    "QCD_ss":       _captions_qcd,
    "QCD_light":    _captions_qcd,
    "QCD_187":      _captions_qcd,
    "QCD_185":      _captions_qcd,
    "QCD":          _captions_qcd,
}

# Also include JetClass-2 X_ classes (if used)
for _cls in ["X_bb", "X_cc", "X_ss", "X_uu", "X_gg", "X_bc", "X_cs", "X_bq", "X_cq"]:
    CLASS_CAPTION_GENERATORS[_cls] = _captions_res2p_bb  # fallback; override if needed


def generate_class_specific_captions(
    jet_meta: dict,
    n_reasoning: int = 3,
    n_descriptive: int = 3,
) -> list[dict]:
    """Generate class-specific physics reasoning captions for one jet.

    Produces `n_reasoning` captions with reasoning-style prompts and
    `n_descriptive` with descriptive prompts, pulling from per-class templates.
    """
    cls = jet_meta["class"]
    gen_fn = CLASS_CAPTION_GENERATORS.get(cls)

    if gen_fn is None:
        # Fallback: generic rule-based for unknown classes
        info = CLASS_INFO.get(cls, {})
        fallback_caption = (
            f"This jet originates from {info.get('process', cls)}. "
            f"pT = {jet_meta.get('jet_pt', 0):.0f} GeV, "
            f"mass = {jet_meta.get('jet_sdmass', 0):.1f} GeV, "
            f"{jet_meta.get('jet_nparticles', 0)} constituents."
        )
        templates = [fallback_caption]
    else:
        templates = gen_fn(jet_meta)

    conversations = []

    # Reasoning-style prompts (the most discriminative)
    for i in range(n_reasoning):
        caption = templates[i % len(templates)]
        conversations.append({
            "id": f"{jet_meta['jet_id']}_cs_r{i}",
            "jet_id": jet_meta["jet_id"],
            "conversations": [
                {"from": "human", "value": random.choice(_REASONING_PROMPTS)},
                {"from": "gpt", "value": caption},
            ],
            "caption_type": "class_specific_reasoning",
        })

    # Descriptive prompts
    for i in range(n_descriptive):
        caption = templates[(n_reasoning + i) % len(templates)]
        conversations.append({
            "id": f"{jet_meta['jet_id']}_cs_d{i}",
            "jet_id": jet_meta["jet_id"],
            "conversations": [
                {"from": "human", "value": random.choice(_DESCRIBE_PROMPTS)},
                {"from": "gpt", "value": caption},
            ],
            "caption_type": "class_specific_descriptive",
        })

    return conversations


# =============================================================================
# Strategy 3: Template-based with slot-filling (kinematic/observational variety)
# =============================================================================

SLOT_FILL_TEMPLATES = [
    # Observational
    "The jet has {jet_nparticles} reconstructed constituents, pT = {jet_pt:.0f} GeV, "
    "and soft-drop mass = {jet_sdmass:.1f} GeV.",

    "Jet kinematics: pT = {jet_pt:.0f} GeV, η = {jet_eta:.2f}, "
    "mass = {jet_sdmass:.1f} GeV, {jet_nparticles} constituents.",

    "The anti-kT R=0.8 jet has pT = {jet_pt:.0f} GeV and groomed mass {jet_sdmass:.1f} GeV. "
    "N-subjettiness: τ₁ = {jet_tau1:.3f}, τ₂ = {jet_tau2:.3f}, τ₃ = {jet_tau3:.3f}.",

    "This {jet_pt:.0f} GeV jet with {jet_nparticles} constituents has soft-drop mass {jet_sdmass:.1f} GeV, "
    "which is in the {mass_window}.",

    # Substructure-focused
    "The N-subjettiness ratio τ₂/τ₁ = {tau21:.3f}: {tau21_interp}. "
    "Jet mass = {jet_sdmass:.1f} GeV, pT = {jet_pt:.0f} GeV, {jet_nparticles} constituents.",

    "Substructure summary: τ₂/τ₁ = {tau21:.3f} ({substructure_desc}), "
    "τ₃/τ₂ = {tau32:.3f}. Mass = {jet_sdmass:.1f} GeV ({mass_window}). pT = {jet_pt:.0f} GeV.",

    # Physics interpretation
    "The jet mass of {jet_sdmass:.1f} GeV is {mass_window}. "
    "The {jet_nparticles}-particle jet is reconstructed at pT = {jet_pt:.0f} GeV.",

    "At pT = {jet_pt:.0f} GeV, this {pt_regime} jet has soft-drop mass {jet_sdmass:.1f} GeV "
    "and τ₂/τ₁ = {tau21:.3f}.",

    # Process + kinematics
    "This {process} jet has pT = {jet_pt:.0f} GeV, mass = {jet_sdmass:.1f} GeV, "
    "and τ₂/τ₁ = {tau21:.3f} consistent with a {n_prong}-prong substructure.",

    "A {process} jet: pT = {jet_pt:.0f} GeV, mass = {jet_sdmass:.1f} GeV, "
    "{jet_nparticles} constituents, τ₂/τ₁ = {tau21:.3f}.",

    "The {particle} jet ({process}) is reconstructed with pT = {jet_pt:.0f} GeV, "
    "mass = {jet_sdmass:.1f} GeV, {jet_nparticles} constituents.",

    # Reasoning caption (generic, connecting observables)
    "The jet mass of {jet_sdmass:.1f} GeV ({mass_window}) and "
    "τ₂/τ₁ = {tau21:.3f} ({substructure_desc}) are the key observables for identifying this {process} jet.",

    "Observables: mass = {jet_sdmass:.1f} GeV, τ₂/τ₁ = {tau21:.3f}, "
    "τ₃/τ₂ = {tau32:.3f}, {jet_nparticles} constituents. "
    "This is consistent with {process}.",

    # Detailed multi-sentence
    (
        "This jet was reconstructed from {jet_nparticles} particles with pT = {jet_pt:.0f} GeV. "
        "The soft-drop mass is {jet_sdmass:.1f} GeV. "
        "The N-subjettiness values are τ₁ = {jet_tau1:.3f}, τ₂ = {jet_tau2:.3f}, τ₃ = {jet_tau3:.3f}, "
        "giving τ₂/τ₁ = {tau21:.3f}. "
        "The jet originates from {process}."
    ),

    # Concise
    "{process}: pT = {jet_pt:.0f} GeV, m = {jet_sdmass:.1f} GeV, τ₂₁ = {tau21:.3f}, "
    "{jet_nparticles} particles.",

    # Comparative
    "Compared to QCD jets, this {process} jet has a higher mass ({jet_sdmass:.1f} GeV) "
    "and {substructure_desc} substructure (τ₂/τ₁ = {tau21:.3f}).",
]

_SLOT_FILL_PROMPTS = [
    "<jet>\nDescribe this jet.",
    "<jet>\nWhat are the properties of this jet?",
    "<jet>\nSummarize the key features of this jet.",
    "<jet>\nWhat can you tell me about this jet?",
    "<jet>\nCharacterize this jet's kinematics and substructure.",
    "<jet>\nReport the properties of this jet.",
    "<jet>\nAnalyze this jet and describe its properties.",
]


def generate_slot_fill_caption(jet_meta: dict) -> str:
    cls = jet_meta["class"]
    info = CLASS_INFO.get(cls, {"particle": cls, "decay": cls, "process": cls, "n_prongs": 1})
    t21 = _tau21(jet_meta)
    t32 = _tau32(jet_meta)
    sdmass = jet_meta.get("jet_sdmass", 0)
    pt = jet_meta.get("jet_pt", 0)

    template_vars = {
        "particle":       info["particle"],
        "process":        info["process"],
        "n_prong":        info["n_prongs"],
        "jet_pt":         pt,
        "jet_eta":        jet_meta.get("jet_eta", 0),
        "jet_phi":        jet_meta.get("jet_phi", 0),
        "jet_energy":     jet_meta.get("jet_energy", 0),
        "jet_sdmass":     sdmass,
        "jet_nparticles": jet_meta.get("jet_nparticles", 0),
        "jet_tau1":       jet_meta.get("jet_tau1", 0),
        "jet_tau2":       jet_meta.get("jet_tau2", 0),
        "jet_tau3":       jet_meta.get("jet_tau3", 0),
        "tau21":          t21,
        "tau32":          t32,
        "substructure_desc": _substructure_desc(t21),
        "tau21_interp":   _n_subjettiness_interp(t21, info["n_prongs"]),
        "mass_window":    _mass_window(sdmass),
        "pt_regime":      _pt_regime(pt),
    }

    template = random.choice(SLOT_FILL_TEMPLATES)
    try:
        return template.format(**template_vars)
    except (KeyError, ValueError):
        return (
            f"Jet from {info['process']}: pT = {pt:.0f} GeV, "
            f"mass = {sdmass:.1f} GeV, τ₂/τ₁ = {t21:.3f}, "
            f"{jet_meta.get('jet_nparticles', 0)} constituents."
        )


# =============================================================================
# Strategy 2: LLM-generated captions (via OpenRouter)
# =============================================================================

LLM_CAPTION_SYSTEM_PROMPT = """\
You are a particle physics expert writing natural-language descriptions of \
particle jets from LHC collisions. Given structured metadata about a jet, \
write a single paragraph (2-5 sentences) that: (1) identifies the physics process \
from the observable properties, (2) explains the reasoning chain connecting \
jet mass, N-subjettiness, and constituent count to the parent particle, and \
(3) notes any distinctive features that distinguish this class from similar classes. \
Vary your language and style. Be specific and discriminative — avoid generic statements \
that would apply equally to all jets. Do NOT use bullet points or lists. \
Just output the caption text, nothing else."""


def _format_jet_for_llm(jet_meta: dict) -> str:
    cls = jet_meta["class"]
    info = CLASS_INFO.get(cls, {"particle": cls, "process": cls, "decay": cls, "n_prongs": 1})
    t21 = _tau21(jet_meta)
    t32 = _tau32(jet_meta)
    return (
        f"Jet class: {cls}\n"
        f"Parent particle: {info['particle']}\n"
        f"Decay process: {info['process']} ({info['decay']})\n"
        f"Expected prong structure: {info['n_prongs']}-prong\n"
        f"Jet pT: {jet_meta.get('jet_pt', 0):.1f} GeV\n"
        f"Jet eta: {jet_meta.get('jet_eta', 0):.2f}\n"
        f"Soft-drop mass: {jet_meta.get('jet_sdmass', 0):.1f} GeV\n"
        f"Number of constituents: {jet_meta.get('jet_nparticles', 0)}\n"
        f"tau_1: {jet_meta.get('jet_tau1', 0):.4f}\n"
        f"tau_2: {jet_meta.get('jet_tau2', 0):.4f}\n"
        f"tau_3: {jet_meta.get('jet_tau3', 0):.4f}\n"
        f"tau21 ratio: {t21:.4f}\n"
        f"tau32 ratio: {t32:.4f}\n"
        f"Mass window: {_mass_window(jet_meta.get('jet_sdmass', 0))}"
    )


def generate_llm_caption(jet_meta: dict, config: dict) -> str | None:
    caption_cfg = config.get("captions", {})
    model = caption_cfg.get("llm_caption_model", "anthropic/claude-sonnet-4")
    max_tokens = caption_cfg.get("llm_caption_max_tokens", 300)
    try:
        return chat_completion(
            messages=[
                {"role": "system", "content": LLM_CAPTION_SYSTEM_PROMPT},
                {"role": "user", "content": _format_jet_for_llm(jet_meta)},
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
    jets: list[dict], config: dict, num_per_class: int = 30
) -> list[dict]:
    sampled = random.sample(jets, min(num_per_class, len(jets)))
    conversations = []
    prompts = _REASONING_PROMPTS + _DESCRIBE_PROMPTS
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
# Per-jet caption generation
# =============================================================================

def generate_captions_for_jet(
    jet_meta: dict,
    num_class_specific_reasoning: int = 3,
    num_class_specific_descriptive: int = 3,
    num_slot_fill: int = 2,
) -> list[dict]:
    """Generate multiple captions for a single jet.

    Returns list of conversation dicts in LLaVA format.
    Total per jet: num_class_specific_reasoning + num_class_specific_descriptive + num_slot_fill
    """
    conversations = []

    # Class-specific physics reasoning captions (most discriminative)
    conversations.extend(
        generate_class_specific_captions(
            jet_meta,
            n_reasoning=num_class_specific_reasoning,
            n_descriptive=num_class_specific_descriptive,
        )
    )

    # Slot-fill kinematic captions (observational variety)
    for i in range(num_slot_fill):
        caption = generate_slot_fill_caption(jet_meta)
        conversations.append({
            "id": f"{jet_meta['jet_id']}_sf_{i}",
            "jet_id": jet_meta["jet_id"],
            "conversations": [
                {"from": "human", "value": random.choice(_SLOT_FILL_PROMPTS)},
                {"from": "gpt", "value": caption},
            ],
            "caption_type": "slot_fill",
        })

    return conversations


# =============================================================================
# Caption generation main function
# =============================================================================

def generate_all_captions(
    data_dir: str,
    config: dict,
    seed: int = 42,
    skip_llm: bool = False,
) -> Path:
    """Generate captions for all tokenized jets."""
    global CLASS_INFO
    CLASS_INFO = build_class_info(config["dataset"]["classes"])

    random.seed(seed)

    from scripts.config import get_paths
    _paths = get_paths(config)
    tokenized_path = _paths["tokenized_dir"] / "tokenized_jets.json"
    with open(tokenized_path) as f:
        all_jets = json.load(f)

    print(f"Generating captions for {len(all_jets)} jets...")

    all_conversations = []
    for jet_meta in all_jets:
        convos = generate_captions_for_jet(
            jet_meta,
            num_class_specific_reasoning=3,
            num_class_specific_descriptive=3,
            num_slot_fill=2,
        )
        all_conversations.extend(convos)

    print(f"  Class-specific + slot-fill: {len(all_conversations)} conversations")

    # LLM-generated captions
    if not skip_llm:
        openrouter_var = config.get("env", {}).get("openrouter_token_var", "OPENROUTER_API_KEY")
        if os.environ.get(openrouter_var):
            print("Generating LLM captions via OpenRouter...")
            num_per_class = config.get("captions", {}).get("num_llm_generated_per_class", 30)
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
    parser.add_argument("--override", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-llm", action="store_true")
    args = parser.parse_args()

    from scripts.config import load_config
    config = load_config(args.config, args.override)
    if args.data_dir is not None:
        config["data_dir"] = args.data_dir
    generate_all_captions(config["data_dir"], config, args.seed, skip_llm=args.skip_llm)


if __name__ == "__main__":
    main()

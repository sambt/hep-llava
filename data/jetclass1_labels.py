"""Class definitions and physics descriptions for the original JetClass dataset.

JetClass (Qu et al., 2022 — arxiv:2202.03772) contains 10 physics processes
simulated with Pythia + Delphes. The data lives in ROOT files organised as::

    {jetclass_path}/train_100M/{ClassName}_{index}.root
    {jetclass_path}/val_5M/{ClassName}_{index}.root
    {jetclass_path}/test_20M/{ClassName}_{index}.root

Class names match the ROOT file prefixes exactly.

References:
    https://arxiv.org/abs/2202.03772
    https://zenodo.org/record/6619768
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Class list (matches ROOT file prefixes)
# ---------------------------------------------------------------------------

CLASSES: list[str] = [
    "HToBB",        # H → bb̄
    "HToCC",        # H → cc̄
    "HToGG",        # H → gg
    "HToWW2Q1L",    # H → WW → qqℓν  (semi-leptonic, 3-prong)
    "HToWW4Q",      # H → WW → qqqq  (fully hadronic, 4-prong)
    "TTBar",        # tt̄ → fully hadronic
    "TTBarLep",     # tt̄ → semi-leptonic
    "WToQQ",        # W → qq̄
    "ZJetsToNuNu",  # Z + jets → νν̄ + jets  (invisible)
    "ZToQQ",        # Z → qq̄
]

# ---------------------------------------------------------------------------
# Physics descriptions (same schema as download_jetclass._LABEL_PHYSICS)
# ---------------------------------------------------------------------------

LABEL_PHYSICS: dict[str, dict] = {
    "HToBB": {
        "particle": "Higgs boson",
        "decay":    "bb̄",
        "process":  "H → bb̄",
        "n_prongs": 2,
    },
    "HToCC": {
        "particle": "Higgs boson",
        "decay":    "cc̄",
        "process":  "H → cc̄",
        "n_prongs": 2,
    },
    "HToGG": {
        "particle": "Higgs boson",
        "decay":    "gg",
        "process":  "H → gg",
        "n_prongs": 2,
    },
    "HToWW2Q1L": {
        "particle": "Higgs boson",
        "decay":    "WW → qqℓν (semi-leptonic)",
        "process":  "H → WW → qqℓν",
        "n_prongs": 3,
    },
    "HToWW4Q": {
        "particle": "Higgs boson",
        "decay":    "WW → qqqq (fully hadronic)",
        "process":  "H → WW → qqqq",
        "n_prongs": 4,
    },
    "TTBar": {
        "particle": "top-antitop quark pair",
        "decay":    "tt̄ (fully hadronic)",
        "process":  "pp → tt̄ → bqq + b̄qq̄",
        "n_prongs": 5,
    },
    "TTBarLep": {
        "particle": "top-antitop quark pair",
        "decay":    "tt̄ (semi-leptonic)",
        "process":  "pp → tt̄ → bqq + b̄ℓν",
        "n_prongs": 4,
    },
    "WToQQ": {
        "particle": "W boson",
        "decay":    "qq̄",
        "process":  "W → qq̄",
        "n_prongs": 2,
    },
    "ZJetsToNuNu": {
        "particle": "Z + jets",
        "decay":    "νν̄ + jets (invisible)",
        "process":  "Z → νν̄ (+ jets)",
        "n_prongs": 1,
    },
    "ZToQQ": {
        "particle": "Z boson",
        "decay":    "qq̄",
        "process":  "Z → qq̄",
        "n_prongs": 2,
    },
}


def physics_for_class(class_name: str) -> dict:
    """Return the physics description dict for a JetClass-I class name.

    Args:
        class_name: One of the 10 JetClass-I class strings.

    Returns:
        Dict with keys ``particle``, ``decay``, ``process``, ``n_prongs``.

    Raises:
        ValueError: If the class name is not recognised.
    """
    if class_name not in LABEL_PHYSICS:
        raise ValueError(
            f"Unknown JetClass-I class {class_name!r}. "
            f"Valid classes: {CLASSES}"
        )
    return LABEL_PHYSICS[class_name]


def build_class_info(class_names: list[str]) -> dict[str, dict]:
    """Build a CLASS_INFO dict from a list of JetClass-I class names.

    Args:
        class_names: Subset of :data:`CLASSES`.

    Returns:
        Dict mapping class name → physics info dict.

    Raises:
        ValueError: If any name is not a valid JetClass-I class.
    """
    unknown = [n for n in class_names if n not in LABEL_PHYSICS]
    if unknown:
        raise ValueError(
            f"Unknown JetClass-I class name(s): {unknown}. "
            f"Valid classes: {CLASSES}"
        )
    return {name: LABEL_PHYSICS[name] for name in class_names}

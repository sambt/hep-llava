"""Download and prepare a subset of JetClass-II data from HuggingFace.

Downloads parquet files directly (no streaming) and samples jets by jet_label.
Class names are taken directly from ``data.jetclass_labels`` (e.g. ``X_bb``,
``QCD_light``) so they match the official JetClass-II label strings.

Label sources (from arxiv:2405.12972):
  Res2P file:  labels   0–14   (2-prong resonance X → pair)
  Res34P file: labels  15–160  (3/4-prong resonances)
  QCD file:    labels 161–187  (QCD sub-types)
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from huggingface_hub import hf_hub_download

from data.jetclass_labels import idx_to_label, label_to_idx


# ---------------------------------------------------------------------------
# Source-file routing
# ---------------------------------------------------------------------------

def _label_to_source_file(label_int: int) -> str:
    """Return the HuggingFace parquet path for a given integer jet_label."""
    if label_int <= 14:
        return "data/Res2P_0000.parquet"
    elif label_int <= 160:
        return "data/Res34P_0000.parquet"
    else:
        return "data/QCD_0000.parquet"


# ---------------------------------------------------------------------------
# Physics descriptions for selected labels
# ---------------------------------------------------------------------------
# Keyed by the official label string from data.jetclass_labels.
# Used by generate_captions.py and generate_qa.py.

_LABEL_PHYSICS: dict[str, dict] = {
    # --- JetClass-I (original dataset, arxiv:2202.03772) ---
    "HToBB":     {"particle": "Higgs boson",           "decay": "bb̄",                     "process": "H → bb̄",            "n_prongs": 2},
    "HToCC":     {"particle": "Higgs boson",           "decay": "cc̄",                     "process": "H → cc̄",            "n_prongs": 2},
    "HToGG":     {"particle": "Higgs boson",           "decay": "gg",                      "process": "H → gg",             "n_prongs": 2},
    "HToWW2Q1L": {"particle": "Higgs boson",           "decay": "WW → qqℓν",               "process": "H → WW → qqℓν",     "n_prongs": 3},
    "HToWW4Q":   {"particle": "Higgs boson",           "decay": "WW → qqqq",               "process": "H → WW → qqqq",     "n_prongs": 4},
    "TTBar":     {"particle": "top-antitop quark pair","decay": "tt̄ (fully hadronic)",     "process": "pp → tt̄ (had.)",    "n_prongs": 5},
    "TTBarLep":  {"particle": "top-antitop quark pair","decay": "tt̄ (semi-leptonic)",      "process": "pp → tt̄ (semilep.)","n_prongs": 4},
    "WToQQ":     {"particle": "W boson",               "decay": "qq̄",                      "process": "W → qq̄",            "n_prongs": 2},
    "ZJetsToNuNu":{"particle": "Z + jets",             "decay": "νν̄ + jets",               "process": "Z → νν̄ (+ jets)",  "n_prongs": 1},
    "ZToQQ":     {"particle": "Z boson",               "decay": "qq̄",                      "process": "Z → qq̄",            "n_prongs": 2},
    # --- 2-prong resonances (Res2P) ---
    "X_bb":  {"particle": "heavy resonance X", "decay": "bb̄",          "process": "X → bb̄",  "n_prongs": 2},
    "X_cc":  {"particle": "heavy resonance X", "decay": "cc̄",          "process": "X → cc̄",  "n_prongs": 2},
    "X_ss":  {"particle": "heavy resonance X", "decay": "ss̄",          "process": "X → ss̄",  "n_prongs": 2},
    "X_qq":  {"particle": "heavy resonance X", "decay": "qq̄ (light)",  "process": "X → qq̄",  "n_prongs": 2},
    "X_bc":  {"particle": "heavy resonance X", "decay": "bc̄",          "process": "X → bc̄",  "n_prongs": 2},
    "X_cs":  {"particle": "heavy resonance X", "decay": "cs̄",          "process": "X → cs̄",  "n_prongs": 2},
    "X_bq":  {"particle": "heavy resonance X", "decay": "bq̄ (light)",  "process": "X → bq̄",  "n_prongs": 2},
    "X_cq":  {"particle": "heavy resonance X", "decay": "cq̄ (light)",  "process": "X → cq̄",  "n_prongs": 2},
    "X_sq":  {"particle": "heavy resonance X", "decay": "sq̄ (light)",  "process": "X → sq̄",  "n_prongs": 2},
    "X_gg":  {"particle": "heavy resonance X", "decay": "gg",           "process": "X → gg",   "n_prongs": 2},
    "X_ee":  {"particle": "heavy resonance X", "decay": "e⁺e⁻",        "process": "X → e⁺e⁻","n_prongs": 2},
    "X_mm":  {"particle": "heavy resonance X", "decay": "μ⁺μ⁻",        "process": "X → μ⁺μ⁻","n_prongs": 2},
    # --- QCD backgrounds ---
    "QCD_light": {"particle": "light quark/gluon", "decay": "QCD multijet",       "process": "q/g → jet",        "n_prongs": 1},
    "QCD_b":     {"particle": "bottom quark",      "decay": "QCD (b)",            "process": "b → jet",          "n_prongs": 1},
    "QCD_c":     {"particle": "charm quark",        "decay": "QCD (c)",            "process": "c → jet",          "n_prongs": 1},
    "QCD_s":     {"particle": "strange quark",      "decay": "QCD (s)",            "process": "s → jet",          "n_prongs": 1},
    "QCD_bb":    {"particle": "bottom quark pair",  "decay": "QCD (bb̄)",          "process": "bb → jet",         "n_prongs": 1},
    "QCD_cc":    {"particle": "charm quark pair",   "decay": "QCD (cc̄)",          "process": "cc → jet",         "n_prongs": 1},
    "QCD_ss":    {"particle": "strange quark pair", "decay": "QCD (ss̄)",          "process": "ss → jet",         "n_prongs": 1},
    "QCD_bc":    {"particle": "bottom-charm pair",  "decay": "QCD (bc̄)",          "process": "bc → jet",         "n_prongs": 1},
    "QCD_bs":    {"particle": "bottom-strange pair","decay": "QCD (bs̄)",          "process": "bs → jet",         "n_prongs": 1},
    "QCD_cs":    {"particle": "charm-strange pair", "decay": "QCD (cs̄)",          "process": "cs → jet",         "n_prongs": 1},
}

def _physics_for_label(label_name: str) -> dict:
    """Return a physics description dict for a label, generating a fallback if needed."""
    if label_name in _LABEL_PHYSICS:
        return _LABEL_PHYSICS[label_name]
    # Generic fallback: derive particle/process from the label name
    if label_name.startswith("QCD_"):
        content = label_name[4:]
        return {
            "particle": "quark/gluon",
            "decay": f"QCD ({content})",
            "process": f"{content} → jet",
            "n_prongs": 1,
        }
    # X_ or X_YY_ resonance
    content = label_name.split("_", 1)[1] if "_" in label_name else label_name
    return {
        "particle": "heavy resonance X",
        "decay": content,
        "process": f"X → {content}",
        "n_prongs": 2,
    }


# ---------------------------------------------------------------------------
# CLASS_INFO: build from a list of official class names
# ---------------------------------------------------------------------------

def build_class_info(class_names: list[str]) -> dict[str, dict]:
    """Build a CLASS_INFO dict from a list of official label name strings.

    Args:
        class_names: List of official label names, e.g. ``["X_bb", "QCD_light"]``.
            Each name must appear in ``data.jetclass_labels.label_to_idx``.

    Returns:
        Dict mapping class name → physics info dict with keys
        ``label``, ``particle``, ``decay``, ``process``, ``n_prongs``.

    Raises:
        ValueError: If any name is not found in the official label list.
    """
    unknown = [n for n in class_names if n not in label_to_idx]
    if unknown:
        raise ValueError(
            f"Unknown class name(s): {unknown}. "
            "Class names must match entries in data/jetclass_labels.py "
            "(e.g. 'X_bb', 'X_gg', 'QCD_light', 'QCD_ss')."
        )
    return {
        name: {"label": label_to_idx[name], **_physics_for_label(name)}
        for name in class_names
    }


# Default 10-class selection:  8 Res2P + 2 QCD
# (X_bb/cc/ss/bc/cs/bq/cq/gg are labels 0–9, excluding 3 and 8;
#  QCD_ss=185, QCD_light=187 are the two most populated QCD sub-types)
_DEFAULT_CLASSES = [
    "X_bb",      # label   0 — X → bb̄
    "X_cc",      # label   1 — X → cc̄
    "X_ss",      # label   2 — X → ss̄
    "X_bc",      # label   4 — X → bc̄
    "X_cs",      # label   5 — X → cs̄
    "X_bq",      # label   6 — X → bq̄
    "X_cq",      # label   7 — X → cq̄
    "X_gg",      # label   9 — X → gg
    "QCD_ss",    # label 185 — QCD (ss̄),     13 847 jets in QCD_0000
    "QCD_light", # label 187 — QCD multijet,  59 442 jets in QCD_0000
]

# Module-level constant for scripts that do `from data.download_jetclass import CLASS_INFO`
CLASS_INFO: dict[str, dict] = build_class_info(_DEFAULT_CLASSES)


# ---------------------------------------------------------------------------
# Columns to keep
# ---------------------------------------------------------------------------

KEEP_COLUMNS = [
    "jet_pt", "jet_eta", "jet_phi", "jet_energy", "jet_sdmass",
    "jet_nparticles", "jet_tau1", "jet_tau2", "jet_tau3",
    "part_deta", "part_dphi", "part_px", "part_py", "part_pz", "part_energy",
    "aux_genpart_pt", "aux_genpart_eta", "aux_genpart_phi",
    "aux_genpart_mass", "aux_genpart_pid",
    "jet_label",
]


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _load_and_filter(
    hf_filename: str,
    label_ints: list[int],
    num_per_label: int,
    rng: np.random.Generator,
    repo_id: str = "jet-universe/jetclass2",
) -> dict[int, pd.DataFrame]:
    """Download one parquet file and extract jets for the requested labels."""
    print(f"  Downloading {hf_filename} …")
    local_path = hf_hub_download(repo_id, hf_filename, repo_type="dataset")
    df = pd.read_parquet(local_path)
    print(f"    Loaded {len(df)} jets from {hf_filename}")

    cols_present = [c for c in KEEP_COLUMNS if c in df.columns]
    df = df[cols_present]

    result: dict[int, pd.DataFrame] = {}
    for lbl in label_ints:
        name = idx_to_label[lbl]
        subset = df[df["jet_label"] == lbl]
        n_available = len(subset)
        if n_available == 0:
            print(f"    WARNING: label {lbl} ({name}) not found in {hf_filename}")
            continue
        if n_available < num_per_label:
            print(
                f"    WARNING: label {lbl} ({name}) only has {n_available} jets "
                f"(requested {num_per_label}); taking all."
            )
            sampled = subset
        else:
            idx = rng.choice(n_available, size=num_per_label, replace=False)
            sampled = subset.iloc[idx]
        result[lbl] = sampled.reset_index(drop=True)
        print(f"    label {lbl:>3d} ({name:<12}): {len(sampled)} jets sampled")

    return result


def download_jetclass2_subset(
    data_dir: str,
    class_names: list[str],
    num_jets_per_class: int = 3000,
    seed: int = 42,
    repo_id: str = "jet-universe/jetclass2",
) -> Path:
    """Download JetClass-II parquet files and extract a balanced subset.

    Saves one parquet file per class under ``{data_dir}/jetclass2_subset/``.
    Also writes ``manifest.json`` and ``class_info.json``.

    Args:
        data_dir: Root data directory.
        class_names: List of official label names (e.g. ``["X_bb", "QCD_light"]``).
        num_jets_per_class: Jets to sample per label.
        seed: Random seed for reproducible sampling.
        repo_id: HuggingFace dataset repository ID.

    Returns:
        Path to the output directory.
    """
    class_info = build_class_info(class_names)
    output_dir = Path(data_dir) / "jetclass2_subset"
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    # Group label integers by source file
    labels_by_file: dict[str, list[int]] = defaultdict(list)
    for name, info in class_info.items():
        lbl = info["label"]
        labels_by_file[_label_to_source_file(lbl)].append(lbl)

    print(f"\nDownloading JetClass-II from HuggingFace ({repo_id})")
    print(f"Target: {num_jets_per_class} jets per class, {len(class_info)} classes\n")

    all_jets_by_label: dict[int, pd.DataFrame] = {}
    for hf_file, lbl_list in labels_by_file.items():
        jets = _load_and_filter(hf_file, lbl_list, num_jets_per_class, rng, repo_id)
        all_jets_by_label.update(jets)

    print(f"\nSaving class parquet files to {output_dir}")

    saved_classes: list[str] = []
    total_jets = 0

    for lbl_int, df in all_jets_by_label.items():
        class_name = idx_to_label[lbl_int]
        out_path = output_dir / f"{class_name}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"  Saved {len(df):>5d} jets → {out_path.name}")
        saved_classes.append(class_name)
        total_jets += len(df)

    # class_info.json
    class_info_path = output_dir / "class_info.json"
    with open(class_info_path, "w") as f:
        json.dump(class_info, f, indent=2, ensure_ascii=False)
    print(f"\nSaved class metadata → {class_info_path}")

    # manifest.json
    manifest = {
        "repo_id": repo_id,
        "num_jets_per_class": num_jets_per_class,
        "seed": seed,
        "classes": sorted(saved_classes),
        "total_jets": total_jets,
        "label_mapping": {str(label_to_idx[n]): n for n in saved_classes},
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Saved manifest      → {manifest_path}")
    print(f"\nDone. Total jets: {total_jets} across {len(saved_classes)} classes.")

    return output_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download JetClass-II subset from HuggingFace"
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--override", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--num-jets-per-class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--classes", type=str, default=None,
        help=(
            "Comma-separated official class names to download "
            "(e.g. X_bb,X_cc,QCD_light). Overrides dataset.classes in config."
        ),
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.override:
        import yaml as _yaml
        with open(args.override) as f:
            override = _yaml.safe_load(f)
        # Shallow-merge dataset section only (full deep-merge available in scripts.config)
        config.update(override)

    data_dir = args.data_dir or config["data_dir"]
    num_jets = args.num_jets_per_class or config["dataset"]["num_jets_per_class"]
    class_names = (
        args.classes.split(",") if args.classes
        else config["dataset"]["classes"]
    )

    download_jetclass2_subset(data_dir, class_names, num_jets, seed=args.seed)


if __name__ == "__main__":
    main()

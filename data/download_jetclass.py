"""Download and prepare a subset of JetClass-II data from HuggingFace.

Downloads parquet files directly (no streaming) and samples jets by jet_label.

Label sources (from arxiv:2405.12972):
  Res2P file: labels 0-14 (2-prong resonance X → pair)
  QCD file:   labels 161-187 (QCD sub-types)
  Res34P file: labels 15-160 (3/4-prong resonances)
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from huggingface_hub import hf_hub_download


# ---------------------------------------------------------------------------
# JetClass-II label mapping (from paper arxiv:2405.12972)
# ---------------------------------------------------------------------------

# Res2P labels: X → 2-prong resonance decays
# Labels 0-9 have ~7000-7700 jets each in Res2P_0000.parquet
# Label 3 is doubled (~14956) due to dd̄ + uū degeneracy — we use 4 (uu) instead
RES2P_LABEL_INFO = {
    0: {
        "name": "Res2P_bb",
        "particle": "heavy resonance X",
        "decay": "bb̄ (two bottom quarks)",
        "process": "X → bb̄",
        "n_prongs": 2,
        "source_file": "data/Res2P_0000.parquet",
    },
    1: {
        "name": "Res2P_cc",
        "particle": "heavy resonance X",
        "decay": "cc̄ (two charm quarks)",
        "process": "X → cc̄",
        "n_prongs": 2,
        "source_file": "data/Res2P_0000.parquet",
    },
    2: {
        "name": "Res2P_ss",
        "particle": "heavy resonance X",
        "decay": "ss̄ (two strange quarks)",
        "process": "X → ss̄",
        "n_prongs": 2,
        "source_file": "data/Res2P_0000.parquet",
    },
    4: {
        "name": "Res2P_uu",
        "particle": "heavy resonance X",
        "decay": "uū (two up quarks)",
        "process": "X → uū",
        "n_prongs": 2,
        "source_file": "data/Res2P_0000.parquet",
    },
    5: {
        "name": "Res2P_gg",
        "particle": "heavy resonance X",
        "decay": "gg (two gluons)",
        "process": "X → gg",
        "n_prongs": 2,
        "source_file": "data/Res2P_0000.parquet",
    },
    6: {
        "name": "Res2P_WW4q",
        "particle": "heavy resonance X",
        "decay": "WW → qqqq (four quarks, fully hadronic)",
        "process": "X → WW → qqqq",
        "n_prongs": 4,
        "source_file": "data/Res2P_0000.parquet",
    },
    7: {
        "name": "Res2P_WWlv",
        "particle": "heavy resonance X",
        "decay": "WW → qqℓν (semi-leptonic)",
        "process": "X → WW → qqℓν",
        "n_prongs": 3,
        "source_file": "data/Res2P_0000.parquet",
    },
    9: {
        "name": "Res2P_ZZ4q",
        "particle": "heavy resonance X",
        "decay": "ZZ → qqqq (four quarks, fully hadronic)",
        "process": "X → ZZ → qqqq",
        "n_prongs": 4,
        "source_file": "data/Res2P_0000.parquet",
    },
}

# QCD labels: top 2 most common from QCD_0000.parquet
# 187: 59442 jets, 185: 13847 jets
QCD_LABEL_INFO = {
    187: {
        "name": "QCD_187",
        "particle": "light quark/gluon",
        "decay": "QCD multijet (generic)",
        "process": "q/g → jet (label 187)",
        "n_prongs": 1,
        "source_file": "data/QCD_0000.parquet",
    },
    185: {
        "name": "QCD_185",
        "particle": "light quark/gluon",
        "decay": "QCD multijet (sub-type 185)",
        "process": "q/g → jet (label 185)",
        "n_prongs": 1,
        "source_file": "data/QCD_0000.parquet",
    },
}

# Combined: all 10 selected classes
# Keys are the integer jet_label values
SELECTED_LABELS: dict[int, dict] = {**RES2P_LABEL_INFO, **QCD_LABEL_INFO}

# CLASS_INFO keyed by class name (string) — used by generate_captions.py / generate_qa.py
CLASS_INFO: dict[str, dict] = {
    info["name"]: {
        "label": label_int,
        "particle": info["particle"],
        "decay": info["decay"],
        "process": info["process"],
        "n_prongs": info["n_prongs"],
    }
    for label_int, info in SELECTED_LABELS.items()
}

# Columns to keep in the output parquet files (subset for downstream pipeline)
KEEP_COLUMNS = [
    # Jet-level kinematics
    "jet_pt",
    "jet_eta",
    "jet_phi",
    "jet_energy",
    "jet_sdmass",
    "jet_nparticles",
    # Jet substructure
    "jet_tau1",
    "jet_tau2",
    "jet_tau3",
    # Particle-level arrays (ragged)
    "part_deta",
    "part_dphi",
    "part_px",
    "part_py",
    "part_pz",
    "part_energy",
    # Truth-level generator particles (optional — present in most files)
    "aux_genpart_pt",
    "aux_genpart_eta",
    "aux_genpart_phi",
    "aux_genpart_mass",
    "aux_genpart_pid",
    # Label (kept for reference / debugging)
    "jet_label",
]


def _load_and_filter(
    hf_filename: str,
    label_ints: list[int],
    num_per_label: int,
    rng: np.random.Generator,
    repo_id: str = "jet-universe/jetclass2",
) -> dict[int, pd.DataFrame]:
    """Download one parquet file and extract jets for the requested labels.

    Args:
        hf_filename: Relative path within the HF dataset repo.
        label_ints: List of integer jet_label values to extract.
        num_per_label: Number of jets to sample per label.
        rng: NumPy random generator for reproducible sampling.
        repo_id: HuggingFace dataset repository ID.

    Returns:
        Dict mapping each label integer to a DataFrame of sampled jets.
    """
    print(f"  Downloading {hf_filename} …")
    local_path = hf_hub_download(repo_id, hf_filename, repo_type="dataset")
    df = pd.read_parquet(local_path)
    print(f"    Loaded {len(df)} jets from {hf_filename}")

    # Keep only needed columns (drop missing ones gracefully)
    cols_present = [c for c in KEEP_COLUMNS if c in df.columns]
    df = df[cols_present]

    result: dict[int, pd.DataFrame] = {}
    for lbl in label_ints:
        subset = df[df["jet_label"] == lbl]
        n_available = len(subset)
        if n_available == 0:
            print(f"    WARNING: label {lbl} not found in {hf_filename}")
            continue
        if n_available < num_per_label:
            print(
                f"    WARNING: label {lbl} only has {n_available} jets "
                f"(requested {num_per_label}); taking all."
            )
            sampled = subset
        else:
            idx = rng.choice(n_available, size=num_per_label, replace=False)
            sampled = subset.iloc[idx]
        result[lbl] = sampled.reset_index(drop=True)
        print(f"    label {lbl} ({SELECTED_LABELS[lbl]['name']}): {len(sampled)} jets sampled")

    return result


def download_jetclass2_subset(
    data_dir: str,
    num_jets_per_class: int = 3000,
    seed: int = 42,
    repo_id: str = "jet-universe/jetclass2",
) -> Path:
    """Download JetClass-II parquet files and extract a balanced subset.

    Saves one parquet file per class under {data_dir}/jetclass2_subset/.
    Also writes manifest.json and class_info.json.

    Args:
        data_dir: Root data directory for storing artifacts.
        num_jets_per_class: Number of jets to sample per class label.
        seed: Random seed for reproducible sampling.
        repo_id: HuggingFace dataset repository ID.

    Returns:
        Path to the output directory containing the prepared data.
    """
    output_dir = Path(data_dir) / "jetclass2_subset"
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    # Group selected labels by source file
    from collections import defaultdict
    labels_by_file: dict[str, list[int]] = defaultdict(list)
    for label_int, info in SELECTED_LABELS.items():
        labels_by_file[info["source_file"]].append(label_int)

    print(f"\nDownloading JetClass-II from HuggingFace ({repo_id})")
    print(f"Target: {num_jets_per_class} jets per class, {len(SELECTED_LABELS)} classes\n")

    # Download each required file and extract jets
    all_jets_by_label: dict[int, pd.DataFrame] = {}
    for hf_file, label_list in labels_by_file.items():
        jets = _load_and_filter(hf_file, label_list, num_jets_per_class, rng, repo_id)
        all_jets_by_label.update(jets)

    print(f"\nSaving class parquet files to {output_dir}")

    saved_classes: list[str] = []
    total_jets = 0

    for label_int, df in all_jets_by_label.items():
        class_name = SELECTED_LABELS[label_int]["name"]
        out_path = output_dir / f"{class_name}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"  Saved {len(df):>5d} jets → {out_path.name}")
        saved_classes.append(class_name)
        total_jets += len(df)

    # Write class_info.json
    class_info_path = output_dir / "class_info.json"
    with open(class_info_path, "w") as f:
        json.dump(CLASS_INFO, f, indent=2, ensure_ascii=False)
    print(f"\nSaved class metadata → {class_info_path}")

    # Write manifest.json
    manifest = {
        "repo_id": repo_id,
        "num_jets_per_class": num_jets_per_class,
        "seed": seed,
        "classes": sorted(saved_classes),
        "total_jets": total_jets,
        "label_mapping": {
            str(k): v["name"] for k, v in SELECTED_LABELS.items()
        },
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Saved manifest      → {manifest_path}")
    print(f"\nDone. Total jets: {total_jets} across {len(saved_classes)} classes.")

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download JetClass-II subset from HuggingFace"
    )
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override data_dir from config",
    )
    parser.add_argument(
        "--num-jets-per-class",
        type=int,
        default=None,
        help="Override num_jets_per_class from config",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_dir = args.data_dir or config["data_dir"]
    num_jets = args.num_jets_per_class or config["dataset"]["num_jets_per_class"]

    download_jetclass2_subset(data_dir, num_jets, seed=args.seed)


if __name__ == "__main__":
    main()

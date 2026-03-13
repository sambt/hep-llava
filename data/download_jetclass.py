"""Download and prepare a subset of JetClass-II data from HuggingFace."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# JetClass-II class labels and their physics descriptions
CLASS_INFO = {
    "Hbb": {
        "label_col": "label_Hbb",
        "particle": "Higgs boson",
        "decay": "bb̄ (two bottom quarks)",
        "process": "H → bb̄",
        "n_prongs": 2,
    },
    "Hcc": {
        "label_col": "label_Hcc",
        "particle": "Higgs boson",
        "decay": "cc̄ (two charm quarks)",
        "process": "H → cc̄",
        "n_prongs": 2,
    },
    "Hgg": {
        "label_col": "label_Hgg",
        "particle": "Higgs boson",
        "decay": "gg (two gluons)",
        "process": "H → gg",
        "n_prongs": 2,
    },
    "H4q": {
        "label_col": "label_H4q",
        "particle": "Higgs boson",
        "decay": "4q (four quarks via WW*/ZZ*)",
        "process": "H → WW*/ZZ* → 4q",
        "n_prongs": 4,
    },
    "Hqql": {
        "label_col": "label_Hqql",
        "particle": "Higgs boson",
        "decay": "qqℓν (quarks + lepton + neutrino via WW*/ZZ*)",
        "process": "H → WW*/ZZ* → qqℓν",
        "n_prongs": 3,
    },
    "Zqq": {
        "label_col": "label_Zqq",
        "particle": "Z boson",
        "decay": "qq̄ (two quarks)",
        "process": "Z → qq̄",
        "n_prongs": 2,
    },
    "Wqq": {
        "label_col": "label_Wqq",
        "particle": "W boson",
        "decay": "qq' (two quarks)",
        "process": "W → qq'",
        "n_prongs": 2,
    },
    "Tbqq": {
        "label_col": "label_Tbqq",
        "particle": "top quark",
        "decay": "bqq' (hadronic)",
        "process": "t → bW → bqq'",
        "n_prongs": 3,
    },
    "Tbl": {
        "label_col": "label_Tbl",
        "particle": "top quark",
        "decay": "bℓν (leptonic)",
        "process": "t → bW → bℓν",
        "n_prongs": 2,
    },
    "QCD": {
        "label_col": "label_QCD",
        "particle": "light quark/gluon",
        "decay": "QCD jet",
        "process": "q/g → jet",
        "n_prongs": 1,
    },
}


def download_jetclass2_subset(
    data_dir: str,
    num_jets_per_class: int = 3000,
    seed: int = 42,
) -> Path:
    """Download JetClass-II from HuggingFace and extract a balanced subset.

    Args:
        data_dir: Root data directory for storing artifacts.
        num_jets_per_class: Number of jets to sample per class.
        seed: Random seed for reproducible sampling.

    Returns:
        Path to the output directory containing the prepared data.
    """
    from datasets import load_dataset

    output_dir = Path(data_dir) / "jetclass2_subset"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading JetClass-II from HuggingFace...")
    # Load the training split (we only need a small subset)
    ds = load_dataset("jet-universe/jetclass2", split="train", streaming=True)

    # Collect jets by class
    rng = np.random.default_rng(seed)
    jets_by_class: dict[str, list] = {cls: [] for cls in CLASS_INFO}
    all_classes_full = False

    print(f"Sampling {num_jets_per_class} jets per class...")
    for example in ds:
        if all_classes_full:
            break

        # Determine which class this jet belongs to
        jet_class = None
        for cls, info in CLASS_INFO.items():
            label_col = info["label_col"]
            if label_col in example and example[label_col] == 1:
                jet_class = cls
                break

        if jet_class is None:
            continue

        if len(jets_by_class[jet_class]) < num_jets_per_class:
            jets_by_class[jet_class].append(example)

        # Check if all classes are full
        all_classes_full = all(
            len(jets) >= num_jets_per_class for jets in jets_by_class.values()
        )

    # Report what we got
    for cls, jets in jets_by_class.items():
        print(f"  {cls}: {len(jets)} jets")

    # Save each class as a separate parquet file
    for cls, jets in jets_by_class.items():
        if len(jets) == 0:
            print(f"  WARNING: No jets found for class {cls}")
            continue
        df = pd.DataFrame(jets)
        out_path = output_dir / f"{cls}.parquet"
        df.to_parquet(out_path)
        print(f"  Saved {len(jets)} jets to {out_path}")

    # Save class info metadata
    meta_path = output_dir / "class_info.json"
    with open(meta_path, "w") as f:
        json.dump(CLASS_INFO, f, indent=2)
    print(f"Saved class metadata to {meta_path}")

    # Save a manifest
    manifest = {
        "num_jets_per_class": num_jets_per_class,
        "seed": seed,
        "classes": list(CLASS_INFO.keys()),
        "total_jets": sum(len(jets) for jets in jets_by_class.values()),
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Download JetClass-II subset")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--override", type=str, default=None, help="Path to an override YAML config")
    parser.add_argument("--data-dir", type=str, default=None, help="Override data_dir from config")
    parser.add_argument("--num-jets-per-class", type=int, default=None)
    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help="Comma-separated list of classes to download (e.g. Res2P_bb,QCD_187). "
             "Overrides config dataset.classes before token_set_name derivation.",
    )
    args = parser.parse_args()

    from scripts.config import load_config, get_paths

    config = load_config(args.config, args.override)

    # Apply --classes override before path resolution (affects token_set_name)
    if args.classes is not None:
        config["dataset"]["classes"] = [c.strip() for c in args.classes.split(",")]
        # Re-derive token_set_name with the updated class list
        config["token_set_name"] = None  # trigger re-derivation in load_config logic
        from scripts.config import derive_token_set_name
        tokenizer_type = config.get("tokenizer", {}).get("type", "omnijet_vqvae")
        config["token_set_name"] = derive_token_set_name(
            config["dataset"]["classes"], tokenizer_type
        )

    if args.data_dir is not None:
        config["data_dir"] = args.data_dir

    paths = get_paths(config)
    num_jets = args.num_jets_per_class or config["dataset"]["num_jets_per_class"]

    output_dir = download_jetclass2_subset(str(paths["data_dir"]), num_jets)
    print(f"\nDone. Data saved to: {output_dir}")


if __name__ == "__main__":
    main()

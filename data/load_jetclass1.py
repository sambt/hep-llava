"""Load jets from the original JetClass ROOT files.

Reads ROOT files produced by Pythia + Delphes (JetClass, arxiv:2202.03772)
and returns a dict of DataFrames in the same schema that
:func:`data.tokenize_jets.preprocess_jet_constituents` expects — so the rest
of the pipeline (tokenization, captioning, QA) is unchanged.

ROOT file layout::

    {jetclass_path}/
    ├── train_100M/   HToBB_000.root … HToBB_099.root  (and 9 other classes)
    ├── val_5M/       HToBB_120.root … HToBB_124.root
    └── test_20M/     HToBB_100.root … HToBB_119.root

Each ROOT tree (named ``tree``) has:
- Jet-level scalars:  jet_pt, jet_eta, jet_phi, jet_energy, jet_sdmass,
                      jet_nparticles, jet_tau1, jet_tau2, jet_tau3
- Particle-level jagged arrays: part_px, part_py, part_pz, part_energy,
                                part_deta, part_dphi (relative to jet axis)

The particle-level features map directly to the OmniJet-alpha VQ-VAE inputs:
    pt_rel  = sqrt(px² + py²) / jet_pt  (computed in preprocess_jet_constituents)
    eta_rel = part_deta                  (already relative)
    phi_rel = part_dphi                  (already relative)
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

from data.jetclass1_labels import CLASSES, LABEL_PHYSICS, physics_for_class


# ROOT branches to read — matches the schema of JetClass-II parquets
_JET_BRANCHES = [
    "jet_pt", "jet_eta", "jet_phi", "jet_energy", "jet_sdmass",
    "jet_nparticles", "jet_tau1", "jet_tau2", "jet_tau3",
]
_PART_BRANCHES = [
    "part_px", "part_py", "part_pz", "part_energy",
    "part_deta", "part_dphi",
]

# Map split names to directory names
_SPLIT_DIRS = {
    "train": "train_100M",
    "val":   "val_5M",
    "test":  "test_20M",
    # Also accept the full directory names
    "train_100M": "train_100M",
    "val_5M":     "val_5M",
    "test_20M":   "test_20M",
}

# Known-bad files to skip
_BAD_FILES = {"TTBar_014.root"}


def _find_root_files(jetclass_path: Path, class_name: str, split: str) -> list[Path]:
    """Return sorted ROOT file paths for a class and split.

    Args:
        jetclass_path: Root directory of the JetClass dataset.
        class_name: Class name (e.g. ``"HToBB"``).
        split: One of ``"train"``, ``"val"``, ``"test"`` (or full dir name).

    Returns:
        Sorted list of existing ROOT file paths.

    Raises:
        ValueError: If the split directory doesn't exist.
        FileNotFoundError: If no ROOT files are found for the class.
    """
    split_dir_name = _SPLIT_DIRS.get(split)
    if split_dir_name is None:
        raise ValueError(
            f"Unknown split {split!r}. "
            f"Valid options: {list(_SPLIT_DIRS.keys())}"
        )

    split_dir = Path(jetclass_path) / split_dir_name
    if not split_dir.exists():
        raise ValueError(f"Split directory not found: {split_dir}")

    files = sorted(
        f for f in split_dir.glob(f"{class_name}_*.root")
        if f.name not in _BAD_FILES
    )
    if not files:
        raise FileNotFoundError(
            f"No ROOT files found for class {class_name!r} in {split_dir}. "
            f"Expected files like {class_name}_000.root"
        )
    return files


def _read_root_file(root_path: Path, max_jets: int | None = None) -> pd.DataFrame:
    """Read jet and particle branches from one ROOT file into a DataFrame.

    Particle-level branches are stored as Python lists (one list per jet),
    matching the structure of JetClass-II parquet files.

    Args:
        root_path: Path to a JetClass ROOT file.
        max_jets: If set, read at most this many entries from the file.

    Returns:
        DataFrame with jet-level scalar columns and particle-level list columns.
    """
    try:
        import uproot
        import awkward as ak
    except ImportError as e:
        raise ImportError(
            "Reading JetClass ROOT files requires uproot and awkward. "
            "Install with: pip install uproot awkward"
        ) from e

    branches = _JET_BRANCHES + _PART_BRANCHES
    with uproot.open(root_path) as f:
        tree = f["tree"]
        n_entries = tree.num_entries
        entry_stop = min(n_entries, max_jets) if max_jets is not None else n_entries

        arrays = tree.arrays(branches, library="ak", entry_stop=entry_stop)

    # Build DataFrame: jet-level columns are scalars, particle-level are lists
    data: dict[str, list] = {}

    for branch in _JET_BRANCHES:
        data[branch] = ak.to_numpy(arrays[branch]).tolist()

    for branch in _PART_BRANCHES:
        # Each entry is a variable-length array → convert to Python list
        data[branch] = [ak.to_list(arrays[branch][i]) for i in range(len(arrays))]

    return pd.DataFrame(data)


def load_jetclass1_class(
    jetclass_path: str | Path,
    class_name: str,
    n_jets: int,
    split: str = "train",
    seed: int = 42,
) -> pd.DataFrame:
    """Load up to ``n_jets`` jets for one JetClass-I class.

    Reads ROOT files sequentially until enough jets are collected, then
    returns a random sample of exactly ``n_jets`` (or all available if fewer).

    Args:
        jetclass_path: Root directory of the JetClass dataset
            (contains ``train_100M/``, ``val_5M/``, ``test_20M/``).
        class_name: JetClass-I class name (e.g. ``"HToBB"``).
        n_jets: Number of jets to return.
        split: Dataset split — ``"train"``, ``"val"``, or ``"test"``.
        seed: Random seed for reproducible sampling.

    Returns:
        DataFrame with ``n_jets`` rows (or fewer if not enough are available).

    Raises:
        ValueError: If ``class_name`` is not a valid JetClass-I class.
        FileNotFoundError: If no ROOT files are found for the class.
    """
    if class_name not in LABEL_PHYSICS:
        raise ValueError(
            f"Unknown JetClass-I class {class_name!r}. Valid: {CLASSES}"
        )

    root_files = _find_root_files(Path(jetclass_path), class_name, split)
    rng = random.Random(seed)

    collected: list[pd.DataFrame] = []
    n_collected = 0

    for root_file in root_files:
        if n_collected >= n_jets:
            break

        still_needed = n_jets - n_collected
        # Read a bit more than needed so we have margin after filtering
        read_limit = still_needed + 500
        try:
            df = _read_root_file(root_file, max_jets=read_limit)
        except Exception as e:
            print(f"WARNING: Could not read {root_file.name}: {e} — skipping")
            continue

        collected.append(df)
        n_collected += len(df)

    if not collected:
        raise RuntimeError(
            f"No jets could be read for class {class_name!r} from {jetclass_path}"
        )

    combined = pd.concat(collected, ignore_index=True)

    # Random sample down to n_jets
    if len(combined) > n_jets:
        combined = combined.sample(n=n_jets, random_state=seed).reset_index(drop=True)

    if len(combined) < n_jets:
        print(
            f"WARNING: Only {len(combined)} jets available for {class_name} "
            f"(requested {n_jets})."
        )

    return combined


def load_jetclass1_subset(
    jetclass_path: str | Path,
    class_names: list[str],
    n_jets_per_class: int,
    split: str = "train",
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Load jets for multiple JetClass-I classes.

    Args:
        jetclass_path: Root directory of the JetClass dataset.
        class_names: List of class names to load (e.g. ``["HToBB", "HToCC"]``).
        n_jets_per_class: Number of jets per class.
        split: Dataset split — ``"train"``, ``"val"``, or ``"test"``.
        seed: Random seed.

    Returns:
        Dict mapping class name → DataFrame with ``n_jets_per_class`` rows.
    """
    unknown = [c for c in class_names if c not in LABEL_PHYSICS]
    if unknown:
        raise ValueError(
            f"Unknown JetClass-I class name(s): {unknown}. Valid: {CLASSES}"
        )

    result: dict[str, pd.DataFrame] = {}
    for i, cls in enumerate(class_names):
        print(f"Loading {cls} ({i+1}/{len(class_names)}) ...")
        # Vary seed per class so we don't always pick the same jets
        df = load_jetclass1_class(
            jetclass_path, cls, n_jets_per_class, split=split, seed=seed + i
        )
        result[cls] = df
        print(f"  Loaded {len(df)} jets for {cls}")

    return result

"""Generate synthetic JetClass-style data for pipeline testing.

Creates realistic-looking jet data with proper kinematics distributions
based on typical LHC boosted jet properties. This is used as a fallback
when the real JetClass-II download takes too long.

The generated data mimics the schema of JetClass-II with appropriate
physics distributions for each jet class.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


# Physics-motivated parameter ranges for each class
# Based on typical boosted jet properties at the LHC
CLASS_PHYSICS = {
    "Hbb": {
        "mass_mean": 125.0, "mass_std": 15.0,
        "pt_mean": 550.0, "pt_std": 150.0,
        "tau21_mean": 0.35, "tau21_std": 0.10,
        "nparticles_mean": 55, "nparticles_std": 15,
    },
    "Hcc": {
        "mass_mean": 125.0, "mass_std": 15.0,
        "pt_mean": 550.0, "pt_std": 150.0,
        "tau21_mean": 0.37, "tau21_std": 0.10,
        "nparticles_mean": 52, "nparticles_std": 14,
    },
    "Hgg": {
        "mass_mean": 125.0, "mass_std": 18.0,
        "pt_mean": 550.0, "pt_std": 150.0,
        "tau21_mean": 0.48, "tau21_std": 0.12,
        "nparticles_mean": 60, "nparticles_std": 18,
    },
    "H4q": {
        "mass_mean": 125.0, "mass_std": 20.0,
        "pt_mean": 550.0, "pt_std": 150.0,
        "tau21_mean": 0.45, "tau21_std": 0.12,
        "nparticles_mean": 68, "nparticles_std": 20,
    },
    "Hqql": {
        "mass_mean": 125.0, "mass_std": 20.0,
        "pt_mean": 550.0, "pt_std": 150.0,
        "tau21_mean": 0.42, "tau21_std": 0.12,
        "nparticles_mean": 58, "nparticles_std": 16,
    },
    "Zqq": {
        "mass_mean": 91.2, "mass_std": 10.0,
        "pt_mean": 500.0, "pt_std": 150.0,
        "tau21_mean": 0.38, "tau21_std": 0.10,
        "nparticles_mean": 48, "nparticles_std": 12,
    },
    "Wqq": {
        "mass_mean": 80.4, "mass_std": 10.0,
        "pt_mean": 500.0, "pt_std": 150.0,
        "tau21_mean": 0.36, "tau21_std": 0.10,
        "nparticles_mean": 45, "nparticles_std": 12,
    },
    "Tbqq": {
        "mass_mean": 173.0, "mass_std": 20.0,
        "pt_mean": 600.0, "pt_std": 150.0,
        "tau21_mean": 0.40, "tau21_std": 0.12,
        "nparticles_mean": 70, "nparticles_std": 20,
    },
    "Tbl": {
        "mass_mean": 173.0, "mass_std": 25.0,
        "pt_mean": 600.0, "pt_std": 150.0,
        "tau21_mean": 0.45, "tau21_std": 0.12,
        "nparticles_mean": 62, "nparticles_std": 18,
    },
    "QCD": {
        "mass_mean": 20.0, "mass_std": 15.0,
        "pt_mean": 500.0, "pt_std": 150.0,
        "tau21_mean": 0.65, "tau21_std": 0.15,
        "nparticles_mean": 40, "nparticles_std": 15,
    },
}


def generate_jet_constituents(
    n_particles: int,
    jet_pt: float,
    jet_eta: float,
    max_constituents: int = 128,
    rng: np.random.Generator = None,
) -> dict:
    """Generate synthetic jet constituent kinematics.

    Creates particles following approximate soft-collinear emission patterns.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = min(n_particles, max_constituents)

    # Generate pt fractions (power-law like distribution)
    pt_fracs = rng.exponential(scale=0.1, size=n)
    pt_fracs = pt_fracs / pt_fracs.sum()  # normalize

    # Sort by pt (descending)
    pt_fracs = np.sort(pt_fracs)[::-1]

    # Particle pT
    part_pt = pt_fracs * jet_pt

    # Angular displacements (Gaussian within jet cone R~0.8)
    sigma_eta = 0.15 + 0.1 * rng.random(n)
    sigma_phi = 0.15 + 0.1 * rng.random(n)

    part_deta = rng.normal(0, sigma_eta, n)
    part_dphi = rng.normal(0, sigma_phi, n)

    # Clip to typical AK8 radius
    part_deta = np.clip(part_deta, -0.8, 0.8)
    part_dphi = np.clip(part_dphi, -0.8, 0.8)

    # Convert to px, py (approximate)
    angles = rng.uniform(0, 2 * np.pi, n)
    part_px = part_pt * np.cos(angles)
    part_py = part_pt * np.sin(angles)

    # pz from eta
    part_eta = jet_eta + part_deta
    part_pz = part_pt * np.sinh(part_eta)

    # Energy (approximate massless)
    part_energy = np.sqrt(part_pt**2 + part_pz**2) * (1 + 0.01 * rng.random(n))

    # Track impact parameters (simplified)
    part_d0val = rng.normal(0, 0.02, n)
    part_d0err = np.abs(rng.normal(0.01, 0.005, n))
    part_dzval = rng.normal(0, 0.05, n)
    part_dzerr = np.abs(rng.normal(0.02, 0.01, n))

    # Particle IDs (simplified: mix of charged hadrons, photons, neutrals)
    # 211 = pion+, -211 = pion-, 22 = photon, 2112 = neutron, 11 = electron
    pid_choices = [211, -211, 211, 22, 2112, 211, -211, 11, -11]
    part_charge = np.array([rng.choice(pid_choices) for _ in range(n)], dtype=np.float32)

    return {
        "part_px": part_px.tolist(),
        "part_py": part_py.tolist(),
        "part_pz": part_pz.tolist(),
        "part_energy": part_energy.tolist(),
        "part_deta": part_deta.tolist(),
        "part_dphi": part_dphi.tolist(),
        "part_d0val": part_d0val.tolist(),
        "part_d0err": part_d0err.tolist(),
        "part_dzval": part_dzval.tolist(),
        "part_dzerr": part_dzerr.tolist(),
        "part_charge": part_charge.tolist(),
    }


def generate_jet_for_class(
    cls: str,
    idx: int,
    rng: np.random.Generator,
    max_constituents: int = 128,
) -> dict:
    """Generate a single synthetic jet for the given class."""
    params = CLASS_PHYSICS[cls]

    # Jet-level kinematics
    jet_pt = max(200.0, rng.normal(params["pt_mean"], params["pt_std"]))
    jet_eta = rng.uniform(-2.0, 2.0)
    jet_phi = rng.uniform(-np.pi, np.pi)

    # Soft-drop mass
    jet_sdmass = max(5.0, rng.normal(params["mass_mean"], params["mass_std"]))

    # N-subjettiness (tau values)
    tau21 = np.clip(rng.normal(params["tau21_mean"], params["tau21_std"]), 0.05, 0.99)
    jet_tau2 = np.clip(rng.normal(0.3, 0.1), 0.05, 0.9)
    jet_tau1 = jet_tau2 / tau21
    jet_tau3 = jet_tau2 * np.clip(rng.normal(0.6, 0.1), 0.3, 0.9)

    # Number of particles
    jet_nparticles = max(5, int(rng.normal(params["nparticles_mean"], params["nparticles_std"])))
    jet_nparticles = min(jet_nparticles, max_constituents)

    # Jet energy
    jet_energy = jet_pt * np.cosh(jet_eta)

    # Generate constituents
    constituents = generate_jet_constituents(
        jet_nparticles, jet_pt, jet_eta, max_constituents, rng
    )

    # Build the full jet record
    record = {
        "jet_pt": float(jet_pt),
        "jet_eta": float(jet_eta),
        "jet_phi": float(jet_phi),
        "jet_energy": float(jet_energy),
        "jet_sdmass": float(jet_sdmass),
        "jet_nparticles": int(jet_nparticles),
        "jet_tau1": float(jet_tau1),
        "jet_tau2": float(jet_tau2),
        "jet_tau3": float(jet_tau3),
        # One-hot label
        f"label_{cls}": 1,
    }

    # Add zero labels for other classes
    all_classes = list(CLASS_PHYSICS.keys())
    for other_cls in all_classes:
        if other_cls != cls:
            record[f"label_{other_cls}"] = 0

    # Add constituent arrays
    record.update(constituents)

    return record


def generate_synthetic_dataset(
    data_dir: str,
    num_jets_per_class: int = 500,
    seed: int = 42,
    max_constituents: int = 128,
) -> Path:
    """Generate synthetic JetClass-style dataset.

    Args:
        data_dir: Root data directory.
        num_jets_per_class: Number of jets to generate per class.
        seed: Random seed.
        max_constituents: Maximum number of jet constituents.

    Returns:
        Path to the output directory.
    """
    output_dir = Path(data_dir) / "jetclass2_subset"
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    all_classes = list(CLASS_PHYSICS.keys())

    print(f"Generating synthetic JetClass-style data ({num_jets_per_class} jets per class)...")

    for cls in all_classes:
        print(f"  Generating {num_jets_per_class} {cls} jets...")
        jets = []
        for i in range(num_jets_per_class):
            jet = generate_jet_for_class(cls, i, rng, max_constituents)
            jets.append(jet)

        df = pd.DataFrame(jets)
        out_path = output_dir / f"{cls}.parquet"
        df.to_parquet(out_path)
        print(f"    Saved to {out_path} ({len(df)} jets, {out_path.stat().st_size / 1024:.1f} KB)")

    # Save class info metadata
    from data.download_jetclass import CLASS_INFO
    meta_path = output_dir / "class_info.json"
    with open(meta_path, "w") as f:
        json.dump(CLASS_INFO, f, indent=2)

    # Save manifest
    manifest = {
        "num_jets_per_class": num_jets_per_class,
        "seed": seed,
        "classes": all_classes,
        "total_jets": num_jets_per_class * len(all_classes),
        "synthetic": True,
        "note": "Synthetic data generated for pipeline demonstration. Replace with real JetClass-II data for production.",
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total = num_jets_per_class * len(all_classes)
    print(f"\nGenerated {total} synthetic jets across {len(all_classes)} classes")
    print(f"Saved to {output_dir}")

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic jet data")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--num-jets-per-class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_dir = args.data_dir or config["data_dir"]
    num_jets = args.num_jets_per_class or config["dataset"]["num_jets_per_class"]

    generate_synthetic_dataset(data_dir, num_jets, args.seed)


if __name__ == "__main__":
    main()

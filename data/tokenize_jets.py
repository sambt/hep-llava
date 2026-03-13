"""Tokenize jets using OmniJet-alpha VQ-VAE.

This script loads prepared JetClass-II jet data, preprocesses constituents
into the format expected by OmniJet-alpha (pt_rel, eta_rel, phi_rel),
and produces discrete VQ-VAE token indices for each jet.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


def setup_omnijet(data_dir: str, repo_url: str) -> Path:
    """Clone OmniJet-alpha repo and return the path.

    If already cloned, just returns the path.
    """
    omnijet_dir = Path(data_dir) / "omnijet_alpha"
    if not omnijet_dir.exists():
        print(f"Cloning OmniJet-alpha to {omnijet_dir}...")
        os.system(f"git clone {repo_url} {omnijet_dir}")
    else:
        print(f"OmniJet-alpha already exists at {omnijet_dir}")

    # Add to Python path so we can import it
    if str(omnijet_dir) not in sys.path:
        sys.path.insert(0, str(omnijet_dir))

    return omnijet_dir


def find_vqvae_checkpoint(omnijet_dir: Path) -> Path:
    """Locate the pretrained VQ-VAE checkpoint in the OmniJet-alpha repo.

    The checkpoint location may vary by repo version. This function
    searches common locations.
    """
    # Common checkpoint locations in the OmniJet-alpha repo
    candidates = [
        omnijet_dir / "checkpoints",
        omnijet_dir / "models",
        omnijet_dir / "pretrained",
        omnijet_dir / "weights",
    ]

    # Search for .pt or .ckpt files
    for candidate_dir in candidates:
        if candidate_dir.exists():
            for ext in ["*.pt", "*.ckpt", "*.pth"]:
                ckpts = list(candidate_dir.glob(f"**/{ext}"))
                # Prefer files with 'vqvae' or 'tokenizer' in the name
                vqvae_ckpts = [
                    c for c in ckpts
                    if "vqvae" in c.name.lower() or "tokenizer" in c.name.lower()
                ]
                if vqvae_ckpts:
                    return vqvae_ckpts[0]
                if ckpts:
                    return ckpts[0]

    # If no checkpoint found, the cluster agent will need to download it
    print("WARNING: No VQ-VAE checkpoint found. The cluster setup agent should")
    print("  download the checkpoint following OmniJet-alpha's README instructions.")
    print(f"  Searched directories: {[str(c) for c in candidates]}")
    return None


def preprocess_jet_constituents(
    jet_data: dict,
    max_constituents: int = 128,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Extract and preprocess jet constituents for OmniJet-alpha.

    OmniJet-alpha expects per-particle features: (pt_rel, eta_rel, phi_rel)
    where pt_rel is relative to the jet pT, and eta/phi are relative to the jet axis.

    Args:
        jet_data: Dict with particle-level arrays from JetClass-II.
        max_constituents: Maximum number of constituents to keep.

    Returns:
        Tuple of (features array [N, 3], mask [N], actual count).
    """
    # Extract raw constituent kinematics
    # JetClass-II stores: part_px, part_py, part_pz, part_energy, part_deta, part_dphi
    part_deta = np.array(jet_data.get("part_deta", []), dtype=np.float32)
    part_dphi = np.array(jet_data.get("part_dphi", []), dtype=np.float32)
    part_px = np.array(jet_data.get("part_px", []), dtype=np.float32)
    part_py = np.array(jet_data.get("part_py", []), dtype=np.float32)

    # Compute per-particle pT
    part_pt = np.sqrt(part_px**2 + part_py**2)

    # Get jet-level pT for normalization
    jet_pt = jet_data.get("jet_pt", np.sum(part_pt))

    # Compute relative pT
    pt_rel = part_pt / (jet_pt + 1e-8)

    # Number of actual particles
    n_particles = len(part_pt)
    n_keep = min(n_particles, max_constituents)

    # Sort by pT (descending) and keep top-N
    pt_order = np.argsort(-part_pt)[:n_keep]
    pt_rel_sorted = pt_rel[pt_order]
    deta_sorted = part_deta[pt_order]
    dphi_sorted = part_dphi[pt_order]

    # Stack into [N, 3] array: (pt_rel, eta_rel, phi_rel)
    features = np.stack([pt_rel_sorted, deta_sorted, dphi_sorted], axis=-1)

    # Pad to max_constituents
    padded_features = np.zeros((max_constituents, 3), dtype=np.float32)
    padded_features[:n_keep] = features
    mask = np.zeros(max_constituents, dtype=np.bool_)
    mask[:n_keep] = True

    return padded_features, mask, n_keep


def tokenize_with_omnijet(
    features_batch: np.ndarray,
    masks_batch: np.ndarray,
    omnijet_dir: Path,
    checkpoint_path: Path | None,
    device: str = "cuda",
) -> np.ndarray:
    """Tokenize a batch of jets using OmniJet-alpha VQ-VAE.

    This function loads the OmniJet-alpha model and encodes jet constituents
    into discrete codebook indices.

    Args:
        features_batch: [B, N, 3] array of (pt_rel, eta_rel, phi_rel).
        masks_batch: [B, N] boolean mask.
        omnijet_dir: Path to the cloned OmniJet-alpha repo.
        checkpoint_path: Path to VQ-VAE checkpoint.
        device: Torch device.

    Returns:
        [B, N] array of codebook indices (int64).
    """
    # Import OmniJet-alpha modules
    # NOTE: The exact import path depends on the repo structure.
    # The cluster agent should verify and adjust these imports.
    try:
        # Try common import patterns for OmniJet-alpha
        from omnijet_alpha.models.vqvae import VQVAE
    except ImportError:
        try:
            from models.vqvae import VQVAE
        except ImportError:
            raise ImportError(
                "Could not import OmniJet-alpha VQVAE. "
                "The cluster agent should inspect the repo structure at "
                f"{omnijet_dir} and fix the import path."
            )

    # Load model
    if checkpoint_path is None:
        raise ValueError(
            "No VQ-VAE checkpoint path provided. "
            "Run setup first or specify checkpoint_path in config."
        )

    print(f"Loading VQ-VAE from {checkpoint_path}...")
    # NOTE: Model loading may need adjustment based on how OmniJet-alpha
    # saves its checkpoints. The cluster agent should verify this.
    model = VQVAE.load_from_checkpoint(str(checkpoint_path))
    model = model.to(device)
    model.eval()

    # Tokenize
    features_tensor = torch.from_numpy(features_batch).to(device)
    masks_tensor = torch.from_numpy(masks_batch).to(device)

    with torch.no_grad():
        # Encode to get codebook indices
        # NOTE: The exact API may differ — cluster agent should verify
        # by inspecting OmniJet-alpha's encode/tokenize methods
        _, indices, _ = model.encode(features_tensor, masks_tensor)

    token_indices = indices.cpu().numpy()

    # Zero out padded positions
    token_indices[~masks_batch] = 0

    return token_indices


def tokenize_all_jets(
    data_dir: str,
    config: dict,
    device: str = "cuda",
    batch_size: int = 256,
) -> Path:
    """Main function: tokenize all downloaded jets.

    Args:
        data_dir: Root data directory.
        config: Full config dict.
        device: Torch device.
        batch_size: Batch size for VQ-VAE encoding.

    Returns:
        Path to the output directory with tokenized data.
    """
    input_dir = Path(data_dir) / "jetclass2_subset"
    output_dir = Path(data_dir) / "tokenized_jets"
    output_dir.mkdir(parents=True, exist_ok=True)

    max_constituents = config["dataset"]["max_constituents"]

    # Setup OmniJet-alpha
    omnijet_dir = setup_omnijet(data_dir, config["tokenizer"]["repo_url"])
    checkpoint_path = config["tokenizer"].get("checkpoint_path")
    if checkpoint_path is None:
        checkpoint_path = find_vqvae_checkpoint(omnijet_dir)

    # Process each class
    all_tokenized = []
    for cls in config["dataset"]["classes"]:
        parquet_path = input_dir / f"{cls}.parquet"
        if not parquet_path.exists():
            print(f"WARNING: {parquet_path} not found, skipping {cls}")
            continue

        print(f"\nProcessing class: {cls}")
        df = pd.read_parquet(parquet_path)

        class_features = []
        class_masks = []
        class_metadata = []

        for idx, row in df.iterrows():
            jet_data = row.to_dict()
            features, mask, n_particles = preprocess_jet_constituents(
                jet_data, max_constituents
            )
            class_features.append(features)
            class_masks.append(mask)

            # Collect metadata for captioning
            metadata = {
                "jet_id": f"{cls}_{idx:06d}",
                "class": cls,
                "jet_pt": float(jet_data.get("jet_pt", 0)),
                "jet_eta": float(jet_data.get("jet_eta", 0)),
                "jet_phi": float(jet_data.get("jet_phi", 0)),
                "jet_energy": float(jet_data.get("jet_energy", 0)),
                "jet_sdmass": float(jet_data.get("jet_sdmass", 0)),
                "jet_nparticles": int(n_particles),
                "jet_tau1": float(jet_data.get("jet_tau1", 0)),
                "jet_tau2": float(jet_data.get("jet_tau2", 0)),
                "jet_tau3": float(jet_data.get("jet_tau3", 0)),
            }

            # Extract truth-level particle info if available (JetClass-II)
            if "aux_genpart_pt" in jet_data:
                metadata["aux_genpart_pt"] = jet_data["aux_genpart_pt"]
                metadata["aux_genpart_eta"] = jet_data["aux_genpart_eta"]
                metadata["aux_genpart_phi"] = jet_data["aux_genpart_phi"]
                metadata["aux_genpart_mass"] = jet_data["aux_genpart_mass"]
                metadata["aux_genpart_pid"] = jet_data["aux_genpart_pid"]

            class_metadata.append(metadata)

        # Stack into batches and tokenize
        features_array = np.stack(class_features)
        masks_array = np.stack(class_masks)

        # Tokenize in batches
        all_tokens = []
        for i in range(0, len(features_array), batch_size):
            batch_features = features_array[i : i + batch_size]
            batch_masks = masks_array[i : i + batch_size]
            tokens = tokenize_with_omnijet(
                batch_features,
                batch_masks,
                omnijet_dir,
                checkpoint_path,
                device,
            )
            all_tokens.append(tokens)

        tokens_array = np.concatenate(all_tokens, axis=0)

        # Save tokenized data
        for j, meta in enumerate(class_metadata):
            meta["token_indices"] = tokens_array[j].tolist()
            meta["mask"] = masks_array[j].tolist()
            all_tokenized.append(meta)

        print(f"  Tokenized {len(class_metadata)} jets for {cls}")

    # Save all tokenized data
    output_path = output_dir / "tokenized_jets.json"
    with open(output_path, "w") as f:
        json.dump(all_tokenized, f)
    print(f"\nSaved {len(all_tokenized)} tokenized jets to {output_path}")

    # Also save as numpy arrays for efficient loading
    np.save(
        output_dir / "token_indices.npy",
        np.array([j["token_indices"] for j in all_tokenized]),
    )
    np.save(
        output_dir / "masks.npy",
        np.array([j["mask"] for j in all_tokenized]),
    )

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Tokenize jets with OmniJet-alpha")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_dir = args.data_dir or config["data_dir"]
    output_dir = tokenize_all_jets(data_dir, config, args.device, args.batch_size)
    print(f"\nDone. Tokenized data saved to: {output_dir}")


if __name__ == "__main__":
    main()

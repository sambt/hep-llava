"""Tokenize jets using OmniJet-alpha VQ-VAE (or simple discretization fallback).

This script loads prepared JetClass-II jet data, preprocesses constituents
into the format expected by OmniJet-alpha (pt_rel, eta_rel, phi_rel),
and produces discrete VQ-VAE token indices for each jet.

If OmniJet-alpha is unavailable, falls back to simple 3D binning discretization.
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


# ============================================================================
# Simple discretization fallback (32^3 = 32768 codebook)
# ============================================================================

def simple_discretize(features: np.ndarray, n_bins: int = 32) -> np.ndarray:
    """Discretize particle features into codebook indices.

    Implements a simple 3D binning approach as a fallback when OmniJet-alpha
    VQ-VAE is unavailable.

    Args:
        features: [B, N, 3] array of (pt_rel, eta_rel, phi_rel).
        n_bins: Number of bins per feature dimension (total codebook = n_bins^3).

    Returns:
        [B, N] array of integer codebook indices in range [0, n_bins^3 - 1].
    """
    # pt_rel: typically [0, 1] (relative to jet pT)
    pt_bins = np.digitize(features[..., 0], np.linspace(0, 1, n_bins + 1)[1:-1])

    # eta_rel: typically [-0.8, 0.8] (relative to jet axis)
    eta_bins = np.digitize(features[..., 1], np.linspace(-0.8, 0.8, n_bins + 1)[1:-1])

    # phi_rel: typically [-0.8, 0.8] (relative to jet axis)
    phi_bins = np.digitize(features[..., 2], np.linspace(-0.8, 0.8, n_bins + 1)[1:-1])

    # Combine into a single index
    indices = pt_bins * n_bins ** 2 + eta_bins * n_bins + phi_bins
    return np.clip(indices, 0, n_bins ** 3 - 1).astype(np.int64)


# ============================================================================
# OmniJet-alpha tokenizer (primary approach)
# ============================================================================

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


def find_vqvae_checkpoint(omnijet_dir: Path) -> Path | None:
    """Locate the pretrained VQ-VAE checkpoint in the OmniJet-alpha repo.

    Searches common locations including the standard checkpoint directory
    which contains the 8192-token VQ-VAE.
    """
    candidates = [
        omnijet_dir / "checkpoints" / "vqvae_8192_tokens" / "model_ckpt.ckpt",
        omnijet_dir / "checkpoints",
        omnijet_dir / "models",
        omnijet_dir / "pretrained",
        omnijet_dir / "weights",
    ]

    # Check the primary expected location first
    primary = omnijet_dir / "checkpoints" / "vqvae_8192_tokens" / "model_ckpt.ckpt"
    if primary.exists():
        print(f"Found VQ-VAE checkpoint at {primary}")
        return primary

    # Search for .pt or .ckpt files
    for candidate_dir in candidates[1:]:
        if candidate_dir.exists():
            for ext in ["*.ckpt", "*.pt", "*.pth"]:
                ckpts = list(candidate_dir.glob(f"**/{ext}"))
                vqvae_ckpts = [
                    c for c in ckpts
                    if "vqvae" in c.name.lower() or "tokenizer" in c.name.lower()
                ]
                if vqvae_ckpts:
                    return vqvae_ckpts[0]
                if ckpts:
                    return ckpts[0]

    print("WARNING: No VQ-VAE checkpoint found.")
    print(f"  Expected: {primary}")
    print("  Will use simple discretization fallback.")
    return None


def try_load_omnijet_model(omnijet_dir: Path, checkpoint_path: Path, device: str):
    """Attempt to load OmniJet-alpha VQ-VAE model.

    Returns the model if successful, None if import/loading fails.
    """
    try:
        from gabbro.models.vqvae import VQVAELightning
        from omegaconf import OmegaConf

        print(f"Loading OmniJet-alpha VQ-VAE from {checkpoint_path}...")
        model = VQVAELightning.load_from_checkpoint(str(checkpoint_path))
        model = model.to(device)
        model.eval()

        # Load preprocessing config
        config_path = checkpoint_path.parent / "config.yaml"
        if config_path.exists():
            cfg = OmegaConf.load(config_path)
            pp_dict = OmegaConf.to_container(cfg.data.dataset_kwargs_common.feature_dict)
        else:
            # Default pp_dict for the 8192-token VQ-VAE
            pp_dict = {
                "part_pt": {"multiply_by": 1, "subtract_by": 1.8, "func": "np.log", "inv_func": "np.exp"},
                "part_etarel": {"multiply_by": 3, "larger_than": -0.8, "smaller_than": 0.8},
                "part_phirel": {"multiply_by": 3, "larger_than": -0.8, "smaller_than": 0.8},
            }

        print("OmniJet-alpha VQ-VAE loaded successfully!")
        return model, pp_dict

    except Exception as e:
        print(f"Could not load OmniJet-alpha: {e}")
        print("Falling back to simple discretization tokenizer.")
        return None, None


# ============================================================================
# Preprocessing
# ============================================================================

def preprocess_jet_constituents(
    jet_data: dict,
    max_constituents: int = 128,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Extract and preprocess jet constituents.

    Args:
        jet_data: Dict with particle-level arrays from JetClass-II.
        max_constituents: Maximum number of constituents to keep.

    Returns:
        Tuple of (features array [N, 3], mask [N], actual count).
    """
    # Extract raw constituent kinematics
    part_deta = np.array(jet_data.get("part_deta", []), dtype=np.float32)
    part_dphi = np.array(jet_data.get("part_dphi", []), dtype=np.float32)
    part_px = np.array(jet_data.get("part_px", []), dtype=np.float32)
    part_py = np.array(jet_data.get("part_py", []), dtype=np.float32)

    # Compute per-particle pT
    part_pt = np.sqrt(part_px**2 + part_py**2)

    # Get jet-level pT for normalization
    jet_pt = jet_data.get("jet_pt", np.sum(part_pt) + 1e-8)
    if isinstance(jet_pt, (list, np.ndarray)):
        jet_pt = float(jet_pt[0]) if len(jet_pt) > 0 else (np.sum(part_pt) + 1e-8)
    jet_pt = float(jet_pt)

    # Compute relative pT
    pt_rel = part_pt / (jet_pt + 1e-8)

    # Number of actual particles
    n_particles = len(part_pt)
    n_keep = min(n_particles, max_constituents)

    if n_keep == 0:
        padded_features = np.zeros((max_constituents, 3), dtype=np.float32)
        mask = np.zeros(max_constituents, dtype=np.bool_)
        return padded_features, mask, 0

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


def tokenize_batch(
    features_batch: np.ndarray,
    masks_batch: np.ndarray,
    omnijet_model=None,
    pp_dict: dict | None = None,
    device: str = "cuda",
    use_simple: bool = False,
    n_bins: int = 32,
) -> np.ndarray:
    """Tokenize a batch of jets.

    If omnijet_model is provided, uses the VQ-VAE. Otherwise uses
    simple discretization.

    Args:
        features_batch: [B, N, 3] array of (pt_rel, eta_rel, phi_rel).
        masks_batch: [B, N] boolean mask.
        omnijet_model: Loaded VQVAELightning model (or None).
        pp_dict: Preprocessing dict for OmniJet-alpha.
        device: Torch device.
        use_simple: Force simple discretization.
        n_bins: Number of bins for simple discretization.

    Returns:
        [B, N] array of codebook indices (int64).
    """
    if not use_simple and omnijet_model is not None:
        try:
            return _tokenize_with_omnijet(features_batch, masks_batch, omnijet_model, pp_dict, device)
        except Exception as e:
            print(f"OmniJet tokenization failed: {e}, falling back to simple discretization")

    # Simple discretization fallback
    indices = simple_discretize(features_batch, n_bins=n_bins)
    # Zero out padded positions
    indices[~masks_batch] = 0
    return indices


def _tokenize_with_omnijet(
    features_batch: np.ndarray,
    masks_batch: np.ndarray,
    model,
    pp_dict: dict,
    device: str = "cuda",
) -> np.ndarray:
    """Tokenize jets using OmniJet-alpha VQ-VAE.

    This uses the VQVAELightning.forward() method directly on preprocessed
    features, returning the codebook indices.

    Args:
        features_batch: [B, N, 3] array of (pt_rel, eta_rel, phi_rel).
        masks_batch: [B, N] boolean mask.
        model: VQVAELightning model.
        pp_dict: Preprocessing dict.
        device: Torch device.

    Returns:
        [B, N] array of codebook indices (int64).
    """
    import awkward as ak
    from gabbro.utils.arrays import ak_pad, ak_select_and_preprocess, ak_to_np_stack, np_to_ak

    B, N, _ = features_batch.shape

    # Build an awkward array in the format OmniJet expects
    # (variable-length per jet, named fields: part_pt, part_etarel, part_phirel)
    jets_list = []
    for b in range(B):
        n_valid = int(masks_batch[b].sum())
        if n_valid == 0:
            jets_list.append({"part_pt": [], "part_etarel": [], "part_phirel": []})
        else:
            feat = features_batch[b, :n_valid]
            jets_list.append({
                "part_pt": feat[:, 0].tolist(),
                "part_etarel": feat[:, 1].tolist(),
                "part_phirel": feat[:, 2].tolist(),
            })

    ak_arr = ak.Array(jets_list)

    # Use OmniJet's tokenize function
    tokens_ak = model.tokenize_ak_array(
        ak_arr=ak_arr,
        pp_dict=pp_dict,
        batch_size=min(B, 256),
        pad_length=N,
        hide_pbar=True,
    )

    # Convert awkward array back to numpy [B, N]
    tokens_padded, token_mask = ak_pad(tokens_ak, maxlen=N, return_mask=True)
    token_indices = tokens_padded.to_numpy().astype(np.int64)

    # Apply the original mask
    token_indices[~masks_batch] = 0

    return token_indices


# ============================================================================
# Main tokenization pipeline
# ============================================================================

def tokenize_all_jets(
    data_dir: str,
    config: dict,
    device: str = "cuda",
    batch_size: int = 256,
    force_simple: bool = False,
) -> Path:
    """Main function: tokenize all downloaded jets.

    Args:
        data_dir: Root data directory.
        config: Full config dict.
        device: Torch device.
        batch_size: Batch size for VQ-VAE encoding.
        force_simple: If True, skip OmniJet and use simple discretization.

    Returns:
        Path to the output directory with tokenized data.
    """
    input_dir = Path(data_dir) / "jetclass2_subset"
    output_dir = Path(data_dir) / "tokenized_jets"
    output_dir.mkdir(parents=True, exist_ok=True)

    max_constituents = config["dataset"]["max_constituents"]
    tokenizer_cfg = config.get("tokenizer", {})
    codebook_size = tokenizer_cfg.get("codebook_size", 8192)

    # Determine tokenization method
    omnijet_model = None
    pp_dict = None
    use_simple = force_simple
    n_bins = 32  # For simple discretization: 32^3 = 32768

    if not force_simple:
        omnijet_dir = setup_omnijet(data_dir, tokenizer_cfg.get("repo_url", ""))
        checkpoint_path = tokenizer_cfg.get("checkpoint_path")
        if checkpoint_path is None:
            checkpoint_path = find_vqvae_checkpoint(omnijet_dir)

        if checkpoint_path is not None:
            omnijet_model, pp_dict = try_load_omnijet_model(
                omnijet_dir, Path(checkpoint_path), device
            )

        if omnijet_model is None:
            use_simple = True
            print("Using simple discretization fallback (32^3 = 32768 codes)")
            # Update codebook size for simple discretization
            codebook_size = n_bins ** 3  # 32768

    tokenizer_type = "simple_discretization" if use_simple else "omnijet_vqvae"
    print(f"Tokenizer: {tokenizer_type}, codebook size: {codebook_size}")

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

            class_metadata.append(metadata)

        # Stack into batches and tokenize
        features_array = np.stack(class_features)
        masks_array = np.stack(class_masks)

        # Tokenize in batches
        all_tokens = []
        for i in range(0, len(features_array), batch_size):
            batch_features = features_array[i: i + batch_size]
            batch_masks = masks_array[i: i + batch_size]
            tokens = tokenize_batch(
                batch_features,
                batch_masks,
                omnijet_model=omnijet_model,
                pp_dict=pp_dict,
                device=device,
                use_simple=use_simple,
                n_bins=n_bins,
            )
            all_tokens.append(tokens)

        tokens_array = np.concatenate(all_tokens, axis=0)

        # Attach token data to metadata
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

    # Save tokenizer metadata
    tokenizer_meta = {
        "tokenizer_type": tokenizer_type,
        "codebook_size": codebook_size,
        "n_bins": n_bins if use_simple else None,
        "total_jets": len(all_tokenized),
        "classes": config["dataset"]["classes"],
    }
    with open(output_dir / "tokenizer_meta.json", "w") as f:
        json.dump(tokenizer_meta, f, indent=2)

    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Tokenize jets with OmniJet-alpha or simple fallback")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--force-simple", action="store_true",
                        help="Skip OmniJet-alpha and use simple discretization")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    data_dir = args.data_dir or config["data_dir"]
    output_dir = tokenize_all_jets(
        data_dir, config, args.device, args.batch_size, args.force_simple
    )
    print(f"\nDone. Tokenized data saved to: {output_dir}")


if __name__ == "__main__":
    main()

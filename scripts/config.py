"""Configuration loading and path resolution utilities for PhysLLaVA.

Supports:
- Loading a base YAML config with optional deep-merge override
- Deriving a token_set_name from dataset classes + tokenizer type
- Resolving all paths for a given config (new run-namespaced structure)
- Saving a snapshot of the effective config to the run directory

No torch imports at module level — safe to import from any environment.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Token-set name derivation
# ---------------------------------------------------------------------------


def derive_token_set_name(classes: list[str], tokenizer_type: str) -> str:
    """Auto-derive a stable token set name from class list and tokenizer type.

    Classes are sorted for stability.  If the resulting name would exceed 80
    characters a stable 8-character SHA-256 prefix is used instead and the
    full description is recorded in ``token_set_index.json``.

    Args:
        classes: List of jet class names (e.g. ``["Hbb", "QCD"]``).
        tokenizer_type: Short identifier for the tokenizer
            (e.g. ``"omnijet_vqvae"``).

    Returns:
        A short, file-system-safe string to use as the token set directory
        name.

    Examples:
        >>> derive_token_set_name(["X_bb", "QCD_light", "X_cc"], "omnijet_vqvae")
        'QCD_light_X_bb_X_cc__omnijet_vqvae'
    """
    sorted_cls = "_".join(sorted(classes))
    full_name = f"{sorted_cls}__{tokenizer_type}"
    if len(full_name) <= 80:
        return full_name

    # Fall back to a hash-based short name
    h = hashlib.sha256(full_name.encode()).hexdigest()[:8]
    return f"cls_{h}"


def _write_token_set_index(
    data_dir: str | Path,
    token_set_name: str,
    classes: list[str],
    tokenizer_type: str,
) -> None:
    """Append an entry to the token_set_index.json for hash-based names.

    Only writes when the name is hash-based (starts with ``cls_``).

    Args:
        data_dir: Root data directory.
        token_set_name: The (possibly hash-based) token set name.
        classes: Original class list.
        tokenizer_type: Tokenizer identifier.
    """
    if not token_set_name.startswith("cls_"):
        return

    index_path = Path(data_dir) / "tokenized" / "token_set_index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, Any] = {}
    if index_path.exists():
        with open(index_path) as f:
            existing = json.load(f)

    existing[token_set_name] = {
        "classes": sorted(classes),
        "tokenizer_type": tokenizer_type,
        "full_name": "_".join(sorted(classes)) + f"__{tokenizer_type}",
    }

    with open(index_path, "w") as f:
        json.dump(existing, f, indent=2)


# ---------------------------------------------------------------------------
# Config loading with deep-merge
# ---------------------------------------------------------------------------


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*.

    Merge semantics:
    - Scalars: override wins.
    - Dicts: recurse.
    - Lists: override replaces entirely.

    Args:
        base: Base configuration dictionary.
        override: Override configuration dictionary.

    Returns:
        Merged dictionary (a new dict; inputs are not modified).
    """
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def load_config(base: str, override: str | None = None) -> dict:
    """Load base config and optionally deep-merge an override YAML on top.

    Args:
        base: Path to the base YAML config file.
        override: Optional path to an override YAML file.  Scalars and lists
            from the override win; nested dicts are merged recursively.

    Returns:
        Merged configuration dictionary.
    """
    with open(base) as f:
        config: dict = yaml.safe_load(f)

    if override is not None:
        with open(override) as f:
            override_dict: dict = yaml.safe_load(f)
        config = _deep_merge(config, override_dict)

    # Resolve token_set_name if not explicitly set
    if config.get("token_set_name") is None:
        classes = config.get("dataset", {}).get("classes", [])
        tokenizer_type = config.get("tokenizer", {}).get("type", "omnijet_vqvae")
        config["token_set_name"] = derive_token_set_name(classes, tokenizer_type)

    return config


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def get_paths(config: dict) -> dict:
    """Return all resolved paths for a given config.

    The new directory layout is::

        {data_dir}/
        ├── jetclass2_subset/
        ├── tokenized/
        │   └── {token_set_name}/
        │       ├── token_indices.npy
        │       ├── masks.npy
        │       ├── tokenized_jets.json
        │       └── tokenizer_meta.json
        ├── llm_captions/
        │   └── {token_set_name}/
        │       └── llm_captions.json
        └── runs/
            └── {run_name}/
                ├── config.yaml
                ├── caption_data/
                ├── checkpoints/
                │   ├── stage1/
                │   └── stage2/
                └── eval_results/

    Args:
        config: Fully loaded (and merged) configuration dictionary.

    Returns:
        Dictionary with keys:
        ``data_dir``, ``raw_dir``, ``tokenized_dir``, ``llm_captions_dir``,
        ``run_dir``, ``caption_data_dir``, ``checkpoints_dir``,
        ``stage1_checkpoint_dir``, ``stage2_checkpoint_dir``, ``eval_dir``.
    """
    data_dir = Path(config["data_dir"])
    run_name: str = config.get("run_name", "default")
    token_set_name: str = config.get("token_set_name", "default")

    paths = {
        "data_dir": data_dir,
        "raw_dir": data_dir / "jetclass2_subset",
        "tokenized_dir": data_dir / "tokenized" / token_set_name,
        "llm_captions_dir": data_dir / "llm_captions" / token_set_name,
        "run_dir": data_dir / "runs" / run_name,
        "caption_data_dir": data_dir / "runs" / run_name / "caption_data",
        "checkpoints_dir": data_dir / "runs" / run_name / "checkpoints",
        "stage1_checkpoint_dir": data_dir / "runs" / run_name / "checkpoints" / "stage1",
        "stage2_checkpoint_dir": data_dir / "runs" / run_name / "checkpoints" / "stage2",
        "eval_dir": data_dir / "runs" / run_name / "eval_results",
    }

    # Convert all Path objects to Path (already are, but keep consistent)
    return {k: Path(v) for k, v in paths.items()}


# ---------------------------------------------------------------------------
# W&B run ID derivation
# ---------------------------------------------------------------------------


def get_wandb_run_id(config: dict) -> str:
    """Return a stable W&B run ID for this configuration.

    The ID is derived from ``run_name`` so that Stage 1 and Stage 2 can
    resume the same W&B run via ``wandb.init(id=..., resume="allow")``.

    An explicit override can be set via ``logging.wandb_run_id`` in the
    config.  The ID is sanitised to the characters W&B allows (letters,
    digits, hyphens, underscores; max 128 chars).

    Args:
        config: Fully loaded configuration dictionary.

    Returns:
        A W&B-safe run ID string.
    """
    import re

    override = config.get("logging", {}).get("wandb_run_id")
    if override:
        raw = str(override)
    else:
        raw = config.get("run_name", "default")

    sanitised = re.sub(r"[^A-Za-z0-9_\-]", "-", raw)
    return sanitised[:128]


# ---------------------------------------------------------------------------
# Saving effective config snapshot
# ---------------------------------------------------------------------------


def save_effective_config(config: dict, paths: dict) -> None:
    """Save a snapshot of the effective config to ``{run_dir}/config.yaml``.

    Creates the run directory if it does not exist.

    Args:
        config: Fully resolved configuration dictionary.
        paths: Path dict from :func:`get_paths`.
    """
    run_dir: Path = paths["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)

    config_snapshot_path = run_dir / "config.yaml"

    # yaml.safe_dump can't handle Path objects — convert to strings
    def _to_serialisable(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {k: _to_serialisable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_serialisable(v) for v in obj]
        return obj

    with open(config_snapshot_path, "w") as f:
        yaml.safe_dump(_to_serialisable(config), f, default_flow_style=False, sort_keys=False)

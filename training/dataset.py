"""PyTorch datasets for PhysLLaVA training.

Handles loading tokenized jets + conversation data and preparing
batches for the two training stages.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class PhysLLaVADataset(Dataset):
    """Dataset for PhysLLaVA training.

    Each sample contains:
        - VQ-VAE token indices for the jet
        - A conversation (human question + assistant response)
        - Tokenized text with <jet> placeholder

    The dataset handles both caption (Stage 1) and QA (Stage 2) data.
    """

    def __init__(
        self,
        conversations_path: str | Path,
        tokenized_jets_path: str | Path,
        token_indices_path: str | Path,
        masks_path: str | Path,
        tokenizer: PreTrainedTokenizer,
        max_text_length: int = 512,
        max_jet_tokens: int = 128,
    ):
        """
        Args:
            conversations_path: Path to JSON with conversation data.
            tokenized_jets_path: Path to tokenized_jets.json (for jet_id -> index mapping).
            token_indices_path: Path to token_indices.npy.
            masks_path: Path to masks.npy.
            tokenizer: LLM tokenizer.
            max_text_length: Max tokens for text.
            max_jet_tokens: Max VQ-VAE tokens per jet.
        """
        # Load conversations
        with open(conversations_path) as f:
            self.conversations = json.load(f)

        # Load tokenized jet data
        with open(tokenized_jets_path) as f:
            jet_meta_list = json.load(f)

        # Build jet_id -> index mapping
        self.jet_id_to_idx = {j["jet_id"]: i for i, j in enumerate(jet_meta_list)}

        # Build class -> integer index mapping (for contrastive loss)
        classes = sorted({j["class"] for j in jet_meta_list})
        self.class_to_idx: dict[str, int] = {cls: idx for idx, cls in enumerate(classes)}
        self.jet_id_to_class: dict[str, str] = {j["jet_id"]: j["class"] for j in jet_meta_list}

        # Load numpy arrays for efficient access
        self.token_indices = np.load(token_indices_path)
        self.masks = np.load(masks_path)

        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.max_jet_tokens = max_jet_tokens

    def __len__(self) -> int:
        return len(self.conversations)

    def __getitem__(self, idx: int) -> dict:
        conv = self.conversations[idx]
        jet_id = conv["jet_id"]

        # Get jet tokens
        jet_idx = self.jet_id_to_idx[jet_id]
        jet_tokens = self.token_indices[jet_idx][:self.max_jet_tokens]
        jet_mask = self.masks[jet_idx][:self.max_jet_tokens]

        # Build the text from conversation turns
        text_parts = []
        for turn in conv["conversations"]:
            if turn["from"] == "human":
                text_parts.append(f"User: {turn['value']}")
            elif turn["from"] == "gpt":
                text_parts.append(f"Assistant: {turn['value']}")

        full_text = "\n".join(text_parts)

        # Tokenize text
        text_encoding = self.tokenizer(
            full_text,
            max_length=self.max_text_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = text_encoding["input_ids"].squeeze(0)
        attention_mask = text_encoding["attention_mask"].squeeze(0)

        # Create labels: -100 for everything except assistant responses
        labels = input_ids.clone()

        # Mask everything that's not the assistant's response
        # Find "Assistant:" tokens and only compute loss on what follows
        assistant_token_ids = self.tokenizer.encode("Assistant:", add_special_tokens=False)
        labels[:] = -100  # Start by masking everything

        # Find assistant response regions
        input_ids_list = input_ids.tolist()
        for i in range(len(input_ids_list) - len(assistant_token_ids)):
            if input_ids_list[i:i + len(assistant_token_ids)] == assistant_token_ids:
                # Unmask everything after "Assistant:" until end or next "User:"
                start = i + len(assistant_token_ids)
                user_token_ids = self.tokenizer.encode("\nUser:", add_special_tokens=False)
                end = len(input_ids_list)
                for j in range(start, len(input_ids_list) - len(user_token_ids)):
                    if input_ids_list[j:j + len(user_token_ids)] == user_token_ids:
                        end = j
                        break
                labels[start:end] = input_ids[start:end]

        # Class index for supervised contrastive loss
        cls = self.jet_id_to_class.get(jet_id, "")
        class_idx = self.class_to_idx.get(cls, 0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "jet_token_indices": torch.from_numpy(jet_tokens.astype(np.int64)),
            "jet_attention_mask": torch.from_numpy(jet_mask.astype(bool)),
            "class_idx": torch.tensor(class_idx, dtype=torch.long),
        }


class CombinedDataset(Dataset):
    """Combines multiple PhysLLaVADataset instances (e.g., captions + QA)."""

    def __init__(self, datasets: list[Dataset]):
        self.datasets = datasets
        self.cumulative_sizes = []
        total = 0
        for ds in datasets:
            total += len(ds)
            self.cumulative_sizes.append(total)

    def __len__(self) -> int:
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, idx: int) -> dict:
        for i, cum_size in enumerate(self.cumulative_sizes):
            if idx < cum_size:
                if i == 0:
                    return self.datasets[i][idx]
                else:
                    return self.datasets[i][idx - self.cumulative_sizes[i - 1]]
        raise IndexError(f"Index {idx} out of range for CombinedDataset of size {len(self)}")


def build_stage1_dataset(
    data_dir: str,
    tokenizer: PreTrainedTokenizer,
    max_text_length: int = 512,
    paths: dict | None = None,
) -> Dataset:
    """Build dataset for Stage 1 (caption alignment).

    Args:
        data_dir: Root data directory (legacy; used when *paths* is not provided).
        tokenizer: LLM tokenizer.
        max_text_length: Max tokens for text input.
        paths: Optional resolved paths dict from :func:`scripts.config.get_paths`.
            When provided, the new directory layout is used; otherwise the legacy
            flat layout under *data_dir* is used for backward compatibility.
    """
    if paths is not None:
        tokenized_dir = paths["tokenized_dir"]
        caption_dir = paths["caption_data_dir"]
    else:
        # Legacy flat layout
        data_path = Path(data_dir)
        tokenized_dir = data_path / "tokenized_jets"
        caption_dir = data_path / "caption_data"

    return PhysLLaVADataset(
        conversations_path=caption_dir / "captions.json",
        tokenized_jets_path=tokenized_dir / "tokenized_jets.json",
        token_indices_path=tokenized_dir / "token_indices.npy",
        masks_path=tokenized_dir / "masks.npy",
        tokenizer=tokenizer,
        max_text_length=max_text_length,
    )


def build_stage2_dataset(
    data_dir: str,
    tokenizer: PreTrainedTokenizer,
    max_text_length: int = 512,
    paths: dict | None = None,
) -> Dataset:
    """Build dataset for Stage 2 (instruction tuning with captions + QA).

    Args:
        data_dir: Root data directory (legacy; used when *paths* is not provided).
        tokenizer: LLM tokenizer.
        max_text_length: Max tokens for text input.
        paths: Optional resolved paths dict from :func:`scripts.config.get_paths`.
            When provided, the new directory layout is used; otherwise the legacy
            flat layout under *data_dir* is used for backward compatibility.
    """
    if paths is not None:
        tokenized_dir = paths["tokenized_dir"]
        caption_dir = paths["caption_data_dir"]
    else:
        # Legacy flat layout
        data_path = Path(data_dir)
        tokenized_dir = data_path / "tokenized_jets"
        caption_dir = data_path / "caption_data"

    caption_ds = PhysLLaVADataset(
        conversations_path=caption_dir / "captions.json",
        tokenized_jets_path=tokenized_dir / "tokenized_jets.json",
        token_indices_path=tokenized_dir / "token_indices.npy",
        masks_path=tokenized_dir / "masks.npy",
        tokenizer=tokenizer,
        max_text_length=max_text_length,
    )

    qa_ds = PhysLLaVADataset(
        conversations_path=caption_dir / "qa_data.json",
        tokenized_jets_path=tokenized_dir / "tokenized_jets.json",
        token_indices_path=tokenized_dir / "token_indices.npy",
        masks_path=tokenized_dir / "masks.npy",
        tokenizer=tokenizer,
        max_text_length=max_text_length,
    )

    return CombinedDataset([caption_ds, qa_ds])

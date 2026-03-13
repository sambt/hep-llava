"""PhysLLaVA: Full model integrating physics encoder + projector + LLM.

Follows the LLaVA architecture where <jet> tokens in the input are replaced
with projected physics embeddings from the encoder.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from model.physics_encoder import PhysicsTokenEncoder
from model.projector import MLPProjector


# Special tokens
JET_TOKEN = "<jet>"
DEFAULT_JET_PATCH_TOKEN = "<jet_patch>"


class PhysLLaVA(nn.Module):
    """PhysLLaVA: LLaVA-style model for particle physics jets.

    Components:
        1. PhysicsTokenEncoder: Transformer over VQ-VAE tokens
        2. MLPProjector: Projects to LLM embedding space
        3. LLM: Llama 3.1 8B Instruct (or similar)

    The model replaces <jet> placeholder tokens in the text input with
    projected physics embeddings, then runs standard autoregressive LM.
    """

    def __init__(
        self,
        physics_encoder_config: dict,
        projector_config: dict,
        llm_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        torch_dtype: str = "bfloat16",
        use_flash_attention: bool = True,
    ):
        super().__init__()

        # Physics encoder
        self.physics_encoder = PhysicsTokenEncoder(**physics_encoder_config)

        # MLP projector
        self.projector = MLPProjector(**projector_config)

        # LLM
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.torch_dtype = dtype_map.get(torch_dtype, torch.bfloat16)

        attn_impl = "flash_attention_2" if use_flash_attention else "sdpa"

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name,
            torch_dtype=self.torch_dtype,
            attn_implementation=attn_impl,
            low_cpu_mem_usage=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)

        # Add special tokens
        special_tokens = {"additional_special_tokens": [JET_TOKEN, DEFAULT_JET_PATCH_TOKEN]}
        self.tokenizer.add_special_tokens(special_tokens)
        self.llm.resize_token_embeddings(len(self.tokenizer))

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Store token IDs
        self.jet_token_id = self.tokenizer.convert_tokens_to_ids(JET_TOKEN)
        self.jet_patch_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_JET_PATCH_TOKEN)

    def encode_jets(
        self,
        jet_token_indices: torch.LongTensor,
        jet_attention_mask: torch.BoolTensor | None = None,
    ) -> torch.Tensor:
        """Encode jet tokens and project to LLM space.

        Args:
            jet_token_indices: [B, N] VQ-VAE codebook indices.
            jet_attention_mask: [B, N] mask for valid tokens.

        Returns:
            [B, N, llm_dim] projected jet embeddings.
        """
        # Physics encoder
        hidden_states = self.physics_encoder(jet_token_indices, jet_attention_mask)
        # Project to LLM dim
        projected = self.projector(hidden_states)
        return projected.to(self.torch_dtype)

    def prepare_inputs_with_jets(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        jet_embeddings: torch.Tensor,
        jet_attention_mask: torch.BoolTensor | None = None,
        labels: torch.LongTensor | None = None,
    ) -> dict:
        """Replace <jet> token in input_ids with projected jet embeddings.

        For each sample in the batch:
        1. Find the position of <jet> token
        2. Replace it with the sequence of projected jet embeddings
        3. Adjust attention mask and labels accordingly

        Args:
            input_ids: [B, L] tokenized text with <jet> placeholder.
            attention_mask: [B, L] text attention mask.
            jet_embeddings: [B, N_jet, D] projected jet embeddings.
            jet_attention_mask: [B, N_jet] mask for valid jet tokens.
            labels: [B, L] labels for language modeling (optional).

        Returns:
            Dict with inputs_embeds, attention_mask, labels ready for the LLM.
        """
        B, L = input_ids.shape
        _, N_jet, D = jet_embeddings.shape

        # Get text embeddings from LLM
        text_embeds = self.llm.get_input_embeddings()(input_ids)  # [B, L, D]

        # For each sample, find <jet> token and splice in jet embeddings
        new_embeds_list = []
        new_mask_list = []
        new_labels_list = []

        for i in range(B):
            # Find position of <jet> token
            jet_positions = (input_ids[i] == self.jet_token_id).nonzero(as_tuple=True)[0]

            if len(jet_positions) == 0:
                # No <jet> token — just use text embeddings as-is
                new_embeds_list.append(text_embeds[i])
                new_mask_list.append(attention_mask[i])
                if labels is not None:
                    new_labels_list.append(labels[i])
                continue

            jet_pos = jet_positions[0].item()

            # Split text embeddings around the <jet> token
            before = text_embeds[i, :jet_pos]  # [jet_pos, D]
            after = text_embeds[i, jet_pos + 1:]  # [L - jet_pos - 1, D]

            # Get valid jet embeddings
            if jet_attention_mask is not None:
                n_valid = jet_attention_mask[i].sum().item()
                jet_emb = jet_embeddings[i, :n_valid]  # [n_valid, D]
            else:
                jet_emb = jet_embeddings[i]  # [N_jet, D]

            # Concatenate: [before] + [jet embeddings] + [after]
            combined = torch.cat([before, jet_emb, after], dim=0)
            new_embeds_list.append(combined)

            # Adjust attention mask
            before_mask = attention_mask[i, :jet_pos]
            after_mask = attention_mask[i, jet_pos + 1:]
            jet_mask_part = torch.ones(jet_emb.shape[0], device=attention_mask.device, dtype=attention_mask.dtype)
            combined_mask = torch.cat([before_mask, jet_mask_part, after_mask], dim=0)
            new_mask_list.append(combined_mask)

            # Adjust labels
            if labels is not None:
                before_labels = labels[i, :jet_pos]
                after_labels = labels[i, jet_pos + 1:]
                # Jet token positions get label -100 (ignore in loss)
                jet_labels = torch.full(
                    (jet_emb.shape[0],), -100,
                    device=labels.device, dtype=labels.dtype,
                )
                combined_labels = torch.cat([before_labels, jet_labels, after_labels], dim=0)
                new_labels_list.append(combined_labels)

        # Pad to same length
        max_len = max(e.shape[0] for e in new_embeds_list)

        padded_embeds = torch.zeros(B, max_len, D, device=text_embeds.device, dtype=text_embeds.dtype)
        padded_mask = torch.zeros(B, max_len, device=attention_mask.device, dtype=attention_mask.dtype)
        padded_labels = torch.full((B, max_len), -100, device=input_ids.device, dtype=torch.long) if labels is not None else None

        for i in range(B):
            seq_len = new_embeds_list[i].shape[0]
            padded_embeds[i, :seq_len] = new_embeds_list[i]
            padded_mask[i, :seq_len] = new_mask_list[i]
            if padded_labels is not None and new_labels_list:
                padded_labels[i, :seq_len] = new_labels_list[i]

        result = {
            "inputs_embeds": padded_embeds,
            "attention_mask": padded_mask,
        }
        if padded_labels is not None:
            result["labels"] = padded_labels

        return result

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        jet_token_indices: torch.LongTensor,
        jet_attention_mask: torch.BoolTensor | None = None,
        labels: torch.LongTensor | None = None,
    ) -> dict:
        """Forward pass: encode jets, splice into text, run LLM.

        Args:
            input_ids: [B, L] tokenized text with <jet> placeholder.
            attention_mask: [B, L] attention mask for text.
            jet_token_indices: [B, N] VQ-VAE codebook indices.
            jet_attention_mask: [B, N] mask for valid jet tokens.
            labels: [B, L] labels for LM loss (optional).

        Returns:
            CausalLMOutput with loss and logits.
        """
        # Encode jets
        jet_embeddings = self.encode_jets(jet_token_indices, jet_attention_mask)

        # Prepare inputs with jet embeddings spliced in
        model_inputs = self.prepare_inputs_with_jets(
            input_ids, attention_mask, jet_embeddings, jet_attention_mask, labels
        )

        # Forward through LLM
        outputs = self.llm(
            inputs_embeds=model_inputs["inputs_embeds"],
            attention_mask=model_inputs["attention_mask"],
            labels=model_inputs.get("labels"),
            return_dict=True,
        )

        return outputs

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        jet_token_indices: torch.LongTensor,
        jet_attention_mask: torch.BoolTensor | None = None,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        **kwargs,
    ) -> torch.LongTensor:
        """Generate text conditioned on jet data.

        Args:
            input_ids: [B, L] tokenized prompt with <jet> placeholder.
            attention_mask: [B, L] attention mask.
            jet_token_indices: [B, N] VQ-VAE codebook indices.
            jet_attention_mask: [B, N] mask for valid jet tokens.
            max_new_tokens: Max tokens to generate.
            temperature: Sampling temperature.

        Returns:
            [B, L'] generated token IDs.
        """
        # Encode jets
        jet_embeddings = self.encode_jets(jet_token_indices, jet_attention_mask)

        # Prepare inputs
        model_inputs = self.prepare_inputs_with_jets(
            input_ids, attention_mask, jet_embeddings, jet_attention_mask
        )

        # Generate
        outputs = self.llm.generate(
            inputs_embeds=model_inputs["inputs_embeds"],
            attention_mask=model_inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=self.tokenizer.pad_token_id,
            **kwargs,
        )

        return outputs

    def freeze_llm(self):
        """Freeze LLM parameters (for Stage 1 training)."""
        for param in self.llm.parameters():
            param.requires_grad = False

    def unfreeze_llm(self):
        """Unfreeze LLM parameters (for Stage 2 training)."""
        for param in self.llm.parameters():
            param.requires_grad = True

    def get_trainable_params(self) -> dict[str, int]:
        """Report trainable vs frozen parameter counts."""
        trainable = 0
        frozen = 0
        by_component = {}

        for name, param in self.named_parameters():
            component = name.split(".")[0]
            n = param.numel()
            if param.requires_grad:
                trainable += n
                by_component[component] = by_component.get(component, 0) + n
            else:
                frozen += n

        return {
            "trainable": trainable,
            "frozen": frozen,
            "total": trainable + frozen,
            "trainable_pct": 100 * trainable / (trainable + frozen),
            "by_component": by_component,
        }

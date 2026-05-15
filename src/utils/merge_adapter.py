# Created by Hao at 2025-07-07
import torch
import logging

from safetensors.torch import load_file, save_file

import sys
import os

logger = logging.getLogger(__name__)

exp_path = sys.argv[1]


def merge_lora_weights_from_safetensors(input_safetensors, output_safetensors, lora_alpha=32, r=16):
    """
    Load weights from a `safetensors` file, merge LoRA adapters into the base weights,
    and save the result to a new `safetensors` file.

    Args:
    - input_safetensors (str): Path to the input `safetensors` file (base weights + LoRA layers).
    - output_safetensors (str): Path where the merged `safetensors` file will be written.
    """
    # LoRA scaling factor
    lora_alpha = lora_alpha
    r = r
    scaling_factor = lora_alpha / r  # scaling factor

    # Load safetensors checkpoint
    logger.info(f"Loading weights from: {input_safetensors}")
    state_dict = load_file(input_safetensors)
    merged_state_dict = {}

    # Walk parameters and merge weights
    for name, param in state_dict.items():
        # Skip standalone LoRA tensors (handled with base_layer)
        if "lora_A" in name or "lora_B" in name:
            continue

        # Merge base_layer.weight
        if "base_layer.weight" in name:
            base_name = name.replace(".base_layer.weight", ".weight")
            lora_A_name = name.replace("base_layer.weight", "lora_A.default.weight")
            lora_B_name = name.replace("base_layer.weight", "lora_B.default.weight")

            if lora_A_name in state_dict and lora_B_name in state_dict:
                # Merge rule: W_final = W_base + scaling_factor * (B @ A)
                base_weight = param
                lora_A = state_dict[lora_A_name]
                lora_B = state_dict[lora_B_name]
                delta_weight = torch.matmul(lora_B, lora_A) * scaling_factor
                merged_weight = base_weight + delta_weight

                # Store merged weight
                merged_state_dict[base_name] = merged_weight
            else:
                merged_state_dict[base_name] = param

        # Merge base_layer.bias
        elif "base_layer.bias" in name:
            base_name = name.replace(".base_layer.bias", ".bias")
            merged_state_dict[base_name] = param.clone()

        # Copy any other tensors unchanged
        else:
            merged_state_dict[name] = param

    metadata = {"format": "pt"}  # extend with any extra metadata you need

    # Write merged weights to safetensors
    logger.info(f"Saving merged weights to: {output_safetensors}")
    save_file(merged_state_dict, output_safetensors, metadata=metadata)
    logger.info("✅ LoRA weights successfully merged and saved.")


# model_name="wavlm-llama-3.2-1B-encoder_unfreeze-decoder_freeze-adater_decoder-libri2mix_n"

model_name = exp_path

input_safetensors = model_name + "/model_unmerge.safetensors"
output_safetensors = model_name + "/model.safetensors"

merge_lora_weights_from_safetensors(input_safetensors, output_safetensors)


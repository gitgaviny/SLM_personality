# Created by Hao at 2025-07-01
# insert_adapters.py
#
# Utility for inserting LoRA adapters into a model.
# If you only need adapters in the decoder, control it with `model_args.adapter_only_decoder`.
#
# Dependencies:
#   pip install peft
#
# Usage example:
#   from insert_adapters import insert_adapters
#   insert_adapters(model, model_args, config)

import logging
from peft import LoraConfig  # Uncomment get_peft_model if you prefer that API
# from peft import get_peft_model
# from your_project.utils import checking_trainable_params  # Adjust import path to your project

logger = logging.getLogger(__name__)

def checking_trainable_params(model):
    """Quick utility to print the number of trainable parameters."""
    trainable, total = 0, 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    logger.info(f"Total params: {total:,}; Trainable: {trainable:,} ({trainable/total:.2%})")

def insert_adapters(model, model_args, config):
    """Insert LoRA adapters into the given model.

    Args:
        model:        The model instance to receive LoRA adapters.
        model_args:   An argument namespace that must contain a boolean attribute
                      `adapter_only_decoder` indicating whether to restrict adapters
                      to the decoder.
        config:       The model configuration, expected to have
                      `decoder.num_hidden_layers` when `adapter_only_decoder` is True.
    """

    # 1. Build the target_modules list
    if getattr(model_args, "adapter_only_decoder", False):
        target_modules = []
        for layer_idx in range(config.decoder.num_hidden_layers):
            target_modules.extend([
                f"decoder.model.layers.{layer_idx}.self_attn.k_proj",
                f"decoder.model.layers.{layer_idx}.self_attn.q_proj",
                f"decoder.model.layers.{layer_idx}.self_attn.v_proj",
                f"decoder.model.layers.{layer_idx}.self_attn.o_proj",
            ])
    else:
        target_modules = ["k_proj", "q_proj", "v_proj", "o_proj"]

    # 2. Construct the LoRA configuration
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=target_modules,
        # Enable modules_to_save if you want to keep certain original weights
        # modules_to_save=["lm_head", "embed_tokens", "embed_positions",
        #                  "layernorm_embedding", "encoder"],
    )

    # 3. Log parameter stats before inserting adapters
    logger.info("=" * 48)
    logger.info("Parameter statistics before inserting adapters")
    checking_trainable_params(model)

    # 4. Insert and enable adapters
    model.add_adapter(lora_config)
    model.enable_adapters()

    # 5. Log parameter stats after inserting adapters
    logger.info("Adapters inserted and enabled.")
    logger.info("=" * 48)
    logger.info("Parameter statistics after inserting adapters")
    checking_trainable_params(model)


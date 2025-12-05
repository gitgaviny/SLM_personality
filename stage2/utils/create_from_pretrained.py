#!/usr/bin/env python3
"""
Created by Yuan at 2025-11-19

build_speech_encoder_decoder.py
=================================
Create a *SpeechEncoderDecoderModelLlama* checkpoint by pairing a WavLM encoder
with a Llama-3 decoder, then save the merged model, tokenizer, and
feature-extractor.  All hyper-parameters and I/O paths are now configurable via
CLI flags so the script can be called conveniently from shell / SLURM jobs.

New in this revision
--------------------
* **--instruct** flag: when _enabled_ the script adds the full set of special
  tokens used by the "Instruct" variant (``<sc>``, ``<pad>``, ``<bos_prompt>``,
  ``<eos_prompt>``, ``<bos_speech>``, ``<eos_speech>``, ``<bos_response>``,
  ``<eos_response>``).  When the flag is **absent**, only the speaker-change
  token ``<sc>`` is injected.
* Replaced ``print`` with structured logging.
* Added ``--log_level`` and ``--check_generate`` convenience flags.

Example usage
~~~~~~~~~~~~
Build with full Instruct prompt tokens::

    python build_speech_encoder_decoder.py \
        --instruct \
        --llm_id Meta-Llama-3.1-8B-Instruct

Classic (non-Instruct) model, add only ``<sc>``::

    python build_speech_encoder_decoder.py \
        --llm_id Meta-Llama-3.1-8B \
        --suffix _v1
"""
from __future__ import annotations
import sys
import os

import argparse
import logging
from pathlib import Path
from typing import List

import torch
from transformers import AutoFeatureExtractor, AutoTokenizer

parent = os.path.abspath(os.path.join(__file__, "..", "..", "models"))
if parent not in sys.path:
    sys.path.insert(0, parent)
from modeling_speech_encoder_decoder_llama import SpeechEncoderDecoderModelLlama  # noqa: E402

###############################################################################
# CLI
###############################################################################


def parse_args() -> argparse.Namespace:  # noqa: D401
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Build WavLM ⟶ Llama SPEAR model")

    # I/O
    parser.add_argument(
        "--encoder_base",
        dest="encoder_base",
        default="microsoft/wavlm-large",
        help="HF identifier or local path for the speech encoder",
    )
    parser.add_argument(
        "--llm_id",
        default="Meta-Llama-3.1-8B-Instruct",
        help="Folder / HF id of the Llama decoder (name only – no parent dir)",
    )
    parser.add_argument(
        "--decoder_base",
        default="/lustre/share/downloaded/models/meta-llama/",
        help="Base directory containing Llama checkpoints; overriden if --decoder_id set",
    )
    parser.add_argument(
        "--decoder_id",
        default=None,
        help="Full path or HF id of the decoder; if omitted, use decoder_base/llm_id",
    )
    parser.add_argument(
        "--save_dir",
        default=None,
        help="Where to save the assembled model/tokenizer/feature-extractor (defaults to wavlm-<llm_id>)",
    )

    # Behaviour switches
    parser.add_argument(
        "--instruct",
        action="store_true",
        help="Add full Instruct prompt-format special tokens (default: add only <sc>)",
    )
    parser.add_argument(
        "--check_generate",
        action="store_true",
        help="Run a single dummy generate() call to verify everything works",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Python logging level (default: INFO)",
    )

    # For multi-speaker settings
    parser.add_argument(
        "--talker_ctc",
        dest="talker_ctc",
        action="store_true",
        help="Whether to use the talker-CTCs (default: False)",
    )

    return parser.parse_args()


###############################################################################
# Utility helpers
###############################################################################


def configure_logging(level: str = "INFO") -> None:  # noqa: D401
    """Configure root logger with timestamp + level."""
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=getattr(logging, level.upper(), logging.INFO),
    )


def ensure_path(path: str | Path) -> Path:  # noqa: D401
    """Create directory if it doesn’t exist and return *Path* object."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def add_special_tokens(tokenizer: AutoTokenizer, instruct: bool) -> List[int]:  # noqa: D401
    """Add special tokens and return their ids (for logging)."""

    if instruct:
        extra_tokens = [
            "<sc>",
            "<neutral>",
            "<positive>",
            "<negative>",
            "<pad>",
            "<bos_prompt>",
            "<eos_prompt>",
            "<bos_speech>",
            "<eos_speech>",
            "<bos_response>",
            "<eos_response>",
            "<openness_low>",
            "<openness_high>",
            "<conscientiousness_low>",
            "<conscientiousness_high>",
            "<extraversion_low>",
            "<extraversion_high>",
            "<agreeableness_low>",
            "<agreeableness_high>",
            "<neuroticism_low>",
            "<neuroticism_high>"
        ]
    else:
        extra_tokens = [
            "<sc>",
            "<pad>",
            "<neutral>",
            "<positive>",
            "<negative>",
            "<openness_low>",
            "<openness_high>",
            "<conscientiousness_low>",
            "<conscientiousness_high>",
            "<extraversion_low>",
            "<extraversion_high>",
            "<agreeableness_low>",
            "<agreeableness_high>",
            "<neuroticism_low>",
            "<neuroticism_high>"
        ]

    # Split pad vs additional_special_tokens
    additional = [tok for tok in extra_tokens if tok != "<pad>"]

    token_ids: List[int] = []
    if additional:
        tokenizer.add_special_tokens({"additional_special_tokens": additional})
        token_ids.extend(tokenizer.convert_tokens_to_ids(additional))
    if "<pad>" in extra_tokens:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        token_ids.append(tokenizer.convert_tokens_to_ids("<pad>"))

    return token_ids


###############################################################################
# Main
###############################################################################


def main() -> None:  # noqa: D401
    args = parse_args()
    configure_logging(args.log_level)

    logging.info("Building SpeechEncoderDecoderModelLlama – instruct=%s", args.instruct)

    # Resolve paths
    decoder_id = args.decoder_id or os.path.join(args.decoder_base, args.llm_id)
    save_dir = args.save_dir or f"hubert-{args.llm_id}"
    save_path = ensure_path(save_dir)

    logging.info("Encoder: %s", args.encoder_base)
    logging.info("Decoder: %s", decoder_id)
    logging.info("Save dir: %s", save_path)

    # ------------------------------------------------------------------
    # Load model: encoder + decoder
    # ------------------------------------------------------------------
    model = SpeechEncoderDecoderModelLlama.from_encoder_decoder_pretrained(
        args.encoder_base,
        decoder_id,
        # encoder_add_adapter=True,
    )

    # Basic config tweaks (kept from original script)
    model.config.encoder.feat_proj_dropout = 0.0
    model.config.encoder.final_dropout = 0.0
    model.config.encoder.mask_time_prob = 0.1
    model.config.encoder.layerdrop = 0.0

    model.config.decoder_start_token_id = model.decoder.config.bos_token_id
    model.config.pad_token_id = model.decoder.config.pad_token_id
    model.config.eos_token_id = model.decoder.config.eos_token_id

    model.config.max_length = 200
    model.config.num_beams = 1
    model.config.use_cache = True
    model.config.processor_class = "Wav2Vec2Processor"

    if args.instruct:
        model.config.instruct = True
        model.config.decoder.instruct = True
    else:
        model.config.instruct = False
        model.config.decoder.instruct = False

    if args.talker_ctc:
        model.config.talker_ctc = True

    # ------------------------------------------------------------------
    # Tokenizer + feature extractor
    # ------------------------------------------------------------------
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.encoder_base)
    feature_extractor.save_pretrained(save_path)

    tokenizer = AutoTokenizer.from_pretrained(decoder_id)

    added_ids = add_special_tokens(tokenizer, args.instruct)
    logging.info("Added %d special tokens", len(added_ids))
    for tok_id in added_ids:
        tok = tokenizer.convert_ids_to_tokens(tok_id)
        logging.debug("Token %-15s -> id %d", tok, tok_id)

    # If we inserted <pad>, wire it up to config
    if "<pad>" in tokenizer.additional_special_tokens or tokenizer.pad_token == "<pad>":
        pad_id = tokenizer.convert_tokens_to_ids("<pad>")
        model.config.decoder.pad_token_id = pad_id
        model.config.pad_token_id = pad_id
        logging.info("Pad token id set to %d", pad_id)

    model.config.sc_token_id = tokenizer.convert_tokens_to_ids("<sc>")
    model.config.decoder.sc_token_id = tokenizer.convert_tokens_to_ids("<sc>")
    model.config.ignore_token_id = -100
    model.config.decoder.ignore_token_id = -100

    if args.instruct:
        model.config.bosp_token_id = tokenizer.convert_tokens_to_ids("<bos_prompt>")
        model.config.eosp_token_id = tokenizer.convert_tokens_to_ids("<eos_prompt>")
        model.config.boss_token_id = tokenizer.convert_tokens_to_ids("<bos_speech>")
        model.config.eoss_token_id = tokenizer.convert_tokens_to_ids("<eos_speech>")
        model.config.bosr_token_id = tokenizer.convert_tokens_to_ids("<bos_response>")
        model.config.eosr_token_id = tokenizer.convert_tokens_to_ids("<eos_response>")

        model.config.decoder.bosp_token_id = tokenizer.convert_tokens_to_ids("<bos_prompt>")
        model.config.decoder.eosp_token_id = tokenizer.convert_tokens_to_ids("<eos_prompt>")
        model.config.decoder.boss_token_id = tokenizer.convert_tokens_to_ids("<bos_speech>")
        model.config.decoder.eoss_token_id = tokenizer.convert_tokens_to_ids("<eos_speech>")
        model.config.decoder.bosr_token_id = tokenizer.convert_tokens_to_ids("<bos_response>")
        model.config.decoder.eosr_token_id = tokenizer.convert_tokens_to_ids("<eos_response>")

    # Resize decoder embeddings to reflect new vocab size
    model.decoder.resize_token_embeddings(len(tokenizer))
    logging.info("New vocab size: %d", model.decoder.config.vocab_size)

    tokenizer.save_pretrained(save_path)

    if args.instruct:
        prompts_idx = torch.ones((1, 10)).long()
    else:
        prompts_idx = None

    # Optional quick generate test (dummy input)
    # if args.check_generate:
    #     logging.info("Running dummy generate() to verify model works …")
    #     out = model.generate(
    #         inputs=torch.ones((1, 2000)),
    #         prompt_ids=prompts_idx,
    #         max_length=150,
    #         num_beams=1,
    #         synced_gpus=False,
    #         use_cache=True,
    #     )
    #     logging.info("Generate output shape: %s", out.shape)

    # Save final model
    model.save_pretrained(save_path)
    logging.info("✅ Model, tokenizer, and feature extractor saved to %s", save_path)


if __name__ == "__main__":
    main()

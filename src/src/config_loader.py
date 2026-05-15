"""
Created by Hao at 2025-07-01

src.config_loader
=======================

Helper to load HuggingFace AutoConfig based on model_args.

Usage:
------
from utils.config_loader import load_config
config = load_config(model_args)
"""
from transformers import AutoConfig
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


def load_config(model_args) -> AutoConfig:
    """Load model config from a pre-trained checkpoint or user config path."""
    config_path = model_args.config_name or model_args.model_name_or_path
    logger.info(f"Loading config from: {config_path}")

    config = AutoConfig.from_pretrained(
        config_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    return config

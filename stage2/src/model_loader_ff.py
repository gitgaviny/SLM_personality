
from __future__ import annotations
import logging
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(current_dir, '..', 'models_ff')
if models_dir not in sys.path:
    sys.path.insert(0, models_dir)

from modeling_speech_encoder_decoder_llama import SpeechEncoderDecoderModelLlama

logger = logging.getLogger(__name__)

def load_aed_model(model_args, config, logger: logging.Logger | None = None):
    """Load SpeechEncoderDecoderModelLlama using model_args and config.

    Parameters
    ----------
    model_args : Namespace | dataclass
        Should include ``model_name_or_path``, ``cache_dir``, ``model_revision``, etc.
    config : transformers.PretrainedConfig
        Preloaded configuration to attach to model.
    logger : logging.Logger, optional
        Custom logger for info output.
    """
    lg = logger or logging.getLogger(__name__)
    lg.info("Loading model from %s", model_args.model_name_or_path)

    model = SpeechEncoderDecoderModelLlama.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    lg.info("Model class: %s", model.__class__.__name__)
    return model


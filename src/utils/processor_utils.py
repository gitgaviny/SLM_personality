# Created by Hao at 2025-07-02
# processor_utils.py
# -----------------------------------------------------------
# Save feature extractor / tokenizer / config once (main process),
# then reload a unified `AutoProcessor` from the output directory.
#
# Usage:
#   from processor_utils import save_and_create_processor
#   processor = save_and_create_processor(
#       training_args, feature_extractor, tokenizer, config
#   )
# -----------------------------------------------------------

import logging
from transformers.trainer_utils import is_main_process
from transformers import AutoProcessor

logger = logging.getLogger(__name__)


def save_and_create_processor(
    training_args,
    feature_extractor,
    tokenizer,
    config,
):
    """
    Save FE / tokenizer / config on the main process, then reload a
    single `AutoProcessor` pointing to `training_args.output_dir`.

    Parameters
    ----------
    training_args : transformers.TrainingArguments
        Must expose `.output_dir`, `.local_rank`,
        and the context manager `main_process_first()`.
    feature_extractor : transformers.FeatureExtractionMixin
    tokenizer         : transformers.PreTrainedTokenizerBase
    config            : transformers.PretrainedConfig

    Returns
    -------
    transformers.AutomaticSpeechProcessor (or relevant subclass)
    """

    # Ensure only rank-0 writes to disk
    with training_args.main_process_first():
        if is_main_process(training_args.local_rank):
            logger.info("Saving feature_extractor, tokenizer, and config to %s",
                        training_args.output_dir)
            feature_extractor.save_pretrained(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)
            config.save_pretrained(training_args.output_dir)

    # All ranks load the processor from disk
    processor = AutoProcessor.from_pretrained(training_args.output_dir)
    logger.info("Processor loaded from %s", training_args.output_dir)
    return processor


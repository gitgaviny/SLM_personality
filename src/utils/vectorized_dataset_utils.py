# Author: Hao at 2025-07-01
# preprocess_utils.py
# -----------------------------------------------------------
# Utilities for step 7: preprocessing speech datasets.
#
#  * Resamples audio if needed       → see `maybe_resample_dataset`
#  * Maps over the dataset to:
#       - load/normalize audio
#       - add attention masks (Whisper + SpecAugment case)
#       - add `input_length`
#       - build prompt+target tokens
#  * Filters by min/max duration
#
# -----------------------------------------------------------

import logging
from typing import Dict

import datasets
from utils.instruction_template_utils import build_prompt_and_input
from datasets import DatasetDict, load_dataset, load_from_disk

logger = logging.getLogger(__name__)


def preprocess_and_filter(
    raw_datasets: datasets.DatasetDict,
    data_args,
    feature_extractor,
    tokenizer,
    config,
    training_args,
    inference_mode=False,
) -> datasets.DatasetDict:
    """
    Map + filter the dataset dict according to step 7 of the training script.

    Returns
    -------
    datasets.DatasetDict
        The vectorized & filtered dataset dict ready for Dataloader.
    """

    # ------------ constants ------------------------------------------
    target_sr = feature_extractor.sampling_rate
    audio_col = data_args.audio_column_name
    text_col  = data_args.text_column_name
    model_input_name = feature_extractor.model_input_names[0]
    max_len = data_args.max_duration_in_seconds * target_sr
    min_len = data_args.min_duration_in_seconds * target_sr
    num_proc = data_args.preprocessing_num_workers
    do_lower = data_args.do_lower_case

    forward_mask = (
        getattr(config, "model_type", None) == "whisper"
        and getattr(config, "apply_spec_augment", False)
        and getattr(config, "mask_time_prob", 0) > 0
    )

    # ------------ optional sample limiting --------------------------


    if inference_mode == False:
        if data_args.max_train_samples is not None and "train" in raw_datasets:
            raw_datasets["train"] = raw_datasets["train"].select(range(data_args.max_train_samples))
        if data_args.max_eval_samples is not None and "eval" in raw_datasets:
            raw_datasets["eval"] = raw_datasets["eval"].select(range(data_args.max_eval_samples))
    else:
        raw_datasets = DatasetDict({
            "eval": raw_datasets
        })

    # ------------ mapping fn ----------------------------------------
    def prepare_example(batch: Dict):
        # 1) audio → features
        audio = batch[audio_col]

        if inference_mode:
            idx = batch['id']
            batch["idx"] = idx

        feats = feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
            return_attention_mask=forward_mask,
        )
        batch[model_input_name] = feats.get(model_input_name)[0]
        batch["input_length"]   = len(audio["array"])
        if forward_mask:
            batch["attention_mask"] = feats["attention_mask"][0]

        # 2) build target sequence with prompt if necessary
        text = batch[text_col].lower() if do_lower else batch[text_col]
        if config.instruct:
            prompt = batch["prompt"].lower() if do_lower else batch["prompt"]
            prompt_str, input_str = build_prompt_and_input(prompt, text)
            batch["labels"] = tokenizer(prompt_str + input_str).input_ids
            batch["prompt_ids"] = tokenizer(prompt_str).input_ids[1:]
        else:
            batch["labels"] = tokenizer(text).input_ids
            batch["prompt_ids"] = []

        return batch

    with training_args.main_process_first(desc="dataset map pre-processing"):
        vectorized = raw_datasets.map(
            prepare_example,
            remove_columns=next(iter(raw_datasets.values())).column_names,
            num_proc=num_proc,
            desc="preprocess train/val dataset",
        )

    # ------------ length-based filtering ----------------------------
    def _in_range(length):
        return min_len < length < max_len

    vectorized = vectorized.filter(
        _in_range,
        num_proc=num_proc,
        input_columns=["input_length"],
    )

    logger.info("Preprocessing complete: %s", {k: len(v) for k, v in vectorized.items()})
    return vectorized


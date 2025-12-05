"""
Created by Hao at 2025-07-01

utils.checkpoint_checking_utils
=========================
Helper utilities for checkpoint detection / auto‑resume logic, split out from
main so the entry‑point stays clean.

Public API
----------
resume_or_raise(training_args, logger)
    Replicates the standard 🤗 Transformers example logic:

    * If ``output_dir`` exists & non‑empty **and** ``do_train`` is True
      * If `--overwrite_output_dir` not set ➔ either resume from last checkpoint
        or raise an error.
      * If a checkpoint is found and ``--resume_from_checkpoint`` is *not*
        provided, it logs the resume message and returns the path.
    * Otherwise returns ``None``.
"""
from __future__ import annotations

import logging
import os
from transformers.trainer_utils import get_last_checkpoint
from pathlib import Path
from typing import Optional

__all__ = ["resume_or_raise"]

def resume_or_raise(training_args, logger: logging.Logger) -> Optional[str]:
    """Detect last checkpoint and decide whether to resume or raise.

    Parameters
    ----------
    training_args : transformers.Seq2SeqTrainingArguments
        The parsed training args.
    logger : logging.Logger
        Logger for info / error messages.

    Returns
    -------
    Optional[str]
        Path to the last checkpoint if resume is required, otherwise ``None``.
    """
    output_dir = Path(training_args.output_dir)
    last_checkpoint = None

    if (
        output_dir.is_dir()
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(str(output_dir))
        if last_checkpoint is None and any(output_dir.iterdir()):
            raise ValueError(
                f"Output directory ({output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to train from scratch or choose a new folder."
            )
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                "Checkpoint detected, resuming training at %s. "
                "To start fresh, change --output_dir or add --overwrite_output_dir.",
                last_checkpoint,
            )
    return last_checkpoint


"""
Created by Hao at 2025-07-01

Utility helpers for unified logging & argument inspection
=========================================================

* configure_root_logger   – one-call root logger setup for HF training scripts
* log_parser_info         – list all CLI options (no defaults)
* log_parsed_values       – dump effective values after parsing
* maybe_resume_from_checkpoint – tiny helper that mimics HF resume logic
"""

from __future__ import annotations
import logging
import os
import sys
from argparse import ArgumentParser, Namespace
from dataclasses import fields, is_dataclass
from typing import Sequence

import datasets
import transformers
from transformers.trainer_utils import is_main_process, get_last_checkpoint


# ---------------------------------------------------------------------
# 1.  Root-logger configuration
# ---------------------------------------------------------------------
def configure_root_logger(training_args, *, logger_name: str = "root") -> logging.Logger:
    """
    Configure and return the root logger.

    * Logs stream to **stdout**
    * Format: ``2025-07-01 10:00:00 - INFO - module - message``
    * Level  : `training_args.get_process_log_level()`
    * On main rank (local_rank ∈ {-1, 0}):
        - sets verbosity for *datasets* / *transformers*
        - enables transformers default handler & explicit format
    * Non-main ranks: demote level to WARNING (to reduce clutter)
    """
    # -----------------------------------------------------------------
    # basicConfig – will be ignored if logging already configured
    # -----------------------------------------------------------------
    logger = logging.getLogger(logger_name)

    # -----------------------------------------------------------------
    # Decide effective level per process
    # -----------------------------------------------------------------
    log_level = training_args.get_process_log_level()
    # override for non-main ranks
    if not is_main_process(training_args.local_rank):
        log_level = logging.WARN

    logger.setLevel(log_level)

    # -----------------------------------------------------------------
    # Harmonise HF sub-library verbosity (main process only)
    # -----------------------------------------------------------------
    if is_main_process(training_args.local_rank):
        datasets.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    else:
        # non-main ranks: silence HF libs further
        datasets.utils.logging.set_verbosity(logging.ERROR)
        transformers.utils.logging.set_verbosity(logging.ERROR)

    return logger


# ---------------------------------------------------------------------
# 2.  Argument-parser inspection helpers
# ---------------------------------------------------------------------
def log_parser_info(parser: ArgumentParser, *, level: int = logging.INFO) -> None:
    """List all registered CLI options (names only)."""
    log = logging.getLogger(__name__)
    for act in parser._actions:               # type: ignore[attr-defined]
        if act.option_strings:                # skip positionals
            log.log(level, "%s (%s)",
                    ", ".join(act.option_strings), act.dest)


def log_parsed_values(
    objs: Sequence[object | Namespace],
    *,
    level: int = logging.INFO,
    skip_defaults: bool = True,
) -> None:
    """Dump the *effective* values after parsing (optionally skipping defaults)."""
    log = logging.getLogger(__name__)
    log.log(level, "===== Effective argument values =====")

    for obj in objs:
        cls_name = type(obj).__name__
        log.log(level, "%s:", cls_name)

        if is_dataclass(obj):
            for f in fields(obj):
                val = getattr(obj, f.name)
                if skip_defaults and val == f.default:
                    continue
                log.log(level, "  %s = %s", f.name, val)
        else:  # argparse.Namespace
            for k, v in vars(obj).items():
                log.log(level, "  %s = %s", k, v)
        log.log(level, "")  # spacer


# ---------------------------------------------------------------------
# 3.  Checkpoint helper
# ---------------------------------------------------------------------
def maybe_resume_from_checkpoint(training_args, logger: logging.Logger):
    """Return last checkpoint path or None; raise if output dir not empty."""
    last_ckpt = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_ckpt = get_last_checkpoint(training_args.output_dir)
        if last_ckpt is None and os.listdir(training_args.output_dir):
            raise ValueError(
                f"Output dir ({training_args.output_dir}) exists and is not empty. "
                "Add --overwrite_output_dir or choose a new path."
            )
        if last_ckpt and training_args.resume_from_checkpoint is None:
            logger.info("Checkpoint detected, resuming at %s", last_ckpt)
    return last_ckpt


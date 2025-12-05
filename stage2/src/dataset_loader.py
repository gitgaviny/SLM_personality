"""
Created by Hao at 2025-07-01

src.dataset_loader
=====================
Centralised helpers for loading a HF `DatasetDict` from disk and validating
mandatory column names.

Usage
-----
```python
from utils.data_loading import load_dataset_or_fail
raw_datasets = load_dataset_or_fail(data_args, logger)
```
This keeps the main training script tidy while ensuring consistent checks.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datasets import DatasetDict, load_from_disk

__all__ = ["load_dataset_or_fail"]


# ---------------------------------------------------------------------
# public helper
# ---------------------------------------------------------------------

def load_dataset_or_fail(data_args, logger: logging.Logger) -> DatasetDict:  # type: ignore[name-defined]
    """Load a DatasetDict from `data_args.dataset_name` and verify columns.

    Parameters
    ----------
    data_args : DataArgs
        Contains *dataset_name*, *audio_column_name*, *text_column_name*.
    logger : logging.Logger
        Logger for info / error messages.

    Returns
    -------
    DatasetDict
        With an added "eval" split that aliases "validation".

    Raises
    ------
    ValueError
        If required columns are missing.
    """

    logger.info("Loading dataset from %s", data_args.dataset_name)
    raw_datasets: DatasetDict = load_from_disk(data_args.dataset_name)

    # Alias validation -> eval
    if "validation" in raw_datasets and "eval" not in raw_datasets:
        raw_datasets["eval"] = raw_datasets["validation"]

    first_split = next(iter(raw_datasets.values()))
    cols = first_split.column_names

    if data_args.audio_column_name not in cols:
        raise ValueError(
            f"--audio_column_name '{data_args.audio_column_name}' not found in dataset '{data_args.dataset_name}'. "
            f"Available columns: {', '.join(cols)}."
        )

    if data_args.text_column_name not in cols:
        raise ValueError(
            f"--text_column_name '{data_args.text_column_name}' not found in dataset '{data_args.dataset_name}'. "
            f"Available columns: {', '.join(cols)}."
        )

    logger.info("Dataset loaded with splits: %s", list(raw_datasets.keys()))
    return raw_datasets


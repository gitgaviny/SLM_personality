"""
Created by Hao at 2025-07-01

src.feature_extractor_loader
=================================
Helper to load a HuggingFace ``AutoFeatureExtractor`` based on fields in
``model_args``.

Typical use
-----------
```python
from utils.feature_extractor_loader import load_feature_extractor
feature_extractor = load_feature_extractor(model_args, logger)
```
This mirrors the logic you had inline and centralises it for reuse.
"""
from __future__ import annotations

import logging
from transformers import AutoFeatureExtractor

logger = logging.getLogger(__name__)

def load_feature_extractor(model_args, logger: logging.Logger | None = None):
    """Load a feature extractor using fallback logic on *model_args* fields.

    Parameters
    ----------
    model_args : Namespace | dataclass
        Must expose attributes:
        * ``feature_extractor_name``
        * ``model_name_or_path``
        * ``cache_dir``
        * ``model_revision``
        * ``token``
        * ``trust_remote_code``
    logger : logging.Logger, optional
        Logger used for info messages; defaults to module logger.
    """
    lg = logger or logging.getLogger(__name__)

    src = (
        model_args.feature_extractor_name
        if getattr(model_args, "feature_extractor_name", None)
        else model_args.model_name_or_path
    )

    lg.info("Loading feature extractor from %s", src)

    fe = AutoFeatureExtractor.from_pretrained(
        src,
        cache_dir=getattr(model_args, "cache_dir", None),
        revision=getattr(model_args, "model_revision", None),
        token=getattr(model_args, "token", None),
        trust_remote_code=getattr(model_args, "trust_remote_code", False),
    )

    lg.info("Feature extractor loaded: %s", fe.__class__.__name__)
    return fe


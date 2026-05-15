"""
Created by Hao at 2025-07-01

src.tokenizer_loader
=========================
Helper to load a HuggingFace ``AutoTokenizer`` based on the commonly used
fields in ``model_args``.

Typical usage
-------------
```python
from utils.tokenizer_loader import load_tokenizer

# returns a tokenizer instance ready for use
tokenizer = load_tokenizer(model_args, logger)
```
This mirrors the logic previously written inline and keeps the main script
concise.
"""
from __future__ import annotations

import logging
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def _select_src(model_args):
    """Return the name/path to feed into ``from_pretrained``.

    Fall‑back order:
    1. ``model_args.tokenizer_name`` if provided
    2. ``model_args.model_name_or_path`` otherwise
    """
    return (
        getattr(model_args, "tokenizer_name", None)
        or getattr(model_args, "model_name_or_path")
    )


def load_tokenizer(model_args, logger: logging.Logger | None = None):
    """Load a tokenizer using the standard HuggingFace fallback logic.

    Parameters
    ----------
    model_args : Namespace | dataclass
        Expected attributes:
        * ``tokenizer_name`` (str | None)
        * ``model_name_or_path`` (str)
        * ``cache_dir`` (str | None)
        * ``model_revision`` (str | None)
        * ``token`` (str | None)
        * ``trust_remote_code`` (bool)
        * ``use_fast_tokenizer`` (bool)
    logger : logging.Logger, optional
        If provided, use it for info messages; otherwise module logger.

    Returns
    -------
    transformers.PreTrainedTokenizer | transformers.PreTrainedTokenizerFast
    """
    lg = logger or logging.getLogger(__name__)

    src = _select_src(model_args)
    lg.info("Loading tokenizer from %s", src)

    tok = AutoTokenizer.from_pretrained(
        src,
        cache_dir=getattr(model_args, "cache_dir", None),
        use_fast=getattr(model_args, "use_fast_tokenizer", True),
        revision=getattr(model_args, "model_revision", None),
        token=getattr(model_args, "token", None),
        trust_remote_code=getattr(model_args, "trust_remote_code", False),
    )

    lg.info("Tokenizer loaded: %s (vocab=%d)", tok.__class__.__name__, len(tok))
    return tok


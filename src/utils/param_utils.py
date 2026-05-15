# Author: Hao at 2025-07-01
# param_utils.py
# -----------------------------------------------------------
# Utility functions for inspecting model parameters.
#
# Provides:
#   checking_trainable_params(model): Logs and returns basic
#       statistics about trainable parameters in a PyTorch model.
#
# -----------------------------------------------------------

import logging
from typing import Dict

logger = logging.getLogger(__name__)

def checking_trainable_params(model) -> Dict[str, float]:
    """Log and return the number of trainable and total parameters.

    Args:
        model: A PyTorch model instance exposing a `.parameters()` iterator.

    """
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    # Avoid division by zero
    percentage = 100.0 * num_trainable_params / total_params if total_params else 0.0

    logger.info(f"Trainable parameters: {num_trainable_params}")
    logger.info(f"Total parameters: {total_params}")
    logger.info(f"Percentage of trainable parameters: {percentage:.2f}%")


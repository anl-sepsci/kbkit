"""Raise errors when KBI convergence fails."""

import warnings
from typing import Literal


class KBIConvergenceError(Exception):
    """Raised when KBI fails to converge according to one or more metrics."""


class LinearityError(KBIConvergenceError):
    """Raised when R² linearity metric is not satisfied."""


def handle_error(error_mode: Literal["raise", "warn", "ignore"], message: str, error_type: type[Exception]) -> None:
    """Implementing errors based on level."""
    mode = error_mode.lower()
    if mode not in ("raise", "warn", "ignore"):
        raise ValueError(f"Error mode {error_mode} is not a valid option: ('raise', 'warn', 'ignore').")
    if mode == "raise":
        raise error_type(message)
    elif mode == "warn":
        warnings.warn(message, RuntimeWarning, stacklevel=2)

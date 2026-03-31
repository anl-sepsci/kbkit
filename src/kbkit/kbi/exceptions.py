"""Raise errors when KBI convergence fails."""

class KBIConvergenceError(Exception):
    """Raised when KBI fails to converge according to one or more metrics."""

class LinearityError(KBIConvergenceError):
    """Raised when R² linearity metric is not satisfied."""

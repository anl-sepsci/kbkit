"""Containers for storing activity coefficient properties and their polynomial fit functions."""

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np


@dataclass
class ActivityCoefficientResult:
    r"""
    Containor for activity coefficient results and derivatives.

    Stores polynomial functions if ``activity_integration_type`` is `polynomial` and evaluates function from :math:`x_i`: 0 :math:`\rightarrow` 1.

    Parameters
    ----------
    mol: str
        Molecule name.
    x: np.ndarray
        Array of mole fractions for ``mol``.
    y: np.ndarray
        Array of values corresponding to ``x``.
    property_type: str
        Type of activity coefficient property. Tags result object with `derivative` or `integrated`.
    fn: Callable, optional
        Optional function that describes activity coefficients. Only if ``activity_integration_typ`` is `polynomial`.
    """

    mol: str  # instead of "name"
    x: np.ndarray
    y: np.ndarray
    property_type: Literal["derivative", "integrated"]  # no changes needed
    fn: Callable | None = None  # instead of "fn"

    @property
    def x_eval(self) -> np.ndarray:
        """np.ndarray: Values to evaluate function at."""
        if not self.fn:
            return None
        return np.arange(0, 1.01, 1)

    @property
    def y_eval(self) -> np.ndarray:
        """np.ndarray: Result of the function evaluated at ``x_eval``."""
        if not self.fn:
            return None
        return self.fn(self.x)

    @property
    def has_fn(self) -> np.ndarray:
        """bool: Check if a fit function is defined."""
        return bool(self.fn)


@dataclass
class ActivityMetadata:
    """
    Container for collection of ActivityCoefficientResult objects.

    Parameters
    ----------
    results: list[ActivityCoefficientResult]
        List of ActivityCoefficientResult objects.
    """

    results: list[ActivityCoefficientResult]

    @property
    def by_types(self) -> dict[str, dict[str, ActivityCoefficientResult]]:
        """
        Group the ActivityCoefficientResult by their type, e.g., if they are a `derivative` or `integrated` property.

        Returns
        -------
        dict
            Nested dictionary of ActivityCoefficientResult sorted by property, then by molecule name.
        """
        data = {}
        for m in self.results:
            data.setdefault(m.property_type, {})[m.mol] = m
        return data

    def get(self, mol: str, property_type: Literal["derivative", "integrated"]) -> ActivityCoefficientResult:
        """
        Get an ActivityCoefficientResult object for a given `property_type` and `mol`.

        Parameters
        ----------
        mol: str
            Molecule name.
        property_type: str
            Type of activity coefficient property.
        """
        key = "derivative" if property_type.lower().startswith("d") else "integrated"
        return self.by_types[key][mol]

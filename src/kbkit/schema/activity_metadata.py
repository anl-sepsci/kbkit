"""Containers for storing activity coefficient properties and their polynomial fit functions."""

from dataclasses import dataclass
from typing import Literal

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
        Optional function that describes activity coefficients. Only if ``activity_integration_type`` is `polynomial`.
    """

    mol: str
    x: np.ndarray
    y: np.ndarray
    property_type: Literal["derivative", "integrated"]
    fn: np.poly1d | None = None  # instead of "fn"

    @property
    def x_eval(self):
        """np.ndarray: Values to evaluate function at."""
        return np.arange(0, 1.01, 0.01) if isinstance(self.fn, np.poly1d) else None

    @property
    def y_eval(self):
        """np.ndarray: Result of the function evaluated at ``x_eval``."""
        return self.fn(self.x_eval) if isinstance(self.fn, np.poly1d) else None

    @property
    def has_fn(self) -> bool:
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
        data: dict[str, dict[str, ActivityCoefficientResult]] = {}
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
        try:
            return self.by_types[key][mol]
        except KeyError as e:
            raise KeyError(f"property type {property_type} and mol {mol} not in {self.by_types}") from e

    def __iter__(self):
        """Creates an iterable type object."""
        return iter(self.results)

    def __len__(self) -> int:
        """Allows len(ActivityMetadata) to return num systems in registry."""
        return len(self.results)

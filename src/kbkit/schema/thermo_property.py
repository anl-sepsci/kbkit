"""Structured representation of scalar properties with units and semantic tags."""

from dataclasses import dataclass, field
from functools import cached_property
from typing import Any


@dataclass
class ThermoProperty:
    """
    Container for a scalar property with units and semantic annotations.

    Designed to store a value alongside its physical units and optional tags
    for classification, filtering, or metadata enrichment.

    Attributes
    ----------
    name: str
        Name of the computed property.
    value : Any
        The raw property value (e.g., float, int, or derived object).
    units : str
        Units associated with the value (e.g., "kJ/mol", "nm", "mol/L").
    """

    name: str
    value: Any
    units: str = field(default_factory=str)


def register_property(name: str, units: str):
    """
    Method decorator for associating metadata and units with a ThermoProperty.

    Parameters
    ----------
    name : str
        Property name.
    units : str
        Property units.

    Returns
    -------
    Callable
        The resulting decorator produces a cached property containing a ThermoProperty instance.
    """

    def decorator(func):
        """
        Internal wrapper for producing a cached ThermoProperty from the decorated method.

        Parameters
        ----------
        func : Callable
            Method to be decorated.

        Returns
        -------
        cached_property
            Cached property containing a ThermoProperty instance.
        """

        @cached_property
        def wrapper(self):
            """
            Cached property containing a ThermoProperty instance with name and units metadata.

            Returns
            -------
            ThermoProperty
                Instance containing the computed value, name, and units.
            """
            return ThermoProperty(name=name, value=func(self), units=units)

        return wrapper

    return decorator

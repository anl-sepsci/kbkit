"""Structured representation of scalar properties with units and semantic tags."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from kbkit.config.unit_registry import load_unit_registry


@dataclass
class PropertyResult:
    """
    Container for calculated thermodynamic properties with metadata.

    Parameters
    ----------
    value: np.ndarray
        Calculated property values.
    property_type: str
        String for type of property calculated.
    units: str
        Units of the property.
    metadata: dict, optional
        Additional calculation metadata, i.e., mixing rules.
    """

    name: str
    value: np.ndarray
    property_type: Literal["ideal", "excess", "pure", "simulated"] | None = None
    units: str | None = None
    metadata: dict | None = None

    def to(self, units: str | None = None):
        """Convert property to desired units."""
        units = units or self.units
        ureg = load_unit_registry()

        # Convert the magnitude
        new_value = ureg.Quantity(self.value, self.units).to(units).magnitude

        # Return a copy with new values and units
        return PropertyResult(
            name=self.name, value=new_value, property_type=self.property_type, units=units, metadata=self.metadata
        )

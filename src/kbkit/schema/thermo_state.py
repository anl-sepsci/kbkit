"""Structured representation of thermodynamic and state properties with units and semantic tags."""

from dataclasses import dataclass, field
from typing import Dict, Any

from kbkit.schema.thermo_property import ThermoProperty


@dataclass
class ThermoState:
    """
    Flexible container for thermodynamic and state properties.
    Properties stored in a dictionary keyed by name.
    """

    properties: Dict[str, ThermoProperty] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the ThermoState dataclass to a dictionary."""
        return {name: prop.value for name, prop in self.properties.items()}

    def get(self, property_name: str) -> ThermoProperty:
        """Retrieve a specific ThermoProperty by name."""
        try:
            return self.properties[property_name]
        except KeyError:
            raise AttributeError(f"ThermoState has no property '{property_name}'")
        
    @classmethod
    def from_sources(cls, *sources: Any) -> "ThermoState":
        """
        Build a ThermoState from one or more source objects.
        Each source can be another dataclass, dict, or object with attributes.
        """
        props = {}
        for src in sources:
            if isinstance(src, dict):
                for k, v in src.items():
                    if isinstance(v, ThermoProperty):
                        props[k] = v 
            else:
                for attr in dir(src):
                    val = getattr(src, attr)
                    if isinstance(val, ThermoProperty):
                        props[attr] = val 
        return cls(properties=props)
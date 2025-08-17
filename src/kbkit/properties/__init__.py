"""SubModule for managing GROMACS properties."""

from kbkit.properties.energy_reader import EnergyReader
from kbkit.properties.topology import TopologyParser

__all__ = ["EnergyReader", "TopologyParser"]

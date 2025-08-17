"""Provide a unified interface to compute topology, electron counts, and GROMACS properties."""

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from kbkit.data import energy_aliases, get_gmx_unit, resolve_attr_key
from kbkit.properties import EnergyReader
from kbkit.properties import TopologyParser
from kbkit.config import load_unit_registry
from kbkit.utils import to_float


class SystemProperties:
    """
    Unified interface for access properties from GROMACS simulations.

    Accesses system properties (both energy and topology) for a GROMACS molecular dynamics simulation.
    Includes specific property calculations from functions in parent classes.

    Parameters
    ----------
    syspath: str
        Absolute path to system.
    ensemble: str, optional
        Ensemble for simulations. Default is '`npt`'.

    See Also
    --------
    :class:`kbkit.properties.topology.TopologyParser`: Topology parent class for molecule names, molecule numbers, and electron numbers.
    :class:`kbkit.properties.energy_reader.EnergyReader`: GROMACS properties to calculate with gmx energy.

    """

    def __init__(self, syspath: str, ensemble: str = "npt") -> None:
        self.topology = TopologyParser(syspath, ensemble)
        self.energy = EnergyReader(syspath, ensemble)  # system path that contains .top and .edr files
        self.ureg = load_unit_registry()  # Load the unit registry for unit conversions

    def __getattr__(self, name: str) -> Callable[..., Any]:
        """
        Dynamically resolves and computes system properties as attributes.

        Parameters
        ----------
        name : str
            Name of the property to resolve.

        Returns
        -------
        function
            A function that computes the requested property with optional arguments for units, time range,
            and output format.
        """
        # get energy attribute
        prop = resolve_attr_key(name.lower(), energy_aliases)
        gmx_units = get_gmx_unit(prop)

        def prop_getter(
            time_units: str = "ns",
            units: str = "",
            start_time: float = 0,
            return_std: bool = False,
            timeseries: bool = False,
        ) -> float | tuple[NDArray[np.float64], NDArray[np.float64]] | tuple[float, float]:
            # get property units
            units = units if units else gmx_units

            # first search for heat capacity (unique case)
            if prop == "heat_capacity":
                return self.energy.heat_capacity(nmol=self.topology.total_molecules, units=units)

            # then if volume and ensemble is nvt
            elif prop == "volume" and self.energy.ensemble == "nvt":
                return self.topology.box_volume(units=units)

            # if timeseries is desired, return arrays instead of floats
            elif timeseries:
                time_qty, val_qty = self.energy.stitch_property_timeseries(
                    prop, start_time=start_time, time_units=time_units, units=units
                )
                return time_qty.magnitude, val_qty.magnitude

            # defaults to the averaged property
            else:
                return self.energy.average_property(prop, start_time=start_time, units=units, return_std=return_std)

        # special case for enthalpy
        if prop == "enthalpy":

            def enthalpy_getter(start_time: float = 0, units: str = "kJ/mol") -> float:
                # get potential energy
                U = self.energy.average_property("potential", start_time=start_time, units="kJ/mol", return_std=False)
                # get pressure
                P = self.energy.average_property("pressure", start_time=start_time, units="kPa", return_std=False)
                # get volume, from gmx energy (npt) or .gro file (nvt)
                if self.energy.ensemble == "npt":
                    V = self.energy.average_property("volume", start_time=start_time, units="m^3", return_std=False)
                elif self.energy.ensemble == "nvt":
                    # For NVT, volume is not directly computed, so we use the box volume from the topology
                    V = self.topology.box_volume(units="m^3")

                U, P, V = to_float(U), to_float(P), to_float(V)
                H = U + P * V
                H /= self.topology.total_molecules  # Convert to per molecule

                # convert units
                try:
                    H = self.ureg.Quantity(H, "kJ/mol").to(units).magnitude
                except Exception as e:
                    raise RuntimeError(f"Failed to convert enthalpy units: {e}") from e

                return float(H)

            return enthalpy_getter

        # Return the property getter function so it can accept optional units argument
        else:
            return prop_getter

    def get(self, name: str, **kwargs: Any) -> Any:
        """
        Get average property from gmx energy with automatic reading of topology information.

        Parameters
        ----------
        name: str
            Property to retrieve from gmx energy.

        Returns
        -------
        float or list[float, float]
            Scalar of average property if `return_std` option is False, else list of (average, standard deviation).
        """
        return getattr(self, name)(**kwargs)

    def plot(self, property_name: str, **kwargs: Any) -> None:
        """Plot gmx energy timeseries property.

        Parameters
        ----------
        property_name: str
            Property to plot from gmx energy

        See Also
        --------
        :meth:`kbkit.properties.energy_reader.EnergyReader.plot_property`: More details on the gmx energy plotting function.
        """
        self.energy.plot_property(property_name, **kwargs)

"""Parser for GROMACS (.edr) and LAMMPS (.log, .lammps) energy files."""

from enum import Enum, auto
from functools import cached_property
from pathlib import Path
from typing import ClassVar

import MDAnalysis as mda
import numpy as np
import pandas as pd
from lammps_logfile import read_log

from kbkit.config.unit_registry import load_unit_registry
from kbkit.utils.format import ENERGY_ALIASES, resolve_attr_key
from kbkit.utils.validation import validate_path


class EnergyFormat(Enum):
    """Formatting for various energy types."""

    LAMMPS = auto()
    GROMACS = auto()


class EnergyParser:
    """
    Interface for extracting energy properties from GROMACS .edr and LAMMPS .log/.lammps files.

    Wraps MDAnalysis to provide access to available properties in .edr file.
    Extracts data from .lammps via lammps_logfile and .log via pandas.
    Supports additional properties, `configurational_enthalpy` and `fluctuation properties` (heat capacity and isothermal compressibility).
    Note that the fluctuation properties return a float object rather than a timeseries.
    """

    LAMMPS_to_GMX: ClassVar = {
        "Step": "step",
        "Temp": "temperature",
        "Press": "pressure",
        "Density": "density",
        "Volume": "volume",
        "PotEng": "potential",
        "KinEng": "kinetic en.",
        "TotEng": "total energy",
    }

    GROMACS_UNITS: ClassVar = {
        "step": "",
        "time": "ps",
        "temperature": "K",
        "pressure": "bar",
        "density": "kg/m^3",
        "volume": "nm^3",
        "potential": "kJ/mol",
        "kinetic en.": "kJ/mol",
        "total energy": "kJ/mol",
        "enthalpy": "kJ/mol",
    }

    LAMMPS_UNITS: ClassVar = {
        "step": "",
        "time": "ps",
        "temperature": "K",
        "pressure": "atm",
        "density": "g/cm^3",
        "volume": "angstrom^3",
        "potential": "kcal/mol",
        "total energy": "kcal/mol",
        "enthalpy": "kcal/mol",
    }

    DEFAULT_UNITS: ClassVar = {
        "step": "",
        "time": "ps",
        "temperature": "K",
        "pressure": "kPa",
        "density": "kg/m^3",
        "volume": "nm^3",
        "potential": "kJ/mol",
        "kinetic en.": "kJ/mol",
        "total energy": "kJ/mol",
        "enthalpy": "kJ/mol",
        "cp": "kJ/mol/K",
        "cv": "kJ/mol/K",
        "isothermal-compressibility": "1/kPa",
        "number-density": "molecule/nm^3",
        "molar-volume": "cm^3/mol",
    }

    FLUCT_PROPS: ClassVar = ("cp", "cv", "isothermal-compressibility")

    def __init__(self, path: str | Path) -> None:
        # validate filepath
        self.filepath = validate_path(path, suffix=Path(path).suffix)
        # setup unit registry
        self.ureg = load_unit_registry()
        self.Q_ = self.ureg.Quantity

    @property
    def _energy_format(self) -> EnergyFormat:
        """Returns energy format for file."""
        if self.filepath.suffix.lower() in (".edr"):
            return EnergyFormat.GROMACS
        elif self.filepath.suffix.lower() in (".log", ".lammps"):
            return EnergyFormat.LAMMPS
        else:
            raise ValueError(
                f'Energy file with suffix: {self.filepath.suffix} not recognized. Acceptable forms include: ".edr, .log, .lammps".'
            )

    @property
    def _md_units(self) -> dict[str, str]:
        """Returns dictionary of unit mapping for MD engine."""
        unitmap = {EnergyFormat.GROMACS: self.GROMACS_UNITS, EnergyFormat.LAMMPS: self.LAMMPS_UNITS}
        return unitmap[self._energy_format]

    @cached_property
    def units(self) -> dict[str, str]:
        """Returns a dictionary mapping properties to their units."""
        return self.DEFAULT_UNITS

    @cached_property
    def data(self) -> dict[str, np.ndarray]:
        """Extract energy data."""
        if self.filepath.suffix in (".log", ".lammps"):
            # get data in pd.DataFrame
            if self.filepath.suffix == ".log":
                d = pd.read_csv(self.filepath, sep=r"\s+", skiprows=2).to_dict(orient="list")
            else:
                d = read_log(self.filepath)
            # convert each object to dictionary of arrays
            d_arr = {}
            for k, v in d.items():
                if k in self.LAMMPS_to_GMX:
                    d_arr[self.LAMMPS_to_GMX[k]] = np.array(v)
            return d_arr

        elif self.filepath.suffix == ".edr":
            aux = mda.auxiliary.EDR.EDRReader(self.filepath)
            d = aux.data_dict
            return {k.lower(): v for k, v in d.items()}

        else:
            raise ValueError(
                f'Energy file with suffix: "{self.filepath.suffix}", is not supported. Supported filetypes: ".edr, .log, .lammps".'
            )

    def available_properties(self) -> list[str]:
        """
        Return a list of available energy properties in the .edr file(s).

        Returns
        -------
        list[str]
            Sorted list of property names in .edr files.
        """
        return list(dict.fromkeys(self.data))

    @property
    def _x_key(self) -> str:
        """Return the data key for x-variable depending on format type."""
        keymap = {EnergyFormat.GROMACS: "time", EnergyFormat.LAMMPS: "step"}
        return keymap[self._energy_format]

    def get(self, name: str, start: int = 0, units: str | None = None) -> np.ndarray:
        r"""
        Extract time series data for a given property.

        Parameters
        ----------
        name : str
            Property name to extract (e.g., "potential", "temperature").
        start : int, optional
            Starting point for when data should be used. GROMACS: time (ps), LAMMPS: timestep (i.e., for 1 fs steps, start=1000--start at 1 ps).
        units: str, optional
            Returns property in desired units. If empty, used default values (See :meth:`units`).

        Returns
        -------
        np.ndarray
            Array of values.

        .. note::
            Filters data based on start_time for reproducibility.
        """
        # first check that property is in edr file
        try:
            # resolves common property names for select gmx properties
            prop_key = resolve_attr_key(name, ENERGY_ALIASES).lower()
            all_values = self.data[prop_key]
        except KeyError:
            try:
                prop_key = name.lower()
                all_values = self.data[prop_key]
            except KeyError as e:
                raise KeyError(f"Property {prop_key} is not available.") from e

        # get default units from GROMACS
        md_units = self._md_units.get(prop_key)
        units = units or self.units.get(prop_key)

        # get values from EDR parser
        values = all_values[self.data[self._x_key] > start]
        # convert to desired units
        return self.Q_(values, md_units).to(units).magnitude

    def molar_volume(
        self, nmol: int, volume: float = 0, start: int = 0, units: str | None = None
    ) -> np.ndarray | float:
        r"""
        Calculate molar volume of a simulation.

        If ensemble is NVT, i.e., `volume` is not accessible in .edr file, an input volume is required (i.e., read from bottom of .gro file in :class:`~kbkit.systems.properties.SystemProperties`).

        Parameters
        ----------
        nmol: int
            Number of total molecules in system.
        volume: float, optional
            Simulation box volume.
        start : int, optional
            Starting point for when data should be used. GROMACS: time (ps), LAMMPS: timestep (i.e., for 1 fs steps, start=1000--start at 1 ps).
        units: str, optional
            Desired output units. Defaults to ``pyedr`` units (kJ/mol).

        Returns
        -------
        np.ndarray
            Molar volume of molecular simulation.
        """
        units = units or self.units.get("molar-volume")

        try:
            V = self.get("volume", start=start, units="nm^3")
        except KeyError:
            print(f"Warning! 'Volume' not found in '{self.filepath}'. Falling back on box volume.")
            V = np.asarray(volume)

        molar_vol = V / nmol
        return self.Q_(molar_vol, "nm^3/molecule").to(units).magnitude

    def configurational_enthalpy(
        self, volume: float | None = None, start: int = 0, units: str | None = None
    ) -> np.ndarray:
        r"""
        Calculate configurational enthalpy from potential energy. Not normalized to molecule number in simulation.

        If ensemble is NVT, i.e., `volume` is not accessible in .edr file, an input volume is required (i.e., read from bottom of .gro file in :class:`~kbkit.systems.properties.SystemProperties`).

        Parameters
        ----------
        volume: float, optional
            Simulation box volume (units: nm:math:`^3`).
        start : int, optional
            Starting point for when data should be used. GROMACS: time (ps), LAMMPS: timestep (i.e., for 1 fs steps, start=1000--start at 1 ps).
        units: str, optional
            Desired output units. (default: kJ/mol).

        Returns
        -------
        np.ndarray
            Enthalpy of molecular simulation.

        Notes
        -----
        Enthalpy, :math:`H`, is calculated from potential energy (:math:`U`) according to:

        .. math::
            H = U + pV

        where:
            - :math:`p` is pressure
            - :math:`V` is volume
        """
        units = units or self.units.get("enthalpy")

        U = self.get("potential", start=start, units="kJ/mol")
        P = self.get("pressure", start=start, units="kPa")
        try:
            V = self.get("volume", start=start, units="nm^3")
        except KeyError:
            print(f"Warning! 'Volume' not found in '{self.filepath}'. Falling back on box volume.")
            if volume is None:
                raise ValueError("Volume cannot be Nonetype!") from None
            V = np.asarray(volume)
        V = self.Q_(V, "nm^3").to("m^3").magnitude

        H = U + P * V
        return self.Q_(H, "kJ/mol").to(units).magnitude

    def molar_enthalpy(self, nmol: int, volume: float = 0, start: int = 0, units: str | None = None) -> np.ndarray:
        r"""
        Calculate molar enthalpy. Configurational enthalpy is normalized to the total molecule number in simulation.

        Parameters
        ----------
        nmol: int
            Number of total molecules in system.
        volume: float, optional
            Simulation box volume (units: nm:math:`^3`).
        start : int, optional
            Starting point for when data should be used. GROMACS: time (ps), LAMMPS: timestep (i.e., for 1 fs steps, start=1000--start at 1 ps).
        units: str, optional
            Desired output units. (default: kJ/mol).

        Returns
        -------
        np.ndarray
            Configurational enthalpy normalized by the total number of molecules.

        See Also
        --------
        :meth:`configurational_enthalpy`
        """
        # get desired units
        units = units or self.units.get("enthalpy")

        H = self.configurational_enthalpy(volume=volume, start=start, units=units)
        return H / nmol

    def heat_capacity_cp(
        self, nmol: int, volume: float | None = None, start: int = 0, units: str | None = None
    ) -> float:
        r"""
        Calculate molar constant pressure heat capacity from :meth:`configurational_enthalpy`. Heat capacity is normalized to the total number of molecules in simulation.

        Parameters
        ----------
        nmol: int
            Number of total molecules in system.
        volume: float, optional
            Simulation box volume (units: nm:math:`^3`).
        start : int, optional
            Starting point for when data should be used. GROMACS: time (ps), LAMMPS: timestep (i.e., for 1 fs steps, start=1000--start at 1 ps).
        units: str, optional
            Desired output units. (default: kJ/mol).

        Returns
        -------
        float
            Constant pressure heat capacity.

        Notes
        -----
        Constant pressure heat capacity, :math:`c_p` is calculated according to:

        .. math::
            \begin{aligned}
            c_p &= \frac{\langle H^2 \rangle - \langle H \rangle ^2}{k_B T^2} \\
                &= \frac{\sigma_H^2}{k_B T^2}
            \end{aligned}

        where:
            - :math:`\langle H^2 \rangle - \langle H \rangle ^2` is the variance of the enthalpy (also writted as :math:`\sigma_H^2`)
            - :math:`k_B` is Boltzmann constant
            - :math:`T` is absolute temperature
        """
        # get desired units
        units = units or self.units.get("cp")

        # get enthalpy from potential energy
        H = self.configurational_enthalpy(volume=volume, start=start, units="kJ/mol")
        T = self.get("temperature", start=start)
        T_avg = T.mean()
        R = self.ureg("R").to("kJ/mol/K").magnitude

        # ddof=1 is for sample variance calculations
        cp = H.var(ddof=1) / nmol / (R * T_avg**2)
        return self.Q_(cp, "kJ/mol/K").to(units).magnitude

    def heat_capacity_cv(self, nmol: int, start: int = 0, units: str | None = None) -> float:
        r"""
        Calculate molar constant volume heat capacity. Heat capacity is normalized to the total number of molecules in simulation.

        Parameters
        ----------
        nmol: int
            Number of total molecules in system.
        start : int, optional
            Starting point for when data should be used. GROMACS: time (ps), LAMMPS: timestep (i.e., for 1 fs steps, start=1000--start at 1 ps).
        units: str, optional
            Desired output units. Defaults to ``pyedr`` units (kJ/mol).

        Returns
        -------
        float
            Constant volume heat capacity.

        Notes
        -----
        Constant volume heat capacity, :math:`c_v` is calculated according to:

        .. math::
            \begin{aligned}
            c_v &= \frac{\langle U^2 \rangle - \langle U \rangle ^2}{k_B T^2} \\
                &= \frac{\sigma_U^2}{k_B T^2}
            \end{aligned}

        where:
            - :math:`\langle U^2 \rangle - \langle U \rangle ^2` is the variance of the potential (also writted as :math:`\sigma_U^2`)
            - :math:`k_B` is Boltzmann constant
            - :math:`T` is absolute temperature
        """
        # get desired units
        units = units or self.units.get("cv")

        # get energy properties from potential energy
        U = self.get("potential", start=start, units="kJ/mol")
        T = self.get("temperature", start=start)
        T_avg = T.mean()
        R = self.ureg("R").to("kJ/mol/K").magnitude

        # ddof=1 is for sample variance calculations
        cv = U.var(ddof=1) / nmol / (R * T_avg**2)
        return self.Q_(cv, "kJ/mol/K").to(units).magnitude

    def isothermal_compressibility(self, start: int = 0, units: str | None = None) -> float:
        r"""
        Isothermal compressibility.

        Parameters
        ----------
        start : int, optional
            Starting point for when data should be used. GROMACS: time (ps), LAMMPS: timestep (i.e., for 1 fs steps, start=1000--start at 1 ps).
        units: str, optional
            Desired output units. Defaults to ``pyedr`` units (kJ/mol).

        Returns
        -------
        float
            Isothermal compressibility.
        """
        try:
            V = self.get("volume", start=start, units="m^3")
        except KeyError as e:
            raise KeyError("Isothermal Compressibility cannot be calculated from constant volume simulation!") from e

        units = units or self.units.get("isothermal-compressibility")

        R = self.ureg("R").to("kJ/mol/K").magnitude
        N_A = self.ureg("N_A").to("1/mol").magnitude
        T = self.get("Temperature", start=start)
        kT = N_A * V.var(ddof=1) / (V.mean() * R * T.mean())
        return self.Q_(kT, "1/kPa").to(units).magnitude

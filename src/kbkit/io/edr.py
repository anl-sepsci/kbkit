"""Parser for GROMACS energy (.edr) files."""

import re
import subprocess
import tempfile
from functools import cached_property
from pathlib import Path
from typing import ClassVar

import numpy as np

from kbkit.config.unit_registry import load_unit_registry
from kbkit.utils.format import ENERGY_ALIASES, resolve_attr_key
from kbkit.utils.validation import validate_path


class EdrParser:
    """
    Interface for extracting energy properties from GROMACS .edr files.

    Wraps `gmx energy` to provide access to available properties in .edr file.
    Supports additional properties, `configurational_enthalpy` and `fluctuation properties` (heat capacity and isothermal compressibility).
    Note that the fluctuation properties return a float object rather than a timeseries.

    Parameters
    ----------
    edr_path : str or list[str]
        Path to an .edr file.
    """

    # Standard GROMACS Unit Mapping
    # Energies: kJ/mol, Temp: K, Press: bar, Density: kg/m^3
    GROMACS_UNITS: ClassVar = {
        "time": "ps",
        "temperature": "K",
        "pressure": "bar",
        "density": "kg/m^3",
        "volume": "nm^3",
        "potential": "kJ/mol",
        "kinetic-en": "kJ/mol",
        "total-energy": "kJ/mol",
        "enthalpy": "kJ/mol",
    }

    DEFAULT_UNITS: ClassVar = {
        "time": "ps",
        "temperature": "K",
        "pressure": "kPa",
        "density": "kg/m^3",
        "volume": "nm^3",
        "potential": "kJ/mol",
        "kinetic-en": "kJ/mol",
        "total-energy": "kJ/mol",
        "enthalpy": "kJ/mol",
        "cp": "kJ/mol/K",
        "cv": "kJ/mol/K",
        "isothermal-compressibility": "1/kPa",
        "number-density": "molecule/nm^3",
        "molar-volume": "cm^3/mol"
    }

    FLUCT_PROPS: ClassVar = ("cp", "cv", "isothermal-compressibility")

    def __init__(self, path: str | Path) -> None:
        # validate edr_path
        self.edr_path = validate_path(path, suffix=".edr")
        # setup unit registry
        self.ureg = load_unit_registry()
        self.Q_ = self.ureg.Quantity

    @cached_property
    def units(self) -> dict[str, str]:
        """Returns a dictionary mapping available properties to their units."""
        return {k: v for k, v in self.DEFAULT_UNITS.items() if k in self.available_properties()}

    def get_gmx_property(self, name: str, avg: bool = False, **kwargs) -> tuple | float:
        """Extract gromacs property from energy file.

        Parameters
        ----------
        name: str
            Name of property to extract using gmx energy.
        kwargs: dict[str, str]
            Dictionary of optional inputs to gmx energy command (e.g., {"-b": 10000})

        Returns
        -------
        tuple
            Tuple of property time series (time, values).
        """
        prop_input = name.lower() + "\n\n"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.xvg', delete=False) as tmp:
            tmpfile = tmp.name

        try:
            # first build input list
            cmd = ["gmx", "energy", "-f", str(self.edr_path), "-o", tmpfile]
            for k, v in kwargs.items():
                cmd.extend([str(k), str(v)])

            subprocess.run(
                cmd,
                input=prop_input,
                text=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )

            time, values = np.loadtxt(tmpfile, comments=["@", "#"], unpack=True)

        finally:
            Path(tmpfile).unlink(missing_ok=True)

        return (time, values) if not avg else values.mean()

    @cached_property
    def data(self) -> dict[str, np.ndarray]:
        """Extract energy data from EDR file."""
        names = set(self._get_property_names())
        subset = ["time"]
        subset.extend([prop for prop in self.GROMACS_UNITS.keys() if prop in names])

        if not subset:
            return {}

        prop_input = "\n".join(subset) + "\n\n"

        with tempfile.NamedTemporaryFile(mode='w', suffix='.xvg', delete=False) as tmp:
            tmpfile = tmp.name

        try:
            subprocess.run(
                ["gmx", "energy", "-f", str(self.edr_path), "-o", tmpfile],
                input=prop_input,
                text=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )

            # Parse XVG header to get column order (excluding time)
            column_names = []
            with open(tmpfile, 'r') as f:
                for line in f:
                    if line.startswith('@ s') and 'legend' in line:
                        name = line.split('"')[1]
                        column_names.append(name)

            # Load data
            raw_data = np.loadtxt(tmpfile, comments=["@", "#"], unpack=True)

            # Build dictionary: column 0 is time, columns 1+ are the legend entries
            data = {"time": raw_data[0]}

            for i, col_name in enumerate(column_names):
                # i corresponds to legend index (s0, s1, s2...)
                # but data column is i+1 (because column 0 is time)
                data_column_index = i + 1

                # Normalize XVG name to match your property naming convention
                # "Kinetic En." -> "Kinetic-En", "Total Energy" -> "Total-Energy"
                normalized_name = col_name.replace(" ", "-").replace(".", "")

                data[normalized_name.lower()] = raw_data[data_column_index]

        finally:
            Path(tmpfile).unlink(missing_ok=True)

        return data


    def _get_property_names(self) -> list[str]:
        """Captures the full list of property names from the GMX menu, handling names with internal spaces or special characters."""
        process = subprocess.run(
            ["gmx", "energy", "-f", str(self.edr_path)],
            check=False, input="\n",
            capture_output=True,
            text=True
        )

        names = []
        # We look for lines that start with an index number.
        # Example line: "  1  Bond             2  Angle          3  Proper-Dih."
        # Note: GMX usually uses a fixed-width format for this menu.

        for line in process.stderr.splitlines():
            # Only process lines that actually look like the menu
            if not re.search(r"^\s*\d+\s+", line):
                continue

            # Split the line into chunks based on the pattern " (number) "
            # This captures the text between the numbers
            parts = re.split(r"\s*(\d+)\s+", line)

            # re.split with a capture group returns: ['', '1', 'Name1', '2', 'Name2'...]
            # We want the elements at indices 2, 4, 6...
            for i in range(2, len(parts), 2):
                prop_name = parts[i].strip()
                if prop_name and not prop_name.isdigit():
                    names.append(prop_name.lower().replace('.', ''))

        return names

    def available_properties(self) -> list[str]:
        """
        Return a list of available energy properties in the .edr file(s).

        Returns
        -------
        list[str]
            Sorted list of property names in .edr files.
        """
        return list(dict.fromkeys(self.data))

    def get(self, name: str, start_time: float = 0, units: str | None = None) -> np.ndarray:
        r"""
        Extract time series data for a given property.

        Parameters
        ----------
        name : str
            Property name to extract (e.g., "potential", "temperature").
        start_time : float, optional
            time (in ps) after which data should be included.
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
        gmx_units = self.GROMACS_UNITS.get(prop_key)
        units = units or self.units.get(prop_key)

        # get values from EDR parser
        values = all_values[self.data["time"] > start_time]
        # convert to desired units
        return self.Q_(values, gmx_units).to(units).magnitude

    def molar_volume(self, nmol: int, volume: float = 0, start_time: float = 0, units: str | None = None) -> np.ndarray | float:
        r"""
        Calculate molar volume of a simulation.

        If ensemble is NVT, i.e., `volume` is not accessible in .edr file, an input volume is required (i.e., read from bottom of .gro file in :class:`~kbkit.systems.properties.SystemProperties`).

        Parameters
        ----------
        nmol: int
            Number of total molecules in system.
        volume: float, optional
            Simulation box volume.
        start_time: float, optional
            Start time for enthalpy calculation.
        units: str, optional
            Desired output units. Defaults to ``pyedr`` units (kJ/mol).

        Returns
        -------
        np.ndarray
            Molar volume of molecular simulation.
        """
        units = units or self.units.get("molar-volume")

        try:
            V = self.get("volume", start_time=start_time, units="nm^3")
        except KeyError:
            print(f"Warning! 'Volume' not found in '{self.edr_path}'. Falling back on box volume.")
            V = np.asarray(volume)

        molar_vol = V / nmol
        return self.Q_(molar_vol, "nm^3/molecule").to(units).magnitude

    def configurational_enthalpy(self, volume: float = 0, start_time: float = 0, units: str | None = None) -> np.ndarray:
        r"""
        Calculate enthalpy from potential energy.

        If ensemble is NVT, i.e., `volume` is not accessible in .edr file, an input volume is required (i.e., read from bottom of .gro file in :class:`~kbkit.systems.properties.SystemProperties`).

        Parameters
        ----------
        volume: float, optional
            Simulation box volume.
        start_time: float, optional
            Start time for enthalpy calculation.
        units: str, optional
            Desired output units. Defaults to ``pyedr`` units (kJ/mol).

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

        U = self.get("potential", start_time=start_time, units="kJ/mol")
        P = self.get("pressure", start_time=start_time, units="kPa")
        try:
            V = self.get("volume", start_time=start_time, units="nm^3")
        except KeyError:
            print(f"Warning! 'Volume' not found in '{self.edr_path}'. Falling back on box volume.")
            V = np.asarray(volume)
        V = self.Q_(V, "nm^3").to("m^3").magnitude

        H = (U + P*V)
        return self.Q_(H, "kJ/mol").to(units).magnitude

    def molar_enthalpy(self, nmol: int, volume: float = 0, start_time: float = 0, units: str | None = None) -> np.ndarray:
        r"""
        Calculate molar enthalpy.

        Parameters
        ----------
        nmol: int
            Number of total molecules in system.
        volume: float, optional
            Simulation box volume.
        start_time: float, optional
            Start time for enthalpy calculation.
        units: str, optional
            Desired output units. Defaults to ``pyedr`` units (kJ/mol).

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

        H = self.configurational_enthalpy(volume=volume, start_time=start_time, units=units)
        return H / nmol

    def cp(self, nmol: int, volume: float = 0, start_time: float = 0, units: str | None = None) -> float:
        r"""
        Calculate constant pressure heat capacity from :meth:`configurational_enthalpy`.

        Parameters
        ----------
        nmol: int
            Number of total molecules in system.
        volume: float, optional
            Simulation box volume.
        start_time: float, optional
            Start time for enthalpy calculation.
        units: str, optional
            Desired output units. Defaults to ``pyedr`` units (kJ/mol).

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
        H = self.configurational_enthalpy(volume=volume, start_time=start_time, units="kJ/mol")
        T = self.get("temperature", start_time=start_time)
        T_avg = T.mean()
        R = self.ureg("R").to("kJ/mol/K").magnitude

        # ddof=1 is for sample variance calculations
        cp = H.var(ddof=1)/nmol/(R*T_avg**2)
        return self.Q_(cp, "kJ/mol/K").to(units).magnitude

    def cv(self, nmol: int, start_time: float = 0, units: str | None = None) -> float:
        r"""
        Calculate constant volume heat capacity.

        Parameters
        ----------
        nmol: int
            Number of total molecules in system.
        start_time: float, optional
            Start time for enthalpy calculation.
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
        U = self.get("potential", start_time=start_time, units="kJ/mol")
        T = self.get("temperature", start_time=start_time)
        T_avg = T.mean()
        R = self.ureg("R").to("kJ/mol/K").magnitude

        # ddof=1 is for sample variance calculations
        cv = U.var(ddof=1)/nmol/(R*T_avg**2)
        return self.Q_(cv, "kJ/mol/K").to(units).magnitude

    def isothermal_compressibility(self, start_time: float = 0, units: str | None = None) -> float:
        r"""
        Isothermal compressibility.

        Parameters
        ----------
        start_time: float, optional
            Start time for enthalpy calculation.
        units: str, optional
            Desired output units. Defaults to ``pyedr`` units (kJ/mol).

        Returns
        -------
        float
            Isothermal compressibility.
        """
        try:
            V = self.get("volume", start_time=start_time, units="m^3")
        except KeyError as e:
            raise KeyError("Isothermal Compressibility cannot be calculated from constant volume simulation!") from e

        units = units or self.units.get("isothermal-compressibility")

        R = self.ureg("R").to("kJ/mol/K").magnitude
        N_A = self.ureg("N_A").to("1/mol").magnitude
        T = self.get("Temperature", start_time=start_time)
        kT = N_A * V.var(ddof=1) / (V.mean() * R * T.mean())
        return self.Q_(kT, "1/kPa").to(units).magnitude


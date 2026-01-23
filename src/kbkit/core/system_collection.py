"""
Container for a set of systems for a given thermodynamic state (e.g., constant temperature, function of composition).

The purpose of `SystemCollection` is to load a set of systems and access `PropertyCalculator` and `SystemProperties` to retrieve molecular dynamics properties as a function of composition.
    * This container first discovers molecular systems based on directory structure and input parameters, creating a list of :class:`~kbkit.schema.system_metadata.SystemMetadata` objects.
    * Then topology and energy properties can be calculated as function of composition.
    * Additionally, this object initializes a :class:`~kbkit.analysis.property_calculator.PropertyCalculator` object for calculating `Excess`, `Simulation`, `Ideal` properties and `KBIs`.
"""

import os
import re
from collections import defaultdict
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np

from kbkit.analysis.property_calculator import PropertyCalculator
from kbkit.analysis.system_properties import SystemProperties
from kbkit.parsers import EdrFileParser
from kbkit.plotter.kbi_analysis import KBIAnalysisPlotter
from kbkit.plotter.timeseries import TimeseriesPlotter
from kbkit.schema.property_result import PropertyResult
from kbkit.schema.system_metadata import SystemMetadata
from kbkit.utils import validate_path
from kbkit.utils.format import ENERGY_ALIASES, resolve_attr_key


class SystemCollection:
    """
    Registry of discovered molecular systems with semantic access patterns.

    Stores and organizes SystemMetadata objects by name and kind, enabling
    reproducible filtering, indexing, and iteration across pure and mixture systems.

    Parameters
    ----------
    systems : list[SystemMetadata]
        List of discovered systems to register.
    molecules: list[str]
        List of global unique molecules present in all systems.
    """

    def __init__(self, systems: list["SystemMetadata"], molecules: list[str]) -> None:
        self._systems = systems
        self._molecules = molecules  # Global unique molecules used for sorting
        self._lookup = {s.name: s for s in systems}

    @classmethod
    def load(
        cls,
        base_path: str | None = None,
        base_systems: list[str] | None = None,
        pure_path: str | None = None,
        pure_systems: list[str] | None = None,
        rdf_dir: str = "",
        start_time: int = 10000,
        include_mode: str = "npt",
    ) -> "SystemCollection":
        """
        Construct a :class:`SystemCollection` object from discovered systems.

        Parameters
        ----------
        pure_path : str or Path
            Path to pure component directory.
        pure_systems: list[str]
            List of pure systems to include.
        base_path : str or Path
            Path to base system directory.
        base_systems : list[str], optional
            Explicit list of system names to include.
        rdf_dir: str, optional
            Explicit directory name that contains rdf files.
        start_time : int, optional
            Start time for time-averaged properties.
        include_mode: str, optional
            Optional string to filter files (.edr, .gro, .top) if multiple are found of a given type.

        Returns
        -------
        SystemCollection
            Registry object containing global molecules and list of :class:`SystemMetadata`.
        """
        valid_base_path = validate_path(base_path or os.getcwd())

        # 1. Resolve Mixture (Base) Systems
        if base_systems:
            mixture_dirs = [valid_base_path / s for s in base_systems if cls._is_valid(valid_base_path / s)]
        else:
            mixture_dirs = [f for f in valid_base_path.iterdir() if cls._is_valid(f)]

        # 2. RESOLVE MOLECULES FROM INSIDE MIXTURE FILES
        # This replaces the failing folder-name logic
        detected_molecules = set()
        for d in mixture_dirs:
            detected_molecules.update(cls._peek_molecules(d))

        # Consistent ordering (alphabetical) for the mol_fraction vector
        ordered_mols = sorted(detected_molecules)

        # 3. Resolve Pure Reference Path
        valid_pure_root = validate_path(pure_path) if pure_path else cls._find_reference_dir(valid_base_path)

        pure_dirs = []
        if pure_systems:
            for name in pure_systems:
                match = (
                    next((f for f in valid_pure_root.iterdir() if f.name == name), None) if valid_pure_root else None
                ) or next((f for f in mixture_dirs if f.name == name), None)
                if match:
                    pure_dirs.append(match)
        elif valid_pure_root and ordered_mols:
            # Use detected molecule names to find pure references
            temp = cls._extract_temp(mixture_dirs[0])
            pure_map = cls._find_pure_systems(valid_pure_root, ordered_mols, temp)
            pure_dirs = list({p for p in pure_map.values() if p is not None})

        # 4. Build Metadata (Finding RDF path before instantiation)
        meta_objects = []
        found_pure_paths = {p.resolve() for p in pure_dirs}

        # Create Pure Metadata
        for p in pure_dirs:
            r_path = cls._resolve_rdf_path(p, rdf_dir, is_pure=True)
            meta_objects.append(
                cls._make_meta(p, kind="pure", rdf_path=r_path, start_time=start_time, include=include_mode)
            )

        # Create Mixture Metadata
        for p in mixture_dirs:
            if p.resolve() not in found_pure_paths:
                r_path = cls._resolve_rdf_path(p, rdf_dir, is_pure=False)
                meta_objects.append(
                    cls._make_meta(p, kind="mixture", rdf_path=r_path, start_time=start_time, include=include_mode)
                )

        # 5. Final Sort
        sorted_meta = cls._sort_systems(meta_objects, ordered_mols)
        return cls(sorted_meta, ordered_mols)

    # --- Ultrafast File Peeking ---

    @staticmethod
    def _peek_molecules(path: Path) -> set:
        """Quickly extracts residue/molecule names from .top or .gro without full parsing."""
        mols = set()
        # Try .top first (cleanest)
        top_file = next(path.glob("*.top"), None)
        if top_file:
            with open(top_file, "r") as f:
                for line in f:
                    if "[ molecules ]" in line.lower():
                        for m_line in f:
                            p = m_line.split()
                            if p and not p[0].startswith(";"):
                                mols.add(p[0])
                        break
        # Fallback to .gro header peek
        if not mols:
            gro_file = next(path.glob("*.gro"), None)
            GRO_LIMIT = 10
            if gro_file:
                with open(gro_file, "r") as f:
                    for _ in range(100):
                        line = f.readline()
                        if len(line) < GRO_LIMIT:
                            continue
                        res = line[5:10].strip()
                        if res and not res.isdigit():
                            mols.add(res)
        return mols

    # --- Search & Scoring ---

    @staticmethod
    def _find_pure_systems(pure_base_path: Path, mixture_molecules: list[str], target_temp: float):
        """Search for pure component systems in a desired path, matching molecules present at a given temperature."""
        pure_subdirs = [p for p in pure_base_path.iterdir() if p.is_dir()]
        TEMP_THRESHOLD = 2.0
        results = {}
        for mol in mixture_molecules:
            potential_dirs = []
            for d in pure_subdirs:
                # Reference folder names usually DO contain the molecule name
                if mol.lower() in d.name.lower():
                    t = SystemCollection._extract_temp(d)
                    if t and abs(t - target_temp) <= TEMP_THRESHOLD:
                        potential_dirs.append(d)
            if potential_dirs:

                def score_dir(folder):
                    return sum(1 for m in mixture_molecules if m.lower() in folder.name.lower())

                results[mol] = max(potential_dirs, key=score_dir)
        return results

    # --- Boilerplate & Attributes ---

    @staticmethod
    def _extract_temp(input: str) -> float:
        """Extract temperature from a string or file."""
        path = Path(input)
        # first try to match temp from filename
        match = re.search(r"(\d{3}(?:\.\d+)?)", path.name)
        if match:
            return float(match.group(1))
        # then get it from edr file
        if path.is_file() and path.suffix == ".edr":
            edr = EdrFileParser(path)
            return edr.get_gmx_property("temperature", avg=True)
        # if its directory;
        elif path.is_dir():
            edr_files = SystemProperties.find_files(suffix=".edr", system_path=input)
            edr = EdrFileParser(edr_files[0])
            return edr.get_gmx_property("temperature", avg=True)
        # if all has failed raise
        else:
            raise ValueError("Temperature is not in pathname and can not be extracted from .edr file!")

    @staticmethod
    def _is_valid(path: Path, deep: bool = False) -> bool:
        """Check if systems are valid; requires it to be a directory and contains the necessary GROMACS output files."""
        pattern = "**/*" if deep else "*"
        return (
            path.is_dir()
            and any(path.glob(f"{pattern}.edr"))
            and (any(path.glob(f"{pattern}.gro")) or any(path.glob(f"{pattern}.top")))
        )

    @staticmethod
    def _find_reference_dir(start_path: Path) -> Path:
        """Search upwards from the ``start_path`` to find pure component parent directory."""
        keywords = ["pure", "single", "ref", "neat"]
        for parent in [start_path, *list(start_path.parents)]:
            for word in keywords:
                for candidate in parent.glob(f"*{word}*"):
                    if SystemCollection._is_valid(candidate, deep=True):
                        return candidate
        return None

    @staticmethod
    def _resolve_rdf_path(path: Path, rdf_dir: str, is_pure: bool) -> Path:
        """Finds the RDF directory before metadata creation."""
        # 1. Check explicit name
        if rdf_dir:
            check_path = path / rdf_dir
            if check_path.is_dir():
                return check_path

        # 2. Search for 'rdf' in subdirectories
        for subdir in path.iterdir():
            if (
                subdir.is_dir()
                and ("rdf" in subdir.name.lower())
                and (any(subdir.glob("*.xvg")) or any(subdir.glob("*.txt")))
            ):
                return subdir

        # 3. Validation
        if not is_pure:
            raise FileNotFoundError(f"No RDF directory found in mixture system: {path}")

        return Path()

    @staticmethod
    def _make_meta(path: Path, kind: str, rdf_path: Path, **props_kwargs) -> "SystemMetadata":
        """Create :class`SystemMetadata` object from inputs."""
        return SystemMetadata(
            name=path.name, kind=kind, path=path, rdf_path=rdf_path, props=SystemProperties(path, **props_kwargs)
        )

    @staticmethod
    def _sort_systems(systems: list[SystemMetadata], molecules: list[str]) -> list[SystemMetadata]:
        """Sorts systems by composition; Note: We force the topology to load here to ensure molecule_count exists."""

        def mol_fr_vector(meta: SystemMetadata):
            # 1. Access topology
            topo = meta.props.topology

            # 2. Get counts (ensure case-insensitivity if needed)
            counts = topo.molecule_count
            total = topo.total_molecules

            if total == 0:
                return tuple(0.0 for _ in molecules)

            # 3. Build vector
            return tuple(counts.get(m, 0) / total for m in molecules)

        # We MUST assign the result of sorted() back to a variable
        return sorted(systems, key=mol_fr_vector)

    # --- Properties & Magic Methods ---

    def __getattr__(self, name: str) -> Any:
        """Get attributes from system metadata or SystemProperties object."""
        if not self._systems:
            return []

        # This will now catch your new 'is_pure' if it's an attribute
        # or we can handle it if it's a method
        sample = self._systems[0]
        if hasattr(sample, name):
            attr = getattr(sample, name)
            if callable(attr):
                # If is_pure is a method, call it for all
                vals = [getattr(s, name)() for s in self._systems]
            else:
                vals = [getattr(s, name) for s in self._systems]
        elif hasattr(sample.props, name):
            vals = [getattr(s.props, name) for s in self._systems]
        else:
            vals = [s.props.get(name) for s in self._systems]

        # Convert numeric/boolean to numpy array
        first = next((v for v in vals if v is not None), None)
        if isinstance(first, (int, float, bool, np.number)):
            return np.array(vals)
        return vals

    def __getitem__(self, key):
        """Enables lookup of a specific system either by its' name or its index in the registry list."""
        return self._lookup[key] if isinstance(key, str) else self._systems[key]

    def __len__(self):
        """Allows len(SystemCollection) to return num systems in registry."""
        return len(self._systems)

    def __iter__(self):
        """Creates an iterable type object."""
        return iter(self._systems)

    @property
    def molecules(self) -> list[str]:
        """list[str]: The global order of molecules used for vectorized properties."""
        return self._molecules

    def get_mol_index(self, mol: str) -> int:
        """Get index of molecule in ``molecules``."""
        try:
            return list(self.molecules).index(mol)
        except ValueError as e:
            raise ValueError(f"Molecule '{mol}' is not in molecules! Molecules: {self.molecules}") from e

    @property
    def n_i(self) -> int:
        """int: Number of components present."""
        return len(self.molecules)

    @property
    def n_sys(self) -> int:
        """int: Number of compositions."""
        return len(self._systems)

    @cached_property
    def x(self) -> np.ndarray:
        """np.ndarray: Returns (N_systems, N_molecules) array of mole fractions, follows the order of self.molecules."""
        data = []
        for s in self._systems:
            counts = s.props.topology.molecule_count
            total = s.props.topology.total_molecules
            row = [counts.get(m, 0) / total if total > 0 else 0.0 for m in self._molecules]
            data.append(row)
        return np.array(data)

    @cached_property
    def units(self) -> dict[str, str]:
        """dict[str, str]: Master dictionary mapping energy properties to their default units."""
        unit_dic = defaultdict(str)
        for meta in self._systems:
            unit_dic.update(meta.props.get("units"))
        return dict(unit_dic)

    @property
    def pures(self) -> list["SystemMetadata"]:
        """list[SystemMetadata]: Returns a list of Metadata objects for systems where is_pure() is True."""
        return [s for s in self._systems if s.is_pure()]

    @property
    def mixtures(self) -> list["SystemMetadata"]:
        """list[SystemMetadata]: Returns a list of Metadata objects for systems where is_pure() is False."""
        return [s for s in self._systems if not s.is_pure()]

    def get_units(self, name: str) -> str:
        """Get default GROMACS units for a given property.

        Parameters
        ----------
        name: str
            Name of property to get units of.

        Returns
        -------
        str
            Units of desired property.
        """
        prop = resolve_attr_key(name, ENERGY_ALIASES)
        return self.units.get(prop, "")

    def get(
        self, name: str, units: str | None = None, avg: bool = True, time_series: bool = False
    ) -> np.ndarray | list[np.float64]:
        """
        Vectorized getter for system properties with unit support via Pint.

        Parameters
        ----------
        name : str
            The name of the property (e.g., 'Density', 'Potential').
        units : str, optional
            The target unit string for Pint conversion.
        avg : bool, default False
            If True, returns the mean value for each system.
            If False, returns the full time-series.
        time_series: bool, optional
            Returns both times and values if True (default: False).

        Returns
        -------
        np.ndarray | list
            Vectorized property of all systems in collection.
        """
        values = [s.props.get(name, units=units, avg=avg, time_series=time_series) for s in self._systems]
        try:
            return np.array(values)
        except ValueError:
            return values

    @cached_property
    def properties(self) -> "PropertyCalculator":
        """
        Create a PropertyCalculator for thermodynamic analysis.

        Returns
        -------
        PropertyCalculator
            Calculator instance for computing ideal, excess, and mixing properties.
        """
        return PropertyCalculator(self)

    def timeseries_plotter(self, system: str, start_time: int = 0) -> TimeseriesPlotter:
        """
        Create a TimeseriesPlotter for visualizing time series data for a given system.

        Parameters
        ----------
        system: str
            System to use for visualizing timeseries.
        start_time: int
            Initial time for plotting.

        Returns
        -------
        TimeseriesPlotter
            Plotter instance for computing simulation energy properties.
        """
        return TimeseriesPlotter.from_collection(self, system_name=system, start_time=start_time)

    def kbi_plotter(self, kbi: PropertyResult, molecule_map: dict[str, str] | None = None) -> KBIAnalysisPlotter:
        """
        Create a KBIAnalysisPlotter for visualizing RDF integration and KBI convergence.

        Parameters
        ----------
        kbi: PropertyResult
            KBI result object, containing KBIMetadata for plotting analysis results.
        molecule_map: dict[str, str], optional
            dictionary mapping molecule names to desired molecule labels in figures.

        Returns
        -------
        KBIAnalysisPlotter
            Plotter instance for inspecting KBI process.
        """
        return KBIAnalysisPlotter(kbi=kbi, molecule_map=molecule_map)

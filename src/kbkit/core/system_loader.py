"""Discovers molecular systems based on directory structure and input parameters."""

import logging
import os
from dataclasses import replace
from pathlib import Path

import numpy as np

from kbkit.core.system_properties import SystemProperties
from kbkit.core.system_registry import SystemRegistry
from kbkit.schema.system_config import SystemConfig
from kbkit.schema.system_metadata import SystemMetadata
from kbkit.utils.logging import get_logger
from kbkit.utils.validation import validate_path


class SystemLoader:
    """
    Discovers and organizes molecular systems for analysis.

    Uses directory structure and ensemble metadata to identify valid systems,
    extract thermodynamic and structural properties, and build a registry for
    simulation workflows.

    Parameters
    ----------
    logger : logging.Logger, optional
        Custom logger for diagnostics and traceability.
    verbose : bool, optional
        If True, enables detailed logging output.
    """

    def __init__(self, logger: logging.Logger | None = None, verbose: bool = False) -> None:
        self.verbose = verbose
        self.logger = logger or get_logger(f"{__name__}.{self.__class__.__name__}", verbose=self.verbose)

    def build_config(
        self,
        pure_path: str | Path,
        pure_systems: list[str],
        base_path: str | Path,
        base_systems: list[str] | None = None,
        ensemble: str = "npt",
        cations: list[str] | None = None,
        anions: list[str] | None = None,
        start_time: int = 0,
    ) -> SystemConfig:
        """
        Construct a :class:`SystemConfig` object from discovered systems.

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
        ensemble : str, optional
            Ensemble name used for file resolution.
        cations : list[str], optional
            List of cation species.
        anions : list[str], optional
            List of anion species.
        start_time : int, optional
            Start time for time-averaged properties.

        Returns
        -------
        SystemConfig
            Configuration object containing registry and metadata.
        """
        # get paths to parent directories
        if not base_path:
            base_path = self._find_base_path()
        else:
            base_path = validate_path(base_path)

        if not pure_path:
            pure_path = self._find_pure_path(base_path)
        else:
            pure_path = validate_path(pure_path)

        # get system paths and corresponding metatdata for base systems
        base_systems = base_systems or self._find_systems(base_path)
        self.logger.debug(f"Discovered base systems: {base_systems}")

        base_metadata = [
            self._get_metadata(base_path, system, ensemble, start_time, "mixture") for system in base_systems
        ]

        # get system paths and corresponding metadata for pure systems
        pure_systems = self._validate_pure_systems(pure_path, pure_systems, base_metadata)
        self.logger.debug(f"Discovered pure systems: {pure_systems}")

        pure_metadata = [self._get_metadata(pure_path, system, ensemble, start_time, "pure") for system in pure_systems]

        # update metadata with rdf path
        metadata = self._update_metadata_rdf(base_metadata + pure_metadata)

        # get molecules in system
        molecules = self._extract_top_molecules(metadata)

        # now sort by topology molecule order and mol fraction
        sorted_metadata = self._sort_systems(metadata, molecules)

        return SystemConfig(
            base_path=base_path,
            pure_path=pure_path,
            ensemble=ensemble,
            cations=cations or [],
            anions=anions or [],
            registry=SystemRegistry(sorted_metadata),
            logger=self.logger,
            molecules=molecules,
        )

    def _find_base_path(self) -> Path:
        """
        Return the default base path for system discovery.

        Returns
        -------
        Path
            Current working directory.
        """
        return Path(os.getcwd())

    def _find_pure_path(self, root: str | Path) -> Path:
        """
        Discover pure component directory within the root path.

        Parameters
        ----------
        root : str or Path
            Root directory to search.

        Returns
        -------
        Path
            Path to pure component directory or fallback to root.

        Notes
        -----
        - Searches for directories containing both "pure" and "comp" in their name.
        - Logs warnings if multiple matches are found.
        """
        root = validate_path(root)
        matches = []
        for path in root.rglob("*"):
            if path.is_dir():
                name = path.name.lower()
                if "pure" in name and "comp" in name:
                    matches.append(path)

        if not matches:
            self.logger.info("No pure component directories found! Assuming pure components are stored in base path.")
            print("No pure component directories found! Assuming pure components are stored in base path.")
            return root

        if len(matches) > 1:
            self.logger.warning(f"Multiple pure component paths found. Using: {matches[0]}")
            print(f"Multiple pure component path found. Using: {matches[0]}")

        return matches[0]

    def _get_metadata(self, path: Path, system: str, ensemble: str, start_time: int, kind: str) -> SystemMetadata:
        """
        Extract SystemMetadata for a given system directory.

        Parameters
        ----------
        path : Path
            Parent directory containing the system.
        system : str
            Name of the system subdirectory.
        ensemble : str
            Ensemble name for file resolution.
        start_time : int
            Start time for time-averaged properties.

        Returns
        -------
        SystemMetadata
            Metadata object with structure, topology, and thermodynamics.
        """
        system_path = validate_path(path / system)

        prop = SystemProperties(system_path=system_path, ensemble=ensemble, start_time=start_time, verbose=self.verbose)

        # force systems with only 1 molecule to be "pure"
        if len(prop.topology.molecules) == 1:
            kind = "pure"
        kind = "pure" if kind.lower() == "pure" else "mixture"

        return SystemMetadata(
            name=system,
            path=system_path,
            props=prop,
            kind=kind,
        )

    def _find_systems(self, path: str | Path, pattern: str = "*") -> list[str]:
        """
        Discover valid system directories within a parent path.

        Parameters
        ----------
        path : str or Path
            Directory to search.
        pattern : str, optional
            Glob pattern for matching subdirectories.

        Returns
        -------
        list[str]
            List of system names containing a .top file.
        """
        # validate path
        path = validate_path(path)
        # get subdirs according to pattern if .top found in any files
        return sorted([p.name for p in path.glob(pattern) if p.is_dir() and any(p.glob("*.top"))])

    def _validate_pure_systems(
        self, pure_path: Path, pure_systems: list[str], base_metadata: list[SystemMetadata]
    ) -> list[str]:
        """
        Validate pure component systems.

        Checks that temperature of pure systems is within tolerance of base systems.

        Parameters
        ----------
        pure_path: Path
            Directory containing candidate pure systems.
        pure_systems: list[str]
            System names for pure systems.

        Returns
        -------
        list[str]
            List of validated pure systems.
        """
        if len(pure_systems) == 0:
            print("WARNING: No pure component systems provided.")
            return pure_systems

        MAX_MOLECULES_PURE = 2

        validated_systems = []
        temps = []
        for system in pure_systems:
            path = Path(pure_path) / system
            if path.is_dir():  # check that path exists and is directory
                validated_systems.append(system)
            else:
                raise FileNotFoundError(f"System '{system}' not found in pure_path: {pure_path}")
            prop = SystemProperties(path)
            if len(prop.topology.molecules) > MAX_MOLECULES_PURE:  # cannot be pure component
                continue
            temps.append(prop.get("temperature", units="K"))

        # get temperature for all base systems // make sure pure component temp matches
        temperature_map = {meta.name: meta.props.get("temperature", units="K") for meta in base_metadata}

        # check that base_meta temps are all close to avg.
        T_avg = np.mean(list(temperature_map.values()))
        if not all(np.isclose(T_avg, t, atol=0.5) for t in temperature_map.values()):
            raise ValueError("Temperature variance in base systems exceeds 0.5 K. Check system temperatures.")

        # check that temps of pure systems are within errors of base systems.
        if not all(np.isclose(T_avg, t, atol=0.5) for t in temps):
            raise ValueError(
                "Pure component temperatures varied from temperature of base systems. Check provided pure_systems."
            )

        return validated_systems

    def _update_metadata_rdf(self, metadata: list[SystemMetadata]) -> list[SystemMetadata]:
        """
        Update metadata with RDF directory paths if available.

        Parameters
        ----------
        metadata : list[SystemMetadata]
            List of system metadata objects.

        Returns
        -------
        list[SystemMetadata]
            Updated metadata with RDF paths.
        """
        updated_metadata = metadata.copy()
        for m, meta in enumerate(metadata):
            new_meta = None  # save initialize
            for subdir in meta.path.iterdir():
                if subdir.is_dir() and any("rdf" in p.name.lower() for p in subdir.iterdir()):
                    new_meta = replace(meta, rdf_path=subdir)
                    if new_meta:
                        updated_metadata[m] = new_meta
                        break
            # raise error if system is mixture and no rdf directory is found
            if not new_meta and meta.kind == "mixture":
                self.logger.error(f"No RDF directory found in: {meta.path}")
                raise FileNotFoundError(f"No RDF directory found in: {meta.path}")
        return updated_metadata

    def _sort_systems(self, systems: list[SystemMetadata], molecules: list[str]) -> list[SystemMetadata]:
        """
        Sort systems by their mol fraction vectors in ascending order.

        Parameters
        ----------
        systems : list[SystemMetadata]
            List of systems to sort.
        molecules : list[str]
            Ordered list of molecule names.

        Returns
        -------
        list[SystemMetadata]
            Sorted list of systems.
        """

        def mol_fr_vector(meta: SystemMetadata) -> tuple[float, ...]:
            counts = meta.props.topology.molecule_count
            total = meta.props.topology.total_molecules
            return tuple(counts.get(mol, 0) / total for mol in molecules)

        return sorted(systems, key=mol_fr_vector)

    def _extract_top_molecules(self, systems: list[SystemMetadata]) -> list[str]:
        """
        Extract a list of unique molecules present across all systems.

        Parameters
        ----------
        systems : list[SystemMetadata]
            List of systems to analyze.

        Returns
        -------
        list[str]
            Unique molecule names.
        """
        mols_present = set()
        for meta in systems:
            mols_present.update(meta.props.topology.molecules)
        return list(mols_present)

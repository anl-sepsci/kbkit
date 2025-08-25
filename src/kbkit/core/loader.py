"""Discovers systems based on directory structure and input parameters."""

from pathlib import Path
import numpy as np
import os
import logging 
from dataclasses import replace

from kbkit.core.properties import SystemProperties
from kbkit.core.registry import SystemRegistry
from kbkit.utils.validation import validate_path
from kbkit.utils.logging import get_logger
from kbkit.schema.system_metadata import SystemMetadata
from kbkit.schema.config import SystemConfig


class SystemLoader:

    """
    SystemLoader discovers molecular systems from directory structure and input parameters.

    It identifies base and pure systems, extracts metadata, and builds a registry for simulation workflows.
    """

    def __init__(self, logger: logging.Logger | None = None, verbose: bool = False) -> None:
        self.verbose = verbose
        self.logger = logger or get_logger(f"{__name__}.{self.__class__.__name__}", verbose=self.verbose)

    def build_config(
        self,
        base_path: str | Path | None = None,
        pure_path: str | Path | None = None,
        ensemble: str = "npt",
        cations: list[str] | None = None,
        anions: list[str] | None = None,
        start_time: int = 0,
        system_names: list[str] | None = None,
    ) -> SystemConfig:
        """Create `SystemConfig` object for a list of systems to analyze with KB analysis."""
        
        # get paths to parent directories
        base_path = base_path or self._find_base_path()
        pure_path = pure_path or self._find_pure_path(base_path)
        
        # get system paths and corresponding metatdata for base systems
        base_systems = system_names or self._find_systems(base_path)
        self.logger.debug(f"Discovered base systems: {base_systems}")

        base_metadata = [self._get_metadata(base_path, system, ensemble, start_time) for system in base_systems]

        # get system paths and corresponding metadata for pure systems
        pure_systems = self._find_pure_systems(pure_path, base_metadata)
        self.logger.debug(f"Discovered pure systems: {pure_systems}")

        pure_metadata = [self._get_metadata(pure_path, system, ensemble, start_time) for system in pure_systems]

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
            molecules=molecules
        )

    def _find_base_path(self) -> Path:
        """Default option for base path is current working directory."""
        return Path(os.getcwd())

    def _find_pure_path(self, root: str | Path) -> Path:
        """Default option for pure path is directory containg pure and comp in parent directory."""
        root = validate_path(root)
        matches = []
        for path in root.rglob("*"):
            if path.is_dir():
                name = path.name.lower()
                if "pure" in name and "comp" in name:
                    matches.append(path)
        
        if not matches:
            self.logger.info("No pure component directories found! Assuming pure components are stored in base path.")
            print(f"No pure component directories found! Assuming pure components are stored in base path.")
            return root
        
        if len(matches) > 1:
            self.logger.warning(f"Multiple pure component paths found. Using: {matches[0]}")
            print(f"Multiple pure component path found. Using: {matches[0]}")
        
        return matches[0]
    
    def _get_metadata(self, path: Path, system: str, ensemble: str, start_time: int) -> SystemMetadata:
        """Get SystemMetadata object."""
        path = validate_path(path)

        prop = SystemProperties(
            system_path=path / system,
            ensemble=ensemble,
            start_time=start_time,
            verbose=self.verbose
        )

        kind = "mixture" if len(prop.topology.molecules) > 1 else "pure"

        return SystemMetadata(
            name=system,
            path=path / system,
            props=prop,
            kind=kind,
        )

    def _find_systems(self, path: str | Path, pattern: str = "*") -> list[str]:
        """
        Find systems in parent path following a pattern.
        
        Defaults to requiring a .top file.
        """
        # validate path
        path = validate_path(path)
        # get subdirs according to pattern if .top found in any files
        return sorted([
            p.name for p in path.glob(pattern) 
            if p.is_dir() and any(p.glob("*.top"))
        ]) 

    def _find_pure_systems(self, pure_path: Path, base_metadata: list[SystemMetadata]) -> list[str]:
        """Discover valid pure systems from pure_path, excluding duplicates already represented in base systems."""

        pure_path = validate_path(pure_path)
        all_pure_names = self._find_systems(pure_path)

        # Build molecule–temperature map from base systems
        base_mol_kind: set[tuple[str, str]] = set()
        for meta in base_metadata:
            for mol in meta.props.topology.molecules:
                base_mol_kind.add((mol, meta.kind))

        # construct temperature map for base_metadata
        temps_by_mol = self._build_temperature_map(base_metadata)

        # Track molecules already assigned a pure system
        assigned_molecules: set[str] = set()
        pure_systems: list[str] = []

        for name in all_pure_names:
            system_path = pure_path / name

            try:
                props = SystemProperties(system_path)
            except Exception as e:
                self.logger.debug(f"Skipping system '{name}' due to error: {e}")
                continue

            molecules = props.topology.molecules
            if len(molecules) != 1:
                continue

            mol = molecules[0]
            temp = props.get("temperature", units="K")
            known_temps = temps_by_mol.get(mol, set())

            # Check temperature match
            if not any(np.isclose(temp, t, atol=0.5) for t in known_temps):
                continue

            # Skip if pure molecule already represented in base systems
            if (mol, "pure") in base_mol_kind:
                continue

            # Skip if molecule already assigned a pure system
            if mol in assigned_molecules:
                self.logger.warning(f"Multiple pure systems found for molecule '{mol}'; using first match only.")
                print(f"WARNING: multiple pure systems found for molecule '{mol}'; using first match only.")
                continue

            # Assign system
            pure_systems.append(name)
            assigned_molecules.add(mol)

        return sorted(pure_systems)


    def _build_temperature_map(self, systems: list[SystemMetadata]) -> dict[str, set[float]]:
        """Build a temperature map that associates each molecule type with the set of simulation temperatures observed across all systems."""
        if not systems:
            return {}

        temperature_map: dict[str, set[float]] = {}

        for system in systems:
            props = system.props
            try:
                molecules = props.topology.molecules
                temp = props.get("temperature", units="K")
            except Exception:
                continue  # Skip systems with invalid topology or temperature

            for mol in molecules:
                temperature_map.setdefault(mol, set()).add(temp)

        temp_map = {mol: set(sorted(temps)) for mol, temps in temperature_map.items()}
        self.logger.debug(f"Temperature map: {temp_map}")
        return temp_map

    def _update_metadata_rdf(self, metadata: list[SystemMetadata]) -> list[SystemMetadata]:
        """Find name for rdf directory in subsystems."""
        updated_metadata = metadata.copy()
        for m, meta in enumerate(metadata):
            new_meta = None # save initialize
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
        """Sort systems by their mol fraction vectors in ascending order."""
        def mol_fr_vector(meta: SystemMetadata) -> tuple[float, ...]:
            counts = meta.props.topology.molecule_count
            total = meta.props.topology.total_molecules
            return tuple(counts.get(mol, 0) / total for mol in molecules)

        return sorted(systems, key=mol_fr_vector)
    
    def _extract_top_molecules(self, systems: list[SystemMetadata]) -> list[str]:
        """list: A list of unique molecules present in all systems."""
        mols_present = set()
        for meta in systems:
            mols_present.update(meta.props.topology.molecules)
        return list(mols_present)


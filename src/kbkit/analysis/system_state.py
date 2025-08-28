"""Extracts thermodynamic and compositional features from a batch of molecular simulation systems."""

import itertools
from functools import cached_property

import numpy as np
from numpy.typing import NDArray

from kbkit.config.unit_registry import load_unit_registry
from kbkit.schema.system_config import SystemConfig


class SystemState:
    """
    Performs analysis and validation on molecular simulation systems.

    The SystemAnalyzer consumes a SystemConfig object and provides
    tools for inspecting system composition, temperature distributions, molecule coverage,
    and semantic consistency across base and pure component systems.

    Parameters
    ----------
    config: SystemConfig
        System configuration for a set of systems.

    """

    def __init__(self, config: SystemConfig) -> None:
        # setup config
        self.config = config

        # set up unit registry
        self.ureg = load_unit_registry()
        self.Q_ = self.ureg.Quantity
    
    @property
    def top_molecules(self) -> list[str]:
        """list: Unique molecules in topology files."""
        return self.config.molecules

    @property
    def n_sys(self) -> int:
        """int: Number of systems present."""
        return len(self.config.registry)

    @cached_property
    def salt_pairs(self) -> list[tuple[str, str]]:
        """list: List of salt pairs as (cation, anion) tuples."""
        # get unique combination of anions/cations in configuration
        salt_pairs = [(cation, anion) for cation, anion in itertools.product(self.config.cations, self.config.anions)]

        # now validate list; checks molecules in pairs are in _top_molecules
        for pair in salt_pairs:
            if not all(mol in self.top_molecules for mol in pair):
                raise ValueError(
                    f"Salt pair {pair} contains molecules not present in top molecules: {self.top_molecules}"
                )
        return salt_pairs

    @cached_property
    def _nosalt_molecules(self) -> list[str]:
        """list: Molecules not part of any salt pair."""
        paired = {mol for pair in self.salt_pairs for mol in pair}
        return [mol for mol in self.top_molecules if mol not in paired]

    @cached_property
    def _salt_molecules(self) -> list[str]:
        """list: Combined molecule names for each salt pair."""
        return [".".join(pair) for pair in self.salt_pairs]

    @cached_property
    def unique_molecules(self) -> list[str]:
        """list: Molecules present after combining salt pairs as single entries."""
        return self._nosalt_molecules + self._salt_molecules

    def _get_mol_idx(self, mol: str, molecule_list: list[str]) -> int:
        """Get index of mol in molecule list."""
        if not isinstance(molecule_list, list):
            try:
                molecule_list = list(molecule_list)
            except TypeError as e:
                raise TypeError(
                    f"Molecule list could not be converted to type(list) from type({type(molecule_list)})"
                ) from e
        if mol not in molecule_list:
            raise ValueError(f"{mol} not in molecule list: {molecule_list}")
        return molecule_list.index(mol)

    @property
    def n_comp(self) -> int:
        """int: Number of unique components (molecules) in all systems."""
        return len(self.unique_molecules)

    @cached_property
    def total_molecules(self) -> NDArray[np.float64]:
        """np.ndarray: Total molecule count for each system."""
        return np.array([meta.props.topology.total_molecules for meta in self.config.registry])

    @cached_property
    def _top_molecule_counts(self) -> NDArray[np.float64]:
        """np.ndarray: Molecule count per system."""
        return np.array(
            [
                [meta.props.topology.molecule_count.get(mol, 0) for mol in self.top_molecules]
                for meta in self.config.registry
            ]
        )

    @cached_property
    def molecule_counts(self) -> NDArray[np.float64]:
        """np.ndarray: Molecule count per system."""
        counts = np.zeros((self.n_sys, self.n_comp))
        for i, mol in enumerate(self.unique_molecules):
            mol_split = mol.split(".")
            if len(mol_split) > 1 and tuple(mol_split) in self.salt_pairs:
                for salt in mol_split:
                    salt_idx = self._get_mol_idx(salt, self.top_molecules)
                    counts[:, i] += self._top_molecule_counts[:, salt_idx]
            else:
                mol_idx = self._get_mol_idx(mol, self.top_molecules)
                counts[:, i] += self._top_molecule_counts[:, mol_idx]
        return counts

    @cached_property
    def pure_molecules(self) -> list[str]:
        """list[str]: Names of molecules considered as pure components."""
        molecules = [".".join(meta.props.topology.molecules) for meta in self.config.registry if meta.kind == "pure"]
        return sorted(molecules)

    @cached_property
    def pure_mol_fr(self) -> NDArray[np.float64]:
        """np.ndarray: Mol fraction array in terms of pure components."""
        arr = np.zeros((self.n_sys, len(self.pure_molecules)))
        for i, mol in enumerate(self.pure_molecules):
            mol_split = mol.split(".")
            if len(mol_split) > 1:
                for salt in mol_split:
                    salt_idx = self._get_mol_idx(salt, self.top_molecules)
                    arr[:, i] += self._top_molecule_counts[:, salt_idx]
            else:
                mol_idx = self._get_mol_idx(mol, self.top_molecules)
                arr[:, i] += self._top_molecule_counts[:, mol_idx]
        # get mol_fr
        arr /= self.total_molecules[:, np.newaxis]
        return arr

    @cached_property
    def n_electrons(self) -> NDArray[np.float64]:
        """np.ndarray: Number of electrons corresponding to pure molecules."""
        elec_map: dict[str, float] = dict.fromkeys(self.pure_molecules, 0)
        for meta in self.config.registry:
            # only for pure systems
            if meta.kind == "pure":
                mols = ".".join(meta.props.topology.molecules)
                # get total electron count for molecules in "pure" system
                elec_map[mols] = sum(meta.props.structure.electron_count.values())

        return np.fromiter(elec_map.values(), dtype=np.float64)

    @cached_property
    def mol_fr(self) -> NDArray[np.float64]:
        """np.ndarray: Mol fraction of molecules in registry."""
        return self.molecule_counts / self.molecule_counts.sum(axis=1)[:, np.newaxis]

    def temperature(self, units: str = "K") -> NDArray[np.float64]:
        """Temperature of each simulation."""
        return np.array([meta.props.get("temperature", units=units) for meta in self.config.registry])

    def volume(self, units: str = "nm^3") -> NDArray[np.float64]:
        """Volume of each simulation."""
        return np.array([meta.props.get("volume", units=units) for meta in self.config.registry])

    def molar_volume(self, units: str = "nm^3 / molecule") -> NDArray[np.float64]:
        """Molar volumes of pure components."""
        vol_unit, N_unit = units.split("/")
        volumes = self.volume(vol_unit)
        # make dict in same order as top molecules
        volumes_map: dict[str, float] = dict.fromkeys(self.pure_molecules, 0)
        for i, meta in enumerate(self.config.registry):
            top = meta.props.topology
            # only for pure systems
            if meta.kind == "pure":
                N = self.Q_(top.total_molecules, "molecule").to(N_unit).magnitude
                volumes_map[".".join(top.molecules)] = volumes[i] / N

        return np.fromiter(volumes_map.values(), dtype=np.float64)

    def enthalpy(self, units: str = "kJ/mol") -> NDArray[np.float64]:
        """Enthalpy of each simulation."""
        return np.array([meta.props.get("enthalpy", units=units) for meta in self.config.registry])

    def pure_enthalpy(self, units: str = "kJ/mol") -> NDArray[np.float64]:
        """Pure component enthalpies."""
        enth: dict[str, float] = dict.fromkeys(self.pure_molecules, 0)
        for meta in self.config.registry:
            if meta.kind == "pure":
                value = meta.props.get("enthalpy", units=units, std=False)
                # make sure value is float
                if isinstance(value, tuple):
                    value = value[0]
                mols = ".".join(meta.props.topology.molecules)
                enth[mols] = float(value)
        return np.fromiter(enth.values(), dtype=np.float64)

    def ideal_enthalpy(self, units: str = "kJ/mol") -> NDArray[np.float64]:
        """Linear combination of simulation enthalpy."""
        return self.pure_mol_fr @ self.pure_enthalpy(units)

    def h_mix(self, units: str = "kJ/mol") -> NDArray[np.float64]:
        """Enthalpy of mixing."""
        return self.enthalpy(units) - self.ideal_enthalpy(units)

    def molecule_rho(self, units: str = "molecule/nm^3") -> NDArray[np.float64]:
        """Compute the number density of each molecule."""
        N_units, vol_units = units.split("/")  # get the target units
        N = self.Q_(self.molecule_counts, "molecule").to(N_units).magnitude
        V = self.volume(vol_units)[:, np.newaxis]
        return np.asarray(N / V)

    def volume_bar(self, units: str = "nm^3/molecule") -> NDArray[np.float64]:
        """Linear combination of molar volumes."""
        return self.pure_mol_fr @ self.molar_volume(units)

    def rho_bar(self, units: str = "molecule/nm^3") -> NDArray[np.float64]:
        """Linear combination of number density."""
        N_units, vol_units = units.split("/")
        return 1 / self.volume_bar(units=f"{vol_units}/{N_units}")

    def rho_ij(self, units: str = "molecule/nm^3") -> NDArray[np.float64]:
        """Pairwise number density of molecules."""
        return self.molecule_rho(units)[:, :, np.newaxis] * self.molecule_rho(units)[:, np.newaxis, :]

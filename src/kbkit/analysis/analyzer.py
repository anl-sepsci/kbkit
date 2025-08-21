"""Extract properties from systems."""



from functools import cached_property
import itertools
from dataclasses import replace
from numpy.typing import NDArray
import numpy as np

from kbkit.config.unit_registry import load_unit_registry
from kbkit.schema.config import SystemConfig
from kbkit.schema.system_metadata import SystemMetadata


class SystemAnalyzer: # go back and add logger flags: self.config.logger

    def __init__(self, config: SystemConfig) -> None:
        self.top_molecules = self._extract_top_molecules(config)
        sorted_registry = self._sort_systems(config.registry.all())
        self.config = replace(config, registry=sorted_registry)

        # set up unit registry
        self.ureg = load_unit_registry()
        self.Q_ = self.ureg.Quantity

    def _extract_top_molecules(self, config: SystemConfig) -> list[str]:
        """list: A list of unique molecules present in all systems. This encompasses all anions and cations individually, before being combined into salt pairs."""
        mols_present = set()
        for meta in config.registry:
            mols_present.update(meta.props.topology.molecules)
        return list(mols_present)
    
    def _sort_systems(self, systems: list[SystemMetadata]) -> list[SystemMetadata]:
        """Sort systems by their mol fraction vectors in ascending order."""
        def mol_fr_vector(meta: SystemMetadata) -> tuple[float, ...]:
            counts = meta.props.topology.molecule_count
            total = meta.props.topology.total_molecules
            return tuple(counts.get(mol, 0) / total for mol in self.top_molecules)

        return sorted(systems, key=mol_fr_vector)
    
    @property
    def n_sys(self) -> int:
        """int: Number of systems present."""
        return len(self.config.registry)
    
    @cached_property
    def salt_pairs(self) -> list[tuple[str, str]]:
        """list: List of salt pairs as (cation, anion) tuples."""
        # get unique combination of anions/cations in configuration
        salt_pairs = [(cation, anion) for cation, anion in itertools.product(self.config.cations, self.config.anions)]

        # now validate list; checks molecules in pairs are in top_molecules
        for pair in salt_pairs:
            if not all(mol in self.top_molecules for mol in pair):
                raise ValueError(
                    f"Salt pair {pair} contains molecules not present in top molecules: {self.top_molecules}"
                )
        return salt_pairs
    
    @cached_property
    def nosalt_molecules(self) -> list[str]:
        """list: Molecules not part of any salt pair."""
        paired = {mol for pair in self.salt_pairs for mol in pair}
        return [mol for mol in self.top_molecules if mol not in paired]

    @cached_property
    def salt_molecules(self) -> list[str]:
        """list: Combined molecule names for each salt pair."""
        return ["-".join(pair) for pair in self.salt_pairs]
    
    @cached_property
    def unique_molecules(self) -> list[str]:
        """list: Molecules present after combining salt pairs as single entries."""
        return self.nosalt_molecules + self.salt_molecules
    
    @property
    def n_comp(self) -> int:
        """int: Number of unique components (molecules) in all systems."""
        return len(self.unique_molecules)
    
    @cached_property
    def total_molecules(self) -> NDArray[np.float64]:
        """np.ndarray: Total molecule count for each system."""
        return np.array([
            meta.props.topology.total_molecules for meta in self.config.registry
        ])
    
    @cached_property
    def molecule_counts(self) -> NDArray[np.float64]:
        """np.ndarray: Molecule count per system."""
        return np.array([
            [meta.props.topology.molecule_count.get(mol, 0) for mol in self.top_molecules]
            for meta in self.config.registry    
        ])
    
    @cached_property
    def n_electrons(self) -> NDArray[np.float64]:
        """np.ndarray: Compute number of electrons in each molecule."""
        electrons = set()
        added_molecules = []
        for meta in self.config.registry:
            for mol, elec in meta.props.structure.electron_count.items():
                if mol not in added_molecules:
                    electrons.update(elec)
                    added_molecules.append(mol)
        return np.ndarray(list(electrons))
    
    @cached_property
    def top_mol_fr(self) -> NDArray[np.float64]:
        """np.ndarray: Mol fraction for molecules in top molecules."""
        return self.molecule_counts / self.molecule_counts.sum(axis=1)
    
    def _get_mol_idx(self, mol: str, molecule_list: list[str]) -> int:
        """Get index of mol in molecule list."""
        if not isinstance(molecule_list, list):
            try:
                molecule_list = list(molecule_list)
            except TypeError as e:
                raise TypeError(f"Molecule list could not be converted to type(list) from type({type(molecule_list)})")
        if mol not in molecule_list:
            raise ValueError(f"{mol} not in molecule list: {molecule_list}")
        return molecule_list.index(mol)
    
    @cached_property
    def mol_fr(self) -> NDArray[np.float64]:
        """np.ndarray: Mol fraction array including salt pair combinations."""
        mfr = np.zeros((self.n_sys, self.n_comp))
        # iterate through systems and molecules in topology to find and ajust for salt-pairs
        for i, meta in enumerate(self.config.registry):
            for j, mol in enumerate(self.unique_molecules):
                # check if molecule is salt-pair
                mol_split = mol.split("-")
                # handle salt-pairs
                if len(mol_split) > 0 and tuple(mol_split) in self.salt_pairs:
                    for salt in mol_split:
                        salt_idx = self._get_mol_idx(salt, self.top_molecules)
                        mfr[i,j] += self.top_mol_fr[i,salt_idx]
                else:
                    mol_idx = self._get_mol_idx(salt, self.top_molecules)
                    mfr[i,j] += self.top_mol_fr[i,mol_idx]
        return mfr
    
    def temperature(self, units: str = "K") -> NDArray[np.float64]:
        """Temperature of each simulation."""
        return np.array([
            meta.props.get("temperature", units) for meta in self.config.registry
        ])
    
    def volume(self, units: str = "nm^3") -> NDArray[np.float64]:
        """Volume of each simulation."""
        return np.array([
            meta.props.get("volume", units) for meta in self.config.registry
        ])
    
    def molar_volume(self, units: str = "nm^3 / molecule") -> NDArray[np.float64]:
        """Molar volumes of pure components."""
        vol_unit, N_unit = units.split('/')
        volumes = self.volume(vol_unit)
        # make dict in same order as top molecules
        volumes_map: dict[str, float] = {mol: 0 for mol in self.top_molecules}
        for i, meta in enumerate(self.config.registry):
            top = meta.props.top
            # only for pure systems
            if meta.kind == "pure":
                N = self.Q_(top.total_molecules, "molecule").to(N_unit).magnitude
                volumes_map[top.molecules[0]] = volumes[i] / N
        return np.fromiter(volumes_map.values(), dtype=np.float64)

    def enthalpy(self, units: str = "kJ/mol") -> NDArray[np.float64]:
        """Enthalpy of each simulation."""
        return np.array([
            meta.props.get("enthalpy", units) for meta in self.config.registry
        ])
    
    def pure_enthalpy(self, units: str = "kJ/mol") -> NDArray[np.float64]:
        """Pure component enthalpies."""
        enth: dict[str, float] = {mol: 0 for mol in self.top_molecules}
        for meta in self.config.registry:
            if meta.kind == "pure":
                enth[meta.props.topology.molecules[0]] = meta.props.get("enthalpy", units)
        return np.fromiter(enth.values(), dtype=np.float64)

    def ideal_enthalpy(self, units: str = "kJ/mol") -> NDArray[np.float64]:
        """Linear combination of simulation enthalpy."""
        return self.top_mol_fr @ self.pure_enthalpy(units)

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
        return self.top_mol_fr @ self.molar_volume(units)
    
    def rho_bar(self, units: str = "molecule/nm^3") -> NDArray[np.float64]:
        """Linear combination of number density."""
        N_units, vol_units = units.split('/')
        return 1 / self.volume_bar(units=f"{vol_units}/{N_units}")
    
    def rho_ij(self, units: str = "molecule/nm^3") -> NDArray[np.float64]:
        """Pairwise number density of molecules."""
        return self.molecule_rho(units)[:,:,np.newaxis] * self.molecule_rho(units)[:,np.newaxis,:]

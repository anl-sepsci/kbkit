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


    Attributes
    ----------
    ureg: UnitRegistry
        Pint unit registry.
    Q_: UnitRegistry.Quantity
        Pint quantity object for unit conversions.
    """

    def __init__(self, config: SystemConfig) -> None:
        # setup config
        self.config = config

        # set up unit registry
        self.ureg = load_unit_registry()
        self.Q_ = self.ureg.Quantity

    @property
    def top_molecules(self) -> list[str]:
        """list[str]: Unique molecules in topology files."""
        return self.config.molecules

    @property
    def n_sys(self) -> int:
        """int: Number of systems present."""
        return len(self.config.registry)

    @cached_property
    def salt_pairs(self) -> list[tuple[str, str]]:
        """list[tuple[str, str]]: List of salt pairs as (cation, anion) tuples."""
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
        """list[str]: Molecules not part of any salt pair."""
        paired = {mol for pair in self.salt_pairs for mol in pair}
        return [mol for mol in self.top_molecules if mol not in paired]

    @cached_property
    def _salt_molecules(self) -> list[str]:
        """list[str]: Combined molecule names for each salt pair."""
        return [".".join(pair) for pair in self.salt_pairs]

    @cached_property
    def unique_molecules(self) -> list[str]:
        """list[str]: Molecules present after combining salt pairs as single entries."""
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
        """int: Total number of unique components."""
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
        uniq_elec_map: dict[str, float] = dict.fromkeys(self.top_molecules, 0)
        for meta in self.config.registry:
            mols = meta.props.topology.molecules
            ecount = meta.props.structure.electron_count
            for mol in mols:
                if uniq_elec_map[mol] == 0 and ecount.get(mol) is not None:
                    uniq_elec_map[mol] = ecount.get(mol, 0)

        elec_map: dict[str, float] = dict.fromkeys(self.pure_molecules, 0)
        for mol_ls in self.pure_molecules:
            mols = mol_ls.split(".")
            elec_map[mol_ls] = sum([uniq_elec_map.get(mol, 0) for mol in mols])

        elec_mapped = np.fromiter(elec_map.values(), dtype=np.float64)
        if not all(elec_mapped > 0):
            elec_mapped = np.full_like(self.pure_molecules, fill_value=np.nan, dtype=float)
        return elec_mapped

    @cached_property
    def mol_fr(self) -> NDArray[np.float64]:
        """np.ndarray: Mol fraction of molecules in registry."""
        return self.molecule_counts / self.molecule_counts.sum(axis=1)[:, np.newaxis]

    def temperature(self, units: str = "K") -> NDArray[np.float64]:
        """Temperature of each simulation.

        Parameters
        ----------
        units: str
            Temperature units (default: K)

        Returns
        -------
        np.ndarray
            1D temperature array as a function of composition.
        """
        return np.array([meta.props.get("temperature", units=units) for meta in self.config.registry])

    def volume(self, units: str = "nm^3") -> NDArray[np.float64]:
        """Volume of each simulation.

        Parameters
        ----------
        units: str
            Volume units (default: nm^3)

        Returns
        -------
        np.ndarray
            1D volume array as a function of composition.
        """
        return np.array([meta.props.get("volume", units=units) for meta in self.config.registry])

    def molar_volume(self, units: str = "nm^3 / molecule") -> NDArray[np.float64]:
        """Molar volumes of pure components.

        Parameters
        ----------
        units: str
            Molar volume units (default: nm^3/molecule)

        Returns
        -------
        np.ndarray
            1D array for each unique molecule.
        """
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
        """Enthalpy of each simulation.

        Parameters
        ----------
        units: str
            Enthalpy units (default: kJ/mol/K)

        Returns
        -------
        np.ndarray
            1D array of system enthalpies as a function of composition.
        """
        return np.array([meta.props.get("enthalpy", units=units) for meta in self.config.registry])

    def pure_enthalpy(self, units: str = "kJ/mol") -> NDArray[np.float64]:
        """Pure component enthalpies.

        Parameters
        ----------
        units: str
            Enthalpy units (default: kJ/mol/K)

        Returns
        -------
        np.ndarray
            1D array of enthalpies for pure components.
        """
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
        r"""Ideal enthalpy as a function of composition.

        Parameters
        ----------
        units: str
            Enthalpy units (default: kJ/mol/K)

        Returns
        -------
        np.ndarray
            1D array of ideal enthalpies as a function of composition.

        .. math::
            H^{id} = \sum_{i=1}^n x_i H_i

        where:
            - :math:`x_i` is mol fraction of molecule :math:`i`
            - :math:`H_i` is the pure component simulation enthalpy of molecule :math:`i`
        """
        return self.pure_mol_fr @ self.pure_enthalpy(units)

    def h_mix(self, units: str = "kJ/mol") -> NDArray[np.float64]:
        """Enthalpy of mixing as a function of composition.

        Parameters
        ----------
        units: str
            Enthalpy units (default: kJ/mol/K)

        Returns
        -------
        np.ndarray
            1D array of mixing enthalpies as a function of composition.

        .. math::
            H_{mix} = H - H^{id}

        where:
            - :math:`H` is the simulation enthlapy for mixtures
            - :math:`H^{id}` is ideal enthalpy
        """
        return self.enthalpy(units) - self.ideal_enthalpy(units)

    def molecule_rho(self, units: str = "molecule/nm^3") -> NDArray[np.float64]:
        """Compute the number density of each molecule for all compositions.

        Parameters
        ----------
        units: str
            Number denisty units (default: molecule/nm^3)

        Returns
        -------
        np.ndarray
            2D array of number density of each molecule as a function of composition.
        """
        N_units, vol_units = units.split("/")  # get the target units
        N = self.Q_(self.molecule_counts, "molecule").to(N_units).magnitude
        V = self.volume(vol_units)[:, np.newaxis]
        return np.asarray(N / V)

    def volume_bar(self, units: str = "nm^3/molecule") -> NDArray[np.float64]:
        """Molar volume of mixture.

        Parameters
        ----------
        units: str
            Molar volume units (default: nm^3/molecule)

        Returns
        -------
        np.ndarray
            1D array of molar volumes as a function of composition.
        """
        return self.pure_mol_fr @ self.molar_volume(units)

    def rho_bar(self, units: str = "molecule/nm^3") -> NDArray[np.float64]:
        """Mixture number density.

        Parameters
        ----------
        units: str
            Number density units (default: molecule/nm^3)

        Returns
        -------
        np.ndarray
            1D array of number density as a function of composition.
        """
        N_units, vol_units = units.split("/")
        return 1 / self.volume_bar(units=f"{vol_units}/{N_units}")

    def rho_ij(self, units: str = "molecule/nm^3") -> NDArray[np.float64]:
        """Pairwise number density of molecules.

        Parameters
        ----------
        units: str
            Number density units (default: molecule/nm^3)

        Returns
        -------
        np.ndarray
            2D array of pairwise mixture number densities as a function of composition.
        """
        return self.molecule_rho(units)[:, :, np.newaxis] * self.molecule_rho(units)[:, np.newaxis, :]

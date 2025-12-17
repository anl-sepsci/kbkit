"""Extracts thermodynamic and compositional features from a batch of molecular simulation systems."""

import itertools
from functools import cached_property

import numpy as np
from numpy.typing import NDArray

from kbkit.config.unit_registry import load_unit_registry
from kbkit.schema.system_config import SystemConfig
from kbkit.schema.thermo_property import ThermoProperty


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
        """int: Total number of :meth:`unique_molecules`."""
        return len(self.unique_molecules)

    @cached_property
    def total_molecules(self) -> NDArray[np.float64]:
        """np.ndarray: Total number of molecules, :math:`N_T`, in each system."""
        return np.array([meta.props.topology.total_molecules for meta in self.config.registry])

    @cached_property
    def molecule_info(self) -> dict[str, dict[str, int]]:
        """dict: Number of molecules of each type in topology mapped to each system."""
        return {meta.name: meta.props.topology.molecule_count for meta in self.config.registry}

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
        """np.ndarray: Molecule count per system, mapped to :meth:`unique_molecules`."""
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
        """np.ndarray: Mol fraction array mapped to :meth:`pure_molecules`."""
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
    def top_electron_map(self) -> dict[str, int]:
        """dict[str, int]: Number of electrons mapped to each molecule type."""
        uniq_elec_map: dict[str, int] = dict.fromkeys(self.top_molecules, 0)
        for meta in self.config.registry:
            mols = meta.props.topology.molecules
            ecount = meta.props.topology.electron_count
            for mol in mols:
                if uniq_elec_map[mol] == 0 and ecount.get(mol) is not None:
                    uniq_elec_map[mol] = ecount.get(mol, 0)
        return uniq_elec_map

    @cached_property
    def unique_electrons(self) -> NDArray[np.float64]:
        r"""np.ndarray: Number of electrons, :math:`Z_i`, mapped to :meth:`unique_molecules`."""
        elec_map: dict[str, float] = dict.fromkeys(self.unique_molecules, 0)
        for mol_ls in self.unique_molecules:
            mols = mol_ls.split(".")
            elec_map[mol_ls] = sum([self.top_electron_map.get(mol, 0) for mol in mols])
        elec_mapped = np.fromiter(elec_map.values(), dtype=np.float64)
        if not all(elec_mapped > 0):
            elec_mapped = np.full_like(self.unique_molecules, fill_value=np.nan, dtype=float)
        return elec_mapped

    @cached_property
    def total_electrons(self) -> NDArray[np.float64]:
        r"""np.ndarray: Linear combination of electron numbers and mol fractions, :math:`\bar{Z} = \sum_i x_i Z_i`, mapped to :meth:`unique_molecules`."""
        return self.mol_fr @ self.unique_electrons

    @cached_property
    def mol_fr(self) -> NDArray[np.float64]:
        """np.ndarray: Mol fraction of :meth:`unique_molecules` in registry."""
        return self.molecule_counts / self.molecule_counts.sum(axis=1)[:, np.newaxis]

    def temperature(self, units: str = "K") -> NDArray[np.float64]:
        r"""Temperature, :math:`\left \langle T \right \rangle`, of each simulation.

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
        r"""Volume, :math:`\left \langle V \right \rangle`, of each simulation.

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

    def enthalpy(self, units: str = "kJ/mol") -> NDArray[np.float64]:
        r"""Enthalpy, :math:`H`, of each simulation.

        Parameters
        ----------
        units: str
            Enthalpy units (default: kJ/mol)

        Returns
        -------
        np.ndarray
            1D array of system enthalpies as a function of composition.
        """
        return np.array([meta.props.get("enthalpy", units=units) for meta in self.config.registry])

    def heat_capacity(self, units: str = "kJ/mol/K") -> NDArray[np.float64]:
        r"""Heat capacity, :math:`c_p`, of each simulation.

        Parameters
        ----------
        units: str
            Heat capacity units (default: kJ/mol/K)

        Returns
        -------
        np.ndarray
            1D array of system heat capacities as a function of composition.
        """
        return np.array([meta.props.get("heat_capacity", units=units) for meta in self.config.registry])

    def isothermal_compressibility(self, units: str = "1/kPa") -> NDArray[np.float64]:
        r"""Isothermal compressiblity, :math:`\kappa_T`, of each simulation.

        Parameters
        ----------
        units: str
            Isothermal compressiblity units (default: 1/kPa)

        Returns
        -------
        np.ndarray
            1D array of system isothermal compressiblities as a function of composition.
        """
        return np.array([meta.props.get("isothermal_compressibility", units=units) for meta in self.config.registry])

    def pure_enthalpy(self, units: str = "kJ/mol") -> NDArray[np.float64]:
        """Pure component enthalpies, :math:`H_i`, mapped to :meth:`pure_molecules` array.

        Parameters
        ----------
        units: str
            Enthalpy units (default: kJ/mol)

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
        r"""Ideal enthalpy, :math:`H^{id}`, as a function of composition.

        Parameters
        ----------
        units: str
            Enthalpy units (default: kJ/mol)

        Returns
        -------
        np.ndarray
            1D array of ideal enthalpies as a function of composition.

        Notes
        -----
        Ideal enthalpy, :math:`H^{id}`, is calculated via:

        .. math::
            H^{id} = \sum_{i=1}^n x_i H_i

        where:
            - :math:`x_i` is mol fraction of molecule :math:`i`
            - :math:`H_i` is the pure component enthalpy of molecule :math:`i`
        """
        return self.pure_mol_fr @ self.pure_enthalpy(units)

    def mixture_enthalpy(self, units: str = "kJ/mol") -> NDArray[np.float64]:
        r"""Enthalpy of mixing, :math:`\Delta H_{mix}`, as a function of composition.

        Parameters
        ----------
        units: str
            Enthalpy units (default: kJ/mol)

        Returns
        -------
        np.ndarray
            1D array of mixing enthalpies as a function of composition.

        Notes
        -----
        Mixing enthalpy, :math:`\Delta H_{mix}`, is calculated via:

        .. math::
            \Delta H_{mix} = H - H^{id}

        where:
            - :math:`H` is the simulation enthlapy for mixtures
            - :math:`H^{id}` is ideal enthalpy
        """
        return self.enthalpy(units) - self.ideal_enthalpy(units)
    
    def molar_volume_map(self, units: str = "nm^3 / molecule") -> dict[str, float]:
        r"""Molar volumes, :math:`V_i`, of mapped to molecule name (for pure components).

        Parameters
        ----------
        units: str
            Molar volume units (default: nm^3/molecule)

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary mapping molar volumes to corresponding molecule
        """
        vol_unit, N_unit = units.split("/")
        volumes = self.volume(vol_unit)
        # make dict in same order as pure molecules
        volumes_map: dict[str, float] = dict.fromkeys(self.pure_molecules, 0)
        for i, meta in enumerate(self.config.registry):
            top = meta.props.topology
            # only for pure systems
            if meta.kind == "pure":
                N = self.Q_(top.total_molecules, "molecule").to(N_unit).magnitude
                volumes_map[".".join(top.molecules)] = volumes[i] / N

        return volumes_map

    def pure_molar_volume(self, units: str = "nm^3 / molecule") -> NDArray[np.float64]:
        r"""Molar volumes, :math:`V_i`, of pure components.

        Parameters
        ----------
        units: str
            Molar volume units (default: nm^3/molecule)

        Returns
        -------
        np.ndarray
            1D array for each unique molecule.
        """
        return np.fromiter(self.molar_volume_map(units).values(), dtype=np.float64)


    def ideal_molar_volume(self, units: str = "nm^3/molecule") -> NDArray[np.float64]:
        r"""Ideal molar volume, :math:`\bar{V}`, of mixture.

        Parameters
        ----------
        units: str
            Molar volume units (default: nm^3/molecule)

        Returns
        -------
        np.ndarray
            1D array of molar volumes as a function of composition.

        Notes
        -----
        Ideal molar volume, :math: `\bar{V}`, is calculated according to:

        .. math::
            \bar{V} = \sum_i x_i V_i

        where:
            - :math:`x_i` is the mole fraction of component `i`
            - :math:`V_i` is the molar volume of component `i`
        """
        return self.pure_mol_fr @ self.pure_molar_volume(units)

    def mixture_molar_volume(self, units: str = "nm^3/molecule") -> NDArray[np.float64]:
        r"""Mixture molar volume, :math:`\Delta V_{mix}`.

        Parameters
        ----------
        units: str
            Molar volume units (default: nm^3/molecule)

        Returns
        -------
        np.ndarray
            1D array of molar volumes as a function of composition.

        Notes
        -----
        Mixture molar volume, :math:`\Delta V_{mix}`, is calculated via:

        .. math::
            \Delta V_{mix} = \frac{\left \langle V \right \rangle}{N_T}

        where:
            - :math:`\left \langle V \right \rangle` is the ensemble average volume
            - :math:`N_T` is total number of molecules present
        """
        vol_unit, N_unit = units.split("/")
        volumes = self.volume(vol_unit)
        molecs = self.Q_(self.total_molecules, "molecule").to(N_unit).magnitude
        return np.asarray(volumes / molecs, dtype=np.float64)

    def excess_molar_volume(self, units: str = "nm^3/molecule") -> NDArray[np.float64]:
        r"""Excess molar volume, :math:`V^{ex}`.

        Parameters
        ----------
        units: str
            Molar volume units (default: nm^3/molecule)

        Returns
        -------
        np.ndarray
            1D array of molar volumes as a function of composition.

        Notes
        -----
        Excess molar volume, :math:`V^{ex}`, is calculated via:

        .. math::
            V^{ex} = \Delta V_{mix} - \bar{V}

        where:
            - :math:`\Delta V_{mix}` is the mixture molar volume
            - :math:`\bar{V}` is ideal molar volume
        """
        return self.mixture_molar_volume(units) - self.ideal_molar_volume(units)
    
    def mixture_number_density(self, units: str = "molecule/nm^3") -> NDArray[np.float64]:
        r"""Mixture number density, :math:`\rho`.

        Parameters
        ----------
        units: str
            Number density units (default: molecule/nm^3)

        Returns
        -------
        np.ndarray
            1D array of number densities as a function of composition.

        Notes
        -----
        Mixture number density, :math:`\rho`, is calculated via:

        .. math::
            \rho = \frac{N_T}{\left \langle V \right \rangle}

        where:
            - :math:`\left \langle V \right \rangle` is the ensemble average volume
            - :math:`N_T` is total number of molecules present
        """
        N_unit, vol_unit = units.split("/")
        volumes = self.volume(vol_unit)
        molecs = self.Q_(self.total_molecules, "molecule").to(N_unit).magnitude
        return np.asarray(molecs/volumes, dtype=np.float64)

    def computed_properties(self) -> dict[str, ThermoProperty]:
        """
        Collects all computed properties from molecular dynamics for current set of systems.

        Returns
        -------
        List[ThermoProperty]
            A list of `ThermoProperty` instances, containing the name, value, and units of the
            computed property from current set of systems. The units are corresponding to GROMACS
            default units.
        """     
        return {
            "top_molecules": ThermoProperty(name="top_molecules", value=self.top_molecules, units=""),
            "salt_pairs": ThermoProperty(name="salt_pairs", value=self.salt_pairs, units=""),
            "unique_molecules": ThermoProperty(name="unique_molecules", value=self.unique_molecules, units=""),
            "total_molecules": ThermoProperty(name="total_molecules", value=self.total_molecules, units="molecule"),
            "molecule_info": ThermoProperty(name="molecule_info", value=self.molecule_info, units=""),
            "molecule_counts": ThermoProperty(name="molecule_counts", value=self.molecule_counts, units="molecule"),
            "pure_molecules": ThermoProperty(name="pure_molecules", value=self.pure_molecules, units=""),
            "pure_mol_fr": ThermoProperty(name="pure_mol_fr", value=self.pure_mol_fr, units=""),
            "electron_map": ThermoProperty(name="electron_map", value=self.top_electron_map, units="electron/molecule"),
            "unique_electrons": ThermoProperty(name="unique_electrons", value=self.unique_electrons, units="electron/molecule"),
            "total_electrons": ThermoProperty(name="total_electrons", value=self.total_electrons, units="electron/molecule"),
            "mol_fr": ThermoProperty(name="mol_fr", value=self.mol_fr, units=""),
            "temperature": ThermoProperty(name="temperature", value=self.temperature(units="K"), units="K"),
            "volume": ThermoProperty(name="volume", value=self.volume(units="nm^3"), units="nm^3"),
            "molar_volume_map": 
                ThermoProperty(name="molar_volume_map", value=self.molar_volume_map(units="cm^3/mol"), units="cm^3/mol"),
            "pure_molar_volume": 
                ThermoProperty(name="pure_molar_volume", value=self.pure_molar_volume(units="cm^3/mol"), units="cm^3/mol"),
            "enthalpy": ThermoProperty(name="enthalpy", value=self.enthalpy(units="kJ/mol"), units="kJ/mol"),
            "heat_capacity": 
                ThermoProperty(name="heat_capacity", value=self.heat_capacity(units="kJ/mol/K"), units="kJ/mol/K"),
            "isothermal_compressibility_md": 
                ThermoProperty(
                    name="isothermal_compressibility_md", value=self.isothermal_compressibility(units="1/kPa"), units="1/kPa"
                ),
            "pure_enthalpy": 
                ThermoProperty(name="pure_enthalpy", value=self.pure_enthalpy(units="kJ/mol"), units="kJ/mol"),
            "ideal_enthalpy": 
                ThermoProperty(name="ideal_enthalpy", value=self.ideal_enthalpy(units="kJ/mol"), units="kJ/mol"),
            "mixture_enthalpy": ThermoProperty(name="mixture_enthalpy", value=self.mixture_enthalpy(units="kJ/mol"), units="kJ/mol"),
            "ideal_molar_volume": ThermoProperty(name="ideal_molar_volume", value=self.ideal_molar_volume(units="cm^3/mol"), units="cm^3/mol"),
            "mixture_molar_volume": ThermoProperty(name="mixture_molar_volume", value=self.mixture_molar_volume(units="cm^3/mol"), units="cm^3/mol"),
            "excess_molar_volume": 
                ThermoProperty(name="excess_molar_volume", value=self.excess_molar_volume(units="cm^3/mol"), units="cm^3/mol"),
            "mixture_number_density": ThermoProperty(name="mixture_number_density", value=self.mixture_number_density(units="molecule/nm^3"), units="molecule/nm^3"),
        }

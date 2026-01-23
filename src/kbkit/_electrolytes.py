# """Structured representation of scalar properties with units and semantic tags."""

# from dataclasses import dataclass, field
# from functools import cached_property, wraps
# from typing import Any

# from kbkit.config.unit_registry import load_unit_registry


# @dataclass
# class ThermoProperty:
#     """
#     Container for a scalar property with units and semantic annotations.

#     Designed to store a value alongside its physical units and optional tags
#     for classification, filtering, or metadata enrichment.

#     Attributes
#     ----------
#     name: str
#         Name of the computed property.
#     value : Any
#         The raw property value (e.g., float, int, or derived object).
#     units : str
#         Units associated with the value (e.g., "kJ/mol", "nm", "mol/L").
#     """

#     name: str
#     value: Any
#     units: str = field(default_factory=str)

#     def to(self, new_units: str):
#         """
#         Unit conversion for property.

#         Parameters
#         ----------
#         new_units: str
#             Units for desired property

#         Returns
#         -------
#         Any
#             Value in new units.
#         """
#         ureg = load_unit_registry()
#         Q_ = ureg.Quantity
#         return Q_(self.value, self.units).to(new_units).magnitude


# def register_property(name: str, units: str):
#     """
#     Method decorator for associating metadata and units with a ThermoProperty.

#     Parameters
#     ----------
#     name : str
#         Property name.
#     units : str
#         Property units.

#     Returns
#     -------
#     Callable
#         The resulting decorator produces a cached property containing a ThermoProperty instance.
#     """

#     def decorator(func):
#         """Recieve decorated method and applies the wrapping logic."""

#         @cached_property
#         @wraps(func)
#         def wrapper(self):
#             """Create and return the ThermoProperty object for a given function."""
#             return ThermoProperty(name=name, value=func(self), units=units)

#         return wrapper

#     return decorator


# ####
# """
# Calculator for Kirkwood-Buff Integrals (KBIs).

# This calculator operates on a :class:`~kbkit.systems.state.SystemState` that contains molecular dynamics properties from structure (.gro) and energy (.edr) files.
# Additional inputs to `KBICalculator` are key parameters used for the KBI corrections provided in :class:`~kbkit.analysis.integrator.KBIntegrator`.

# The purpose of the `KBICalculator` is the following:
#     * Computes a KBI matrix for all molecular pairs in each system in the :class:`~kbkit.systems.registry.SystemRegistry` object.
#     * Applies electrolyte corrections to KBI matrix if electrolytes are present.
#     * Stores results for each system into a :class:`~kbkit.schema.kbit_metadata.KBIMetadata` container.
# """

# import numpy as np
# from numpy.typing import NDArray

# from kbkit.analysis.integrator import KBIntegrator
# from kbkit.schema.kbi_metadata import KBIMetadata
# from kbkit.systems.state import SystemState
# from kbkit.utils.file_resolver import FileResolver


# class KBICalculator:
#     """
#     Computes Kirkwood-Buff integrals for molecular systems using RDF data.

#     Interfaces with RDFParser and KBIntegrator to extract pairwise KBIs,
#     populate metadata, and apply corrections for electrolyte systems.

#     Parameters
#     ----------
#     state : SystemState
#         SystemState object providing molecule indexing, salt pairs, and composition.
#     use_fixed_r : bool, optional
#         If True, uses a fixed cutoff radius for KBI calculations (default: False).
#     ignore_convergence_errors : bool, optional
#         If True, ingnores convergence errors and forces KBI calculations to skip entire systems with non-converged RDFs (default: False).
#     rdf_convergence_threshold: float, optional
#         Value of the slope of RDF tail for the RDF to be considered as converged (default: 0.005).
#     correct_rdf_convergence: bool, optional
#         Whether to correct RDF for excess/depletion, i.e., Ganguly correction (default: True).
#     apply_damping: bool, optional
#         Whether to apply damping function to correlation function, i.e., Kruger correction (default: True).
#     extrapolate_thermodynamic_limit: bool, optional
#         Whether to extrapolate KBI value to the thermodynamic limit (default: True).

#     Attributes
#     ----------
#     kbi_metadata : dict[str, list[KBIMetadata]]
#         Dictionary mapping system names to lists of KBI metadata objects.
#     """

#     def __init__(
#         self,
#         state: SystemState,
#         use_fixed_r: bool = False,
#         ignore_convergence_errors: bool = False,
#         rdf_convergence_threshold: float = 0.005,
#         correct_rdf_convergence: bool = True,
#         apply_damping: bool = True,
#         extrapolate_thermodynamic_limit: bool = True,
#     ) -> None:
#         self.state = state
#         self.use_fixed_r = use_fixed_r
#         self.ignore_convergence_errors = ignore_convergence_errors
#         self.rdf_convergence_threshold = rdf_convergence_threshold
#         self.correct_rdf_convergence = correct_rdf_convergence
#         self.apply_damping = apply_damping
#         self.extrapolate_thermodynamic_limit = extrapolate_thermodynamic_limit
#         self.kbi_metadata: dict[str, list[KBIMetadata]] = {}

#     def compute_kbi_matrix(self, apply_electrolyte_correction: bool = True) -> NDArray[np.float64]:
#         r"""
#         Runs the full KBI computation workflow by orchestrating the process of computing the raw KBI matrix and, if specified, applies the electrolyte correction.

#         Parameters
#         ----------
#         apply_electrolyte_correction: bool, optional
#             If True (default), applies corrections for salt-salt and salt-other interactions. If False, returns the raw, uncorrected KBI matrix.

#         Returns
#         -------
#         np.ndarray
#             A 3D numpy array representing the final KBI matrix.

#         Notes
#         -----
#         First the raw KBI matrix is computed for all systems.
#         Each KBI value :math:`G_{ij}` is computed by integrating the RDF between molecule types :math:`i, j`:

#         .. math::
#             G_{ij} = \int_0^{\infty} 4\pi r^2 (g_{ij}(r) - 1) dr

#         * If an RDF directory is missing, the corresponding system's values remain NaN, if ignore_convergence_errors is True.
#         * Populates `kbi_metadata` with integration results for each RDF file.

#         Then if electrolyte correction is desired and an electrolyte is present the following electrolyte corrections are applied.
#         The electrolyte corrections modifies the KBI matrix to account for salt-salt and salt-other interactions using mole fraction-weighted combinations of cation and anion contributions.

#         Salt-salt interactions :math:`G_{ss}` are computed as:

#         .. math::
#             G_{ss} = x_c^2 G_{cc} + x_a^2 G_{aa} + x_c x_a (G_{ca} + G_{ac})

#         Salt-other interactions :math:`G_{si}` are computed as:

#         .. math::
#             G_{si} = x_c G_{ic} + x_a G_{ia}

#         where:
#             * :math:`x_c = \frac{N_c}{N_c + N_a}` is the mole fraction of the cation
#             * :math:`x_a = \frac{N_a}{N_c + N_a}` is the mole fraction of the anion
#             * :math:`G_{ij}` are the raw KBIs between molecule types :math:`i` and :math:`j`


#         See Also
#         --------
#         :class:`~kbkit.analysis.integrator.KBIntegrator` : For derivation and detailed formulas for RDF integration and KBI corrections.
#         """
#         # calculate kbis for each unique molecule in topology file
#         kbis = self._compute_raw_kbi_matrix()
#         # check if any electrolytes are present
#         electrolyte_chk = True if any("." in x for x in self.state.unique_molecules) else False
#         # correct for electrolytes if present
#         if apply_electrolyte_correction and electrolyte_chk:
#             return self._compute_electrolyte_corrected_kbi_matrix(kbis)
#         # return raw kbi matrix if electrolytes are detected but user doesn't want corrected kbi matrix
#         else:
#             return kbis

#     def _compute_raw_kbi_matrix(self) -> NDArray[np.float64]:
#         r"""
#         Compute the raw KBI matrix for all systems.

#         Returns
#         -------
#         np.ndarray
#             A 3D matrix of KBIs with shape ``(n_sys, n_mols, n_mols)``, where:
#             ``n_sys`` is the number of systems and``n_mols`` is the number of unique molecules.
#         """
#         kbis = np.full(
#             (self.state.n_sys, len(self.state.top_molecules), len(self.state.top_molecules)), fill_value=np.nan
#         )

#         # iterate through all systems
#         for s, meta in enumerate(self.state.config.registry):
#             # if rdf dir not in system, skip
#             if not meta.has_rdf():
#                 continue

#             # get all rdf files present
#             file_res = FileResolver(filepath=meta.rdf_path)
#             rdf_files = file_res.get_all(role="rdf")

#             # read all rdf_files
#             for filepath in rdf_files:
#                 # integrate rdf --> kbi calc
#                 integrator = KBIntegrator(
#                     rdf_file=filepath,
#                     system_properties=meta.props,
#                     use_fixed_rmin=self.use_fixed_r,
#                     convergence_threshold=self.rdf_convergence_threshold,
#                     correct_rdf_convergence=self.correct_rdf_convergence,
#                     apply_damping=self.apply_damping,
#                     extrapolate_thermodynamic_limit=self.extrapolate_thermodynamic_limit,
#                 )

#                 # get molecules present in rdf
#                 mol_i, mol_j = integrator.rdf_molecules

#                 # get molecule indices
#                 i = self.state._get_mol_idx(mol_i, self.state.top_molecules)
#                 j = self.state._get_mol_idx(mol_j, self.state.top_molecules)

#                 # if convergence is met, store kbi value
#                 if integrator.rdf.is_converged:
#                     kbis[s, i, j] = integrator.compute_kbi(mol_j=mol_j)
#                     kbis[s, j, i] = integrator.compute_kbi(mol_j=mol_i)
#                 # override convergence check to skip system if not converged
#                 else:  # for not converged rdf
#                     msg = f"RDF for system '{meta.name}' and pair {integrator.rdf_molecules} did not converge."
#                     if self.ignore_convergence_errors:
#                         print(f"WARNING: {msg} Skipping this system.")
#                         continue
#                     else:
#                         raise RuntimeError(msg)

#                 # add values to metadata
#                 self._populate_kbi_metadata(system=meta.name, integrator=integrator)

#         return kbis

#     def _populate_kbi_metadata(self, system: str, integrator: KBIntegrator) -> None:
#         r"""
#         Populate KBI metadata dictionary with integration results for a given RDF file.

#         Stores both raw and corrected KBI values, including:

#         * :math:`r` — radial distances
#         * :math:`g(r)` — RDF values
#         * :math:`G(r)` — cumulative KBI curve
#         * :math:`\lambda(r)` — finite-size correction factor
#         * :math:`\lambda(r) \cdot G(r)` — corrected KBI curve
#         * :math:`G_{\infty}` — extrapolated KBI at infinite dilution

#         Parameters
#         ----------
#         system : str
#             Name of the system being processed.
#         integrator : KBIntegrator
#             Integrator object containing RDF and KBI data.
#         """
#         self.kbi_metadata.setdefault(system, []).append(
#             KBIMetadata(
#                 mols=tuple(integrator.rdf_molecules),
#                 r=integrator.rdf.r,
#                 g=integrator.rdf.g,
#                 rkbi=(integrator.rkbi()),
#                 scaled_rkbi=(integrator.scaled_rkbi()),
#                 r_fit=(rfit := integrator.rdf.r_fit),
#                 scaled_rkbi_fit=integrator.scaled_rkbi_fit(),
#                 scaled_rkbi_est=np.polyval(integrator.fit_limit_params(), rfit),
#                 kbi_limit=integrator.compute_kbi(),
#             )
#         )

#     def get_metadata(self, system: str, mol_pair: tuple[str, str]) -> KBIMetadata | None:
#         """
#         Retrieve metadata for a specific system and molecular pair.

#         Parameters
#         ----------
#         system: str
#             System name.
#         mol_pair: tuple[str, str]
#             Molecule pair.

#         Returns
#         -------
#         KBIMetadata or None
#             Metadata object if found.
#         """
#         for meta in self.kbi_metadata.get(system, []):
#             if set(meta.mols) == set(mol_pair):
#                 return meta
#         return None

#     def _compute_electrolyte_corrected_kbi_matrix(self, kbi_matrix) -> NDArray[np.float64]:
#         r"""
#         Apply electrolyte correction to the input KBI matrix.

#         This method modifies the KBI matrix to account for salt-salt and salt-other interactions
#         using mole fraction-weighted combinations of cation and anion contributions.

#         Parameters
#         ----------
#         kbi_matrix: np.ndarray
#             Input KBI matrix to be corrected for electrolytes.

#         Returns
#         -------
#         np.ndarray
#             Corrected KBI matrix with additional rows/columns for salt interactions.
#         """
#         # This method first computes the raw matrix then corrects it
#         salt_pairs = self.state.salt_pairs
#         top_molecules = self.state.top_molecules
#         unique_molecules = self.state.unique_molecules
#         nosalt_molecules = self.state._nosalt_molecules
#         molecule_counts = self.state.molecule_counts

#         # if no salt pairs detected return original matrix
#         if len(salt_pairs) == 0:
#             return kbi_matrix

#         # create new kbi-matrix
#         adj = len(salt_pairs) - len(top_molecules)
#         kbi_el = np.full((self.state.n_sys, self.state.n_comp + adj, self.state.n_comp + adj), fill_value=np.nan)

#         for cat, an in salt_pairs:
#             # get index of anion and cation in topology molecules
#             cat_idx = top_molecules.index(cat)
#             an_idx = top_molecules.index(an)

#             # mol fraction of anion/cation in anion-cation pair
#             x_cat = molecule_counts[:, cat_idx] / (molecule_counts[:, cat_idx] + molecule_counts[:, an_idx])
#             x_an = molecule_counts[:, an_idx] / (molecule_counts[:, cat_idx] + molecule_counts[:, an_idx])

#             # for salt-salt interactions add to kbi-matrix
#             salt_idx = next(
#                 (i for i, val in enumerate(unique_molecules) if val in {f"{cat}-{an}", f"{an}-{cat}"}),
#                 -1,  # default if not found
#             )

#             if salt_idx == -1:
#                 raise ValueError(f"Neither f'{cat}-{an}' nor f'{an}-{cat}' found in unique_molecules.")

#             # calculate KBI for salt-salt pairs
#             kbi_el[salt_idx, salt_idx] = (
#                 x_cat**2 * kbi_matrix[cat_idx, cat_idx]
#                 + x_an**2 * kbi_matrix[an_idx, an_idx]
#                 + x_cat * x_an * (kbi_matrix[cat_idx, an_idx] + kbi_matrix[an_idx, cat_idx])
#             )

#             # for salt-other interactions
#             for m1, mol1 in enumerate(nosalt_molecules):
#                 m1j = top_molecules.index(mol1)
#                 for m2, mol2 in enumerate(nosalt_molecules):
#                     m2j = top_molecules.index(mol2)
#                     kbi_el[m1, m2] = kbi_matrix[m1j, m2j]
#                 # adjusted KBI for mol-salt interactions
#                 kbi_el[m1, salt_idx] = x_cat * kbi_matrix[m1, cat_idx] + x_an * kbi_matrix[m1, salt_idx]
#                 kbi_el[salt_idx, m1] = x_cat * kbi_matrix[cat_idx, m1] + x_an * kbi_matrix[an_idx, m1]

#         return kbi_el
    
# #####
# """
# Represent the thermodynamic state of a multicomponent mixture at fixed temperature, providing all metadata and simulation-derived properties required for Kirkwood-Buff analysis.

# `SystemState` aggregates species identities, compositions, densities, and concentration-dependent metadata in a consistent, queryable structure.
# It also exposes mixture properties computed directly from simulation via the :class:`~kbkit.system.SystemConfig` object.
# These properties are derived from structure (.gro) or energy (.edr) files and processed through :class:`~kbkit.systems.properties.SystemProperties`.

# The class enforces internal consistency between mole fractions, densities, and derived quantities (e.g., molar concentrations), ensuring that all downstream thermodynamic calculations operate on a coherent and validated state description.

# Notes
# -----
#     * `SystemState` does not perform thermodynamic calculations itself; it provides validated state information and simulation-derived properties to components such as `KBICalculator` and `KBThermo`.
#     * All arrays and properties follow a consistent species ordering to ensure reproducibility across workflows.
#     * Designed to support automated mixture sweeps, concentration series, and multicomponent KB analyses.

# .. note::
#     For mixing enthalpy and excess molar volume calculations, pure-component systems must be supplied during :class:`~kbkit.schema.system_config.SystemConfig` initialization for each molecule type present in the simulation.
# """

# import itertools
# from functools import cached_property

# import numpy as np
# from numpy.typing import NDArray

# from kbkit.config.unit_registry import load_unit_registry
# from kbkit.schema.system_config import SystemConfig
# from kbkit.schema.thermo_property import ThermoProperty, register_property


# class SystemState:
#     """
#     The `SystemState` consumes a `SystemConfig` object and provides tools for inspecting tabulated properties as a function of composition.

#     Parameters
#     ----------
#     config: SystemConfig
#         System configuration for a set of systems.


#     Attributes
#     ----------
#     ureg: UnitRegistry
#         Pint unit registry.
#     Q_: UnitRegistry.Quantity
#         Pint quantity object for unit conversions.
#     """

#     def __init__(self, config: SystemConfig) -> None:
#         # setup config
#         self.config = config

#         # set up unit registry
#         self.ureg = load_unit_registry()
#         self.Q_ = self.ureg.Quantity

#     @property
#     def top_molecules(self) -> list[str]:
#         """list[str]: Unique molecules in topology files."""
#         return self.config.molecules

#     @property
#     def n_sys(self) -> int:
#         """int: Number of systems present."""
#         return len(self.config.registry)

#     @cached_property
#     def salt_pairs(self) -> list[tuple[str, str]]:
#         """list[tuple[str, str]]: List of salt pairs as (cation, anion) tuples."""
#         # get unique combination of anions/cations in configuration
#         salt_pairs = [(cation, anion) for cation, anion in itertools.product(self.config.cations, self.config.anions)]

#         # now validate list; checks molecules in pairs are in _top_molecules
#         for pair in salt_pairs:
#             if not all(mol in self.top_molecules for mol in pair):
#                 raise ValueError(
#                     f"Salt pair {pair} contains molecules not present in top molecules: {self.top_molecules}"
#                 )
#         return salt_pairs

#     @cached_property
#     def _nosalt_molecules(self) -> list[str]:
#         """list[str]: Molecules not part of any salt pair."""
#         paired = {mol for pair in self.salt_pairs for mol in pair}
#         return [mol for mol in self.top_molecules if mol not in paired]

#     @cached_property
#     def _salt_molecules(self) -> list[str]:
#         """list[str]: Combined molecule names for each salt pair."""
#         return [".".join(pair) for pair in self.salt_pairs]

#     @cached_property
#     def unique_molecules(self) -> list[str]:
#         """list[str]: Molecules present after combining salt pairs as single entries."""
#         return self._nosalt_molecules + self._salt_molecules

#     def _get_mol_idx(self, mol: str, molecule_list: list[str]) -> int:
#         """Get index of mol in molecule list."""
#         if not isinstance(molecule_list, list):
#             try:
#                 molecule_list = list(molecule_list)
#             except TypeError as e:
#                 raise TypeError(
#                     f"Molecule list could not be converted to type(list) from type({type(molecule_list)})"
#                 ) from e
#         if mol not in molecule_list:
#             raise ValueError(f"{mol} not in molecule list: {molecule_list}")
#         return molecule_list.index(mol)

#     @property
#     def n_comp(self) -> int:
#         """int: Total number of :meth:`unique_molecules`."""
#         return len(self.unique_molecules)

#     @cached_property
#     def total_molecules(self) -> NDArray[np.float64]:
#         """np.ndarray: Total number of molecules, :math:`N_T`, in each system."""
#         return np.array([meta.props.topology.total_molecules for meta in self.config.registry])

#     @cached_property
#     def molecule_info(self) -> dict[str, dict[str, int]]:
#         """dict: Number of molecules of each type in topology mapped to each system."""
#         return {meta.name: meta.props.topology.molecule_count for meta in self.config.registry}

#     @cached_property
#     def _top_molecule_counts(self) -> NDArray[np.float64]:
#         """np.ndarray: Molecule count per system."""
#         return np.array(
#             [
#                 [meta.props.topology.molecule_count.get(mol, 0) for mol in self.top_molecules]
#                 for meta in self.config.registry
#             ]
#         )

#     @cached_property
#     def molecule_counts(self) -> NDArray[np.float64]:
#         """np.ndarray: Molecule count per system, mapped to :meth:`unique_molecules`."""
#         counts = np.zeros((self.n_sys, self.n_comp))
#         for i, mol in enumerate(self.unique_molecules):
#             mol_split = mol.split(".")
#             if len(mol_split) > 1 and tuple(mol_split) in self.salt_pairs:
#                 for salt in mol_split:
#                     salt_idx = self._get_mol_idx(salt, self.top_molecules)
#                     counts[:, i] += self._top_molecule_counts[:, salt_idx]
#             else:
#                 mol_idx = self._get_mol_idx(mol, self.top_molecules)
#                 counts[:, i] += self._top_molecule_counts[:, mol_idx]
#         return counts

#     @cached_property
#     def pure_molecules(self) -> list[str]:
#         """list[str]: Names of molecules considered as pure components."""
#         molecules = [".".join(meta.props.topology.molecules) for meta in self.config.registry if meta.kind == "pure"]
#         return sorted(molecules)

#     @cached_property
#     def pure_mol_fr(self) -> NDArray[np.float64]:
#         """np.ndarray: Mol fraction array mapped to :meth:`pure_molecules`."""
#         arr = np.zeros((self.n_sys, len(self.pure_molecules)))
#         for i, mol in enumerate(self.pure_molecules):
#             mol_split = mol.split(".")
#             if len(mol_split) > 1:
#                 for salt in mol_split:
#                     salt_idx = self._get_mol_idx(salt, self.top_molecules)
#                     arr[:, i] += self._top_molecule_counts[:, salt_idx]
#             else:
#                 mol_idx = self._get_mol_idx(mol, self.top_molecules)
#                 arr[:, i] += self._top_molecule_counts[:, mol_idx]
#         # get mol_fr
#         arr /= self.total_molecules[:, np.newaxis]
#         return arr

#     @cached_property
#     def top_electron_map(self) -> dict[str, int]:
#         """dict[str, int]: Number of electrons mapped to each molecule type."""
#         uniq_elec_map: dict[str, int] = dict.fromkeys(self.top_molecules, 0)
#         for meta in self.config.registry:
#             mols = meta.props.topology.molecules
#             ecount = meta.props.topology.electron_count
#             for mol in mols:
#                 if uniq_elec_map[mol] == 0 and ecount.get(mol) is not None:
#                     uniq_elec_map[mol] = ecount.get(mol, 0)
#         return uniq_elec_map

#     @cached_property
#     def unique_electrons(self) -> NDArray[np.float64]:
#         r"""np.ndarray: Number of electrons, :math:`Z_i`, mapped to :meth:`unique_molecules`."""
#         elec_map: dict[str, float] = dict.fromkeys(self.unique_molecules, 0)
#         for mol_ls in self.unique_molecules:
#             mols = mol_ls.split(".")
#             elec_map[mol_ls] = sum([self.top_electron_map.get(mol, 0) for mol in mols])
#         elec_mapped = np.fromiter(elec_map.values(), dtype=np.float64)
#         if not all(elec_mapped > 0):
#             elec_mapped = np.full_like(self.unique_molecules, fill_value=np.nan, dtype=float)
#         return elec_mapped

#     @cached_property
#     def total_electrons(self) -> NDArray[np.float64]:
#         r"""np.ndarray: Linear combination of electron numbers and mol fractions, :math:`\bar{Z} = \sum_i x_i Z_i`, mapped to :meth:`unique_molecules`."""
#         return self.mol_fr @ self.unique_electrons

#     @cached_property
#     def mol_fr(self) -> NDArray[np.float64]:
#         """np.ndarray: Mol fraction of :meth:`unique_molecules` in registry."""
#         return self.molecule_counts / self.molecule_counts.sum(axis=1)[:, np.newaxis]

#     @register_property("temperature", "K")
#     def temperature(self) -> NDArray[np.float64]:
#         r"""Temperature, :math:`\left \langle T \right \rangle`, of each simulation.

#         Parameters
#         ----------
#         units: str
#             Temperature units (default: K)

#         Returns
#         -------
#         np.ndarray
#             1D temperature array as a function of composition.
#         """
#         return np.array([meta.props.get("temperature", units="K") for meta in self.config.registry])

#     @register_property("volume", "nm^3")
#     def volume(self) -> NDArray[np.float64]:
#         r"""Volume, :math:`\left \langle V \right \rangle`, of each simulation.

#         Parameters
#         ----------
#         units: str
#             Volume units (default: nm^3)

#         Returns
#         -------
#         np.ndarray
#             1D volume array as a function of composition.
#         """
#         return np.array([meta.props.get("volume", units="nm^3") for meta in self.config.registry])

#     @register_property("enthalpy", "kJ/mol")
#     def enthalpy(self) -> NDArray[np.float64]:
#         r"""Enthalpy, :math:`H`, of each simulation.

#         Parameters
#         ----------
#         units: str
#             Enthalpy units (default: kJ/mol)

#         Returns
#         -------
#         np.ndarray
#             1D array of system enthalpies as a function of composition.
#         """
#         return np.array([meta.props.get("enthalpy", units="kJ/mol") for meta in self.config.registry])

#     @register_property("heat_capacity", "kJ/mol/K")
#     def heat_capacity(self) -> NDArray[np.float64]:
#         r"""Heat capacity, :math:`c_p`, of each simulation.

#         Parameters
#         ----------
#         units: str
#             Heat capacity units (default: kJ/mol/K)

#         Returns
#         -------
#         np.ndarray
#             1D array of system heat capacities as a function of composition.
#         """
#         return np.array([meta.props.get("heat_capacity", units="kJ/mol/K") for meta in self.config.registry])

#     @register_property("isothermal_compressibility_md", "1/kPa")
#     def isothermal_compressibility(self) -> NDArray[np.float64]:
#         r"""Isothermal compressiblity, :math:`\kappa_T`, of each simulation.

#         Parameters
#         ----------
#         units: str
#             Isothermal compressiblity units (default: 1/kPa)

#         Returns
#         -------
#         np.ndarray
#             1D array of system isothermal compressiblities as a function of composition.
#         """
#         return np.array([meta.props.get("isothermal_compressibility", units="1/kPa") for meta in self.config.registry])

#     @register_property("pure_enthalpy", "kJ/mol")
#     def pure_enthalpy(self) -> NDArray[np.float64]:
#         """Pure component enthalpies, :math:`H_i`, mapped to :meth:`pure_molecules` array.

#         Parameters
#         ----------
#         units: str
#             Enthalpy units (default: kJ/mol)

#         Returns
#         -------
#         np.ndarray
#             1D array of enthalpies for pure components.
#         """
#         enth: dict[str, float] = dict.fromkeys(self.pure_molecules, 0)
#         for meta in self.config.registry:
#             if meta.kind == "pure":
#                 value = meta.props.get("enthalpy", units="kJ/mol", std=False)
#                 # make sure value is float
#                 if isinstance(value, tuple):
#                     value = value[0]
#                 mols = ".".join(meta.props.topology.molecules)
#                 enth[mols] = float(value)
#         return np.fromiter(enth.values(), dtype=np.float64)

#     @register_property("pure_enthalpy", "kJ/mol")
#     def ideal_enthalpy(self) -> NDArray[np.float64]:
#         r"""Ideal enthalpy, :math:`H^{id}`, as a function of composition.

#         Parameters
#         ----------
#         units: str
#             Enthalpy units (default: kJ/mol)

#         Returns
#         -------
#         np.ndarray
#             1D array of ideal enthalpies as a function of composition.

#         Notes
#         -----
#         Ideal enthalpy, :math:`H^{id}`, is calculated via:

#         .. math::
#             H^{id} = \sum_{i=1}^n x_i H_i

#         where:
#             - :math:`x_i` is mol fraction of molecule :math:`i`
#             - :math:`H_i` is the pure component enthalpy of molecule :math:`i`
#         """
#         return self.pure_mol_fr @ self.pure_enthalpy.to("kJ/mol")

#     @register_property("mixture_enthalpy", "kJ/mol")
#     def mixture_enthalpy(self) -> NDArray[np.float64]:
#         r"""Enthalpy of mixing, :math:`\Delta H_{mix}`, as a function of composition.

#         Parameters
#         ----------
#         units: str
#             Enthalpy units (default: kJ/mol)

#         Returns
#         -------
#         np.ndarray
#             1D array of mixing enthalpies as a function of composition.

#         Notes
#         -----
#         Mixing enthalpy, :math:`\Delta H_{mix}`, is calculated via:

#         .. math::
#             \Delta H_{mix} = H - H^{id}

#         where:
#             - :math:`H` is the simulation enthlapy for mixtures
#             - :math:`H^{id}` is ideal enthalpy
#         """
#         return self.enthalpy.to("kJ/mol") - self.ideal_enthalpy.to("kJ/mol")

#     def molar_volume_map(self, units: str = "cm^3/mol") -> dict[str, float]:
#         r"""Molar volumes, :math:`V_i`, of mapped to molecule name (for pure components).

#         Parameters
#         ----------
#         units: str
#             Molar volume units (default: cm^3/mol)

#         Returns
#         -------
#         dict[str, np.ndarray]
#             Dictionary mapping molar volumes to corresponding molecule
#         """
#         vol_unit, N_unit = units.split("/")
#         volumes = self.volume.to(vol_unit)
#         # make dict in same order as pure molecules
#         volumes_map: dict[str, float] = dict.fromkeys(self.pure_molecules, 0)
#         for i, meta in enumerate(self.config.registry):
#             top = meta.props.topology
#             # only for pure systems
#             if meta.kind == "pure":
#                 N = self.Q_(top.total_molecules, "molecule").to(N_unit).magnitude
#                 volumes_map[".".join(top.molecules)] = volumes[i] / N

#         return volumes_map

#     @register_property("pure_molar_volume", "cm^3/mol")
#     def pure_molar_volume(self) -> NDArray[np.float64]:
#         r"""Molar volumes, :math:`V_i`, of pure components.

#         Parameters
#         ----------
#         units: str
#             Molar volume units (default: cm^3/mol)

#         Returns
#         -------
#         np.ndarray
#             1D array for each unique molecule.
#         """
#         return np.fromiter(self.molar_volume_map("cm^3/mol").values(), dtype=np.float64)

#     @register_property("ideal_molar_volume", "cm^3/mol")
#     def ideal_molar_volume(self) -> NDArray[np.float64]:
#         r"""Ideal molar volume, :math:`\bar{V}`, of mixture.

#         Parameters
#         ----------
#         units: str
#             Molar volume units (default: cm^3/mol)

#         Returns
#         -------
#         np.ndarray
#             1D array of molar volumes as a function of composition.

#         Notes
#         -----
#         Ideal molar volume, :math: `\bar{V}`, is calculated according to:

#         .. math::
#             \bar{V} = \sum_i x_i V_i

#         where:
#             - :math:`x_i` is the mole fraction of component `i`
#             - :math:`V_i` is the molar volume of component `i`
#         """
#         return self.pure_mol_fr @ self.pure_molar_volume.to("cm^3/mol")

#     @register_property("mixture_molar_volume", "cm^3/mol")
#     def mixture_molar_volume(self) -> NDArray[np.float64]:
#         r"""Mixture molar volume, :math:`\Delta V_{mix}`.

#         Parameters
#         ----------
#         units: str
#             Molar volume units (default: cm^3/mol)

#         Returns
#         -------
#         np.ndarray
#             1D array of molar volumes as a function of composition.

#         Notes
#         -----
#         Mixture molar volume, :math:`\Delta V_{mix}`, is calculated via:

#         .. math::
#             \Delta V_{mix} = \frac{\left \langle V \right \rangle}{N_T}

#         where:
#             - :math:`\left \langle V \right \rangle` is the ensemble average volume
#             - :math:`N_T` is total number of molecules present
#         """
#         volumes = self.volume.to("cm^3")
#         molecs = self.Q_(self.total_molecules, "molecule").to("mol").magnitude
#         return np.asarray(volumes / molecs, dtype=np.float64)

#     @register_property("excess_molar_volume", "cm^3/mol")
#     def excess_molar_volume(self) -> NDArray[np.float64]:
#         r"""Excess molar volume, :math:`V^{ex}`.

#         Parameters
#         ----------
#         units: str
#             Molar volume units (default: nm^3/molecule)

#         Returns
#         -------
#         np.ndarray
#             1D array of molar volumes as a function of composition.

#         Notes
#         -----
#         Excess molar volume, :math:`V^{ex}`, is calculated via:

#         .. math::
#             V^{ex} = \Delta V_{mix} - \bar{V}

#         where:
#             - :math:`\Delta V_{mix}` is the mixture molar volume
#             - :math:`\bar{V}` is ideal molar volume
#         """
#         return self.mixture_molar_volume.to("cm^3/mol") - self.ideal_molar_volume.to("cm^3/mol")

#     @register_property("mixture_number_density", "molecule/nm^3")
#     def mixture_number_density(self) -> NDArray[np.float64]:
#         r"""Mixture number density, :math:`\rho`.

#         Parameters
#         ----------
#         units: str
#             Number density units (default: molecule/nm^3)

#         Returns
#         -------
#         np.ndarray
#             1D array of number densities as a function of composition.

#         Notes
#         -----
#         Mixture number density, :math:`\rho`, is calculated via:

#         .. math::
#             \rho = \frac{N_T}{\left \langle V \right \rangle}

#         where:
#             - :math:`\left \langle V \right \rangle` is the ensemble average volume
#             - :math:`N_T` is total number of molecules present
#         """
#         volumes = self.volume.to("nm^3")
#         return np.asarray(self.total_molecules / volumes, dtype=np.float64)

#     def computed_properties(self) -> dict[str, ThermoProperty]:
#         """
#         Collects all computed properties from molecular dynamics for current set of systems.

#         Returns
#         -------
#         List[ThermoProperty]
#             A list of `ThermoProperty` instances, containing the name, value, and units of the
#             computed property from current set of systems. The units are corresponding to GROMACS
#             default units.
#         """
#         return {
#             "top_molecules": ThermoProperty(name="top_molecules", value=self.top_molecules, units=""),
#             "salt_pairs": ThermoProperty(name="salt_pairs", value=self.salt_pairs, units=""),
#             "unique_molecules": ThermoProperty(name="unique_molecules", value=self.unique_molecules, units=""),
#             "total_molecules": ThermoProperty(name="total_molecules", value=self.total_molecules, units="molecule"),
#             "molecule_info": ThermoProperty(name="molecule_info", value=self.molecule_info, units=""),
#             "molecule_counts": ThermoProperty(name="molecule_counts", value=self.molecule_counts, units="molecule"),
#             "pure_molecules": ThermoProperty(name="pure_molecules", value=self.pure_molecules, units=""),
#             "pure_mol_fr": ThermoProperty(name="pure_mol_fr", value=self.pure_mol_fr, units=""),
#             "electron_map": ThermoProperty(name="electron_map", value=self.top_electron_map, units="electron/molecule"),
#             "unique_electrons": ThermoProperty(
#                 name="unique_electrons", value=self.unique_electrons, units="electron/molecule"
#             ),
#             "total_electrons": ThermoProperty(
#                 name="total_electrons", value=self.total_electrons, units="electron/molecule"
#             ),
#             "mol_fr": ThermoProperty(name="mol_fr", value=self.mol_fr, units=""),
#             "molar_volume_map": ThermoProperty(
#                 name="molar_volume_map", value=self.molar_volume_map("cm^3/mol"), units="cm^3/mol"
#             ),
#         }
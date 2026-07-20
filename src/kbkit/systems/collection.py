"""
Container for a set of systems for a given thermodynamic state (e.g., constant temperature, function of composition).

The purpose of `SystemCollection` is to load a set of systems and access :class:`~kbkit.systems.properties.SystemProperties` to retrieve molecular dynamics properties as a function of composition.
    * This container first discovers molecular systems based on directory structure and input parameters, creating a list of :class:`~kbkit.schema.system_metadata.SystemMetadata` objects.
    * Then topology and energy properties can be calculated as function of composition.
    * Additionally, this object is used to calculating `Excess`, `Simulation`, and `Ideal` properties.
"""

import itertools
from collections import defaultdict
from functools import cached_property
from pathlib import Path
from typing import Any, Literal

import numpy as np

from kbkit.schema.property_result import PropertyResult
from kbkit.schema.system_metadata import SystemMetadata
from kbkit.systems.properties import SystemProperties
from kbkit.utils.decorators import cached_property_value
from kbkit.utils.format import ENERGY_ALIASES, resolve_attr_key
from kbkit.utils.validation import validate_path
from kbkit.visualization.timeseries import TimeseriesPlotter


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
    charges: dict[str, int], optional
        Optional charge dictionary for ions. If provided, enables electrolyte basis.
    """

    def __init__(
        self, systems: list["SystemMetadata"], molecules: list[str], charges: dict[str, int] | None = None
    ) -> None:
        self._systems = self._sort_systems(systems=systems, molecules=molecules)
        self._residue_molecules = molecules  # Global unique molecules used for sorting
        self._lookup = {s.name: s for s in self._systems}
        self._cache: dict[tuple, PropertyResult] = {}
        # user-provided charges; if None or empty -> neutral behavior
        self.charges: dict[str, int] = charges or {}
        self.system_names = list(
            self._lookup.keys()
        )  # just get list of system names, without iterating through objects

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

    def __len__(self) -> int:
        """Allows len(SystemCollection) to return num systems in registry."""
        return len(self._systems)

    def __iter__(self):
        """Creates an iterable type object."""
        return iter(self._systems)

    @classmethod
    def load(
        cls,
        base_path: str | None = None,
        base_systems: list[str] | None = None,
        pure_path: str | None = None,
        pure_systems: list[str] | None = None,
        rdf_dir: str = "",
        start: int = 10000,
        include_mode: str = "npt",
        charges: dict[str, int] | None = None,
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
        start : int, optional
            Start time for time-averaged properties.
        include_mode: str, optional
            Optional string to filter energy and topology files, if multiple are found of a given type.
        charges: dict[str, int], optional
            Optional charge dictionary for ions.

        Returns
        -------
        SystemCollection
            Registry object containing global molecules and list of :class:`~kbkit.schema.system_metadata.SystemMetadata`.
        """
        # validate paths
        valid_base_path = validate_path(base_path or Path(".").resolve())
        valid_pure_path = validate_path(pure_path or Path(".").resolve())

        # Resolve Mixture (Base) Systems
        if base_systems is not None:
            mixture_dirs = [valid_base_path / s for s in base_systems if cls._is_valid(valid_base_path / s)]
        else:
            mixture_dirs = [f for f in valid_base_path.iterdir() if cls._is_valid(f)]

        # Now repeat for pure systems
        pure_dirs = []
        if pure_systems is not None:
            for name in pure_systems:
                match = (
                    next((f for f in valid_pure_path.iterdir() if f.name == name), None) if valid_pure_path else None
                ) or next((f for f in mixture_dirs if f.name == name), None)
                if match:
                    pure_dirs.append(match)

        # Build Metadata (Finding RDF path before instantiation)
        meta_objects = []
        found_pure_paths = {p.resolve() for p in pure_dirs}

        # Create Pure Metadata
        for p in pure_dirs:
            r_path = cls._resolve_rdf_path(p, rdf_dir, is_pure=True)
            meta_objects.append(cls._make_meta(p, kind="pure", rdf_path=r_path, start=start, include=include_mode))

        # Create Mixture Metadata
        ordered_mols = set()
        for p in mixture_dirs:
            if p.resolve() not in found_pure_paths:
                r_path = cls._resolve_rdf_path(p, rdf_dir, is_pure=False)
                meta_p = cls._make_meta(p, kind="mixture", rdf_path=r_path, start=start, include=include_mode)
                meta_objects.append(meta_p)

                mols_present = meta_p.props.topology.molecules
                for mol in mols_present:
                    ordered_mols.add(mol)

        return cls(meta_objects, ordered_mols, charges=charges)

    # --- Setting up files/systems for system metadata ---

    @staticmethod
    def _is_valid(path: Path, deep: bool = False) -> bool:
        """Check if systems are valid; requires it to be a directory and contains the necessary GROMACS output files."""
        pattern = "**/*" if deep else "*"
        return (
            path.is_dir()
            and (
                any(path.glob(f"{pattern}.edr"))
                or any(path.glob(f"{pattern}.log"))
                or any(path.glob(f"{pattern}.lammps"))
            )
            and (
                any(path.glob(f"{pattern}.gro")) or any(path.glob(f"{pattern}.top")) or any(path.glob(f"{pattern}.lmp"))
            )
        )

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
            name=path.name, kind=kind, path=path, rdf_path=rdf_path, props=SystemProperties(str(path), **props_kwargs)
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

    # --- electrolyte helpers ---

    def _validate_charges(self) -> None:
        """Ensure all charged species exist in residue_molecules."""
        for ion in self.charges:
            if ion not in self._residue_molecules:
                raise ValueError(
                    f"Charge declared for '{ion}', but it is not in residue_molecules: {self._residue_molecules}"
                )

    def _build_salt_pairs(self) -> list[tuple[str, str]]:
        """Return list of (cation, anion) pairs based on charges."""
        cations = [ion for ion, q in self.charges.items() if q > 0]
        anions = [ion for ion, q in self.charges.items() if q < 0]
        if not cations and not anions:
            return []
        return [(c, a) for c, a in itertools.product(cations, anions)]

    def _build_nu_matrix(self, salt_pairs: list[tuple[str, str]]) -> np.ndarray:
        """Build stoichiometric matrix nu (residue_molecules x nsalts)."""
        nmol = len(self._residue_molecules)
        nsalts = len(salt_pairs)
        nu = np.zeros((nmol, nsalts))

        for i, (cat, an) in enumerate(salt_pairs):
            try:
                cat_idx = list(self._residue_molecules).index(cat)
                an_idx = list(self._residue_molecules).index(an)
            except ValueError as e:
                raise ValueError(f"Salt component '{cat}' or '{an}' not found in residue_molecules.") from e

            q_cat = self.charges[cat]
            q_an = self.charges[an]
            if q_cat <= 0 or q_an >= 0:
                raise ValueError(
                    f"Inconsistent charges for salt pair ({cat}, {an}): "
                    f"q_cat={q_cat}, q_an={q_an}. Expected cation>0, anion<0."
                )

            nu[cat_idx, i] = abs(q_an)
            nu[an_idx, i] = abs(q_cat)

        return nu

    def _solve_salt_counts(self, nu: np.ndarray, N: np.ndarray) -> np.ndarray:
        """Solve for salt counts for each system given nu and residue counts N."""
        if nu.shape[1] == 0:
            return np.zeros((N.shape[0], 0))

        salt_counts = np.linalg.lstsq(nu, N.T, rcond=None)[0].T
        salt_counts[salt_counts < 0] = 0.0
        return salt_counts

    def _canonical_salt_names(self, salt_pairs: list[tuple[str, str]], nu: np.ndarray) -> list[str]:
        """Build canonical salt names like: - Na.Cl - Ca.Cl2."""
        names: list[str] = []
        for col_idx, (c, a) in enumerate(salt_pairs):
            c_idx = list(self._residue_molecules).index(c)
            a_idx = list(self._residue_molecules).index(a)
            n_c = int(nu[c_idx, col_idx])
            n_a = int(nu[a_idx, col_idx])
            # we encode stoichiometry on anion side: Ca.Cl2, Na.Cl
            c_part = c if n_c == 1 else f"{c}{n_c}"
            a_part = a if n_a == 1 else f"{a}{n_a}"
            names.append(f"{c_part}.{a_part}")
        return names

    # ---------- Basis accessors ----------

    @property
    def residue_molecules(self) -> list[str]:
        """Raw MD residue basis (unique residues from topology)."""
        return self._residue_molecules

    @cached_property
    def residue_counts(self) -> np.ndarray:
        """np.ndarray: (N_systems, N_residues) mole fractions in residue basis."""
        # return self.x * self.total_molecules[:, np.newaxis]
        data = []
        for s in self._systems:
            counts = s.props.topology.molecule_count
            total = s.props.topology.total_molecules
            row = [counts.get(m, 0) if total > 0 else 0.0 for m in self._residue_molecules]
            data.append(row)
        return np.array(data)

    @cached_property
    def residue_x(self) -> np.ndarray:
        """np.ndarray: (N_systems, N_residues) mole fractions in residue basis."""
        data = []
        for s in self._systems:
            counts = s.props.topology.molecule_count
            total = s.props.topology.total_molecules
            row = [counts.get(m, 0) / total if total > 0 else 0.0 for m in self._residue_molecules]
            data.append(row)
        return np.array(data)

    @cached_property
    def electrolyte_basis(self) -> dict[str, np.ndarray]:
        """Build electrolyte basis.

        - new_molecules: neutral molecules + salts.
        - new_N: counts in new basis.
        - new_x: mole fractions in new basis.
        - nu: stoichiometric matrix (residue x salts) Returns None if no charges.
        """
        if not self.charges:
            return {}

        self._validate_charges()
        salt_pairs = self._build_salt_pairs()
        if not salt_pairs:
            return {
                "molecules": np.array(self._residue_molecules),
                "N": self.residue_counts,
                "x": self.residue_x,
                "nu": np.zeros((len(self._residue_molecules), 0)),
            }

        nu = self._build_nu_matrix(salt_pairs)
        N: np.ndarray = (self.residue_x).astype(float)

        neutral_mask = np.all(nu == 0, axis=1)
        salt_counts = self._solve_salt_counts(nu, N)

        neutral_counts = N[:, neutral_mask]
        new_N = np.column_stack((neutral_counts, salt_counts))

        totals = new_N.sum(axis=1)[:, np.newaxis]
        if np.any(totals == 0):
            raise ValueError("At least one system has total count zero after salt reconstruction.")
        new_x = new_N / totals

        neutral_names = list(np.array(list(self._residue_molecules))[neutral_mask])
        salt_names = list(self._canonical_salt_names(salt_pairs, nu))
        new_molecules = neutral_names + salt_names

        return {"molecules": np.array(new_molecules), "N": new_N, "x": new_x, "nu": nu}

    @property
    def electrolyte_molecules(self) -> list[str]:
        """List of molecule names for electrolyte basis (neutral molecules + salts)."""
        if not self.charges:
            raise ValueError("No charges provided; electrolyte basis unavailable.")
        assert self.electrolyte_basis is not None
        return list(self.electrolyte_basis["molecules"])

    @property
    def electrolyte_x(self) -> np.ndarray:
        """Mole fractions for electrolyte basis."""
        if not self.charges:
            raise ValueError("No charges provided; electrolyte basis unavailable.")
        assert self.electrolyte_basis is not None
        return self.electrolyte_basis["x"]

    @property
    def nu(self) -> np.ndarray:
        """Stoichiometric matrix (residue basis x salts) if charges provided."""
        if not self.charges:
            raise ValueError("No charges provided; stoichiometric matrix unavailable.")
        assert self.electrolyte_basis is not None
        return self.electrolyte_basis["nu"]

    # --- user-facing basis (switches on charges) ---

    @property
    def molecules(self) -> list[str]:
        """list[str]: The global order of molecules used for vectorized properties."""
        return self.electrolyte_molecules if self.charges else self.residue_molecules

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
        return self.electrolyte_x if self.charges else self.residue_x

    @cached_property
    def units(self) -> dict[str, str]:
        """dict[str, str]: Master dictionary mapping energy properties to their default units."""
        unit_dic: dict[str, str] = defaultdict(str)
        for meta in self._systems:
            meta_units = meta.props.get("units")
            if isinstance(meta_units, dict):
                unit_dic.update(meta_units)
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
        """Get default units for a given energy property.

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
    ) -> np.ndarray | list:
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

    def _get_from_cache(self, key: tuple, target_units: str):
        """Check cache and return converted result if found."""
        if key in self._cache:
            return self._cache[key].to(target_units)
        return None

    def has_all_required_pures(self) -> bool:
        """Check that collection has required pure components for excess properties calculation."""
        return True if len(self.pures) == len(self.molecules) else False

    @cached_property_value()
    def simulated_property(self, name: str, units: str | None = None, avg: bool = True) -> np.ndarray:
        """
        Extract raw values directly from MD simulation (EDR files).

        Returns
        -------
        np.ndarray
            Values as simulated in the MD engine.
        """
        units = units or self.get_units(name)
        return np.asarray(self.get(name, units=units, avg=avg))

    @cached_property_value()
    def pure_property(self, name: str, units: str | None = None, avg: bool = True) -> np.ndarray:
        """
        Extract pure component properties.

        Parameters
        ----------
        name : str
            Property name (e.g., 'Density', 'Volume').
        units : str, optional
            Target units for conversion.
        avg : bool, default True
            Return time-averaged values.

        Returns
        -------
        np.ndarray
            Pure component property values with metadata.
        """
        units = units or self.get_units(name)

        pure_dict = self._build_pure_lookup(name, units, avg)
        values = np.full(len(self.molecules), fill_value=np.nan)
        for i, mol in enumerate(self.molecules):
            try:
                values[i] = pure_dict[mol]
            except KeyError:
                continue
        return values

    @cached_property_value()
    def ideal_property(
        self,
        name: str,
        mixing_rule: Literal["linear", "volume_weighted"] = "linear",
        units: str | None = None,
        avg: bool = True,
    ) -> np.ndarray:
        r"""
        Calculate ideal mixing property using specified mixing rule.

        Linear mixing rule:

        .. math::
            \bar{P} = \sum_i x_i P_i^{pure}

        Volume-weighted mixing rule:

        .. math::
            \bar{P} = \sum_i \left(\frac{x_i}{P_i^{pure}} \right)^{-1}

        where:
            - :math:`x_i` is the mole fraction of molecule :math:`i`
            - :math:`P_i` is the pure component property
            - :math:`\bar{P}` is the ideal property according to the mixing rule

        Parameters
        ----------
        name : str
            Property name.
        mixing_rule : {"linear", "volume_weighted"}, default "linear"
            Mixing rule to apply.
        units : str, optional
            Target units.
        avg : bool, default True
            Use time-averaged values.

        Returns
        -------
        np.ndarray
            Ideal property values for each mixture composition.
        """
        units = units or self.get_units(name)
        pure_res = self.pure_property(name=name, units=units, avg=avg)
        compositions = self.x

        if "lin" in mixing_rule.lower():
            ideal_values = compositions @ pure_res
        elif "vol" in mixing_rule.lower():
            ideal_values = 1.0 / (compositions @ (1.0 / pure_res))
        else:
            raise ValueError(f"Unknown mixing rule: {mixing_rule}")

        return ideal_values

    @cached_property_value()
    def excess_property(
        self,
        name: str,
        mixing_rule: Literal["linear", "volume_weighted"] = "linear",
        units: str | None = None,
        avg: bool = True,
    ) -> np.ndarray:
        r"""
        Calculate excess property: Excess = Real - Ideal.

        Parameters
        ----------
        name : str
            Property name.
        mixing_rule : {"linear", "volume_weighted"}, default "linear"
            Mixing rule for ideal calculation.
        units : str, optional
            Target units.
        avg : bool, default True
            Use time-averaged values.

        Returns
        -------
        np.ndarray
            Excess property values.

        Notes
        -----
        For a given property, :math:`P`, the excess property, :math:`P^{E}`, is calculated according to:

        .. math::
            P^{E} = P - \bar{P}

        where:
            - :math:`x_i` is the mole fraction of molecule :math:`i`
            - :math:`P` is the property directly from simulation
            - :math:`\bar{P}` is the ideal property according to the mixing rule
        """
        units = units or self.get_units(name)
        sim_res = self.simulated_property(name=name, units=units, avg=avg)
        ideal_res = self.ideal_property(name=name, units=units, mixing_rule=mixing_rule, avg=avg)
        return sim_res - ideal_res

    @cached_property
    def results(self) -> dict[str, PropertyResult]:
        """Dictionary of :class:`~kbkit.schema.property_result.PropertyResult` with mapped names and values.

        Returns
        -------
        dict[str, PropertyResult]
            Mapped property result objects for properties.
        """

        def add_property(name: str, units: str | None = None) -> dict[str, PropertyResult]:
            """Compute simulated, ideal, and excess PropertyResult objects for a given property."""
            values = {
                "simulated": self.simulated_property(name=name, units=units, avg=True),
                "ideal": self.ideal_property(name=name, units=units, avg=True),
                "excess": self.excess_property(name=name, units=units, avg=True),
            }

            prop_res = {}
            for ptype, val in values.items():
                key = f"{ptype}_{prop.lower().replace('-', '_')}"
                prop_res[key] = PropertyResult(name=key, value=val, units=units, property_type=ptype)

            return prop_res

        results = {
            "molecules": PropertyResult(name="molecules", value=np.asarray(self.molecules)),
            "n_i": PropertyResult(name="n_i", value=np.asarray(self.n_i)),
            "n_sys": PropertyResult(name="n_sys", value=np.asarray(self.n_sys)),
            "x": PropertyResult(name="x", value=self.x),
        }
        for prop, units in self.units.items():
            if ("time" in prop.lower()) or ("step" in prop.lower()):
                continue
            results.update(add_property(prop, units))

        return results

    def _build_pure_lookup(
        self, name: str, units: str | None = None, avg: bool = True
    ) -> dict[str, float | np.ndarray | list[np.ndarray]]:
        r"""
        Build a lookup dictionary mapping molecule names to pure property values.

        For electrolytes, a pure system may contain multiple residues but must reduce to exactly one component (neutral or salt) under the electrolyte basis.

        Parameters
        ----------
        name : str
            Property name.
        units : str, optional
            Target units.
        avg : bool, default True
            Use time-averaged values.

        Returns
        -------
        dict[str, float]
            Mapping of molecule name to pure property value.
        """
        pure_lookup: dict[str, Any] = {}
        for pure_sys in self.pures:
            mol_counts = pure_sys.props.topology.molecule_count
            residue_names = list(mol_counts.keys())

            if self.charges:
                # electrolyte-aware reduction
                # reuse internal helpers on a per-system basis
                # build a temporary salt composition for this pure system
                temp_collection = SystemCollection(
                    systems=[pure_sys],
                    molecules=residue_names,
                    charges=self.charges,
                )
                basis = temp_collection.electrolyte_basis
                assert basis is not None
                new_molecules = basis["molecules"]
                if len(new_molecules) != 1:
                    raise ValueError(
                        f"Pure system {pure_sys.name} does not reduce to a single component in electrolyte basis: "
                        f"{new_molecules}"
                    )
                comp_name = str(new_molecules[0])
            else:
                # neutral case: must be a single residue
                if len(mol_counts) != 1:
                    raise ValueError(f"Pure system {pure_sys.name} contains multiple molecules: {mol_counts}")
                comp_name = str(residue_names[0])

            pure_value = pure_sys.props.get(name, units=units, avg=avg)
            if isinstance(pure_value, dict):
                pure_value = pure_value.get(comp_name, next(iter(pure_value.values())))
            pure_lookup[comp_name] = pure_value

        return pure_lookup

    def timeseries_plotter(self, system: str, start: int = 0) -> TimeseriesPlotter:
        """
        Create a TimeseriesPlotter for visualizing time series data for a given system.

        Parameters
        ----------
        system: str
            System to use for visualizing timeseries.
        start: int
            Initial time for plotting.

        Returns
        -------
        TimeseriesPlotter
            Plotter instance for computing simulation energy properties.
        """
        return TimeseriesPlotter.from_collection(self, system_name=system, start=start)

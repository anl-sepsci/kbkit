"""
Container for a set of systems for a given thermodynamic state (e.g., constant temperature, function of composition).

The purpose of `SystemCollection` is to load a set of systems and access :class:`~kbkit.systems.properties.SystemProperties` to retrieve molecular dynamics properties as a function of composition.
    * This container first discovers molecular systems based on directory structure and input parameters, creating a list of :class:`~kbkit.schema.system_metadata.SystemMetadata` objects.
    * Then topology and energy properties can be calculated as function of composition.
    * Additionally, this object is used to calculating `Excess`, `Simulation`, and `Ideal` properties.
"""

import itertools
import os
import re
from collections import defaultdict
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from kbkit.io import EdrParser
from kbkit.schema.system_metadata import SystemMetadata
from kbkit.systems.properties import SystemProperties
from kbkit.utils.decorators import cached_property_result
from kbkit.utils.format import ENERGY_ALIASES, resolve_attr_key
from kbkit.utils.validation import validate_path
from kbkit.visualization.timeseries import TimeseriesPlotter

if TYPE_CHECKING:
    from kbkit.schema.property_result import PropertyResult

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

    def __init__(self, systems: list["SystemMetadata"], molecules: list[str], charges: dict[str, int] | None = None) -> None:
        self._systems = systems
        self._residue_molecules = molecules  # Global unique molecules used for sorting
        self._lookup = {s.name: s for s in systems}
        self._cache: dict[tuple, PropertyResult] = {}
        # user-provided charges; if None or empty -> neutral behavior
        self.charges: dict[str, int] = charges or {}

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
        start_time: int = 10000,
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
        start_time : int, optional
            Start time for time-averaged properties.
        include_mode: str, optional
            Optional string to filter files (.edr, .gro, .top) if multiple are found of a given type.
        charges: dict[str, int], optional
            Optional charge dictionary for ions.

        Returns
        -------
        SystemCollection
            Registry object containing global molecules and list of :class:`~kbkit.schema.system_metadata.SystemMetadata`.
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
        return cls(sorted_meta, ordered_mols, charges=charges)

    # --- Setting up files/systems for system metadata ---

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

    @staticmethod
    def _extract_temp(input: str | Path) -> float:
        """Extract temperature from a string or file."""
        path = Path(input)
        # first try to match temp from filename
        match = re.search(r"(\d{3}(?:\.\d+)?)", path.name)
        if match:
            return float(match.group(1))
        # then get it from edr file
        if path.is_file() and path.suffix == ".edr":
            edr = EdrParser(str(path))
        # if its directory;
        elif path.is_dir():
            edr_files = SystemProperties.find_files(suffix=".edr", system_path=input)
            edr = EdrParser(str(edr_files[0]))
        # if all has failed raise
        else:
            raise ValueError("Temperature is not in pathname and can not be extracted from .edr file!")

        temp = edr.get_gmx_property("temperature", avg=True)
        if isinstance(temp, float):
            return temp
        raise TypeError(f"Expected float, {type(temp)} observed.")


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
        raise FileNotFoundError(f"No parent directories for pure-components were found containing keywords: {keywords}.")

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
                raise ValueError(f"Charge declared for '{ion}', but it is not in residue_molecules: {self._residue_molecules}")

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
                cat_idx = self._residue_molecules.index(cat)
                an_idx = self._residue_molecules.index(an)
            except ValueError as e:
                raise ValueError(f"Salt component '{cat}' or '{an}' not found in residue_molecules.") from e

            q_cat = self.charges[cat]
            q_an = self.charges[an]
            if q_cat <= 0 or q_an >= 0:
                raise ValueError( f"Inconsistent charges for salt pair ({cat}, {an}): " f"q_cat={q_cat}, q_an={q_an}. Expected cation>0, anion<0." )

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
            c_idx = self._residue_molecules.index(c)
            a_idx = self._residue_molecules.index(a)
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
        return self.x * self.total_molecules[:,np.newaxis]

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
        N = (self.residue_x).astype(float)

        neutral_mask = np.all(nu == 0, axis=1)
        salt_counts = self._solve_salt_counts(nu, N)

        neutral_counts = N[:, neutral_mask]
        new_N = np.column_stack((neutral_counts, salt_counts))

        totals = new_N.sum(axis=1)[:, np.newaxis]
        if np.any(totals == 0):
            raise ValueError("At least one system has total count zero after salt reconstruction.")
        new_x = new_N / totals

        neutral_names = list(np.array(self._residue_molecules)[neutral_mask])
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

    @cached_property_result()
    def simulated_property(self, name: str, units: str | None = None, avg: bool = True):
        """
        Extract raw values directly from MD simulation (EDR files).

        Returns
        -------
        PropertyResult
            Values as simulated in the MD engine.
        """
        units = units or self.get_units(name)
        return self.get(name, units=units, avg=avg)

    @cached_property_result()
    def pure_property(self, name: str, units: str | None = None, avg: bool = True):
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
        PropertyResult
            Pure component property values with metadata.
        """
        units = units or self.get_units(name)

        pure_dict = self._build_pure_lookup(name, units, avg)
        return np.array([pure_dict[mol] for mol in self.molecules])

    @cached_property_result()
    def ideal_property(
        self,
        name: str,
        mixing_rule: Literal["linear", "volume_weighted"] = "linear",
        units: str | None = None,
        avg: bool = True,
    ):
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
        PropertyResult
            Ideal property values for each mixture composition.
        """
        units = units or self.get_units(name)
        pure_res = self.pure_property(name=name, units=units, avg=avg)
        compositions = self.x

        if "lin" in mixing_rule.lower():
            ideal_values = compositions @ pure_res.value
        elif "vol" in mixing_rule.lower():
            ideal_values = 1.0 / (compositions @ (1.0 / pure_res.value))
        else:
            raise ValueError(f"Unknown mixing rule: {mixing_rule}")

        return ideal_values

    @cached_property_result()
    def excess_property(
        self,
        name: str,
        mixing_rule: Literal["linear", "volume_weighted"] = "linear",
        units: str | None = None,
        avg: bool = True,
    ):
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
        PropertyResult
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
        return sim_res.value - ideal_res.value

    def _build_pure_lookup(self, name: str, units: str | None = None, avg: bool = True) -> dict[str, float | np.ndarray | list[np.ndarray]]:
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
                temp_collection = SystemCollection(systems=[pure_sys], molecules=residue_names, charges=self.charges,)
                basis = temp_collection.electrolyte_basis
                assert basis is not None
                new_molecules = basis["molecules"]
                if len(new_molecules) != 1:
                    raise ValueError( f"Pure system {pure_sys.name} does not reduce to a single component in electrolyte basis: " f"{new_molecules}" )
                comp_name = new_molecules[0]
            else:
                # neutral case: must be a single residue
                if len(mol_counts) != 1:
                    raise ValueError(f"Pure system {pure_sys.name} contains multiple molecules: {mol_counts}")
                comp_name = next(iter(mol_counts.keys()))

            pure_value = pure_sys.props.get(name, units=units, avg=avg)
            if isinstance(pure_value, dict):
                pure_value = pure_value.get(comp_name, next(iter(pure_value.values())))
            pure_lookup[comp_name] = pure_value

        return pure_lookup

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



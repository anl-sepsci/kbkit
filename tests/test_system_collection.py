"""Unit tests for SystemCollection class."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Module path — adjust if layout changes
# ---------------------------------------------------------------------------
_MODULE = "kbkit.systems.collection"

_VALIDATE_PATH    = f"{_MODULE}.validate_path"
_SYSTEM_PROPS     = f"{_MODULE}.SystemProperties"
_SYSTEM_META      = f"{_MODULE}.SystemMetadata"
_TIMESERIES       = f"{_MODULE}.TimeseriesPlotter"
_RESOLVE_KEY      = f"{_MODULE}.resolve_attr_key"

from kbkit.systems.collection import SystemCollection
from kbkit.schema.system_metadata import SystemMetadata
from kbkit.schema.property_result import PropertyResult


# ===========================================================================
# Helpers / Factories
# ===========================================================================

def _make_mock_topology(
    molecule_count: dict,
    total_molecules: int | None = None,
    box_volume: float = 30.0,
    electron_count: int = 50,
    molecules: list[str] | None = None,
):
    """Return a MagicMock mimicking TopologyParser."""
    topo = MagicMock()
    topo.molecule_count = molecule_count
    topo.total_molecules = total_molecules if total_molecules is not None else sum(molecule_count.values())
    topo.box_volume = box_volume
    topo.electron_count = electron_count
    topo.molecules = molecules or list(molecule_count.keys())
    return topo


def _make_mock_props(
    molecule_count: dict,
    total_molecules: int | None = None,
    get_return: float = 1.0,
    units: dict | None = None,
):
    """Return a MagicMock mimicking SystemProperties."""
    props = MagicMock()
    props.topology = _make_mock_topology(molecule_count, total_molecules)
    props.get.return_value = get_return
    props.get.side_effect = None
    props.units = units or {"temperature": "K", "density": "kg/m^3"}
    return props


def _make_meta(
    name: str,
    kind: str = "mixture",
    molecule_count: dict | None = None,
    get_return: float = 1.0,
    units: dict | None = None,
    path: Path | None = None,
) -> MagicMock:
    """Return a MagicMock mimicking SystemMetadata."""
    molecule_count = molecule_count or {"MOL": 100}
    meta = MagicMock(spec=SystemMetadata)
    meta.name = name
    meta.kind = kind
    meta.path = path or Path(f"/fake/{name}")
    meta.props = _make_mock_props(molecule_count, get_return=get_return, units=units)
    meta.is_pure.return_value = kind == "pure"
    return meta


def _make_binary_collection(
    mol_a: str = "A",
    mol_b: str = "B",
    n_systems: int = 3,
    charges: dict | None = None,
) -> SystemCollection:
    """
    Build a minimal binary SystemCollection with n_systems mixture systems.
    Mole fractions go from pure A → pure B.
    """
    systems = []
    for i in range(n_systems):
        frac = i / max(n_systems - 1, 1)
        n_a = int(round((1 - frac) * 100))
        n_b = int(round(frac * 100))
        mc = {mol_a: n_a, mol_b: n_b}
        systems.append(_make_meta(f"sys_{i}", kind="mixture", molecule_count=mc))

    return SystemCollection(systems=systems, molecules=[mol_a, mol_b], charges=charges)


# ===========================================================================
# 1. __init__ and basic dunder methods
# ===========================================================================

class TestInit:
    """Tests for SystemCollection.__init__ and dunder methods."""

    def test_len_returns_number_of_systems(self):
        sc = _make_binary_collection(n_systems=4)
        assert len(sc) == 4

    def test_iter_yields_all_systems(self):
        sc = _make_binary_collection(n_systems=3)
        items = list(sc)
        assert len(items) == 3

    def test_getitem_by_index(self):
        sc = _make_binary_collection(n_systems=3)
        item = sc[0]
        assert item is sc._systems[0]

    def test_getitem_by_name(self):
        sc = _make_binary_collection(n_systems=3)
        name = sc._systems[1].name
        assert sc[name] is sc._systems[1]

    def test_system_names_populated(self):
        sc = _make_binary_collection(n_systems=3)
        assert len(sc.system_names) == 3
        assert all(isinstance(n, str) for n in sc.system_names)

    def test_charges_defaults_to_empty_dict(self):
        sc = _make_binary_collection()
        assert sc.charges == {}

    def test_charges_stored_when_provided(self):
        sc = _make_binary_collection(charges={"Na": 1, "Cl": -1})
        assert sc.charges == {"Na": 1, "Cl": -1}

    def test_lookup_maps_name_to_metadata(self):
        sc = _make_binary_collection(n_systems=2)
        for name in sc.system_names:
            assert sc._lookup[name].name == name

    def test_cache_starts_empty(self):
        sc = _make_binary_collection()
        assert sc._cache == {}


# ===========================================================================
# 2. _sort_systems
# ===========================================================================

class TestSortSystems:
    """Tests for SystemCollection._sort_systems."""

    def test_systems_sorted_by_mole_fraction(self):
        """Systems should be ordered by increasing mole fraction of first molecule."""
        systems = [
            _make_meta("high_A", molecule_count={"A": 90, "B": 10}),
            _make_meta("low_A",  molecule_count={"A": 10, "B": 90}),
            _make_meta("mid_A",  molecule_count={"A": 50, "B": 50}),
        ]
        sorted_sys = SystemCollection._sort_systems(systems, molecules=["A", "B"])
        fracs = [s.props.topology.molecule_count["A"] / s.props.topology.total_molecules
                 for s in sorted_sys]
        assert fracs == sorted(fracs)

    def test_zero_total_molecules_handled(self):
        """Systems with zero total molecules should not raise."""
        meta = _make_meta("empty", molecule_count={"A": 0, "B": 0})
        meta.props.topology.total_molecules = 0
        result = SystemCollection._sort_systems([meta], molecules=["A", "B"])
        assert len(result) == 1

    def test_single_system_unchanged(self):
        systems = [_make_meta("only", molecule_count={"A": 100})]
        result = SystemCollection._sort_systems(systems, molecules=["A"])
        assert result[0].name == "only"


# ===========================================================================
# 3. _is_valid
# ===========================================================================

class TestIsValid:
    """Tests for SystemCollection._is_valid (static method)."""

    def test_valid_directory_with_edr_and_gro(self, tmp_path):
        (tmp_path / "run.edr").touch()
        (tmp_path / "run.gro").touch()
        assert SystemCollection._is_valid(tmp_path) is True

    def test_valid_directory_with_log_and_top(self, tmp_path):
        (tmp_path / "run.log").touch()
        (tmp_path / "run.top").touch()
        assert SystemCollection._is_valid(tmp_path) is True

    def test_valid_directory_with_lammps_and_lmp(self, tmp_path):
        (tmp_path / "run.lammps").touch()
        (tmp_path / "run.lmp").touch()
        assert SystemCollection._is_valid(tmp_path) is True

    def test_invalid_when_missing_energy_file(self, tmp_path):
        (tmp_path / "run.gro").touch()
        assert SystemCollection._is_valid(tmp_path) is False

    def test_invalid_when_missing_topology_file(self, tmp_path):
        (tmp_path / "run.edr").touch()
        assert SystemCollection._is_valid(tmp_path) is False

    def test_invalid_when_path_is_file_not_dir(self, tmp_path):
        f = tmp_path / "run.edr"
        f.touch()
        assert SystemCollection._is_valid(f) is False

    def test_invalid_empty_directory(self, tmp_path):
        assert SystemCollection._is_valid(tmp_path) is False


# ===========================================================================
# 4. _resolve_rdf_path
# ===========================================================================

class TestResolveRdfPath:
    """Tests for SystemCollection._resolve_rdf_path (static method)."""

    def test_explicit_rdf_dir_found(self, tmp_path):
        rdf_dir = tmp_path / "rdf_data"
        rdf_dir.mkdir()
        result = SystemCollection._resolve_rdf_path(tmp_path, rdf_dir="rdf_data", is_pure=False)
        assert result == rdf_dir

    def test_auto_discovers_rdf_subdir_with_xvg(self, tmp_path):
        rdf_dir = tmp_path / "rdf_output"
        rdf_dir.mkdir()
        (rdf_dir / "gr.xvg").touch()
        result = SystemCollection._resolve_rdf_path(tmp_path, rdf_dir="", is_pure=False)
        assert result == rdf_dir

    def test_auto_discovers_rdf_subdir_with_txt(self, tmp_path):
        rdf_dir = tmp_path / "rdf_results"
        rdf_dir.mkdir()
        (rdf_dir / "gr.txt").touch()
        result = SystemCollection._resolve_rdf_path(tmp_path, rdf_dir="", is_pure=False)
        assert result == rdf_dir

    def test_raises_for_mixture_when_no_rdf_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No RDF directory"):
            SystemCollection._resolve_rdf_path(tmp_path, rdf_dir="", is_pure=False)

    def test_returns_empty_path_for_pure_when_no_rdf(self, tmp_path):
        result = SystemCollection._resolve_rdf_path(tmp_path, rdf_dir="", is_pure=True)
        assert result == Path()

    def test_explicit_rdf_dir_takes_priority(self, tmp_path):
        explicit = tmp_path / "explicit_rdf"
        explicit.mkdir()
        auto = tmp_path / "rdf_auto"
        auto.mkdir()
        (auto / "gr.xvg").touch()
        result = SystemCollection._resolve_rdf_path(tmp_path, rdf_dir="explicit_rdf", is_pure=False)
        assert result == explicit


# ===========================================================================
# 5. Basis accessors: residue_x, residue_counts, molecules, x
# ===========================================================================

class TestBasisAccessors:
    """Tests for mole fraction and count accessors."""

    def test_residue_x_shape(self):
        sc = _make_binary_collection(n_systems=3)
        x = sc.residue_x
        assert x.shape == (3, 2)

    def test_residue_x_rows_sum_to_one(self):
        sc = _make_binary_collection(n_systems=4)
        row_sums = sc.residue_x.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_residue_counts_shape(self):
        sc = _make_binary_collection(n_systems=3)
        counts = sc.residue_counts
        assert counts.shape == (3, 2)

    def test_residue_counts_non_negative(self):
        sc = _make_binary_collection(n_systems=3)
        assert np.all(sc.residue_counts >= 0)

    def test_molecules_returns_residue_molecules_when_no_charges(self):
        sc = _make_binary_collection(mol_a="W", mol_b="E")
        assert sc.molecules == ["W", "E"]

    def test_x_equals_residue_x_when_no_charges(self):
        sc = _make_binary_collection(n_systems=3)
        np.testing.assert_array_equal(sc.x, sc.residue_x)

    def test_n_i_returns_number_of_components(self):
        sc = _make_binary_collection(mol_a="A", mol_b="B")
        assert sc.n_i == 2

    def test_n_sys_returns_number_of_systems(self):
        sc = _make_binary_collection(n_systems=5)
        assert sc.n_sys == 5

    def test_get_mol_index_returns_correct_index(self):
        sc = _make_binary_collection(mol_a="A", mol_b="B")
        assert sc.get_mol_index("A") == 0
        assert sc.get_mol_index("B") == 1

    def test_get_mol_index_raises_for_unknown_molecule(self):
        sc = _make_binary_collection(mol_a="A", mol_b="B")
        with pytest.raises(ValueError, match="not in molecules"):
            sc.get_mol_index("X")


# ===========================================================================
# 6. pures / mixtures
# ===========================================================================

class TestPuresMixtures:
    """Tests for pures and mixtures properties."""

    def _sc_with_kinds(self, kinds: list[str]) -> SystemCollection:
        systems = [_make_meta(f"sys_{i}", kind=k) for i, k in enumerate(kinds)]
        return SystemCollection(systems=systems, molecules=["MOL"])

    def test_pures_returns_only_pure_systems(self):
        sc = self._sc_with_kinds(["pure", "mixture", "pure"])
        assert len(sc.pures) == 2
        assert all(s.is_pure() for s in sc.pures)

    def test_mixtures_returns_only_mixture_systems(self):
        sc = self._sc_with_kinds(["pure", "mixture", "mixture"])
        assert len(sc.mixtures) == 2
        assert all(not s.is_pure() for s in sc.mixtures)

    def test_has_all_required_pures_true(self):
        """One pure per molecule → True."""
        pure_a = _make_meta("A", kind="pure", molecule_count={"A": 100})
        pure_b = _make_meta("B", kind="pure", molecule_count={"B": 100})
        sc = SystemCollection(systems=[pure_a, pure_b], molecules=["A", "B"])
        assert sc.has_all_required_pures() is True

    def test_has_all_required_pures_false(self):
        """Only one pure for two molecules → False."""
        pure_a = _make_meta("A", kind="pure", molecule_count={"A": 100})
        sc = SystemCollection(systems=[pure_a], molecules=["A", "B"])
        assert sc.has_all_required_pures() is False


# ===========================================================================
# 7. get() and units
# ===========================================================================

class TestGet:
    """Tests for SystemCollection.get()."""

    def test_get_returns_array_of_correct_length(self):
        sc = _make_binary_collection(n_systems=4)
        for meta in sc._systems:
            meta.props.get.return_value = 300.0
        result = sc.get("temperature", avg=True)
        assert len(result) == 4

    def test_get_returns_numpy_array_when_possible(self):
        sc = _make_binary_collection(n_systems=3)
        for meta in sc._systems:
            meta.props.get.return_value = 1.0
        result = sc.get("density", avg=True)
        assert isinstance(result, np.ndarray)

    def test_get_returns_list_when_values_are_ragged(self):
        sc = _make_binary_collection(n_systems=2)
        # ragged arrays cannot be stacked into ndarray
        sc._systems[0].props.get.return_value = np.array([1.0, 2.0])
        sc._systems[1].props.get.return_value = np.array([3.0, 4.0, 5.0])
        result = sc.get("temperature", avg=False)
        # numpy will raise ValueError and fall back to list
        assert isinstance(result, (list, np.ndarray))

    def test_get_passes_units_to_props(self):
        sc = _make_binary_collection(n_systems=2)
        sc.get("density", units="g/cm^3", avg=True)
        for meta in sc._systems:
            meta.props.get.assert_called_with("density", units="g/cm^3", avg=True, time_series=False)

    def test_get_units_returns_string(self):
        sc = _make_binary_collection()
        # units dict is built from mock props
        result = sc.get_units("density")
        assert isinstance(result, str)

    def test_units_cached_property_aggregates_all_systems(self):
        sc = _make_binary_collection(n_systems=3)
        for meta in sc._systems:
            meta.props.get.return_value = {"temperature": "K", "density": "kg/m^3"}
        units = sc.units
        assert isinstance(units, dict)


# ===========================================================================
# 8. simulated_property / ideal_property / excess_property
# ===========================================================================

class TestDerivedProperties:
    """Tests for simulated, ideal, and excess property calculations."""

    def _sc_with_pures(self, pure_val_a: float = 2.0, pure_val_b: float = 4.0) -> SystemCollection:
        """
        Build a collection with 2 pure systems and 1 mixture (50/50).
        """
        pure_a = _make_meta("A", kind="pure",    molecule_count={"A": 100, "B": 0},   get_return=pure_val_a)
        pure_b = _make_meta("B", kind="pure",    molecule_count={"A": 0,   "B": 100}, get_return=pure_val_b)
        mix    = _make_meta("M", kind="mixture", molecule_count={"A": 50,  "B": 50},  get_return=3.5)

        # topology.molecule_count must not include zero-count molecules for pure lookup
        pure_a.props.topology.molecule_count = {"A": 100}
        pure_b.props.topology.molecule_count = {"B": 100}

        sc = SystemCollection(systems=[pure_a, pure_b, mix], molecules=["A", "B"])
        # patch get_units so it returns a consistent unit string
        sc.__dict__["units"] = {"density": "kg/m^3"}
        return sc

    def test_simulated_property_returns_array(self):
        sc = self._sc_with_pures()
        result = sc.simulated_property(name="density", units="kg/m^3")
        assert isinstance(result, np.ndarray)
        assert len(result) == 3

    def test_ideal_property_linear_mixing(self):
        sc = self._sc_with_pures(pure_val_a=2.0, pure_val_b=4.0)
        # 50/50 mixture → ideal = 0.5*2 + 0.5*4 = 3.0
        ideal = sc.ideal_property(name="density", units="kg/m^3", mixing_rule="linear")
        # check the mixture system (last after sorting)
        mix_idx = next(i for i, s in enumerate(sc._systems) if s.name == "M")
        assert ideal[mix_idx] == pytest.approx(3.0, abs=0.1)

    def test_ideal_property_volume_weighted_mixing(self):
        sc = self._sc_with_pures(pure_val_a=2.0, pure_val_b=4.0)
        ideal = sc.ideal_property(name="density", units="kg/m^3", mixing_rule="volume_weighted")
        assert isinstance(ideal, np.ndarray)
        assert len(ideal) == 3

    def test_ideal_property_raises_for_unknown_mixing_rule(self):
        sc = self._sc_with_pures()
        with pytest.raises(ValueError, match="Unknown mixing rule"):
            sc.ideal_property(name="density", units="kg/m^3", mixing_rule="unknown_rule")

    def test_excess_property_is_simulated_minus_ideal(self):
        sc = self._sc_with_pures(pure_val_a=2.0, pure_val_b=4.0)
        sim   = sc.simulated_property(name="density", units="kg/m^3")
        ideal = sc.ideal_property(name="density",    units="kg/m^3", mixing_rule="linear")
        excess = sc.excess_property(name="density",  units="kg/m^3", mixing_rule="linear")
        np.testing.assert_allclose(excess, sim - ideal, atol=1e-10)

    def test_pure_property_returns_nan_for_missing_component(self):
        """If a pure system is missing, the corresponding entry should be nan."""
        pure_a = _make_meta("A", kind="pure", molecule_count={"A": 100}, get_return=5.0)
        pure_a.props.topology.molecule_count = {"A": 100}
        mix = _make_meta("M", kind="mixture", molecule_count={"A": 50, "B": 50}, get_return=4.0)
        sc = SystemCollection(systems=[pure_a, mix], molecules=["A", "B"])
        sc.__dict__["units"] = {"density": "kg/m^3"}
        result = sc.pure_property(name="density", units="kg/m^3")
        assert np.isnan(result[1])   # B has no pure system → nan
        assert result[0] == pytest.approx(5.0)


# ===========================================================================
# 9. Electrolyte basis
# ===========================================================================

class TestElectrolyteBasis:
    """Tests for electrolyte basis construction."""

    def _sc_electrolyte(self) -> SystemCollection:
        """Na-Cl binary with charges."""
        pure_na = _make_meta("Na", kind="pure",    molecule_count={"Na": 100, "Cl": 0})
        pure_cl = _make_meta("Cl", kind="pure",    molecule_count={"Na": 0,   "Cl": 100})
        mix     = _make_meta("M",  kind="mixture", molecule_count={"Na": 50,  "Cl": 50})
        pure_na.props.topology.molecule_count = {"Na": 100}
        pure_cl.props.topology.molecule_count = {"Cl": 100}
        return SystemCollection(
            systems=[pure_na, pure_cl, mix],
            molecules=["Na", "Cl"],
            charges={"Na": 1, "Cl": -1},
        )

    def test_electrolyte_basis_returns_dict(self):
        sc = self._sc_electrolyte()
        basis = sc.electrolyte_basis
        assert isinstance(basis, dict)
        assert "molecules" in basis
        assert "x" in basis
        assert "N" in basis
        assert "nu" in basis

    def test_nu_property_returns_matrix(self):
        sc = self._sc_electrolyte()
        nu = sc.nu
        assert isinstance(nu, np.ndarray)
        assert nu.ndim == 2

    def test_electrolyte_x_rows_sum_to_one(self):
        sc = self._sc_electrolyte()
        row_sums = sc.electrolyte_x.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_electrolyte_molecules_returns_list(self):
        sc = self._sc_electrolyte()
        mols = sc.electrolyte_molecules
        assert isinstance(mols, list)
        assert len(mols) >= 1

    def test_electrolyte_properties_raise_when_no_charges(self):
        sc = _make_binary_collection()  # no charges
        with pytest.raises(ValueError, match="No charges provided"):
            _ = sc.electrolyte_molecules
        with pytest.raises(ValueError, match="No charges provided"):
            _ = sc.electrolyte_x
        with pytest.raises(ValueError, match="No charges provided"):
            _ = sc.nu

    def test_validate_charges_raises_for_unknown_ion(self):
        sc = _make_binary_collection(mol_a="A", mol_b="B", charges={"X": 1, "Y": -1})
        with pytest.raises(ValueError, match="not in residue_molecules"):
            sc._validate_charges()

    def test_build_salt_pairs_returns_correct_pairs(self):
        sc = _make_binary_collection(
            mol_a="Na", mol_b="Cl", charges={"Na": 1, "Cl": -1}
        )
        pairs = sc._build_salt_pairs()
        assert ("Na", "Cl") in pairs

    def test_build_salt_pairs_empty_when_no_charges(self):
        sc = _make_binary_collection()
        pairs = sc._build_salt_pairs()
        assert pairs == []

    def test_canonical_salt_names_nacl(self):
        sc = _make_binary_collection(
            mol_a="Na", mol_b="Cl", charges={"Na": 1, "Cl": -1}
        )
        pairs = [("Na", "Cl")]
        nu = sc._build_nu_matrix(pairs)
        names = sc._canonical_salt_names(pairs, nu)
        assert names == ["Na.Cl"]

    def test_build_nu_matrix_shape(self):
        sc = _make_binary_collection(
            mol_a="Na", mol_b="Cl", charges={"Na": 1, "Cl": -1}
        )
        pairs = sc._build_salt_pairs()
        nu = sc._build_nu_matrix(pairs)
        assert nu.shape == (2, 1)   # 2 residues, 1 salt pair

    def test_build_nu_matrix_raises_for_wrong_charges(self):
        """Passing a cation with negative charge should raise."""
        sc = _make_binary_collection(
            mol_a="Na", mol_b="Cl", charges={"Na": -1, "Cl": 1}  # intentionally swapped
        )
        pairs = [("Na", "Cl")]
        with pytest.raises(ValueError, match="Inconsistent charges"):
            sc._build_nu_matrix(pairs)


# ===========================================================================
# 10. __getattr__ delegation
# ===========================================================================

class TestGetattr:
    """Tests for __getattr__ delegation to SystemMetadata / SystemProperties."""

    def test_getattr_delegates_to_metadata_attribute(self):
        sc = _make_binary_collection(n_systems=2)
        # 'name' is an attribute on each SystemMetadata mock
        names = sc.name
        assert isinstance(names, list)
        assert len(names) == 2

    def test_getattr_returns_numpy_array_for_numeric_values(self):
        sc = _make_binary_collection(n_systems=3)
        for meta in sc._systems:
            meta.props.get.return_value = 300.0
        # 'temperature' is not on metadata or props directly → falls to props.get
        result = sc.temperature
        assert isinstance(result, (list, np.ndarray))

    def test_getattr_returns_empty_list_when_no_systems(self):
        sc = SystemCollection(systems=[], molecules=[])
        result = sc.anything
        assert result == []


# ===========================================================================
# 11. timeseries_plotter
# ===========================================================================

class TestTimeseriesPlotter:
    """Tests for SystemCollection.timeseries_plotter."""

    def test_returns_plotter_instance(self):
        sc = _make_binary_collection(n_systems=2)
        mock_plotter = MagicMock()
        with patch(_TIMESERIES) as MockTS:
            MockTS.from_collection.return_value = mock_plotter
            result = sc.timeseries_plotter(system="sys_0", start=100)
            MockTS.from_collection.assert_called_once_with(sc, system_name="sys_0", start=100)
        assert result is mock_plotter

    def test_default_start_is_zero(self):
        sc = _make_binary_collection(n_systems=2)
        with patch(_TIMESERIES) as MockTS:
            MockTS.from_collection.return_value = MagicMock()
            sc.timeseries_plotter(system="sys_0")
            _, kwargs = MockTS.from_collection.call_args
            assert kwargs.get("start", 0) == 0


# ===========================================================================
# 12. results cached_property
# ===========================================================================

class TestResults:
    """Tests for SystemCollection.results cached property."""

    def _sc_for_results(self) -> SystemCollection:
        pure_a = _make_meta("A", kind="pure",    molecule_count={"A": 100}, get_return=2.0)
        pure_b = _make_meta("B", kind="pure",    molecule_count={"B": 100}, get_return=4.0)
        mix    = _make_meta("M", kind="mixture", molecule_count={"A": 50, "B": 50}, get_return=3.0)
        pure_a.props.topology.molecule_count = {"A": 100}
        pure_b.props.topology.molecule_count = {"B": 100}
        sc = SystemCollection(systems=[pure_a, pure_b, mix], molecules=["A", "B"])
        sc.__dict__["units"] = {}   # skip energy property loop
        return sc

    def test_results_returns_dict(self):
        sc = self._sc_for_results()
        r = sc.results
        assert isinstance(r, dict)

    def test_results_contains_required_keys(self):
        sc = self._sc_for_results()
        r = sc.results
        assert "molecules" in r
        assert "n_i" in r
        assert "n_sys" in r
        assert "x" in r

    def test_results_values_are_property_result_objects(self):
        sc = self._sc_for_results()
        r = sc.results
        assert all(isinstance(v, PropertyResult) for v in r.values())

    def test_results_x_shape_matches_collection(self):
        sc = self._sc_for_results()
        x_result = sc.results["x"].value
        assert x_result.shape == sc.x.shape


# ===========================================================================
# 13. _build_pure_lookup
# ===========================================================================

class TestBuildPureLookup:
    """Tests for SystemCollection._build_pure_lookup."""

    def test_lookup_maps_molecule_to_value(self):
        pure_a = _make_meta("A", kind="pure", molecule_count={"A": 100}, get_return=7.5)
        pure_a.props.topology.molecule_count = {"A": 100}
        sc = SystemCollection(systems=[pure_a], molecules=["A"])
        lookup = sc._build_pure_lookup("density", units="kg/m^3")
        assert "A" in lookup
        assert lookup["A"] == pytest.approx(7.5)

    def test_lookup_raises_when_pure_has_multiple_molecules(self):
        """Neutral pure system with >1 molecule type should raise."""
        bad_pure = _make_meta("AB", kind="pure", molecule_count={"A": 50, "B": 50})
        bad_pure.props.topology.molecule_count = {"A": 50, "B": 50}
        sc = SystemCollection(systems=[bad_pure], molecules=["A", "B"])
        with pytest.raises(ValueError, match="multiple molecules"):
            sc._build_pure_lookup("density")

    def test_lookup_empty_when_no_pures(self):
        mix = _make_meta("M", kind="mixture", molecule_count={"A": 50, "B": 50})
        sc = SystemCollection(systems=[mix], molecules=["A", "B"])
        lookup = sc._build_pure_lookup("density")
        assert lookup == {}

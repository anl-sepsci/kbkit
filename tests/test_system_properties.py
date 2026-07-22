"""Unit tests for SystemProperties class."""

import warnings
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Correct module path — adjust this single constant if your layout changes
# ---------------------------------------------------------------------------
_MODULE = "kbkit.systems.properties"

# Patch targets: patch where the name is USED (imported into), not where defined
_VALIDATE_PATH = f"{_MODULE}.validate_path"
_ENERGY_PARSER = f"{_MODULE}.EnergyParser"
_TOPO_PARSER = f"{_MODULE}.TopologyParser"
_LOAD_UREG = f"{_MODULE}.load_unit_registry"
_TIMESERIES = f"{_MODULE}.TimeseriesPlotter"
_RESOLVE_KEY = f"{_MODULE}.resolve_attr_key"

# Import the class under test AFTER defining patch targets
from kbkit.io import EnergyParser  # needed for FLUCT_PROPS reference
from kbkit.systems.properties import SystemProperties

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_energy_parser(
    units=None,
    get_return=None,
    x_key="Time",
    heat_capacity_cp=1.0,
    heat_capacity_cv=1.0,
    molar_enthalpy=1.0,
    isothermal_compressibility=1.0,
    molar_volume=0.5,
):
    """Return a fully-configured MagicMock that mimics EnergyParser."""
    parser = MagicMock(spec=EnergyParser)
    parser._x_key = x_key
    parser.units = units or {"molar-volume": "L/mol", "temperature": "K"}
    parser.get.return_value = get_return if get_return is not None else np.array([300.0, 301.0, 299.0])
    parser.heat_capacity_cp.return_value = heat_capacity_cp
    parser.heat_capacity_cv.return_value = heat_capacity_cv
    parser.molar_enthalpy.return_value = molar_enthalpy
    parser.isothermal_compressibility.return_value = isothermal_compressibility
    parser.molar_volume.return_value = molar_volume
    return parser


def _make_mock_topology_parser(
    box_volume=30.0,
    total_molecules=100,
    electron_count=50,
    extra_props=None,
):
    """Return a fully-configured MagicMock that mimics TopologyParser."""
    topo = MagicMock()
    topo.box_volume = box_volume
    topo.total_molecules = total_molecules
    topo.electron_count = electron_count
    if extra_props:
        for k, v in extra_props.items():
            setattr(topo, k, v)
    return topo


def _inject_sp(mock_topo, mock_energy_list, start=0):
    """
    Build a SystemProperties instance without calling __init__.
    Injects topology and energy via __dict__ to bypass cached_property.
    """
    sp = object.__new__(SystemProperties)
    sp.start = start
    sp.ureg = MagicMock()
    sp.Q_ = MagicMock()
    sp.__dict__["topology"] = mock_topo
    sp.__dict__["energy"] = mock_energy_list
    return sp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_system(tmp_path):
    """Create a minimal fake GROMACS system directory."""
    (tmp_path / "system.edr").touch()
    (tmp_path / "system.gro").touch()
    return tmp_path


@pytest.fixture
def mock_energy_parser():
    return _make_mock_energy_parser()


@pytest.fixture
def mock_topology_parser():
    return _make_mock_topology_parser()


@pytest.fixture
def fluct_props_empty():
    """Patch EnergyParser.FLUCT_PROPS to an empty list for all energy-branch tests."""
    with patch.object(EnergyParser, "FLUCT_PROPS", new_callable=lambda: property(lambda self: [])):
        yield


# ===========================================================================
# 1. _get_abspath
# ===========================================================================


class TestGetAbspath:
    """Tests for SystemProperties._get_abspath (static method)."""

    def test_returns_absolute_path_when_file_exists(self, tmp_path):
        f = tmp_path / "data.edr"
        f.touch()
        result = SystemProperties._get_abspath(str(tmp_path), "data.edr")
        assert result == f.resolve()

    def test_resolves_relative_to_parent_path(self, tmp_path):
        f = tmp_path / "sub.edr"
        f.touch()
        result = SystemProperties._get_abspath(str(tmp_path), "sub.edr")
        assert result.is_file()

    def test_raises_file_not_found_when_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            SystemProperties._get_abspath(str(tmp_path), "ghost.edr")

    def test_raises_file_not_found_when_path_is_none(self, tmp_path):
        # file doesn't exist at cwd either
        with pytest.raises(FileNotFoundError):
            SystemProperties._get_abspath(None, "definitely_missing_xyz.edr")

    def test_absolute_filename_ignores_path_arg(self, tmp_path):
        f = tmp_path / "run.edr"
        f.touch()
        result = SystemProperties._get_abspath(None, str(f))
        assert result == f.resolve()


# ===========================================================================
# 2. _find_files_in_path
# ===========================================================================


class TestFindFilesInPath:
    """Tests for SystemProperties._find_files_in_path (static method)."""

    def test_raises_when_path_is_none(self):
        with pytest.raises(ValueError, match="valid 'filepath'"):
            SystemProperties._find_files_in_path(suffix=".edr", path=None)

    def test_returns_single_file(self, tmp_path):
        (tmp_path / "run.edr").touch()
        result = SystemProperties._find_files_in_path(suffix=".edr", path=tmp_path)
        assert len(result) == 1
        assert result[0].name == "run.edr"

    def test_returns_all_files_when_no_include_filter(self, tmp_path):
        for name in ("a.edr", "b.edr", "c.edr"):
            (tmp_path / name).touch()
        result = SystemProperties._find_files_in_path(suffix=".edr", path=tmp_path, include="")
        assert len(result) == 3

    def test_include_filter_applied_when_multiple_files(self, tmp_path):
        (tmp_path / "prod.edr").touch()
        (tmp_path / "other.edr").touch()
        result = SystemProperties._find_files_in_path(suffix=".edr", path=tmp_path, include="prod")
        assert all("prod" in f.name for f in result)

    def test_include_filter_returns_all_when_no_match(self, tmp_path):
        """If include filter yields nothing, fall back to all files."""
        for name in ("a.edr", "b.edr"):
            (tmp_path / name).touch()
        result = SystemProperties._find_files_in_path(suffix=".edr", path=tmp_path, include="xyz")
        assert len(result) == 2

    def test_exclude_filter_removes_default_patterns(self, tmp_path):
        (tmp_path / "prod_run.edr").touch()
        (tmp_path / "prod_eq.edr").touch()
        (tmp_path / "prod_em.edr").touch()
        result = SystemProperties._find_files_in_path(suffix=".edr", path=tmp_path, include="prod")
        names = [f.name for f in result]
        assert "prod_run.edr" in names
        assert "prod_eq.edr" not in names
        assert "prod_em.edr" not in names

    def test_returns_empty_list_when_no_files(self, tmp_path):
        result = SystemProperties._find_files_in_path(suffix=".edr", path=tmp_path)
        assert result == []

    def test_strip_leading_dot_from_suffix(self, tmp_path):
        (tmp_path / "run.edr").touch()
        result = SystemProperties._find_files_in_path(suffix="edr", path=tmp_path)
        assert len(result) == 1

    def test_sorted_output(self, tmp_path):
        for name in ("c.edr", "a.edr", "b.edr"):
            (tmp_path / name).touch()
        result = SystemProperties._find_files_in_path(suffix=".edr", path=tmp_path)
        names = [f.name for f in result]
        assert names == sorted(names)


# ===========================================================================
# 3. _get_files
# ===========================================================================


class TestGetFiles:
    """Tests for SystemProperties._get_files (static method)."""

    def test_raises_when_no_path_and_no_filename(self):
        with pytest.raises(ValueError, match="path is required"):
            SystemProperties._get_files(path=None, filename=None, suffixes=[".edr"], include="")

    def test_returns_file_when_filename_given_and_exists(self, tmp_path):
        f = tmp_path / "run.edr"
        f.touch()
        result = SystemProperties._get_files(path=str(tmp_path), filename="run.edr", suffixes=[".edr"], include="")
        assert result == [f.resolve()]

    def test_falls_back_to_path_search_when_filename_not_found(self, tmp_path):
        (tmp_path / "run.edr").touch()
        result = SystemProperties._get_files(path=str(tmp_path), filename="missing.edr", suffixes=[".edr"], include="")
        assert len(result) == 1

    def test_raises_when_no_files_found_for_any_suffix(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No files found"):
            SystemProperties._get_files(path=str(tmp_path), filename=None, suffixes=[".edr", ".lammps"], include="")

    def test_priority_order_of_suffixes(self, tmp_path):
        """First matching suffix wins."""
        (tmp_path / "run.edr").touch()
        (tmp_path / "run.log").touch()
        result = SystemProperties._get_files(path=str(tmp_path), filename=None, suffixes=[".edr", ".log"], include="")
        assert all(f.suffix == ".edr" for f in result)

    def test_returns_path_object(self, tmp_path):
        f = tmp_path / "run.edr"
        f.touch()
        result = SystemProperties._get_files(path=str(tmp_path), filename="run.edr", suffixes=[".edr"], include="")
        assert isinstance(result, list)


# ===========================================================================
# 4. __init__
# ===========================================================================


class TestInit:
    """Tests for SystemProperties.__init__."""

    def test_init_sets_start(self, tmp_path):
        """start attribute is stored correctly."""
        edr = tmp_path / "run.edr"
        gro = tmp_path / "run.gro"
        edr.touch()
        gro.touch()
        # patch load_unit_registry where it is imported in the module
        with patch(_LOAD_UREG, return_value=MagicMock()) as mock_ureg:
            mock_ureg.return_value.Quantity = MagicMock()
            sp = SystemProperties(energy=str(edr), topology=str(gro), start=42)
        assert sp.start == 42

    def test_init_with_explicit_files(self, tmp_path):
        edr = tmp_path / "run.edr"
        gro = tmp_path / "run.gro"
        edr.touch()
        gro.touch()
        with patch(_LOAD_UREG, return_value=MagicMock()):
            sp = SystemProperties(energy=str(edr), topology=str(gro))
        # energy_files and topology_files should be populated
        assert sp.energy_files is not None
        assert sp.topology_files is not None

    def test_init_raises_when_no_path_and_no_files(self):
        with patch(_LOAD_UREG, return_value=MagicMock()):
            with pytest.raises((ValueError, FileNotFoundError)):
                SystemProperties()

    def test_parent_inferred_when_all_files_share_parent(self, tmp_path):
        edr = tmp_path / "run.edr"
        gro = tmp_path / "run.gro"
        edr.touch()
        gro.touch()
        with patch(_LOAD_UREG, return_value=MagicMock()):
            sp = SystemProperties(energy=str(edr), topology=str(gro))
        assert sp.parent == tmp_path.resolve()

    def test_init_with_path_directory(self, tmp_path):
        (tmp_path / "run.edr").touch()
        (tmp_path / "run.gro").touch()
        with patch(_LOAD_UREG, return_value=MagicMock()):
            sp = SystemProperties(path=str(tmp_path))
        assert sp.energy_files is not None
        assert sp.topology_files is not None

    def test_default_start_is_zero(self, tmp_path):
        edr = tmp_path / "run.edr"
        gro = tmp_path / "run.gro"
        edr.touch()
        gro.touch()
        with patch(_LOAD_UREG, return_value=MagicMock()):
            sp = SystemProperties(energy=str(edr), topology=str(gro))
        assert sp.start == 0


# ===========================================================================
# 5. cached_property: energy / topology
# ===========================================================================


class TestCachedProperties:
    """Tests for the `energy` and `topology` cached properties."""

    def _build_sp(self, tmp_path):
        """Construct a SystemProperties with real files but mocked unit registry."""
        edr = tmp_path / "run.edr"
        gro = tmp_path / "run.gro"
        edr.touch()
        gro.touch()
        with patch(_LOAD_UREG, return_value=MagicMock()):
            sp = SystemProperties(energy=str(edr), topology=str(gro))
        return sp

    def test_energy_returns_list_of_parsers(self, tmp_path):
        sp = self._build_sp(tmp_path)
        mock_instance = MagicMock()
        # patch EnergyParser where it is imported in the module
        with patch(_ENERGY_PARSER, return_value=mock_instance):
            # clear cached value so the property re-runs
            sp.__dict__.pop("energy", None)
            parsers = sp.energy
        assert isinstance(parsers, list)
        assert parsers[0] is mock_instance

    def test_topology_returns_single_parser(self, tmp_path):
        sp = self._build_sp(tmp_path)
        mock_instance = MagicMock()
        with patch(_TOPO_PARSER, return_value=mock_instance):
            sp.__dict__.pop("topology", None)
            topo = sp.topology
        assert topo is mock_instance

    def test_energy_is_cached(self, tmp_path):
        sp = self._build_sp(tmp_path)
        mock_instance = MagicMock()
        with patch(_ENERGY_PARSER, return_value=mock_instance):
            sp.__dict__.pop("energy", None)
            first = sp.energy
            second = sp.energy  # should NOT call EnergyParser again
        assert first is second


# ===========================================================================
# 6. topology_properties
# ===========================================================================


class TestTopologyProperties:
    """Tests for the topology_properties property."""

    def test_returns_list_of_strings(self, mock_topology_parser):
        sp = _inject_sp(mock_topology_parser, [])
        props = sp.topology_properties
        assert isinstance(props, list)
        assert all(isinstance(p, str) for p in props)

    def test_excludes_private_attributes(self, mock_topology_parser):
        sp = _inject_sp(mock_topology_parser, [])
        props = sp.topology_properties
        assert not any(p.startswith("_") for p in props)


# ===========================================================================
# 7. get() — topology branch
# ===========================================================================


class TestGetTopologyBranch:
    """Tests for SystemProperties.get() when the property lives in topology."""

    def test_returns_topology_attribute(self, mock_topology_parser, mock_energy_parser):
        mock_topology_parser.box_volume = 42.0
        sp = _inject_sp(mock_topology_parser, [mock_energy_parser])
        # make 'box_volume' appear in topology_properties
        with patch.object(
            type(sp),
            "topology_properties",
            new_callable=PropertyMock,
            return_value=["box_volume"],
        ):
            result = sp.get("box_volume")
        assert result == 42.0

    def test_electron_count_returned_for_elec_alias(self, mock_topology_parser, mock_energy_parser):
        mock_topology_parser.electron_count = 88
        sp = _inject_sp(mock_topology_parser, [mock_energy_parser])
        with patch.object(
            type(sp),
            "topology_properties",
            new_callable=PropertyMock,
            return_value=[],
        ):
            result = sp.get("elec_count")
        assert result == 88

    def test_electron_count_returned_for_z_prefix(self, mock_topology_parser, mock_energy_parser):
        mock_topology_parser.electron_count = 77
        sp = _inject_sp(mock_topology_parser, [mock_energy_parser])
        with patch.object(
            type(sp),
            "topology_properties",
            new_callable=PropertyMock,
            return_value=[],
        ):
            result = sp.get("z_density")
        assert result == 77


# ===========================================================================
# 8. get() — energy branch (generic property)
# ===========================================================================


class TestGetEnergyBranch:
    """Tests for SystemProperties.get() when the property comes from EnergyParser."""

    def _patch_topo_props(self, sp, props=None):
        return patch.object(
            type(sp),
            "topology_properties",
            new_callable=PropertyMock,
            return_value=props or [],
        )

    # --- units ---

    def test_get_units_returns_unit_dict(self, mock_topology_parser, mock_energy_parser):
        sp = _inject_sp(mock_topology_parser, [mock_energy_parser])
        with self._patch_topo_props(sp):
            result = sp.get("units")
        assert result == mock_energy_parser.units

    # --- averaged scalar ---

    def test_get_avg_returns_float(self, mock_topology_parser, mock_energy_parser):
        mock_energy_parser.get.return_value = np.array([100.0, 200.0, 300.0])
        sp = _inject_sp(mock_topology_parser, [mock_energy_parser])
        with self._patch_topo_props(sp):
            with patch.object(EnergyParser, "FLUCT_PROPS", new=[]):
                with patch(_RESOLVE_KEY, return_value="temperature"):
                    result = sp.get("temperature", avg=True)
        assert isinstance(result, float)
        assert result == pytest.approx(200.0)

    # --- time series ---

    def test_get_time_series_returns_tuple(self, mock_topology_parser, mock_energy_parser):
        times = np.array([0.0, 1.0, 2.0])
        values = np.array([10.0, 20.0, 30.0])
        mock_energy_parser._x_key = "Time"
        mock_energy_parser.get.side_effect = lambda key, start=0, units=None: times if key == "Time" else values
        sp = _inject_sp(mock_topology_parser, [mock_energy_parser])
        with self._patch_topo_props(sp):
            with patch.object(EnergyParser, "FLUCT_PROPS", new=[]):
                with patch(_RESOLVE_KEY, return_value="temperature"):
                    t, v = sp.get("temperature", avg=False, time_series=True)
        assert len(t) == len(v)

    # --- array (no time series) ---

    def test_get_array_returns_ndarray(self, mock_topology_parser, mock_energy_parser):
        times = np.array([0.0, 1.0, 2.0])
        values = np.array([10.0, 20.0, 30.0])
        mock_energy_parser._x_key = "Time"
        mock_energy_parser.get.side_effect = lambda key, start=0, units=None: times if key == "Time" else values
        sp = _inject_sp(mock_topology_parser, [mock_energy_parser])
        with self._patch_topo_props(sp):
            with patch.object(EnergyParser, "FLUCT_PROPS", new=[]):
                with patch(_RESOLVE_KEY, return_value="temperature"):
                    result = sp.get("temperature", avg=False, time_series=False)
        assert isinstance(result, np.ndarray)

    # --- multiple parsers averaged ---

    def test_get_averages_across_multiple_parsers(self, mock_topology_parser):
        p1 = _make_mock_energy_parser(get_return=np.array([100.0]))
        p2 = _make_mock_energy_parser(get_return=np.array([200.0]))
        sp = _inject_sp(mock_topology_parser, [p1, p2])
        with patch.object(
            type(sp),
            "topology_properties",
            new_callable=PropertyMock,
            return_value=[],
        ):
            with patch.object(EnergyParser, "FLUCT_PROPS", new=[]):
                with patch(_RESOLVE_KEY, return_value="temperature"):
                    result = sp.get("temperature", avg=True)
        assert result == pytest.approx(150.0)

    # --- empty values → nan ---

    def test_get_returns_nan_when_no_values(self, mock_topology_parser, mock_energy_parser):
        """When the parser returns an empty array, get() should return nan."""
        mock_energy_parser.get.return_value = np.array([])
        sp = _inject_sp(mock_topology_parser, [mock_energy_parser])
        with self._patch_topo_props(sp):
            with patch.object(EnergyParser, "FLUCT_PROPS", new=[]):
                with patch(_RESOLVE_KEY, return_value="temperature"):
                    # Suppress the expected RuntimeWarning from np.mean([])
                    with pytest.warns(RuntimeWarning, match="invalid value encountered"):
                        result = sp.get("temperature", avg=True)
        assert np.isnan(result)


# ===========================================================================
# 9. get() — derived thermodynamic properties
# ===========================================================================


class TestGetDerivedProperties:
    """Tests for Cp, Cv, enthalpy, compressibility, molar-volume, number-density."""

    def _call(self, sp, resolved_key, prop_key="dummy", **kwargs):
        with patch.object(
            type(sp),
            "topology_properties",
            new_callable=PropertyMock,
            return_value=[],
        ):
            with patch.object(EnergyParser, "FLUCT_PROPS", new=[]):
                with patch(_RESOLVE_KEY, return_value=resolved_key):
                    return sp.get(prop_key, avg=True, **kwargs)

    def test_cp(self, mock_topology_parser, mock_energy_parser):
        mock_energy_parser.heat_capacity_cp.return_value = 75.3
        sp = _inject_sp(mock_topology_parser, [mock_energy_parser])
        result = self._call(sp, "cp")
        assert result == pytest.approx(75.3)
        mock_energy_parser.heat_capacity_cp.assert_called_once()

    def test_cv(self, mock_topology_parser, mock_energy_parser):
        mock_energy_parser.heat_capacity_cv.return_value = 60.1
        sp = _inject_sp(mock_topology_parser, [mock_energy_parser])
        result = self._call(sp, "cv")
        assert result == pytest.approx(60.1)
        mock_energy_parser.heat_capacity_cv.assert_called_once()

    def test_enthalpy(self, mock_topology_parser, mock_energy_parser):
        mock_energy_parser.molar_enthalpy.return_value = 5000.0
        sp = _inject_sp(mock_topology_parser, [mock_energy_parser])
        result = self._call(sp, "enthalpy")
        assert result == pytest.approx(5000.0)
        mock_energy_parser.molar_enthalpy.assert_called_once()

    def test_isothermal_compressibility(self, mock_topology_parser, mock_energy_parser):
        mock_energy_parser.isothermal_compressibility.return_value = 4.5e-10
        sp = _inject_sp(mock_topology_parser, [mock_energy_parser])
        result = self._call(sp, "isothermal-compressibility")
        assert result == pytest.approx(4.5e-10)
        mock_energy_parser.isothermal_compressibility.assert_called_once()

    def test_molar_volume(self, mock_topology_parser, mock_energy_parser):
        mock_energy_parser.molar_volume.return_value = 0.018
        mock_energy_parser.units = {"molar-volume": "L/mol"}
        sp = _inject_sp(mock_topology_parser, [mock_energy_parser])
        result = self._call(sp, "molar-volume")
        assert result == pytest.approx(0.018)
        mock_energy_parser.molar_volume.assert_called_once()

    def test_number_density(self, mock_topology_parser, mock_energy_parser):
        mock_energy_parser.molar_volume.return_value = 0.5
        mock_energy_parser.units = {"molar-volume": "L/mol"}
        sp = _inject_sp(mock_topology_parser, [mock_energy_parser])
        result = self._call(sp, "number-density")
        assert result == pytest.approx(2.0)  # 1 / 0.5


# ===========================================================================
# 10. timeseries_plotter
# ===========================================================================


class TestTimeseriesPlotter:
    """Tests for SystemProperties.timeseries_plotter."""

    def test_returns_plotter_instance(self):
        sp = object.__new__(SystemProperties)
        mock_instance = MagicMock()
        # patch TimeseriesPlotter where it is imported in the module
        with patch(_TIMESERIES, return_value=mock_instance) as MockPlotter:
            result = sp.timeseries_plotter(start=100)
            MockPlotter.assert_called_once_with(sp, start=100)
        assert result is mock_instance

    def test_default_start_is_zero(self):
        sp = object.__new__(SystemProperties)
        with patch(_TIMESERIES, return_value=MagicMock()) as MockPlotter:
            sp.timeseries_plotter()
            _, kwargs = MockPlotter.call_args
            assert kwargs.get("start", 0) == 0


# ===========================================================================
# 11. Edge cases
# ===========================================================================


class TestEdgeCases:
    """Miscellaneous edge-case and regression tests."""

    def test_duplicate_times_removed_in_array_output(self, mock_topology_parser):
        """Duplicate time entries must be deduplicated in array output."""
        times = np.array([0.0, 0.0, 1.0, 2.0])  # duplicate at t=0
        values = np.array([10.0, 99.0, 20.0, 30.0])

        parser = _make_mock_energy_parser()
        parser._x_key = "Time"
        parser.get.side_effect = lambda key, start=0, units=None: times if key == "Time" else values

        sp = _inject_sp(mock_topology_parser, [parser])

        with patch.object(
            type(sp),
            "topology_properties",
            new_callable=PropertyMock,
            return_value=[],
        ):
            with patch.object(EnergyParser, "FLUCT_PROPS", new=[]):
                with patch(_RESOLVE_KEY, return_value="temperature"):
                    result = sp.get("temperature", avg=False, time_series=False)

        # After dedup, t=0 appears once → 3 rows total
        assert len(result) == 3
        assert result[0] == pytest.approx(10.0)  # first occurrence kept

    def test_get_abspath_with_absolute_filename(self, tmp_path):
        f = tmp_path / "run.edr"
        f.touch()
        result = SystemProperties._get_abspath(None, str(f))
        assert result == f.resolve()

    def test_find_files_sorted_output(self, tmp_path):
        for name in ("c.edr", "a.edr", "b.edr"):
            (tmp_path / name).touch()
        result = SystemProperties._find_files_in_path(suffix=".edr", path=tmp_path)
        names = [f.name for f in result]
        assert names == sorted(names)

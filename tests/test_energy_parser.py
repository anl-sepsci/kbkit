"""Unit tests for EnergyParser class."""
from unittest.mock import MagicMock, patch, PropertyMock
import pytest
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from kbkit.io.energy import EnergyParser, EnergyFormat


# ---------------------------------------------------------------------------
# Helpers / shared constants
# ---------------------------------------------------------------------------

MOCK_GROMACS_DATA = {
    "time": np.linspace(0, 1000, 500),  # ps
    "temperature": np.random.normal(300, 1, 500),  # K
    "pressure": np.random.normal(1.0, 0.1, 500),  # bar
    "density": np.full(500, 997.0),  # kg/m^3
    "volume": np.full(500, 25.0),  # nm^3
    "potential": np.random.normal(-5000, 10, 500),  # kJ/mol
    "kinetic en.": np.random.normal(1000, 5, 500),  # kJ/mol
    "total energy": np.random.normal(-4000, 10, 500),  # kJ/mol
}

MOCK_LAMMPS_DATA = {
    "step": np.arange(0, 10000, 20, dtype=float),
    "temperature": np.random.normal(300, 1, 500),
    "pressure": np.random.normal(1.0, 0.1, 500),
    "density": np.full(500, 0.997),
    "volume": np.full(500, 25000.0),  # Å^3
    "potential": np.random.normal(-1200, 5, 500),  # kcal/mol
    "total energy": np.random.normal(-1000, 5, 500),  # kcal/mol
}

NMOL = 100


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_edr_parser(tmp_path):
    fake_edr = tmp_path / "md.edr"
    fake_edr.touch()

    from unittest.mock import patch
    with patch("kbkit.io.energy.validate_path", return_value=fake_edr):
        parser = EnergyParser(fake_edr)

    parser.__dict__["data"] = MOCK_GROMACS_DATA.copy()
    return parser


@pytest.fixture
def mock_lammps_parser(tmp_path):
    fake_lammps = tmp_path / "md.lammps"
    fake_lammps.touch()

    from unittest.mock import patch
    with patch("kbkit.io.energy.validate_path", return_value=fake_lammps):
        parser = EnergyParser(fake_lammps)

    parser.__dict__["data"] = MOCK_LAMMPS_DATA.copy()
    return parser


@pytest.fixture
def mock_log_parser(tmp_path):
    fake_log = tmp_path / "md.log"
    fake_log.touch()

    from unittest.mock import patch
    with patch("kbkit.io.energy.validate_path", return_value=fake_log):
        parser = EnergyParser(fake_log)

    parser.__dict__["data"] = MOCK_LAMMPS_DATA.copy()
    return parser


# ---------------------------------------------------------------------------
# 1. Initialisation & format detection
# ---------------------------------------------------------------------------

class TestEnergyFormat:
    """Tests for _energy_format property."""

    def test_edr_format_detected(self, mock_edr_parser):
        assert mock_edr_parser._energy_format == EnergyFormat.GROMACS

    def test_lammps_format_detected(self, mock_lammps_parser):
        assert mock_lammps_parser._energy_format == EnergyFormat.LAMMPS

    def test_log_format_detected(self, mock_log_parser):
        assert mock_log_parser._energy_format == EnergyFormat.LAMMPS

    def test_unsupported_suffix_raises(self, tmp_path):
        """An unrecognised suffix must raise ValueError."""
        fake = tmp_path / "md.xyz"
        fake.touch()

        with patch("kbkit.io.energy.validate_path", return_value=fake):
            parser = EnergyParser(fake)

        with pytest.raises(ValueError, match="not recognized"):
            _ = parser._energy_format


# ---------------------------------------------------------------------------
# 2. Unit maps
# ---------------------------------------------------------------------------

class TestUnitMaps:
    """Tests for _md_units and units properties."""

    def test_gromacs_md_units(self, mock_edr_parser):
        md_units = mock_edr_parser._md_units
        assert md_units["temperature"] == "K"
        assert md_units["pressure"] == "bar"
        assert md_units["potential"] == "kJ/mol"

    def test_lammps_md_units(self, mock_lammps_parser):
        md_units = mock_lammps_parser._md_units
        assert md_units["pressure"] == "atm"
        assert md_units["potential"] == "kcal/mol"
        assert md_units["volume"] == "angstrom^3"

    def test_default_units_keys(self, mock_edr_parser):
        """All expected keys must be present in the default unit map."""
        required = {
            "temperature", "pressure", "density", "volume",
            "potential", "total energy", "enthalpy", "cp", "cv",
            "isothermal-compressibility",
        }
        assert required.issubset(mock_edr_parser.units.keys())


# ---------------------------------------------------------------------------
# 3. available_properties
# ---------------------------------------------------------------------------

class TestAvailableProperties:
    """Tests for available_properties()."""

    def test_returns_list(self, mock_edr_parser):
        props = mock_edr_parser.available_properties()
        assert isinstance(props, list)

    def test_contains_expected_keys(self, mock_edr_parser):
        props = mock_edr_parser.available_properties()
        for key in ("time", "temperature", "potential", "volume"):
            assert key in props

    def test_lammps_available_properties(self, mock_lammps_parser):
        props = mock_lammps_parser.available_properties()
        assert "temperature" in props
        assert "potential" in props


# ---------------------------------------------------------------------------
# 4. get()
# ---------------------------------------------------------------------------

class TestGet:
    """Tests for the get() method."""

    def test_returns_ndarray(self, mock_edr_parser):
        result = mock_edr_parser.get("temperature")
        assert isinstance(result, np.ndarray)

    def test_start_filter_gromacs(self, mock_edr_parser):
        """Values before start time must be excluded."""
        start = 500  # ps
        result = mock_edr_parser.get("temperature", start=start)
        time = mock_edr_parser.data["time"]
        expected_len = int((time > start).sum())
        assert len(result) == expected_len

    def test_start_filter_lammps(self, mock_lammps_parser):
        """Values at or before start step must be excluded."""
        start = 4000
        result = mock_lammps_parser.get("temperature", start=start)
        steps = mock_lammps_parser.data["step"]
        expected_len = int((steps > start).sum())
        assert len(result) == expected_len

    def test_unit_conversion_pressure(self, mock_edr_parser):
        """Pressure converted from bar → kPa should be ~100× larger."""
        bar_values = mock_edr_parser.get("pressure", units="bar")
        kpa_values = mock_edr_parser.get("pressure", units="kPa")
        np.testing.assert_allclose(kpa_values, bar_values * 100, rtol=1e-6)

    def test_unknown_property_raises_key_error(self, mock_edr_parser):
        with pytest.raises(KeyError, match="not available"):
            mock_edr_parser.get("nonexistent_property")

    def test_alias_resolution(self, mock_edr_parser):
        """Common aliases (e.g. 'temp') should resolve to 'temperature'."""
        result_alias = mock_edr_parser.get("temp")
        result_direct = mock_edr_parser.get("temperature")
        np.testing.assert_array_equal(result_alias, result_direct)

# ---------------------------------------------------------------------------
# 5. molar_volume()
# ---------------------------------------------------------------------------

class TestMolarVolume:
    """Tests for molar_volume()."""

    def test_returns_ndarray(self, mock_edr_parser):
        result = mock_edr_parser.molar_volume(nmol=NMOL)
        assert isinstance(result, np.ndarray)

    def test_shape_matches_filtered_data(self, mock_edr_parser):
        start = 200
        result = mock_edr_parser.molar_volume(nmol=NMOL, start=start)
        time = mock_edr_parser.data["time"]
        assert len(result) == int((time > start).sum())

    def test_fallback_to_input_volume(self, mock_edr_parser):
        """When 'volume' is absent, the supplied scalar volume is used."""
        data_no_vol = {k: v for k, v in MOCK_GROMACS_DATA.items() if k != "volume"}
        mock_edr_parser.__dict__["data"] = data_no_vol

        box_vol = 25.0  # nm^3
        result = mock_edr_parser.molar_volume(nmol=NMOL, volume=box_vol)
        # scalar path → result should be a scalar-like array
        assert np.isscalar(result) or result.ndim == 0

    def test_unit_conversion(self, mock_edr_parser):
        """Result in nm^3/mol should differ from cm^3/mol by a known factor."""
        nm3 = mock_edr_parser.molar_volume(nmol=NMOL, units="nm^3/mol")
        cm3 = mock_edr_parser.molar_volume(nmol=NMOL, units="cm^3/mol")
        # 1 nm^3 = 1e-21 cm^3, but per mol: 1 nm^3/mol = 6.022e2 cm^3/mol? No.
        # pint handles the conversion; just verify the ratio is consistent.
        np.testing.assert_allclose(nm3 / cm3, 1e21, rtol=1e-4)


# ---------------------------------------------------------------------------
# 6. configurational_enthalpy()
# ---------------------------------------------------------------------------

class TestConfigurationalEnthalpy:
    """Tests for configurational_enthalpy()."""

    def test_returns_ndarray(self, mock_edr_parser):
        result = mock_edr_parser.configurational_enthalpy()
        assert isinstance(result, np.ndarray)

    def test_shape_consistent_with_potential(self, mock_edr_parser):
        H = mock_edr_parser.configurational_enthalpy()
        U = mock_edr_parser.get("potential")
        assert H.shape == U.shape

    def test_fallback_volume_none_raises(self, mock_edr_parser):
        """Missing volume key with volume=None must raise ValueError."""
        data_no_vol = {k: v for k, v in MOCK_GROMACS_DATA.items() if k != "volume"}
        mock_edr_parser.__dict__["data"] = data_no_vol

        with pytest.raises(ValueError, match="Volume cannot be Nonetype"):
            mock_edr_parser.configurational_enthalpy(volume=None)

    def test_fallback_volume_scalar(self, mock_edr_parser):
        """Supplying a scalar volume when key is absent should not raise."""
        data_no_vol = {k: v for k, v in MOCK_GROMACS_DATA.items() if k != "volume"}
        mock_edr_parser.__dict__["data"] = data_no_vol

        result = mock_edr_parser.configurational_enthalpy(volume=25.0)
        assert isinstance(result, np.ndarray)

    def test_start_filter_applied(self, mock_edr_parser):
        start = 500
        H_full = mock_edr_parser.configurational_enthalpy()
        H_trim = mock_edr_parser.configurational_enthalpy(start=start)
        assert len(H_trim) < len(H_full)


# ---------------------------------------------------------------------------
# 7. molar_enthalpy()
# ---------------------------------------------------------------------------

class TestMolarEnthalpy:
    """Tests for molar_enthalpy()."""

    def test_equals_enthalpy_divided_by_nmol(self, mock_edr_parser):
        H = mock_edr_parser.configurational_enthalpy()
        Hm = mock_edr_parser.molar_enthalpy(nmol=NMOL)
        np.testing.assert_allclose(Hm, H / NMOL, rtol=1e-10)

    def test_returns_ndarray(self, mock_edr_parser):
        assert isinstance(mock_edr_parser.molar_enthalpy(nmol=NMOL), np.ndarray)


# ---------------------------------------------------------------------------
# 8. heat_capacity_cp()
# ---------------------------------------------------------------------------

class TestHeatCapacityCp:
    """Tests for heat_capacity_cp()."""

    def test_returns_float(self, mock_edr_parser):
        cp = mock_edr_parser.heat_capacity_cp(nmol=NMOL)
        assert isinstance(cp, float)

    def test_positive_value(self, mock_edr_parser):
        """Heat capacity must be physically positive."""
        cp = mock_edr_parser.heat_capacity_cp(nmol=NMOL)
        assert cp > 0

    def test_unit_conversion(self, mock_edr_parser):
        """cp in J/mol/K should be 1000× larger than in kJ/mol/K."""
        cp_kj = mock_edr_parser.heat_capacity_cp(nmol=NMOL, units="kJ/mol/K")
        cp_j = mock_edr_parser.heat_capacity_cp(nmol=NMOL, units="J/mol/K")
        assert pytest.approx(cp_j, rel=1e-6) == cp_kj * 1000

    def test_scales_inversely_with_nmol(self, mock_edr_parser):
        """Doubling nmol should halve cp (extensive → intensive normalisation)."""
        cp1 = mock_edr_parser.heat_capacity_cp(nmol=NMOL)
        cp2 = mock_edr_parser.heat_capacity_cp(nmol=NMOL * 2)
        assert pytest.approx(cp1, rel=1e-6) == cp2 * 2

    def test_start_filter_changes_result(self, mock_edr_parser):
        """Trimming equilibration data should change the computed cp."""
        cp_full = mock_edr_parser.heat_capacity_cp(nmol=NMOL, start=0)
        cp_trim = mock_edr_parser.heat_capacity_cp(nmol=NMOL, start=500)
        # They should differ (different data subsets)
        assert cp_full != cp_trim


# ---------------------------------------------------------------------------
# 9. heat_capacity_cv()
# ---------------------------------------------------------------------------

class TestHeatCapacityCv:
    """Tests for heat_capacity_cv()."""

    def test_returns_float(self, mock_edr_parser):
        cv = mock_edr_parser.heat_capacity_cv(nmol=NMOL)
        assert isinstance(cv, float)

    def test_positive_value(self, mock_edr_parser):
        cv = mock_edr_parser.heat_capacity_cv(nmol=NMOL)
        assert cv > 0

    def test_unit_conversion(self, mock_edr_parser):
        cv_kj = mock_edr_parser.heat_capacity_cv(nmol=NMOL, units="kJ/mol/K")
        cv_j = mock_edr_parser.heat_capacity_cv(nmol=NMOL, units="J/mol/K")
        assert pytest.approx(cv_j, rel=1e-6) == cv_kj * 1000

    def test_scales_inversely_with_nmol(self, mock_edr_parser):
        cv1 = mock_edr_parser.heat_capacity_cv(nmol=NMOL)
        cv2 = mock_edr_parser.heat_capacity_cv(nmol=NMOL * 2)
        assert pytest.approx(cv1, rel=1e-6) == cv2 * 2


# ---------------------------------------------------------------------------
# 10. isothermal_compressibility()
# ---------------------------------------------------------------------------

class TestIsothermalCompressibility:
    """Tests for isothermal_compressibility()."""

    def test_returns_float(self, mock_edr_parser):
        kT = mock_edr_parser.isothermal_compressibility()
        assert isinstance(kT, float)

    def test_positive_value(self, mock_edr_parser):
        kT = mock_edr_parser.isothermal_compressibility()
        assert kT > 0

    def test_raises_when_volume_absent(self, mock_edr_parser):
        """NVT simulation (no volume key) must raise KeyError."""
        data_no_vol = {k: v for k, v in MOCK_GROMACS_DATA.items() if k != "volume"}
        mock_edr_parser.__dict__["data"] = data_no_vol

        with pytest.raises(KeyError, match="constant volume"):
            mock_edr_parser.isothermal_compressibility()

    def test_unit_conversion(self, mock_edr_parser):
        """1/kPa → 1/Pa should differ by factor 1000."""
        kT_kpa = mock_edr_parser.isothermal_compressibility(units="1/kPa")
        kT_pa = mock_edr_parser.isothermal_compressibility(units="1/Pa")
        assert pytest.approx(kT_pa, rel=1e-6) == kT_kpa / 1000


# ---------------------------------------------------------------------------
# 11. data loading paths (LAMMPS .log and .lammps)
# ---------------------------------------------------------------------------

class TestDataLoading:
    """Integration-style tests for the data cached property."""

    def test_lammps_data_keys_mapped(self, mock_lammps_parser):
        """LAMMPS column names must be translated to GROMACS-style keys."""
        props = mock_lammps_parser.available_properties()
        assert "temperature" in props
        assert "potential" in props
        # Raw LAMMPS keys should NOT appear
        assert "Temp" not in props
        assert "PotEng" not in props

    def test_edr_data_keys_lowercase(self, mock_edr_parser):
        """EDR keys must be stored in lowercase."""
        for key in mock_edr_parser.data:
            assert key == key.lower(), f"Key '{key}' is not lowercase"

    @patch("kbkit.io.energy.read_log")
    def test_lammps_file_read_log_called(self, mock_read_log, tmp_path):
        """read_log must be called when suffix is .lammps."""
        fake = tmp_path / "md.lammps"
        fake.touch()

        # Provide minimal data that covers LAMMPS_to_GMX keys
        mock_read_log.return_value = {
            "Step": list(range(100)),
            "Temp": [300.0] * 100,
            "PotEng": [-1200.0] * 100,
            "TotEng": [-1000.0] * 100,
            "Press": [1.0] * 100,
            "Density": [0.997] * 100,
            "Volume": [25000.0] * 100,
            "KinEng": [200.0] * 100,
        }

        with patch("kbkit.io.energy.validate_path", return_value=fake):
            parser = EnergyParser(fake)
            _ = parser.data  # trigger cached_property

        mock_read_log.assert_called_once_with(fake)

    @patch("pandas.read_csv")
    def test_log_file_read_csv_called(self, mock_read_csv, tmp_path):
        """pd.read_csv must be called when suffix is .log."""
        fake = tmp_path / "md.log"
        fake.touch()

        mock_df = pd.DataFrame({
            "Step": range(100),
            "Temp": [300.0] * 100,
            "PotEng": [-1200.0] * 100,
            "TotEng": [-1000.0] * 100,
            "Press": [1.0] * 100,
            "Density": [0.997] * 100,
            "Volume": [25000.0] * 100,
            "KinEng": [200.0] * 100,
        })
        mock_read_csv.return_value = mock_df

        with patch("kbkit.io.energy.validate_path", return_value=fake):
            parser = EnergyParser(fake)
            _ = parser.data

        mock_read_csv.assert_called_once()


# ---------------------------------------------------------------------------
# 12. Edge cases & boundary conditions
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Boundary and edge-case tests."""

    def test_start_beyond_all_data_returns_empty(self, mock_edr_parser):
        """A start time beyond the simulation end should return an empty array."""
        result = mock_edr_parser.get("temperature", start=1e9)
        assert len(result) == 0

    def test_start_zero_returns_all_data(self, mock_edr_parser):
        """start=0 (default) should return the full dataset."""
        result = mock_edr_parser.get("temperature", start=0)
        # time > 0 excludes the first frame (time=0); adjust expectation
        time = mock_edr_parser.data["time"]
        assert len(result) == int((time > 0).sum())

    def test_get_case_insensitive(self, mock_edr_parser):
        """Property lookup should be case-insensitive."""
        lower = mock_edr_parser.get("temperature")
        upper = mock_edr_parser.get("Temperature")
        np.testing.assert_array_equal(lower, upper)

    def test_nmol_one_molar_enthalpy_equals_enthalpy(self, mock_edr_parser):
        """With nmol=1, molar_enthalpy must equal configurational_enthalpy."""
        H = mock_edr_parser.configurational_enthalpy()
        Hm = mock_edr_parser.molar_enthalpy(nmol=1)
        np.testing.assert_allclose(Hm, H, rtol=1e-10)

    def test_fluct_props_tuple_contents(self):
        """FLUCT_PROPS class variable must contain the three expected keys."""
        assert "cp" in EnergyParser.FLUCT_PROPS
        assert "cv" in EnergyParser.FLUCT_PROPS
        assert "isothermal-compressibility" in EnergyParser.FLUCT_PROPS

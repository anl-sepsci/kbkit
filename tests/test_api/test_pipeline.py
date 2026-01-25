"""
Unit tests for the Pipeline module.

This test suite provides comprehensive coverage of the Pipeline class,
including initialization, system building, property calculations, and plotting.
"""
import warnings

# Suppress NumPy/SciPy compatibility warning
warnings.filterwarnings('ignore', message='numpy.ndarray size changed', category=RuntimeWarning)

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from kbkit.api.pipeline import Pipeline
from kbkit.kbi.calculator import KBICalculator
from kbkit.kbi.thermodynamics import KBThermo
from kbkit.schema.property_result import PropertyResult
from kbkit.systems.collection import SystemCollection
from kbkit.visualization.kbi import KBIAnalysisPlotter
from kbkit.visualization.thermo import ThermoPlotter
from kbkit.visualization.timeseries import TimeseriesPlotter


@pytest.fixture
def mock_system_collection():
    """Create a mock SystemCollection."""
    mock_sc = Mock(spec=SystemCollection)
    mock_sc.molecules = ["MOL1", "MOL2"]
    mock_sc.n_i = 2
    mock_sc.n_sys = 3
    mock_sc.x = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    mock_sc.units = {
        "Temperature": "K",
        "Pressure": "bar",
        "Density": "kg/m^3",
        "Time": "ps"  # Should be skipped
    }

    # Mock property methods
    mock_sc.simulated_property.return_value = PropertyResult(
        name="Temperature",
        value=np.array([298.15, 298.15, 298.15]),
        units="K",
        property_type="simulated"
    )
    mock_sc.ideal_property.return_value = PropertyResult(
        name="Temperature",
        value=np.array([298.15, 298.15, 298.15]),
        units="K",
        property_type="ideal"
    )
    mock_sc.excess_property.return_value = PropertyResult(
        name="Temperature",
        value=np.array([0.0, 0.0, 0.0]),
        units="K",
        property_type="excess"
    )

    return mock_sc


@pytest.fixture
def mock_kbi_calculator():
    """Create a mock KBICalculator."""
    mock_calc = Mock(spec=KBICalculator)

    # Mock KBI result
    mock_kbi_result = PropertyResult(
        name="kbi",
        value=np.random.rand(3, 2, 2),
        units="cm^3/mol",
        metadata={}
    )
    mock_calc.kbi.return_value = mock_kbi_result

    return mock_calc


@pytest.fixture
def mock_kb_thermo():
    """Create a mock KBThermo."""
    mock_thermo = Mock(spec=KBThermo)

    # Mock results
    mock_thermo.results = {
        "kbi": PropertyResult(
            name="kbi",
            value=np.random.rand(3, 2, 2),
            units="cm^3/mol"
        ),
        "ln_activity_coef": PropertyResult(
            name="ln_activity_coef",
            value=np.random.rand(3, 2),
            units="dimensionless"
        ),
        "g_ex": PropertyResult(
            name="g_ex",
            value=np.random.rand(3),
            units="kJ/mol"
        ),
        "non_property_result": "should be filtered"
    }

    return mock_thermo


class TestPipelineInitialization:
    """Test Pipeline initialization."""

    def test_init_with_minimal_parameters(self):
        """Test initialization with minimal parameters."""
        pipeline = Pipeline()

        assert pipeline.base_path is None
        assert pipeline.base_systems is None
        assert pipeline.pure_path is None
        assert pipeline.pure_systems is None
        assert pipeline.rdf_dir == ""
        assert pipeline.start_time == 10000
        assert pipeline.include_mode == "npt"

    def test_init_with_all_parameters(self):
        """Test initialization with all parameters."""
        pipeline = Pipeline(
            base_path="/path/to/base",
            base_systems=["sys1", "sys2"],
            pure_path="/path/to/pure",
            pure_systems=["pure1", "pure2"],
            rdf_dir="rdfs",
            start_time=5000,
            include_mode="nvt",
            ignore_convergence_errors=True,
            rdf_convergence_thresholds=(1e-4, 1e-3),
            rdf_tail_length=2.5,
            kbi_correct_rdf_convergence=False,
            kbi_apply_damping=False,
            kbi_extrapolate_thermodynamic_limit=False,
            activity_integration_type="polynomial",
            activity_polynomial_degree=7,
            molecule_map={"MOL1": "Molecule 1"}
        )

        assert pipeline.base_path == "/path/to/base"
        assert pipeline.base_systems == ["sys1", "sys2"]
        assert pipeline.pure_path == "/path/to/pure"
        assert pipeline.pure_systems == ["pure1", "pure2"]
        assert pipeline.rdf_dir == "rdfs"
        assert pipeline.start_time == 5000
        assert pipeline.include_mode == "nvt"
        assert pipeline.ignore_convergence_errors is True
        assert pipeline.rdf_convergence_thresholds == (1e-4, 1e-3)
        assert pipeline.rdf_tail_length == 2.5
        assert pipeline.kbi_correct_rdf_convergence is False
        assert pipeline.kbi_apply_damping is False
        assert pipeline.kbi_extrapolate_thermodynamic_limit is False
        assert pipeline.activity_integration_type == "polynomial"
        assert pipeline.activity_polynomial_degree == 7
        assert pipeline.molecule_map == {"MOL1": "Molecule 1"}

    def test_init_converts_activity_integration_type_to_string(self):
        """Test that activity_integration_type is converted to string."""
        pipeline = Pipeline(activity_integration_type="numerical")

        assert isinstance(pipeline.activity_integration_type, str)
        assert pipeline.activity_integration_type == "numerical"

    def test_init_converts_polynomial_degree_to_int(self):
        """Test that activity_polynomial_degree is converted to int."""
        pipeline = Pipeline(activity_polynomial_degree=5.0)

        assert isinstance(pipeline.activity_polynomial_degree, int)
        assert pipeline.activity_polynomial_degree == 5


class TestPipelineBuildSystems:
    """Test _build_systems method."""

    @patch('kbkit.api.pipeline.SystemCollection.load')
    def test_build_systems_calls_load(self, mock_load):
        """Test that _build_systems calls SystemCollection.load."""
        mock_sc = Mock(spec=SystemCollection)
        mock_load.return_value = mock_sc

        pipeline = Pipeline(
            base_path="/path/to/base",
            base_systems=["sys1"],
            pure_path="/path/to/pure",
            pure_systems=["pure1"],
            rdf_dir="rdfs",
            start_time=5000,
            include_mode="nvt"
        )

        result = pipeline._build_systems()

        mock_load.assert_called_once_with(
            base_path="/path/to/base",
            base_systems=["sys1"],
            pure_path="/path/to/pure",
            pure_systems=["pure1"],
            rdf_dir="rdfs",
            start_time=5000,
            include_mode="nvt"
        )
        assert result == mock_sc

    @patch('kbkit.api.pipeline.SystemCollection.load')
    def test_build_systems_with_defaults(self, mock_load):
        """Test _build_systems with default parameters."""
        mock_sc = Mock(spec=SystemCollection)
        mock_load.return_value = mock_sc

        pipeline = Pipeline()
        result = pipeline._build_systems()

        mock_load.assert_called_once_with(
            base_path=None,
            base_systems=None,
            pure_path=None,
            pure_systems=None,
            rdf_dir="",
            start_time=10000,
            include_mode="npt"
        )


class TestPipelineSystemsProperty:
    """Test systems cached property."""

    @patch('kbkit.api.pipeline.SystemCollection.load')
    def test_systems_property_calls_build_systems(self, mock_load, mock_system_collection):
        """Test that systems property calls _build_systems."""
        mock_load.return_value = mock_system_collection

        pipeline = Pipeline()

        result = pipeline.systems

        assert result == mock_system_collection
        mock_load.assert_called_once()

    @patch('kbkit.api.pipeline.SystemCollection.load')
    def test_systems_property_is_cached(self, mock_load, mock_system_collection):
        """Test that systems property is cached."""
        mock_load.return_value = mock_system_collection

        pipeline = Pipeline()

        result1 = pipeline.systems
        result2 = pipeline.systems

        assert result1 is result2
        mock_load.assert_called_once()


class TestPipelineCalculatorProperty:
    """Test calculator cached property."""

    @patch('kbkit.api.pipeline.KBICalculator')
    @patch('kbkit.api.pipeline.SystemCollection.load')
    def test_calculator_property_creates_kbi_calculator(
        self, mock_load, mock_calc_class, mock_system_collection, mock_kbi_calculator
    ):
        """Test that calculator property creates KBICalculator."""
        mock_load.return_value = mock_system_collection
        mock_calc_class.return_value = mock_kbi_calculator

        pipeline = Pipeline(
            ignore_convergence_errors=True,
            rdf_convergence_thresholds=(1e-4, 1e-3),
            rdf_tail_length=2.5,
            kbi_correct_rdf_convergence=False,
            kbi_apply_damping=False,
            kbi_extrapolate_thermodynamic_limit=False
        )

        result = pipeline.calculator

        mock_calc_class.assert_called_once_with(
            systems=mock_system_collection,
            ignore_convergence_errors=True,
            convergence_thresholds=(1e-4, 1e-3),
            tail_length=2.5,
            correct_rdf_convergence=False,
            apply_damping=False,
            extrapolate_thermodynamic_limit=False
        )
        assert result == mock_kbi_calculator

    @patch('kbkit.api.pipeline.KBICalculator')
    @patch('kbkit.api.pipeline.SystemCollection.load')
    def test_calculator_property_is_cached(
        self, mock_load, mock_calc_class, mock_system_collection, mock_kbi_calculator
    ):
        """Test that calculator property is cached."""
        mock_load.return_value = mock_system_collection
        mock_calc_class.return_value = mock_kbi_calculator

        pipeline = Pipeline()

        result1 = pipeline.calculator
        result2 = pipeline.calculator

        assert result1 is result2
        mock_calc_class.assert_called_once()


class TestPipelineKBIResProperty:
    """Test kbi_res property."""

    @patch('kbkit.api.pipeline.KBICalculator')
    @patch('kbkit.api.pipeline.SystemCollection.load')
    def test_kbi_res_property_calls_calculator_kbi(
        self, mock_load, mock_calc_class, mock_system_collection, mock_kbi_calculator
    ):
        """Test that kbi_res property calls calculator.kbi()."""
        mock_load.return_value = mock_system_collection
        mock_calc_class.return_value = mock_kbi_calculator

        mock_kbi_result = PropertyResult(
            name="kbi",
            value=np.random.rand(3, 2, 2),
            units="cm^3/mol"
        )
        mock_kbi_calculator.kbi.return_value = mock_kbi_result

        pipeline = Pipeline()

        result = pipeline.kbi_res

        mock_kbi_calculator.kbi.assert_called_once_with(units="cm^3/mol")
        assert result == mock_kbi_result

    @patch('kbkit.api.pipeline.KBICalculator')
    @patch('kbkit.api.pipeline.SystemCollection.load')
    def test_kbi_res_property_not_cached(
        self, mock_load, mock_calc_class, mock_system_collection, mock_kbi_calculator
    ):
        """Test that kbi_res property is not cached (calls kbi each time)."""
        mock_load.return_value = mock_system_collection
        mock_calc_class.return_value = mock_kbi_calculator

        mock_kbi_result = PropertyResult(
            name="kbi",
            value=np.random.rand(3, 2, 2),
            units="cm^3/mol"
        )
        mock_kbi_calculator.kbi.return_value = mock_kbi_result

        pipeline = Pipeline()

        result1 = pipeline.kbi_res
        result2 = pipeline.kbi_res

        # Should call kbi twice (not cached)
        assert mock_kbi_calculator.kbi.call_count == 2


class TestPipelineThermoProperty:
    """Test thermo cached property."""

    @patch('kbkit.api.pipeline.KBThermo')
    @patch('kbkit.api.pipeline.KBICalculator')
    @patch('kbkit.api.pipeline.SystemCollection.load')
    def test_thermo_property_creates_kb_thermo(
        self, mock_load, mock_calc_class, mock_thermo_class,
        mock_system_collection, mock_kbi_calculator, mock_kb_thermo
    ):
        """Test that thermo property creates KBThermo."""
        mock_load.return_value = mock_system_collection
        mock_calc_class.return_value = mock_kbi_calculator
        mock_thermo_class.return_value = mock_kb_thermo

        mock_kbi_result = PropertyResult(
            name="kbi",
            value=np.random.rand(3, 2, 2),
            units="cm^3/mol"
        )
        mock_kbi_calculator.kbi.return_value = mock_kbi_result

        pipeline = Pipeline(
            activity_integration_type="polynomial",
            activity_polynomial_degree=7
        )

        result = pipeline.thermo

        mock_thermo_class.assert_called_once_with(
            systems=mock_system_collection,
            kbi=mock_kbi_result,
            activity_integration_type="polynomial",
            activity_polynomial_degree=7
        )
        assert result == mock_kb_thermo

    @patch('kbkit.api.pipeline.KBThermo')
    @patch('kbkit.api.pipeline.KBICalculator')
    @patch('kbkit.api.pipeline.SystemCollection.load')
    def test_thermo_property_is_cached(
        self, mock_load, mock_calc_class, mock_thermo_class,
        mock_system_collection, mock_kbi_calculator, mock_kb_thermo
    ):
        """Test that thermo property is cached."""
        mock_load.return_value = mock_system_collection
        mock_calc_class.return_value = mock_kbi_calculator
        mock_thermo_class.return_value = mock_kb_thermo

        mock_kbi_result = PropertyResult(
            name="kbi",
            value=np.random.rand(3, 2, 2),
            units="cm^3/mol"
        )
        mock_kbi_calculator.kbi.return_value = mock_kbi_result

        pipeline = Pipeline()

        result1 = pipeline.thermo
        result2 = pipeline.thermo

        assert result1 is result2
        mock_thermo_class.assert_called_once()


class TestPipelineResultsProperty:
    """Test results cached property."""

    @patch('kbkit.api.pipeline.KBThermo')
    @patch('kbkit.api.pipeline.KBICalculator')
    @patch('kbkit.api.pipeline.SystemCollection.load')
    def test_results_property_includes_thermo_results(
        self, mock_load, mock_calc_class, mock_thermo_class,
        mock_system_collection, mock_kbi_calculator, mock_kb_thermo
    ):
        """Test that results includes KBThermo results."""
        mock_load.return_value = mock_system_collection
        mock_calc_class.return_value = mock_kbi_calculator
        mock_thermo_class.return_value = mock_kb_thermo

        mock_kbi_result = PropertyResult(
            name="kbi",
            value=np.random.rand(3, 2, 2),
            units="cm^3/mol"
        )
        mock_kbi_calculator.kbi.return_value = mock_kbi_result

        pipeline = Pipeline()

        results = pipeline.results

        # Should include PropertyResult objects from thermo.results
        assert "kbi" in results
        assert "ln_activity_coef" in results
        assert "g_ex" in results
        # Should not include non-PropertyResult objects
        assert "non_property_result" not in results

    @patch('kbkit.api.pipeline.KBThermo')
    @patch('kbkit.api.pipeline.KBICalculator')
    @patch('kbkit.api.pipeline.SystemCollection.load')
    def test_results_property_includes_system_properties(
        self, mock_load, mock_calc_class, mock_thermo_class,
        mock_system_collection, mock_kbi_calculator, mock_kb_thermo
    ):
        """Test that results includes system properties."""
        mock_load.return_value = mock_system_collection
        mock_calc_class.return_value = mock_kbi_calculator
        mock_thermo_class.return_value = mock_kb_thermo

        mock_kbi_result = PropertyResult(
            name="kbi",
            value=np.random.rand(3, 2, 2),
            units="cm^3/mol"
        )
        mock_kbi_calculator.kbi.return_value = mock_kbi_result

        pipeline = Pipeline()

        results = pipeline.results

        # Should include simulated, ideal, and excess for each property
        assert "simulated_temperature" in results
        assert "ideal_temperature" in results
        assert "excess_temperature" in results
        assert "simulated_pressure" in results
        assert "ideal_pressure" in results
        assert "excess_pressure" in results
        assert "simulated_density" in results
        assert "ideal_density" in results
        assert "excess_density" in results

    @patch('kbkit.api.pipeline.KBThermo')
    @patch('kbkit.api.pipeline.KBICalculator')
    @patch('kbkit.api.pipeline.SystemCollection.load')
    def test_results_property_skips_time_properties(
        self, mock_load, mock_calc_class, mock_thermo_class,
        mock_system_collection, mock_kbi_calculator, mock_kb_thermo
    ):
        """Test that results skips properties with 'time' in name."""
        mock_load.return_value = mock_system_collection
        mock_calc_class.return_value = mock_kbi_calculator
        mock_thermo_class.return_value = mock_kb_thermo

        mock_kbi_result = PropertyResult(
            name="kbi",
            value=np.random.rand(3, 2, 2),
            units="cm^3/mol"
        )
        mock_kbi_calculator.kbi.return_value = mock_kbi_result

        pipeline = Pipeline()

        results = pipeline.results

        # Should not include time-related properties
        assert "simulated_time" not in results
        assert "ideal_time" not in results
        assert "excess_time" not in results

    @patch('kbkit.api.pipeline.KBThermo')
    @patch('kbkit.api.pipeline.KBICalculator')
    @patch('kbkit.api.pipeline.SystemCollection.load')
    def test_results_property_normalizes_property_names(
        self, mock_load, mock_calc_class, mock_thermo_class,
        mock_system_collection, mock_kbi_calculator, mock_kb_thermo
    ):
        """Test that results normalizes property names (lowercase, replace -)."""
        mock_load.return_value = mock_system_collection
        mock_calc_class.return_value = mock_kbi_calculator
        mock_thermo_class.return_value = mock_kb_thermo

        # Add property with uppercase and hyphens
        mock_system_collection.units["Coul-SR"] = "kJ/mol"

        mock_kbi_result = PropertyResult(
            name="kbi",
            value=np.random.rand(3, 2, 2),
            units="cm^3/mol"
        )
        mock_kbi_calculator.kbi.return_value = mock_kbi_result

        pipeline = Pipeline()

        results = pipeline.results

        # Should normalize to lowercase and replace hyphens
        assert "simulated_coul_sr" in results


class TestPipelineTimeseriesPlotter:
    """Test timeseries_plotter method."""

    @patch('kbkit.api.pipeline.SystemCollection.load')
    def test_timeseries_plotter_calls_systems_method(
        self, mock_load, mock_system_collection
    ):
        """Test that timeseries_plotter calls systems.timeseries_plotter."""
        mock_load.return_value = mock_system_collection

        mock_plotter = Mock(spec=TimeseriesPlotter)
        mock_system_collection.timeseries_plotter.return_value = mock_plotter

        pipeline = Pipeline()

        result = pipeline.timeseries_plotter("system_1", start_time=5000)

        mock_system_collection.timeseries_plotter.assert_called_once_with(
            "system_1", 5000
        )
        assert result == mock_plotter


class TestPipelineKBIPlotter:
    """Test kbi_plotter property."""

    @patch('kbkit.api.pipeline.KBICalculator')
    @patch('kbkit.api.pipeline.SystemCollection.load')
    def test_kbi_plotter_calls_calculator_method(
        self, mock_load, mock_calc_class, mock_system_collection, mock_kbi_calculator
    ):
        """Test that kbi_plotter calls calculator.kbi_plotter."""
        mock_load.return_value = mock_system_collection
        mock_calc_class.return_value = mock_kbi_calculator

        mock_kbi_result = PropertyResult(
            name="kbi",
            value=np.random.rand(3, 2, 2),
            units="cm^3/mol"
        )
        mock_kbi_calculator.kbi.return_value = mock_kbi_result

        mock_plotter = Mock(spec=KBIAnalysisPlotter)
        mock_kbi_calculator.kbi_plotter.return_value = mock_plotter

        pipeline = Pipeline(molecule_map={"MOL1": "Molecule 1"})

        result = pipeline.kbi_plotter

        mock_kbi_calculator.kbi_plotter.assert_called_once_with(
            molecule_map={"MOL1": "Molecule 1"}
        )
        assert result == mock_plotter


class TestPipelineThermoPlotter:
    """Test thermo_plotter property."""

    @patch('kbkit.api.pipeline.KBThermo')
    @patch('kbkit.api.pipeline.KBICalculator')
    @patch('kbkit.api.pipeline.SystemCollection.load')
    def test_thermo_plotter_calls_thermo_method(
        self, mock_load, mock_calc_class, mock_thermo_class,
        mock_system_collection, mock_kbi_calculator, mock_kb_thermo
    ):
        """Test that thermo_plotter calls thermo.plotter."""
        mock_load.return_value = mock_system_collection
        mock_calc_class.return_value = mock_kbi_calculator
        mock_thermo_class.return_value = mock_kb_thermo

        mock_kbi_result = PropertyResult(
            name="kbi",
            value=np.random.rand(3, 2, 2),
            units="cm^3/mol"
        )
        mock_kbi_calculator.kbi.return_value = mock_kbi_result

        mock_plotter = Mock(spec=ThermoPlotter)
        mock_kb_thermo.plotter.return_value = mock_plotter

        pipeline = Pipeline(molecule_map={"MOL1": "Molecule 1"})

        result = pipeline.thermo_plotter

        mock_kb_thermo.plotter.assert_called_once_with(
            molecule_map={"MOL1": "Molecule 1"}
        )
        assert result == mock_plotter


class TestPipelineMakeFigures:
    """Test make_figures method."""

    @patch('kbkit.api.pipeline.validate_path')
    @patch('kbkit.api.pipeline.KBThermo')
    @patch('kbkit.api.pipeline.KBICalculator')
    @patch('kbkit.api.pipeline.SystemCollection.load')
    def test_make_figures_creates_directories(self, mock_load, mock_calc_class, mock_thermo_class, mock_validate, tmp_path):
        """Test that make_figures creates necessary directories."""
        mock_sc = Mock(spec=SystemCollection)
        mock_mixture = Mock()
        mock_mixture.path = tmp_path / "mixture"
        mock_sc.mixtures = [mock_mixture]
        mock_load.return_value = mock_sc

        mock_calc = Mock(spec=KBICalculator)
        mock_calc.kbi.return_value = PropertyResult(name="kbi", value=[1.0], units="cm^3/mol")
        mock_kbi_plotter = Mock()
        mock_calc.kbi_plotter.return_value = mock_kbi_plotter
        mock_calc_class.return_value = mock_calc

        mock_thermo = Mock(spec=KBThermo)
        mock_thermo_plotter = Mock()
        mock_thermo.plotter.return_value = mock_thermo_plotter
        mock_thermo_class.return_value = mock_thermo

        save_path = tmp_path / "figures"
        mock_validate.return_value = save_path

        pipeline = Pipeline()
        pipeline.make_figures(savepath=str(save_path))

        assert save_path.exists()
        assert (save_path / "system_figures").exists()

    @patch('kbkit.api.pipeline.validate_path')
    @patch('kbkit.api.pipeline.KBThermo')
    @patch('kbkit.api.pipeline.KBICalculator')
    @patch('kbkit.api.pipeline.SystemCollection.load')
    def test_make_figures_calls_kbi_plotter(self, mock_load, mock_calc_class, mock_thermo_class, mock_validate, tmp_path):
        """Test that make_figures calls kbi_plotter.plot_all."""
        mock_sc = Mock(spec=SystemCollection)
        mock_mixture = Mock()
        mock_mixture.path = tmp_path / "mixture"
        mock_sc.mixtures = [mock_mixture]
        mock_load.return_value = mock_sc

        mock_calc = Mock(spec=KBICalculator)
        mock_calc.kbi.return_value = PropertyResult(name="kbi", value=[1.0], units="cm^3/mol")
        mock_kbi_plotter = Mock()
        mock_calc.kbi_plotter.return_value = mock_kbi_plotter
        mock_calc_class.return_value = mock_calc

        mock_thermo = Mock(spec=KBThermo)
        mock_thermo_plotter = Mock()
        mock_thermo.plotter.return_value = mock_thermo_plotter
        mock_thermo_class.return_value = mock_thermo

        save_path = tmp_path / "figures"
        mock_validate.return_value = save_path

        pipeline = Pipeline()
        pipeline.make_figures(savepath=str(save_path))

        mock_kbi_plotter.plot_all.assert_called_once()
        call_kwargs = mock_kbi_plotter.plot_all.call_args[1]
        assert call_kwargs['units'] == "cm^3/mol"
        assert call_kwargs['show'] is False

    @patch('kbkit.api.pipeline.validate_path')
    @patch('kbkit.api.pipeline.KBThermo')
    @patch('kbkit.api.pipeline.KBICalculator')
    @patch('kbkit.api.pipeline.SystemCollection.load')
    def test_make_figures_calls_thermo_plotter(self, mock_load, mock_calc_class, mock_thermo_class, mock_validate, tmp_path):
        """Test that make_figures calls thermo_plotter.make_figures."""
        mock_sc = Mock(spec=SystemCollection)
        mock_mixture = Mock()
        mock_mixture.path = tmp_path / "mixture"
        mock_sc.mixtures = [mock_mixture]
        mock_load.return_value = mock_sc

        mock_calc = Mock(spec=KBICalculator)
        mock_calc.kbi.return_value = PropertyResult(name="kbi", value=[1.0], units="cm^3/mol")
        mock_kbi_plotter = Mock()
        mock_calc.kbi_plotter.return_value = mock_kbi_plotter
        mock_calc_class.return_value = mock_calc

        mock_thermo = Mock(spec=KBThermo)
        mock_thermo_plotter = Mock()
        mock_thermo.plotter.return_value = mock_thermo_plotter
        mock_thermo_class.return_value = mock_thermo

        save_path = tmp_path / "figures"
        mock_validate.return_value = save_path

        pipeline = Pipeline()
        pipeline.make_figures(xmol="Water", cmap="viridis", savepath=str(save_path))

        mock_thermo_plotter.make_figures.assert_called_once_with(
            xmol="Water",
            cmap="viridis",
            savepath=str(save_path)
        )

    @patch('kbkit.api.pipeline.KBThermo')
    @patch('kbkit.api.pipeline.KBICalculator')
    @patch('kbkit.api.pipeline.SystemCollection.load')
    def test_make_figures_uses_default_path_when_none(self, mock_load, mock_calc_class, mock_thermo_class, tmp_path):
        """Test that make_figures uses default path when savepath is None."""
        mock_sc = Mock(spec=SystemCollection)
        mock_mixture = Mock()
        mock_mixture.path = tmp_path / "mixture"
        mock_sc.mixtures = [mock_mixture]
        mock_load.return_value = mock_sc

        mock_calc = Mock(spec=KBICalculator)
        mock_calc.kbi.return_value = PropertyResult(name="kbi", value=[1.0], units="cm^3/mol")
        mock_kbi_plotter = Mock()
        mock_calc.kbi_plotter.return_value = mock_kbi_plotter
        mock_calc_class.return_value = mock_calc

        mock_thermo = Mock(spec=KBThermo)
        mock_thermo_plotter = Mock()
        mock_thermo.plotter.return_value = mock_thermo_plotter
        mock_thermo_class.return_value = mock_thermo

        pipeline = Pipeline()
        pipeline.make_figures()

        # Should create kb_analysis directory in parent of first mixture
        expected_path = tmp_path / "kb_analysis"
        assert expected_path.exists()

class TestPipelineIntegration:
    """Integration tests for Pipeline."""

    @patch('kbkit.api.pipeline.KBThermo')
    @patch('kbkit.api.pipeline.KBICalculator')
    @patch('kbkit.api.pipeline.SystemCollection.load')
    def test_full_workflow(
        self, mock_load, mock_calc_class, mock_thermo_class,
        mock_system_collection, mock_kbi_calculator, mock_kb_thermo
    ):
        """Test complete pipeline workflow."""
        mock_load.return_value = mock_system_collection
        mock_calc_class.return_value = mock_kbi_calculator
        mock_thermo_class.return_value = mock_kb_thermo

        mock_kbi_result = PropertyResult(
            name="kbi",
            value=np.random.rand(3, 2, 2),
            units="cm^3/mol"
        )
        mock_kbi_calculator.kbi.return_value = mock_kbi_result

        # Create pipeline
        pipeline = Pipeline(
            base_path="/path/to/base",
            activity_integration_type="polynomial"
        )

        # Access all major components
        systems = pipeline.systems
        calculator = pipeline.calculator
        kbi_res = pipeline.kbi_res
        thermo = pipeline.thermo
        results = pipeline.results

        # Verify all components were created
        assert systems == mock_system_collection
        assert calculator == mock_kbi_calculator
        assert kbi_res == mock_kbi_result
        assert thermo == mock_kb_thermo
        assert isinstance(results, dict)
        assert len(results) > 0

    @patch('kbkit.api.pipeline.KBThermo')
    @patch('kbkit.api.pipeline.KBICalculator')
    @patch('kbkit.api.pipeline.SystemCollection.load')
    def test_pipeline_with_all_plotters(
        self, mock_load, mock_calc_class, mock_thermo_class,
        mock_system_collection, mock_kbi_calculator, mock_kb_thermo
    ):
        """Test pipeline with all plotter types."""
        mock_load.return_value = mock_system_collection
        mock_calc_class.return_value = mock_kbi_calculator
        mock_thermo_class.return_value = mock_kb_thermo

        mock_kbi_result = PropertyResult(
            name="kbi",
            value=np.random.rand(3, 2, 2),
            units="cm^3/mol"
        )
        mock_kbi_calculator.kbi.return_value = mock_kbi_result

        # Mock plotters
        mock_ts_plotter = Mock(spec=TimeseriesPlotter)
        mock_system_collection.timeseries_plotter.return_value = mock_ts_plotter

        mock_kbi_plotter = Mock(spec=KBIAnalysisPlotter)
        mock_kbi_calculator.kbi_plotter.return_value = mock_kbi_plotter

        mock_thermo_plotter = Mock(spec=ThermoPlotter)
        mock_kb_thermo.plotter.return_value = mock_thermo_plotter

        pipeline = Pipeline()

        # Access all plotters
        ts_plotter = pipeline.timeseries_plotter("system_1")
        kbi_plotter = pipeline.kbi_plotter
        thermo_plotter = pipeline.thermo_plotter

        assert ts_plotter == mock_ts_plotter
        assert kbi_plotter == mock_kbi_plotter
        assert thermo_plotter == mock_thermo_plotter

"""
Unit tests for the KBICalculator module.

This test suite provides comprehensive coverage of the KBICalculator class,
including KBI computation, caching, and error handling.
"""
import warnings

# Suppress NumPy/SciPy compatibility warning (harmless with NumPy 2.x + SciPy 1.16+)
warnings.filterwarnings('ignore', message='numpy.ndarray size changed', category=RuntimeWarning)

from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

from kbkit.io.rdf import RdfParser
from kbkit.kbi.calculator import KBICalculator
from kbkit.kbi.integrator import KBIntegrator
from kbkit.schema.kbi_metadata import KBIMetadata
from kbkit.schema.property_result import PropertyResult
from kbkit.schema.system_metadata import SystemMetadata
from kbkit.systems.collection import SystemCollection
from kbkit.systems.properties import SystemProperties


@pytest.fixture
def mock_system_metadata():
    """Create a mock SystemMetadata object."""
    mock_meta = Mock(spec=SystemMetadata)
    mock_meta.name = "mixture_50_50"
    mock_meta.rdf_path = Path("/path/to/rdf")

    mock_props = Mock(spec=SystemProperties)
    mock_topology = Mock()
    mock_topology.molecule_count = {"MOL1": 50, "MOL2": 50}
    mock_topology.total_molecules = 100
    mock_props.topology = mock_topology
    mock_props.get.return_value = 1.0

    mock_meta.props = mock_props
    mock_meta.has_rdf.return_value = True

    return mock_meta


@pytest.fixture
def mock_pure_metadata():
    """Create a mock pure SystemMetadata object."""
    mock_meta = Mock(spec=SystemMetadata)
    mock_meta.name = "pure_MOL1"
    mock_meta.rdf_path = Path()

    mock_props = Mock(spec=SystemProperties)
    mock_topology = Mock()
    mock_topology.molecule_count = {"MOL1": 100}
    mock_topology.total_molecules = 100
    mock_topology.molecules = ["MOL1"]
    mock_props.topology = mock_topology
    mock_props.get.return_value = 1.0

    mock_meta.props = mock_props
    mock_meta.has_rdf.return_value = False

    return mock_meta


@pytest.fixture
def mock_system_collection(mock_system_metadata, mock_pure_metadata):
    """Create a mock SystemCollection object."""
    mock_sc = Mock(spec=SystemCollection)
    mock_sc._systems = [mock_pure_metadata, mock_system_metadata]
    mock_sc.molecules = ["MOL1", "MOL2"]
    mock_sc.pures = [mock_pure_metadata]
    mock_sc.mixtures = [mock_system_metadata]
    mock_sc.x = np.array([[1.0, 0.0], [0.5, 0.5]])
    mock_sc.get_units.return_value = "kg/m^3"
    mock_sc.get.return_value = np.array([1000.0, 950.0])
    mock_sc.__iter__ = Mock(return_value=iter([mock_pure_metadata, mock_system_metadata]))
    mock_sc.__len__ = Mock(return_value=2)

    return mock_sc


# In test_property_calculator.py

class TestKBICalculatorInitialization:
    """Test KBICalculator initialization."""

    def test_init_with_system_collection(self, mock_system_collection):
        """Test initialization with SystemCollection."""
        calc = KBICalculator(mock_system_collection)

        assert calc.systems == mock_system_collection
        assert isinstance(calc._cache, dict)
        assert len(calc._cache) == 0

    def test_init_creates_empty_cache(self, mock_system_collection):
        """Test that initialization creates empty cache."""
        calc = KBICalculator(mock_system_collection)

        assert calc._cache == {}

    def test_init_with_custom_parameters(self, mock_system_collection):
        """Test initialization with custom parameters."""
        calc = KBICalculator(
            mock_system_collection,
            ignore_convergence_errors=True,
            convergence_thresholds=(1e-4, 1e-3),
            tail_length=2.5,
            correct_rdf_convergence=False,
            apply_damping=False,
            extrapolate_thermodynamic_limit=False
        )

        # Fixed: no longer tuples
        assert calc.ignore_convergence_errors is True
        assert calc.convergence_thresholds == (1e-4, 1e-3)
        assert calc.tail_length == 2.5
        assert calc.correct_rdf_convergence is False
        assert calc.apply_damping is False
        assert calc.extrapolate_thermodynamic_limit is False



class TestKBICalculatorCaching:
    """Test the caching mechanism."""

    def test_get_from_cache_returns_none_when_empty(self, mock_system_collection):
        """Test _get_from_cache returns None when cache is empty."""
        calc = KBICalculator(mock_system_collection)

        cache_key = ("test", "key")
        result = calc._cache.get(cache_key)

        assert result is None

    def test_get_from_cache_returns_cached_result(self, mock_system_collection):
        """Test _get_from_cache returns cached result."""
        calc = KBICalculator(mock_system_collection)

        # Create a mock PropertyResult
        mock_result = Mock(spec=PropertyResult)
        mock_result.to.return_value = mock_result

        # Add to cache
        key = ("kbi",)
        calc._cache[key] = mock_result

        # Retrieve from cache
        result = calc._cache[key].to("cm^3/mol")

        assert result == mock_result
        mock_result.to.assert_called_once_with("cm^3/mol")


class TestKBICalculatorKBI:
    """Test the kbi method."""

    @patch('kbkit.kbi.calculator.KBIntegrator')
    @patch('kbkit.kbi.calculator.RdfParser')
    def test_kbi_computes_matrix(self, mock_rdf_class, mock_integrator_class,
                                 mock_system_collection, tmp_path):
        """Test that kbi computes KBI matrix."""
        calc = KBICalculator(mock_system_collection)

        # Setup RDF files
        mixture = mock_system_collection.mixtures[0]
        rdf_path = tmp_path / "rdf"
        rdf_path.mkdir()
        rdf_file = rdf_path / "rdf_MOL1_MOL2.xvg"
        rdf_file.touch()
        mixture.rdf_path = rdf_path

        # Mock RDF parser
        mock_rdf = Mock(spec=RdfParser)
        mock_rdf.is_converged = True
        mock_rdf.r = np.linspace(0, 3, 100)
        mock_rdf.g = np.ones(100)
        mock_rdf.r_tail = np.linspace(2, 3, 50)
        mock_rdf_class.return_value = mock_rdf

        # Mock integrator
        mock_integrator = Mock(spec=KBIntegrator)
        mock_integrator.rdf_molecules = ["MOL1", "MOL2"]
        mock_integrator.compute_kbi.return_value = 1.5
        mock_integrator.rkbi.return_value = np.ones(100)
        mock_integrator.scaled_rkbi.return_value = np.ones(100)
        mock_integrator.scaled_rkbi_fit.return_value = np.ones(50)
        mock_integrator.fit_limit_params.return_value = np.array([1.5, 0.1])
        mock_integrator_class.from_system_properties.return_value = mock_integrator

        result = calc.kbi(units="nm^3/molecule")

        assert isinstance(result, PropertyResult)
        assert result.value.shape == (2, 2, 2)  # (n_systems, n_mols, n_mols)
        assert result.name == "kbi"
        assert result.units == "nm^3/molecule"

    @patch('kbkit.kbi.calculator.KBIntegrator')
    @patch('kbkit.kbi.calculator.RdfParser')
    def test_kbi_skips_systems_without_rdf(self, mock_rdf_class, mock_integrator_class,
                                           mock_system_collection):
        """Test that kbi skips systems without RDF."""
        calc = KBICalculator(mock_system_collection)

        # Set mixture to have no RDF
        mixture = mock_system_collection.mixtures[0]
        mixture.has_rdf.return_value = False

        result = calc.kbi(units="nm^3/molecule")

        # Should have NaN values for systems without RDF
        assert np.all(np.isnan(result.value[1]))  # mixture index

    @patch('kbkit.kbi.calculator.KBIntegrator')
    @patch('kbkit.kbi.calculator.RdfParser')
    def test_kbi_raises_on_non_converged_rdf(self, mock_rdf_class, mock_integrator_class,
                                             mock_system_collection, tmp_path):
        """Test that kbi raises error for non-converged RDF."""
        calc = KBICalculator(mock_system_collection, ignore_convergence_errors=False)

        # Setup RDF files
        mixture = mock_system_collection.mixtures[0]
        rdf_path = tmp_path / "rdf"
        rdf_path.mkdir()
        rdf_file = rdf_path / "rdf_MOL1_MOL2.xvg"
        rdf_file.touch()
        mixture.rdf_path = rdf_path

        # Mock non-converged RDF
        mock_rdf = Mock(spec=RdfParser)
        mock_rdf.is_converged = False
        mock_rdf_class.return_value = mock_rdf

        mock_integrator = Mock(spec=KBIntegrator)
        mock_integrator.rdf_molecules = ["MOL1", "MOL2"]
        mock_integrator_class.from_system_properties.return_value = mock_integrator

        with pytest.raises(RuntimeError, match="did not converge"):
            calc.kbi(units="nm^3/molecule")

    @patch('kbkit.kbi.calculator.KBIntegrator')
    @patch('kbkit.kbi.calculator.RdfParser')
    def test_kbi_ignores_convergence_errors(self, mock_rdf_class, mock_integrator_class,
                                           mock_system_collection, tmp_path, capsys):
        """Test that kbi can ignore convergence errors."""
        calc = KBICalculator(mock_system_collection, ignore_convergence_errors=True)

        # Setup RDF files
        mixture = mock_system_collection.mixtures[0]
        rdf_path = tmp_path / "rdf"
        rdf_path.mkdir()
        rdf_file = rdf_path / "rdf_MOL1_MOL2.xvg"
        rdf_file.touch()
        mixture.rdf_path = rdf_path

        # Mock non-converged RDF
        mock_rdf = Mock(spec=RdfParser)
        mock_rdf.is_converged = False
        mock_rdf_class.return_value = mock_rdf

        mock_integrator = Mock(spec=KBIntegrator)
        mock_integrator.rdf_molecules = ["MOL1", "MOL2"]
        mock_integrator_class.from_system_properties.return_value = mock_integrator

        result = calc.kbi(units="nm^3/molecule")

        # Should print warning
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "Skipping this system" in captured.out

    @patch('kbkit.kbi.calculator.KBIntegrator')
    @patch('kbkit.kbi.calculator.RdfParser')
    def test_kbi_populates_metadata(self, mock_rdf_class, mock_integrator_class,
                                    mock_system_collection, tmp_path):
        """Test that kbi populates metadata."""
        calc = KBICalculator(mock_system_collection)

        # Setup RDF files
        mixture = mock_system_collection.mixtures[0]
        rdf_path = tmp_path / "rdf"
        rdf_path.mkdir()
        rdf_file = rdf_path / "rdf_MOL1_MOL2.xvg"
        rdf_file.touch()
        mixture.rdf_path = rdf_path

        # Mock RDF parser
        mock_rdf = Mock(spec=RdfParser)
        mock_rdf.is_converged = True
        mock_rdf.r = np.linspace(0, 3, 100)
        mock_rdf.g = np.ones(100)
        mock_rdf.r_tail = np.linspace(2, 3, 50)
        mock_rdf_class.return_value = mock_rdf

        # Mock integrator
        mock_integrator = Mock(spec=KBIntegrator)
        mock_integrator.rdf_molecules = ["MOL1", "MOL2"]
        mock_integrator.compute_kbi.return_value = 1.5
        mock_integrator.rkbi.return_value = np.ones(100)
        mock_integrator.scaled_rkbi.return_value = np.ones(100)
        mock_integrator.scaled_rkbi_fit.return_value = np.ones(50)
        mock_integrator.fit_limit_params.return_value = np.array([1.5, 0.1])
        mock_integrator_class.from_system_properties.return_value = mock_integrator

        result = calc.kbi(units="nm^3/molecule")

        assert result.metadata is not None
        assert "mixture_50_50" in result.metadata
        assert "MOL1.MOL2" in result.metadata["mixture_50_50"]

    @patch('kbkit.kbi.calculator.KBIntegrator')
    @patch('kbkit.kbi.calculator.RdfParser')
    def test_kbi_with_custom_parameters(self, mock_rdf_class, mock_integrator_class,
                                       mock_system_collection, tmp_path):
        """Test kbi with custom correction parameters."""
        calc = KBICalculator(
            mock_system_collection,
            convergence_thresholds=(1e-4, 1e-3),
            tail_length=2.5,
            correct_rdf_convergence=False,
            apply_damping=False,
            extrapolate_thermodynamic_limit=False
        )

        # Setup RDF files
        mixture = mock_system_collection.mixtures[0]
        rdf_path = tmp_path / "rdf"
        rdf_path.mkdir()
        rdf_file = rdf_path / "rdf_MOL1_MOL2.xvg"
        rdf_file.touch()
        mixture.rdf_path = rdf_path

        mock_rdf = Mock(spec=RdfParser)
        mock_rdf.is_converged = True
        mock_rdf.r = np.linspace(0, 3, 100)
        mock_rdf.g = np.ones(100)
        mock_rdf.r_tail = np.linspace(2, 3, 50)
        mock_rdf_class.return_value = mock_rdf

        mock_integrator = Mock(spec=KBIntegrator)
        mock_integrator.rdf_molecules = ["MOL1", "MOL2"]
        mock_integrator.compute_kbi.return_value = 1.5
        mock_integrator.rkbi.return_value = np.ones(100)
        mock_integrator.scaled_rkbi.return_value = np.ones(100)
        mock_integrator.scaled_rkbi_fit.return_value = np.ones(50)
        mock_integrator.fit_limit_params.return_value = np.array([1.5, 0.1])
        mock_integrator_class.from_system_properties.return_value = mock_integrator

        calc.kbi(units="cm^3/mol")

        # Verify RdfParser was called with correct parameters
        mock_rdf_class.assert_called()
        call_kwargs = mock_rdf_class.call_args[1]
        assert call_kwargs['convergence_thresholds'] == (1e-4, 1e-3)
        assert call_kwargs['tail_length'] == 2.5

        # Verify KBIntegrator was called with correct parameters
        mock_integrator_class.from_system_properties.assert_called()
        call_kwargs = mock_integrator_class.from_system_properties.call_args[1]
        assert call_kwargs['correct_rdf_convergence'] is False
        assert call_kwargs['apply_damping'] is False
        assert call_kwargs['extrapolate_thermodynamic_limit'] is False

    @patch('kbkit.kbi.calculator.KBIntegrator')
    @patch('kbkit.kbi.calculator.RdfParser')
    def test_kbi_caching(self, mock_rdf_class, mock_integrator_class,
                        mock_system_collection, tmp_path):
        """Test that kbi uses caching."""
        calc = KBICalculator(mock_system_collection)

        # Setup RDF files
        mixture = mock_system_collection.mixtures[0]
        rdf_path = tmp_path / "rdf"
        rdf_path.mkdir()
        rdf_file = rdf_path / "rdf_MOL1_MOL2.xvg"
        rdf_file.touch()
        mixture.rdf_path = rdf_path

        mock_rdf = Mock(spec=RdfParser)
        mock_rdf.is_converged = True
        mock_rdf.r = np.linspace(0, 3, 100)
        mock_rdf.g = np.ones(100)
        mock_rdf.r_tail = np.linspace(2, 3, 50)
        mock_rdf_class.return_value = mock_rdf

        mock_integrator = Mock(spec=KBIntegrator)
        mock_integrator.rdf_molecules = ["MOL1", "MOL2"]
        mock_integrator.compute_kbi.return_value = 1.5
        mock_integrator.rkbi.return_value = np.ones(100)
        mock_integrator.scaled_rkbi.return_value = np.ones(100)
        mock_integrator.scaled_rkbi_fit.return_value = np.ones(50)
        mock_integrator.fit_limit_params.return_value = np.array([1.5, 0.1])
        mock_integrator_class.from_system_properties.return_value = mock_integrator

        # First call
        result1 = calc.kbi(units="nm^3/molecule")
        # Second call
        result2 = calc.kbi(units="nm^3/molecule")

        # Should use cache on second call
        assert result1 is result2
        # RdfParser should only be called once
        assert mock_rdf_class.call_count == 1


class TestKBICalculatorIntegration:
    """Integration tests for KBICalculator."""

    @patch('kbkit.kbi.calculator.KBIntegrator')
    @patch('kbkit.kbi.calculator.RdfParser')
    def test_kbi_metadata_structure(self, mock_rdf_class, mock_integrator_class,
                                    mock_system_collection, tmp_path):
        """Test that KBI metadata has correct structure."""
        calc = KBICalculator(mock_system_collection)

        # Setup RDF files
        mixture = mock_system_collection.mixtures[0]
        rdf_path = tmp_path / "rdf"
        rdf_path.mkdir()
        rdf_file = rdf_path / "rdf_MOL1_MOL2.xvg"
        rdf_file.touch()
        mixture.rdf_path = rdf_path

        # Mock RDF parser
        mock_rdf = Mock(spec=RdfParser)
        mock_rdf.is_converged = True
        mock_rdf.r = np.linspace(0, 3, 100)
        mock_rdf.g = np.ones(100)
        mock_rdf.r_tail = np.linspace(2, 3, 50)
        mock_rdf_class.return_value = mock_rdf

        # Mock integrator
        mock_integrator = Mock(spec=KBIntegrator)
        mock_integrator.rdf_molecules = ["MOL1", "MOL2"]
        mock_integrator.compute_kbi.return_value = 1.5
        mock_integrator.rkbi.return_value = np.ones(100)
        mock_integrator.scaled_rkbi.return_value = np.ones(100)
        mock_integrator.scaled_rkbi_fit.return_value = np.ones(50)
        mock_integrator.fit_limit_params.return_value = np.array([1.5, 0.1])
        mock_integrator_class.from_system_properties.return_value = mock_integrator

        result = calc.kbi(units="nm^3/molecule")

        # Check metadata structure
        metadata = result.metadata["mixture_50_50"]["MOL1.MOL2"]
        assert isinstance(metadata, KBIMetadata)
        assert metadata.mols == ("MOL1", "MOL2")
        assert len(metadata.r) == 100
        assert len(metadata.g) == 100
        assert metadata.kbi_limit == 1.5


class TestKBICalculatorEdgeCases:
    """Test edge cases and error conditions."""

    @patch('kbkit.kbi.calculator.RdfParser')
    def test_kbi_with_no_rdf_files(self, mock_rdf_class, mock_system_collection, tmp_path):
        """Test kbi when RDF directory is empty."""
        calc = KBICalculator(mock_system_collection)

        # Setup empty RDF directory
        mixture = mock_system_collection.mixtures[0]
        rdf_path = tmp_path / "rdf"
        rdf_path.mkdir()
        mixture.rdf_path = rdf_path

        result = calc.kbi(units="nm^3/molecule")

        # Should have NaN values
        assert np.all(np.isnan(result.value[1]))  # mixture index

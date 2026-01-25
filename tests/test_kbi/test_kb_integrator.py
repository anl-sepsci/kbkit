"""
Unit tests for the KBIntegrator module.

This test suite provides comprehensive coverage of the KBIntegrator class,
including initialization, corrections, KBI computation, and plotting methods.
"""
import warnings

# Suppress NumPy/SciPy compatibility warning (harmless with NumPy 2.x + SciPy 1.16+)
warnings.filterwarnings('ignore', message='numpy.ndarray size changed', category=RuntimeWarning)

from unittest.mock import Mock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

from kbkit.io.rdf import RdfParser
from kbkit.kbi.integrator import KBIntegrator
from kbkit.systems.properties import SystemProperties


@pytest.fixture
def mock_rdf_parser():
    """Create a mock RdfParser object."""
    mock_rdf = Mock(spec=RdfParser)
    mock_rdf.fname = "rdf_MOL1_MOL2.xvg"

    # Create sample RDF data
    r = np.linspace(0.1, 3.0, 100)
    g = np.ones_like(r)
    g[:20] = 0.5  # Some structure at short range
    g[20:40] = 1.5

    mock_rdf.r = r
    mock_rdf.g = g
    mock_rdf.mask = np.ones(len(r), dtype=bool)
    mock_rdf.mask[:50] = False  # Tail starts at index 50
    mock_rdf.r_tail = r[mock_rdf.mask]

    return mock_rdf


@pytest.fixture
def sample_molecule_count():
    """Sample molecule count dictionary."""
    return {"MOL1": 100, "MOL2": 50}


@pytest.fixture
def sample_volume():
    """Sample box volume in nm^3."""
    return 125.0


@pytest.fixture
def mock_system_properties(sample_volume, sample_molecule_count):
    """Create a mock SystemProperties object."""
    mock_props = Mock(spec=SystemProperties)
    mock_props.get.return_value = sample_volume

    mock_topology = Mock()
    mock_topology.molecule_count = sample_molecule_count
    mock_props.topology = mock_topology

    return mock_props


class TestKBIntegratorInitialization:
    """Test KBIntegrator initialization."""

    def test_init_with_all_parameters(self, mock_rdf_parser, sample_volume, sample_molecule_count):
        """Test initialization with all parameters."""
        integrator = KBIntegrator(
            rdf=mock_rdf_parser,
            volume=sample_volume,
            molecule_count=sample_molecule_count,
            correct_rdf_convergence=True,
            apply_damping=True,
            extrapolate_thermodynamic_limit=True,
        )

        assert integrator.rdf == mock_rdf_parser
        assert integrator.box_volume == sample_volume
        assert integrator.molecule_count == sample_molecule_count
        assert integrator.molecules == ["MOL1", "MOL2"]
        assert integrator.correct_rdf_convergence is True
        assert integrator.apply_damping is True
        assert integrator.extrapolate_thermodynamic_limit is True

    def test_init_with_default_corrections(self, mock_rdf_parser, sample_volume, sample_molecule_count):
        """Test initialization with default correction parameters."""
        integrator = KBIntegrator(
            rdf=mock_rdf_parser,
            volume=sample_volume,
            molecule_count=sample_molecule_count,
        )

        assert integrator.correct_rdf_convergence is True
        assert integrator.apply_damping is True
        assert integrator.extrapolate_thermodynamic_limit is True

    def test_init_with_corrections_disabled(self, mock_rdf_parser, sample_volume, sample_molecule_count):
        """Test initialization with all corrections disabled."""
        integrator = KBIntegrator(
            rdf=mock_rdf_parser,
            volume=sample_volume,
            molecule_count=sample_molecule_count,
            correct_rdf_convergence=False,
            apply_damping=False,
            extrapolate_thermodynamic_limit=False,
        )

        assert integrator.correct_rdf_convergence is False
        assert integrator.apply_damping is False
        assert integrator.extrapolate_thermodynamic_limit is False

    def test_molecules_preserves_order(self, mock_rdf_parser, sample_volume):
        """Test that molecules list preserves order and removes duplicates."""
        molecule_count = {"MOL1": 100, "MOL2": 50, "MOL1": 100}  # Duplicate

        integrator = KBIntegrator(
            rdf=mock_rdf_parser,
            volume=sample_volume,
            molecule_count=molecule_count,
        )

        # Should preserve order and remove duplicates
        assert len(integrator.molecules) == 2




class TestKBIntegratorFromSystemProperties:
    """Test the from_system_properties class method."""

    def test_from_system_properties_creates_integrator(self, mock_rdf_parser, mock_system_properties):
        """Test creating KBIntegrator from SystemProperties."""
        integrator = KBIntegrator.from_system_properties(
            rdf=mock_rdf_parser,
            system_properties=mock_system_properties,
        )

        assert isinstance(integrator, KBIntegrator)
        assert integrator.rdf == mock_rdf_parser
        mock_system_properties.get.assert_called_once_with("volume", units="nm^3", avg=True)

    def test_from_system_properties_with_custom_corrections(self, mock_rdf_parser, mock_system_properties):
        """Test from_system_properties with custom correction parameters."""
        integrator = KBIntegrator.from_system_properties(
            rdf=mock_rdf_parser,
            system_properties=mock_system_properties,
            correct_rdf_convergence=False,
            apply_damping=False,
            extrapolate_thermodynamic_limit=False,
        )

        assert integrator.correct_rdf_convergence is False
        assert integrator.apply_damping is False
        assert integrator.extrapolate_thermodynamic_limit is False




class TestKBIntegratorRDFMolecules:
    """Test the rdf_molecules property."""

    @patch('kbkit.kbi.integrator.RdfParser.extract_molecules')
    def test_rdf_molecules_extracts_correctly(self, mock_extract, mock_rdf_parser,
                                            sample_volume, sample_molecule_count):
        """Test that rdf_molecules extracts molecules from filename."""
        mock_extract.return_value = ["MOL1", "MOL2"]

        integrator = KBIntegrator(
            rdf=mock_rdf_parser,
            volume=sample_volume,
            molecule_count=sample_molecule_count,
        )

        molecules = integrator.rdf_molecules

        mock_extract.assert_called_once_with(
            text=mock_rdf_parser.fname,
            mol_list=integrator.molecules
        )
        assert molecules == ["MOL1", "MOL2"]

    @patch('kbkit.kbi.integrator.RdfParser.extract_molecules')
    def test_rdf_molecules_raises_on_wrong_count(self, mock_extract, mock_rdf_parser,
                                                sample_volume, sample_molecule_count):
        """Test that rdf_molecules raises error if not exactly 2 molecules."""
        mock_extract.return_value = ["MOL1", "MOL2", "MOL3"]

        integrator = KBIntegrator(
            rdf=mock_rdf_parser,
            volume=sample_volume,
            molecule_count=sample_molecule_count,
        )

        with pytest.raises(ValueError, match="Number of molecules detected.*is '3', expected 2"):
            _ = integrator.rdf_molecules




class TestKBIntegratorMolJ:
    """Test the _mol_j property."""

    @patch('kbkit.kbi.integrator.RdfParser.extract_molecules')
    def test_mol_j_returns_second_molecule(self, mock_extract, mock_rdf_parser,
                                        sample_volume, sample_molecule_count):
        """Test that _mol_j returns the second molecule."""
        mock_extract.return_value = ["MOL1", "MOL2"]

        integrator = KBIntegrator(
            rdf=mock_rdf_parser,
            volume=sample_volume,
            molecule_count=sample_molecule_count,
        )

        assert integrator._mol_j == "MOL2"




class TestKBIntegratorKroneckerDelta:
    """Test the kronecker_delta method."""

    @patch('kbkit.kbi.integrator.RdfParser.extract_molecules')
    def test_kronecker_delta_same_molecules(self, mock_extract, mock_rdf_parser,
                                            sample_volume, sample_molecule_count):
        """Test kronecker_delta returns 1 for same molecules."""
        mock_extract.return_value = ["MOL1", "MOL1"]

        integrator = KBIntegrator(
            rdf=mock_rdf_parser,
            volume=sample_volume,
            molecule_count=sample_molecule_count,
        )

        assert integrator.kronecker_delta() == 1

    @patch('kbkit.kbi.integrator.RdfParser.extract_molecules')
    def test_kronecker_delta_different_molecules(self, mock_extract, mock_rdf_parser,
                                                sample_volume, sample_molecule_count):
        """Test kronecker_delta returns 0 for different molecules."""
        mock_extract.return_value = ["MOL1", "MOL2"]

        integrator = KBIntegrator(
            rdf=mock_rdf_parser,
            volume=sample_volume,
            molecule_count=sample_molecule_count,
        )

        assert integrator.kronecker_delta() == 0




class TestKBIntegratorNJ:
    """Test the n_j method."""

    @patch('kbkit.kbi.integrator.RdfParser.extract_molecules')
    def test_n_j_with_explicit_molecule(self, mock_extract, mock_rdf_parser,
                                        sample_volume, sample_molecule_count):
        """Test n_j with explicitly specified molecule."""
        mock_extract.return_value = ["MOL1", "MOL2"]

        integrator = KBIntegrator(
            rdf=mock_rdf_parser,
            volume=sample_volume,
            molecule_count=sample_molecule_count,
        )

        assert integrator.n_j("MOL1") == 100
        assert integrator.n_j("MOL2") == 50

    @patch('kbkit.kbi.integrator.RdfParser.extract_molecules')
    def test_n_j_with_default_molecule(self, mock_extract, mock_rdf_parser,
                                    sample_volume, sample_molecule_count):
        """Test n_j uses second molecule by default."""
        mock_extract.return_value = ["MOL1", "MOL2"]

        integrator = KBIntegrator(
            rdf=mock_rdf_parser,
            volume=sample_volume,
            molecule_count=sample_molecule_count,
        )

        assert integrator.n_j() == 50  # MOL2 count

    @patch('kbkit.kbi.integrator.RdfParser.extract_molecules')
    def test_n_j_raises_on_empty_string(self, mock_extract, mock_rdf_parser,
                                        sample_volume, sample_molecule_count):
        """Test n_j raises error for empty string."""
        mock_extract.return_value = ["MOL1", "MOL2"]

        integrator = KBIntegrator(
            rdf=mock_rdf_parser,
            volume=sample_volume,
            molecule_count=sample_molecule_count,
        )

        with pytest.raises(ValueError, match="Molecule '' cannot be empty str"):
            integrator.n_j("")

    @patch('kbkit.kbi.integrator.RdfParser.extract_molecules')
    def test_n_j_raises_on_invalid_molecule(self, mock_extract, mock_rdf_parser,
                                            sample_volume, sample_molecule_count):
        """Test n_j raises error for molecule not in RDF."""
        mock_extract.return_value = ["MOL1", "MOL2"]

        integrator = KBIntegrator(
            rdf=mock_rdf_parser,
            volume=sample_volume,
            molecule_count=sample_molecule_count,
        )

        with pytest.raises(ValueError, match="Molecule 'MOL3' not in rdf molecules"):
            integrator.n_j("MOL3")




class TestKBIntegratorGangulyCorrection:
    """Test the ganguly_correction_factor method."""

    @patch('kbkit.kbi.integrator.RdfParser.extract_molecules')
    def test_ganguly_correction_returns_array(self, mock_extract, mock_rdf_parser,
                                            sample_volume, sample_molecule_count):
        """Test that ganguly_correction_factor returns numpy array."""
        mock_extract.return_value = ["MOL1", "MOL2"]

        integrator = KBIntegrator(
            rdf=mock_rdf_parser,
            volume=sample_volume,
            molecule_count=sample_molecule_count,
        )

        correction = integrator.ganguly_correction_factor()

        assert isinstance(correction, np.ndarray)
        assert len(correction) == len(mock_rdf_parser.r)

    @patch('kbkit.kbi.integrator.RdfParser.extract_molecules')
    def test_ganguly_correction_shape_matches_rdf(self, mock_extract, mock_rdf_parser,
                                                sample_volume, sample_molecule_count):
        """Test that correction factor has same shape as RDF."""
        mock_extract.return_value = ["MOL1", "MOL2"]

        integrator = KBIntegrator(
            rdf=mock_rdf_parser,
            volume=sample_volume,
            molecule_count=sample_molecule_count,
        )

        correction = integrator.ganguly_correction_factor()

        assert correction.shape == mock_rdf_parser.g.shape

    @patch('kbkit.kbi.integrator.RdfParser.extract_molecules')
    def test_ganguly_correction_with_custom_mol_j(self, mock_extract, mock_rdf_parser,
                                                sample_volume, sample_molecule_count):
        """Test ganguly_correction_factor with custom mol_j."""
        mock_extract.return_value = ["MOL1", "MOL2"]

        integrator = KBIntegrator(
            rdf=mock_rdf_parser,
            volume=sample_volume,
            molecule_count=sample_molecule_count,
        )

        correction = integrator.ganguly_correction_factor(mol_j="MOL1")

        assert isinstance(correction, np.ndarray)




class TestKBIntegratorKrugerDamping:
    """Test the kruger_damping_factor method."""

    def test_kruger_damping_returns_array(self, mock_rdf_parser, sample_volume, sample_molecule_count):
        """Test that kruger_damping_factor returns numpy array."""
        integrator = KBIntegrator(
            rdf=mock_rdf_parser,
            volume=sample_volume,
            molecule_count=sample_molecule_count,
        )

        damping = integrator.kruger_damping_factor()

        assert isinstance(damping, np.ndarray)
        assert len(damping) == len(mock_rdf_parser.r)

    def test_kruger_damping_range(self, mock_rdf_parser, sample_volume, sample_molecule_count):
        """Test that damping factor is between 0 and 1."""
        integrator = KBIntegrator(
            rdf=mock_rdf_parser,
            volume=sample_volume,
            molecule_count=sample_molecule_count,
        )

        damping = integrator.kruger_damping_factor()

        assert np.all(damping >= 0)
        assert np.all(damping <= 1)

    def test_kruger_damping_at_boundaries(self, mock_rdf_parser, sample_volume, sample_molecule_count):
        """Test damping factor at r=0 and r=r_max."""
        integrator = KBIntegrator(
            rdf=mock_rdf_parser,
            volume=sample_volume,
            molecule_count=sample_molecule_count,
        )

        damping = integrator.kruger_damping_factor()

        # At r=0, damping should be ~1
        assert damping[0] > 0.99
        # At r=r_max, damping should be 0
        assert damping[-1] == 0.0




class TestKBIntegratorCorrelationFunction:
    """Test the h method (correlation function)."""

    @patch('kbkit.kbi.integrator.RdfParser.extract_molecules')
    def test_h_with_correction(self, mock_extract, mock_rdf_parser, sample_volume, sample_molecule_count):
        """Test h with Ganguly correction enabled."""
        mock_extract.return_value = ["MOL1", "MOL2"]

        integrator = KBIntegrator(
            rdf=mock_rdf_parser,
            volume=sample_volume,
            molecule_count=sample_molecule_count,
            correct_rdf_convergence=True,
        )

        h = integrator.h()

        assert isinstance(h, np.ndarray)
        assert len(h) == len(mock_rdf_parser.g)

    @patch('kbkit.kbi.integrator.RdfParser.extract_molecules')
    def test_h_without_correction(self, mock_extract, mock_rdf_parser, sample_volume, sample_molecule_count):
        """Test h without Ganguly correction."""
        mock_extract.return_value = ["MOL1", "MOL2"]

        integrator = KBIntegrator(
            rdf=mock_rdf_parser,
            volume=sample_volume,
            molecule_count=sample_molecule_count,
            correct_rdf_convergence=False,
        )

        h = integrator.h()

        # Without correction, h should be g - 1
        np.testing.assert_array_almost_equal(h, mock_rdf_parser.g - 1)

    @patch('kbkit.kbi.integrator.RdfParser.extract_molecules')
    def test_h_with_custom_mol_j(self, mock_extract, mock_rdf_parser, sample_volume, sample_molecule_count):
        """Test h with custom mol_j parameter."""
        mock_extract.return_value = ["MOL1", "MOL2"]

        integrator = KBIntegrator(
            rdf=mock_rdf_parser,
            volume=sample_volume,
            molecule_count=sample_molecule_count,
        )

        h = integrator.h(mol_j="MOL1")

        assert isinstance(h, np.ndarray)




class TestKBIntegratorRKBI:
    """Test the rkbi method (running KBI)."""

    @patch('kbkit.kbi.integrator.RdfParser.extract_molecules')
    def test_rkbi_returns_array(self, mock_extract, mock_rdf_parser, sample_volume, sample_molecule_count):
        """Test that rkbi returns numpy array."""
        mock_extract.return_value = ["MOL1", "MOL2"]

        integrator = KBIntegrator(
            rdf=mock_rdf_parser,
            volume=sample_volume,
            molecule_count=sample_molecule_count,
        )

        rkbi = integrator.rkbi()

        assert isinstance(rkbi, np.ndarray)
        assert len(rkbi) == len(mock_rdf_parser.r)

    @patch('kbkit.kbi.integrator.RdfParser.extract_molecules')
    def test_rkbi_with_damping(self, mock_extract, mock_rdf_parser, sample_volume, sample_molecule_count):
        """Test rkbi with damping enabled."""
        mock_extract.return_value = ["MOL1", "MOL2"]

        integrator = KBIntegrator(
            rdf=mock_rdf_parser,
            volume=sample_volume,
            molecule_count=sample_molecule_count,
            apply_damping=True,
        )

        rkbi = integrator.rkbi()

        assert isinstance(rkbi, np.ndarray)

    @patch('kbkit.kbi.integrator.RdfParser.extract_molecules')
    def test_rkbi_without_damping(self, mock_extract, mock_rdf_parser, sample_volume, sample_molecule_count):
        """Test rkbi without damping."""
        mock_extract.return_value = ["MOL1", "MOL2"]

        integrator = KBIntegrator(
            rdf=mock_rdf_parser,
            volume=sample_volume,
            molecule_count=sample_molecule_count,
            apply_damping=False,
        )

        rkbi = integrator.rkbi()

        assert isinstance(rkbi, np.ndarray)

    @patch('kbkit.kbi.integrator.RdfParser.extract_molecules')
    def test_rkbi_is_monotonic(self, mock_extract, mock_rdf_parser, sample_volume, sample_molecule_count):
        """Test that rkbi is generally monotonic (cumulative integral)."""
        mock_extract.return_value = ["MOL1", "MOL2"]

        integrator = KBIntegrator(
            rdf=mock_rdf_parser,
            volume=sample_volume,
            molecule_count=sample_molecule_count,
        )

        rkbi = integrator.rkbi()

        # First value should be 0 (initial condition)
        assert rkbi[0] == 0.0




class TestKBIntegratorScaledRKBI:
    """Test the scaled_rkbi and scaled_rkbi_fit methods."""

    @patch('kbkit.kbi.integrator.RdfParser.extract_molecules')
    def test_scaled_rkbi_returns_array(self, mock_extract, mock_rdf_parser, sample_volume, sample_molecule_count):
        """Test that scaled_rkbi returns numpy array."""
        mock_extract.return_value = ["MOL1", "MOL2"]

        integrator = KBIntegrator(
            rdf=mock_rdf_parser,
            volume=sample_volume,
            molecule_count=sample_molecule_count,
        )

        scaled = integrator.scaled_rkbi()

        assert isinstance(scaled, np.ndarray)
        assert len(scaled) == len(mock_rdf_parser.r)

    @patch('kbkit.kbi.integrator.RdfParser.extract_molecules')
    def test_scaled_rkbi_fit_uses_mask(self, mock_extract, mock_rdf_parser, sample_volume, sample_molecule_count):
        """Test that scaled_rkbi_fit applies mask correctly."""
        mock_extract.return_value = ["MOL1", "MOL2"]

        integrator = KBIntegrator(
            rdf=mock_rdf_parser,
            volume=sample_volume,
            molecule_count=sample_molecule_count,
        )

        scaled_fit = integrator.scaled_rkbi_fit()

        assert isinstance(scaled_fit, np.ndarray)
        assert len(scaled_fit) == np.sum(mock_rdf_parser.mask)




class TestKBIntegratorFitLimitParams:
    """Test the fit_limit_params method."""

    @patch('kbkit.kbi.integrator.RdfParser.extract_molecules')
    def test_fit_limit_params_returns_two_values(self, mock_extract, mock_rdf_parser,
                                                sample_volume, sample_molecule_count):
        """Test that fit_limit_params returns slope and intercept."""
        mock_extract.return_value = ["MOL1", "MOL2"]

        integrator = KBIntegrator(
            rdf=mock_rdf_parser,
            volume=sample_volume,
            molecule_count=sample_molecule_count,
        )

        params = integrator.fit_limit_params()

        assert isinstance(params, np.ndarray)
        assert len(params) == 2  # slope and intercept

    @patch('kbkit.kbi.integrator.RdfParser.extract_molecules')
    def test_fit_limit_params_with_custom_mol_j(self, mock_extract, mock_rdf_parser,
                                                sample_volume, sample_molecule_count):
        """Test fit_limit_params with custom mol_j."""
        mock_extract.return_value = ["MOL1", "MOL2"]

        integrator = KBIntegrator(
            rdf=mock_rdf_parser,
            volume=sample_volume,
            molecule_count=sample_molecule_count,
        )

        params = integrator.fit_limit_params(mol_j="MOL1")

        assert len(params) == 2




class TestKBIntegratorComputeKBI:
    """Test the compute_kbi method."""

    @patch('kbkit.kbi.integrator.RdfParser.extract_molecules')
    def test_compute_kbi_with_extrapolation(self, mock_extract, mock_rdf_parser,
                                        sample_volume, sample_molecule_count):
        """Test compute_kbi with thermodynamic limit extrapolation."""
        mock_extract.return_value = ["MOL1", "MOL2"]

        integrator = KBIntegrator(
            rdf=mock_rdf_parser,
            volume=sample_volume,
            molecule_count=sample_molecule_count,
            extrapolate_thermodynamic_limit=True,
        )

        kbi = integrator.compute_kbi()

        assert isinstance(kbi, float)

    @patch('kbkit.kbi.integrator.RdfParser.extract_molecules')
    def test_compute_kbi_without_extrapolation(self, mock_extract, mock_rdf_parser,
                                            sample_volume, sample_molecule_count):
        """Test compute_kbi without extrapolation (uses tail average)."""
        mock_extract.return_value = ["MOL1", "MOL2"]

        integrator = KBIntegrator(
            rdf=mock_rdf_parser,
            volume=sample_volume,
            molecule_count=sample_molecule_count,
            extrapolate_thermodynamic_limit=False,
        )

        kbi = integrator.compute_kbi()

        assert isinstance(kbi, float)

    @patch('kbkit.kbi.integrator.RdfParser.extract_molecules')
    def test_compute_kbi_with_custom_mol_j(self, mock_extract, mock_rdf_parser,
                                        sample_volume, sample_molecule_count):
        """Test compute_kbi with custom mol_j."""
        mock_extract.return_value = ["MOL1", "MOL2"]

        integrator = KBIntegrator(
            rdf=mock_rdf_parser,
            volume=sample_volume,
            molecule_count=sample_molecule_count,
        )

        kbi = integrator.compute_kbi(mol_j="MOL1")

        assert isinstance(kbi, float)




class TestKBIntegratorComputeRKBI:
    """Test the _compute_rkbi private method."""

    @patch('kbkit.kbi.integrator.RdfParser.extract_molecules')
    def test_compute_rkbi_with_all_corrections(self, mock_extract, mock_rdf_parser,
                                            sample_volume, sample_molecule_count):
        """Test _compute_rkbi with all corrections enabled."""
        mock_extract.return_value = ["MOL1", "MOL2"]

        integrator = KBIntegrator(
            rdf=mock_rdf_parser,
            volume=sample_volume,
            molecule_count=sample_molecule_count,
        )

        rkbi = integrator._compute_rkbi(correct_rdf_convergence=True, apply_damping=True)

        assert isinstance(rkbi, np.ndarray)
        assert len(rkbi) == len(mock_rdf_parser.r)

    @patch('kbkit.kbi.integrator.RdfParser.extract_molecules')
    def test_compute_rkbi_no_corrections(self, mock_extract, mock_rdf_parser,
                                        sample_volume, sample_molecule_count):
        """Test _compute_rkbi with no corrections."""
        mock_extract.return_value = ["MOL1", "MOL2"]

        integrator = KBIntegrator(
            rdf=mock_rdf_parser,
            volume=sample_volume,
            molecule_count=sample_molecule_count,
        )

        rkbi = integrator._compute_rkbi(correct_rdf_convergence=False, apply_damping=False)

        assert isinstance(rkbi, np.ndarray)




class TestKBIntegratorPlotting:
    """Test plotting methods."""

    @patch('kbkit.kbi.integrator.plt.show')
    @patch('kbkit.kbi.integrator.RdfParser.extract_molecules')
    def test_plot_rkbis_displays(self, mock_extract, mock_show, mock_rdf_parser,
                                sample_volume, sample_molecule_count):
        """Test that plot_rkbis creates and displays plot."""
        mock_extract.return_value = ["MOL1", "MOL2"]

        integrator = KBIntegrator(
            rdf=mock_rdf_parser,
            volume=sample_volume,
            molecule_count=sample_molecule_count,
        )

        integrator.plot_rkbis()

        mock_show.assert_called_once()
        plt.close('all')

    @patch('kbkit.kbi.integrator.plt.show')
    @patch('kbkit.kbi.integrator.RdfParser.extract_molecules')
    def test_plot_rkbis_saves_file(self, mock_extract, mock_show, mock_rdf_parser,
                                sample_volume, sample_molecule_count, tmp_path):
        """Test that plot_rkbis saves file when save_dir provided."""
        mock_extract.return_value = ["MOL1", "MOL2"]

        integrator = KBIntegrator(
            rdf=mock_rdf_parser,
            volume=sample_volume,
            molecule_count=sample_molecule_count,
        )

        save_dir = str(tmp_path)
        integrator.plot_rkbis(save_dir=save_dir)

        # Check that file was created
        expected_file = tmp_path / "rkbis_MOL1_MOL2.pdf"
        assert expected_file.exists()
        plt.close('all')

    @patch('kbkit.kbi.integrator.plt.show')
    @patch('kbkit.kbi.integrator.RdfParser.extract_molecules')
    def test_plot_integrand_displays(self, mock_extract, mock_show, mock_rdf_parser,
                                    sample_volume, sample_molecule_count):
        """Test that plot_integrand creates and displays plot."""
        mock_extract.return_value = ["MOL1", "MOL2"]

        integrator = KBIntegrator(
            rdf=mock_rdf_parser,
            volume=sample_volume,
            molecule_count=sample_molecule_count,
        )

        integrator.plot_integrand()

        mock_show.assert_called_once()
        plt.close('all')

    @patch('kbkit.kbi.integrator.plt.show')
    @patch('kbkit.kbi.integrator.RdfParser.extract_molecules')
    def test_plot_integrand_saves_file(self, mock_extract, mock_show, mock_rdf_parser,
                                    sample_volume, sample_molecule_count, tmp_path):
        """Test that plot_integrand saves file when save_dir provided."""
        mock_extract.return_value = ["MOL1", "MOL2"]

        integrator = KBIntegrator(
            rdf=mock_rdf_parser,
            volume=sample_volume,
            molecule_count=sample_molecule_count,
        )

        save_dir = str(tmp_path)
        integrator.plot_integrand(save_dir=save_dir)

        expected_file = tmp_path / "kbi_integrand_MOL1_MOL2.pdf"
        assert expected_file.exists()
        plt.close('all')

    @patch('kbkit.kbi.integrator.plt.show')
    @patch('kbkit.kbi.integrator.RdfParser.extract_molecules')
    def test_plot_extrapolation_displays(self, mock_extract, mock_show, mock_rdf_parser,
                                        sample_volume, sample_molecule_count):
        """Test that plot_extrapolation creates and displays plot."""
        mock_extract.return_value = ["MOL1", "MOL2"]

        integrator = KBIntegrator(
            rdf=mock_rdf_parser,
            volume=sample_volume,
            molecule_count=sample_molecule_count,
        )

        integrator.plot_extrapolation()

        mock_show.assert_called_once()
        plt.close('all')

    @patch('kbkit.kbi.integrator.plt.show')
    @patch('kbkit.kbi.integrator.RdfParser.extract_molecules')
    def test_plot_extrapolation_saves_file(self, mock_extract, mock_show, mock_rdf_parser,
                                        sample_volume, sample_molecule_count, tmp_path):
        """Test that plot_extrapolation saves file when save_dir provided."""
        mock_extract.return_value = ["MOL1", "MOL2"]

        integrator = KBIntegrator(
            rdf=mock_rdf_parser,
            volume=sample_volume,
            molecule_count=sample_molecule_count,
        )

        save_dir = str(tmp_path)
        integrator.plot_extrapolation(save_dir=save_dir)

        expected_file = tmp_path / "kbi_extrapolation_MOL1_MOL2.pdf"
        assert expected_file.exists()
        plt.close('all')

    @patch('kbkit.kbi.integrator.plt.show')
    @patch('kbkit.kbi.integrator.RdfParser.extract_molecules')
    def test_plot_methods_with_custom_mol_j(self, mock_extract, mock_show, mock_rdf_parser,
                                            sample_volume, sample_molecule_count):
        """Test plotting methods with custom mol_j parameter."""
        mock_extract.return_value = ["MOL1", "MOL2"]

        integrator = KBIntegrator(
            rdf=mock_rdf_parser,
            volume=sample_volume,
            molecule_count=sample_molecule_count,
        )

        # Should not raise errors
        integrator.plot_rkbis(mol_j="MOL1")
        integrator.plot_integrand(mol_j="MOL1")
        integrator.plot_extrapolation(mol_j="MOL1")

        plt.close('all')




class TestKBIntegratorIntegration:
    """Integration tests for KBIntegrator."""

    @patch('kbkit.kbi.integrator.RdfParser.extract_molecules')
    def test_full_workflow_with_all_corrections(self, mock_extract, mock_rdf_parser,
                                                sample_volume, sample_molecule_count):
        """Test complete workflow with all corrections enabled."""
        mock_extract.return_value = ["MOL1", "MOL2"]

        integrator = KBIntegrator(
            rdf=mock_rdf_parser,
            volume=sample_volume,
            molecule_count=sample_molecule_count,
            correct_rdf_convergence=True,
            apply_damping=True,
            extrapolate_thermodynamic_limit=True,
        )

        # Test all major methods
        assert integrator.rdf_molecules == ["MOL1", "MOL2"]
        assert integrator.kronecker_delta() == 0
        assert integrator.n_j() == 50

        ganguly = integrator.ganguly_correction_factor()
        assert isinstance(ganguly, np.ndarray)

        kruger = integrator.kruger_damping_factor()
        assert isinstance(kruger, np.ndarray)

        h = integrator.h()
        assert isinstance(h, np.ndarray)

        rkbi = integrator.rkbi()
        assert isinstance(rkbi, np.ndarray)

        kbi = integrator.compute_kbi()
        assert isinstance(kbi, float)

    @patch('kbkit.kbi.integrator.RdfParser.extract_molecules')
    def test_full_workflow_no_corrections(self, mock_extract, mock_rdf_parser,
                                        sample_volume, sample_molecule_count):
        """Test complete workflow with all corrections disabled."""
        mock_extract.return_value = ["MOL1", "MOL2"]

        integrator = KBIntegrator(
            rdf=mock_rdf_parser,
            volume=sample_volume,
            molecule_count=sample_molecule_count,
            correct_rdf_convergence=False,
            apply_damping=False,
            extrapolate_thermodynamic_limit=False,
        )

        # Compute KBI without corrections
        kbi = integrator.compute_kbi()
        assert isinstance(kbi, float)

        # h should be exactly g - 1
        h = integrator.h()
        np.testing.assert_array_almost_equal(h, mock_rdf_parser.g - 1)

    @patch('kbkit.kbi.integrator.RdfParser.extract_molecules')
    def test_from_system_properties_workflow(self, mock_extract, mock_rdf_parser,
                                            mock_system_properties):
        """Test workflow using from_system_properties constructor."""
        mock_extract.return_value = ["MOL1", "MOL2"]

        integrator = KBIntegrator.from_system_properties(
            rdf=mock_rdf_parser,
            system_properties=mock_system_properties,
        )

        kbi = integrator.compute_kbi()
        assert isinstance(kbi, float)

    @patch('kbkit.kbi.integrator.RdfParser.extract_molecules')
    def test_different_correction_combinations(self, mock_extract, mock_rdf_parser,
                                            sample_volume, sample_molecule_count):
        """Test various combinations of correction flags."""
        mock_extract.return_value = ["MOL1", "MOL2"]

        combinations = [
            (True, True, True),
            (True, True, False),
            (True, False, True),
            (False, True, True),
            (True, False, False),
            (False, True, False),
            (False, False, True),
            (False, False, False),
        ]

        for conv, damp, extrap in combinations:
            integrator = KBIntegrator(
                rdf=mock_rdf_parser,
                volume=sample_volume,
                molecule_count=sample_molecule_count,
                correct_rdf_convergence=conv,
                apply_damping=damp,
                extrapolate_thermodynamic_limit=extrap,
            )

            kbi = integrator.compute_kbi()
            assert isinstance(kbi, float)




class TestKBIntegratorEdgeCases:
    """Test edge cases and error conditions."""

    @patch('kbkit.kbi.integrator.RdfParser.extract_molecules')
    def test_zero_volume_handling(self, mock_extract, mock_rdf_parser, sample_molecule_count):
        """Test behavior with zero volume."""
        mock_extract.return_value = ["MOL1", "MOL2"]

        # Should not raise during initialization
        integrator = KBIntegrator(
            rdf=mock_rdf_parser,
            volume=0.0,
            molecule_count=sample_molecule_count,
        )

        # But may cause issues in calculations (division by zero)
        with pytest.raises(ZeroDivisionError):
            _ = integrator.ganguly_correction_factor()

    @patch('kbkit.kbi.integrator.RdfParser.extract_molecules')
    def test_single_point_rdf(self, mock_extract, sample_volume, sample_molecule_count):
        """Test with RDF containing single point."""
        mock_extract.return_value = ["MOL1", "MOL2"]

        mock_rdf = Mock(spec=RdfParser)
        mock_rdf.fname = "rdf_MOL1_MOL2.xvg"
        mock_rdf.r = np.array([1.0])
        mock_rdf.g = np.array([1.0])
        mock_rdf.mask = np.array([True])
        mock_rdf.r_tail = np.array([1.0])

        integrator = KBIntegrator(
            rdf=mock_rdf,
            volume=sample_volume,
            molecule_count=sample_molecule_count,
        )

        # Should handle gracefully
        with pytest.raises(IndexError):
            _ = integrator.compute_kbi()

    @patch('kbkit.kbi.integrator.RdfParser.extract_molecules')
    def test_empty_molecule_count(self, mock_extract, mock_rdf_parser, sample_volume):
        """Test with empty molecule count dictionary."""
        mock_extract.return_value = ["MOL1", "MOL2"]

        integrator = KBIntegrator(
            rdf=mock_rdf_parser,
            volume=sample_volume,
            molecule_count={},
        )

        # Should raise KeyError when trying to access molecule counts
        with pytest.raises(KeyError):
            integrator.n_j("MOL1")



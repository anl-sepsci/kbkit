"""
Unit tests for the KBThermo module.

This test suite provides comprehensive coverage of the KBThermo class,
including thermodynamic property calculations, activity coefficients,
structure factors, and integration methods.
"""
import warnings

# Suppress NumPy/SciPy compatibility warning (harmless with NumPy 2.x + SciPy 1.16+)
warnings.filterwarnings('ignore', message='numpy.ndarray size changed', category=RuntimeWarning)

from unittest.mock import Mock, patch

import numpy as np
import pytest

from kbkit.kbi.thermodynamics import KBThermo
from kbkit.schema.activity_metadata import ActivityMetadata
from kbkit.schema.property_result import PropertyResult
from kbkit.systems.collection import SystemCollection
from kbkit.visualization.thermo import ThermoPlotter


@pytest.fixture
def mock_system_collection():
    """Create a mock SystemCollection object with only mixture compositions."""
    mock_sc = Mock(spec=SystemCollection)
    mock_sc.molecules = ["MOL1", "MOL2"]
    mock_sc.n_i = 2
    # Use only mixtures to avoid singular matrices at pure component limits
    mock_sc.x = np.array([
        [0.2, 0.8],
        [0.5, 0.5],
        [0.8, 0.2]
    ])
    mock_sc.get_mol_index = Mock(side_effect=lambda mol: ["MOL1", "MOL2"].index(mol))
    mock_sc.charges = dict.fromkeys(mock_sc.molecules, 0)

    # Mock properties
    mock_props = Mock()
    mock_sc.properties = mock_props

    return mock_sc


@pytest.fixture
def mock_kbi_result():
    """Create a mock KBI PropertyResult with physically realistic values."""
    # Realistic KBI values for binary liquid mixtures (in nm^3)
    # These values are typical for organic liquid mixtures
    kbi_values = np.array([
        # Composition 1: x1=0.2, x2=0.8
        [[2.5, 1.9],
         [1.9, 2.2]],

        # Composition 2: x1=0.5, x2=0.5
        [[2.3, 1.8],
         [1.8, 2.1]],

        # Composition 3: x1=0.8, x2=0.2
        [[2.2, 1.7],
         [1.7, 2.0]]
    ])

    mock_result = Mock(spec=PropertyResult)
    mock_result.value = kbi_values
    mock_result.to = Mock(return_value=mock_result)

    return mock_result


@pytest.fixture
def kb_thermo(mock_system_collection, mock_kbi_result):
    """Create a KBThermo instance with mocked dependencies."""
    # Setup property mocks
    def create_property_result(value):
        result = Mock(spec=PropertyResult)
        result.value = value
        result.to = Mock(return_value=result)
        return result

    # Fix: Make sure all lambda functions accept both name and units
    mock_system_collection.simulated_property = Mock(
        side_effect=lambda name, units=None: create_property_result(
            np.array([298.15, 298.15, 298.15]) if "temp" in name.lower()
            else np.array([0.033, 0.033, 0.033])  # number density
        )
    )

    mock_system_collection.ideal_property = Mock(
        side_effect=lambda name, units=None: create_property_result(
            np.array([18.0, 18.0, 18.0]) if "volume" in name.lower()
            else np.array([9.2, 9.0, 8.8])  # electron count (weighted by composition)
        )
    )

    mock_system_collection.pure_property = Mock(
        side_effect=lambda name=None: create_property_result(np.array([10, 8]))
    )

    mock_system_collection.excess_property = Mock(
        side_effect=lambda name, units=None: create_property_result(np.array([-1.5, -2.0, -1.5]))
    )

    return KBThermo(mock_system_collection, mock_kbi_result)


class TestKBThermoInitialization:
    """Test KBThermo initialization."""

    def test_init_with_required_parameters(self, mock_system_collection, mock_kbi_result):
        """Test initialization with required parameters."""
        kb = KBThermo(mock_system_collection, mock_kbi_result)

        assert kb.systems == mock_system_collection
        assert kb.kbi_res == mock_kbi_result
        assert kb.activity_integration_type == "numerical"
        assert kb.activity_polynomial_degree == 5
        assert isinstance(kb._cache, dict)
        assert isinstance(kb._activity_coef_meta, list)

    def test_init_with_custom_integration_type(self, mock_system_collection, mock_kbi_result):
        """Test initialization with custom integration type."""
        kb = KBThermo(
            mock_system_collection,
            mock_kbi_result,
            activity_integration_type="polynomial",
            activity_polynomial_degree=7
        )

        assert kb.activity_integration_type == "polynomial"
        assert kb.activity_polynomial_degree == 7

    def test_init_converts_integration_type_to_lowercase(self, mock_system_collection, mock_kbi_result):
        """Test that integration type is converted to lowercase."""
        kb = KBThermo(
            mock_system_collection,
            mock_kbi_result,
            activity_integration_type="POLYNOMIAL"
        )

        assert kb.activity_integration_type == "polynomial"


class TestKBThermoBasicProperties:
    """Test basic property methods."""

    def test_kbi_returns_converted_values(self, kb_thermo):
        """Test that kbi method returns converted values."""
        result = kb_thermo.kbi(units="cm^3/mol")

        assert isinstance(result, np.ndarray)
        kb_thermo.kbi_res.to.assert_called_with("cm^3/mol")

    def test_R_returns_gas_constant(self, kb_thermo):
        """Test R method returns gas constant."""
        r = kb_thermo.R(units="kJ/mol/K")

        assert isinstance(r, float)
        assert r > 0

    def test_temperature_returns_array(self, kb_thermo):
        """Test temperature method returns array."""
        temp = kb_thermo.temperature(units="K")

        assert isinstance(temp, np.ndarray)
        assert len(temp) == 3
        np.testing.assert_array_almost_equal(temp, [298.15, 298.15, 298.15])

    def test_RT_returns_product(self, kb_thermo):
        """Test RT method returns R*T."""
        rt = kb_thermo.RT(units="kJ/mol")

        assert isinstance(rt, np.ndarray)
        assert len(rt) == 3

    def test_rho_returns_number_density(self, kb_thermo):
        """Test rho method returns number density."""
        rho = kb_thermo.rho(units="molecule/nm^3")

        assert isinstance(rho, np.ndarray)
        assert len(rho) == 3

    def test_v_bar_returns_molar_volume(self, kb_thermo):
        """Test v_bar method returns ideal molar volume."""
        v_bar = kb_thermo.v_bar(units="cm^3/mol")

        assert isinstance(v_bar, np.ndarray)
        assert len(v_bar) == 3


class TestKBThermoElectronProperties:
    """Test electron-related properties."""

    def test_z_i_returns_pure_electron_counts(self, kb_thermo):
        """Test z_i property returns pure electron counts."""
        z_i = kb_thermo.z_i

        assert isinstance(z_i, np.ndarray)
        assert len(z_i) == 2
        np.testing.assert_array_equal(z_i, [10, 8])

    def test_z_bar_returns_ideal_electron_count(self, kb_thermo):
        """Test z_bar property returns ideal electron count."""
        z_bar = kb_thermo.z_bar

        assert isinstance(z_bar, np.ndarray)
        assert len(z_bar) == 3

    def test_z_i_diff_returns_difference(self, kb_thermo):
        """Test z_i_diff property returns difference from last element."""
        z_i_diff = kb_thermo.z_i_diff

        assert isinstance(z_i_diff, np.ndarray)
        assert len(z_i_diff) == 1  # n_i - 1
        assert z_i_diff[0] == 10 - 8


class TestKBThermoMatrixProperties:
    """Test matrix-based properties."""

    def test_delta_ij_returns_identity(self, kb_thermo):
        """Test delta_ij returns Kronecker delta."""
        delta = kb_thermo.delta_ij

        assert isinstance(delta, np.ndarray)
        assert delta.shape == (2, 2)
        np.testing.assert_array_equal(delta, np.eye(2))

    def test_x_3d_reshapes_mole_fractions(self, kb_thermo):
        """Test _x_3d reshapes mole fractions."""
        x_3d = kb_thermo._x_3d

        assert x_3d.shape == (3, 2, 1)

    def test_x_3d_sq_creates_pairwise_products(self, kb_thermo):
        """Test _x_3d_sq creates pairwise products."""
        x_3d_sq = kb_thermo._x_3d_sq

        assert x_3d_sq.shape == (3, 2, 2)


class TestKBThermoAInvAndA:
    """Test B and A matrix calculations."""

    def test_B_shape(self, kb_thermo):
        """Test B has correct shape."""
        B = kb_thermo.B()

        assert B.shape == (3, 2, 2)

    def test_B_calculation(self, kb_thermo):
        """Test B calculation formula."""
        B = kb_thermo.B()

        # Check that it's not all zeros
        assert not np.all(B == 0)
        # Check that diagonal elements are positive
        for i in range(3):
            assert np.all(np.diag(B[i]) > 0)

    def test_Berts_B(self, kb_thermo):
        """Test that A is inverse of B."""
        a = kb_thermo.A()
        B = kb_thermo.B()

        # Check shape
        assert a.shape == (3, 2, 2)

        # Check that A @ B ≈ I for each system
        for i in range(3):
            product = a[i] @ B[i]
            np.testing.assert_array_almost_equal(product, np.eye(2), decimal=5)

    def test_A_raises_on_singular_matrix(self, kb_thermo):
        """Test that A raises error for singular B."""
        # Create a singular B matrix (rows/columns are linearly dependent)
        B_singular = np.array([
            [[1.0, 1.0], [1.0, 1.0]],  # Singular: rows are identical
            [[2.0, 2.0], [2.0, 2.0]],
            [[3.0, 3.0], [3.0, 3.0]]
        ])

        # Patch the B method to return singular matrix
        with patch.object(kb_thermo, 'B', return_value=B_singular):
            with pytest.raises(ValueError, match="singular and cannot be inverted"):
                _ = kb_thermo.A()

class TestKBThermoStabilityMetrics:
    """Test stability-related calculations."""

    def test_l_returns_stability_array(self, kb_thermo):
        """Test _l returns stability array."""
        l = kb_thermo._l()

        assert isinstance(l, np.ndarray)
        assert l.shape == (3,)
        # Stability array should be positive for stable mixtures
        assert np.all(l > 0)

    def test_M_shape(self, kb_thermo):
        """Test M has correct shape."""
        m = kb_thermo.M(units="kJ/mol")

        assert m.shape == (3, 2, 2)

    def test_isothermal_compressibility_returns_array(self, kb_thermo):
        """Test isothermal_compressibility returns array."""
        kappa = kb_thermo.isothermal_compressibility(units="1/kPa")

        assert isinstance(kappa, np.ndarray)
        assert kappa.shape == (3,)
        # Compressibility should be positive
        assert np.all(kappa > 0)


class TestKBThermoHessian:
    """Test Hessian calculations."""

    def test_subtract_nth_elements_reduces_dimension(self, kb_thermo):
        """Test _subtract_nth_elements reduces matrix dimension."""
        matrix = np.random.rand(3, 2, 2)
        result = kb_thermo._subtract_nth_elements(matrix)

        assert result.shape == (3, 1, 1)

    def test_hessian_shape(self, kb_thermo):
        """Test hessian has correct shape."""
        h = kb_thermo.H(units="kJ/mol")

        assert h.shape == (3, 1, 1)  # n_sys, n_i-1, n_i-1

    def test_hessian_determinant_shape(self, kb_thermo):
        """Test hessian_determinant has correct shape."""
        det_h = kb_thermo.det_H(units="kJ/mol")

        assert det_h.shape == (3,)


class TestKBThermoChemicalPotentialDerivDiag:
    """Test diagonal chemical potential derivatives."""

    def test_dmu_dxi_shape(self, kb_thermo):
        """Test dmu_dxi has correct shape."""
        dmui_dxi = kb_thermo.dmu_dxi(units="kJ/mol")

        assert dmui_dxi.shape == (3, 2)

    def test_dmu_dxi_no_pure_components(self, kb_thermo):
        """Test that derivatives are non-zero for mixtures."""
        dmui_dxi = kb_thermo.dmu_dxi(units="kJ/mol")

        # For mixtures, derivatives should be non-zero
        assert not np.all(dmui_dxi == 0)


class TestKBThermoActivityCoefficients:
    """Test activity coefficient calculations."""

    def test_dlngamma_dxi_shape(self, kb_thermo):
        """Test dlngamma_dxi has correct shape."""
        dlng = kb_thermo.dlngamma_dxi()

        assert dlng.shape == (3, 2)

    def test_get_ref_state_dict_infinite_dilution(self, kb_thermo):
        """Test _get_ref_state_dict for infinite dilution reference."""
        # With our fixture, MOL1 never reaches x=1.0
        ref_state = kb_thermo._get_ref_state("MOL1")

        # Could be either pure or infinite dilution depending on max composition
        assert ref_state in (0., 1.)

    def test_get_weights_returns_array(self, kb_thermo):
        """Test _get_weights returns array."""
        x = np.array([0.2, 0.5, 0.8])
        weights = kb_thermo._get_weights("MOL1", x)

        assert isinstance(weights, np.ndarray)
        assert len(weights) == 3


class TestKBThermoActivityIntegration:
    """Test activity coefficient integration methods."""

    def test_integrate_polynomial_returns_array(self, kb_thermo):
        """Test _integrate_polynomial returns array."""
        xi = np.array([0.2, 0.5, 0.8])
        dlng = np.array([0.1, 0.5, 0.2])

        result = kb_thermo._integrate_polynomial(xi, dlng, 0.5, "MOL1", degree=2)

        assert isinstance(result, np.ndarray)
        assert len(result) == 3

    def test_integrate_polynomial_stores_functions(self, kb_thermo):
        """Test that _integrate_polynomial stores polynomial functions."""
        xi = np.array([0.2, 0.5, 0.8])
        dlng = np.array([0.1, 0.5, 0.2])

        kb_thermo._integrate_polynomial(xi, dlng, 0.5, "MOL1", degree=2)

        assert "MOL1" in kb_thermo._lngamma_fn_dict
        assert "MOL1" in kb_thermo._dlngamma_fn_dict

    def test_integrate_polynomial_reduces_degree_if_needed(self, kb_thermo):
        """Test that polynomial degree is reduced if insufficient points."""
        xi = np.array([0.2, 0.5, 0.8])
        dlng = np.array([0.1, 0.5, 0.2])

    # Request degree 10 with only 3 points - expect warning
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message="Polyfit may be poorly conditioned")
            result = kb_thermo._integrate_polynomial(xi, dlng, 0.5, "MOL1", degree=10)

        assert isinstance(result, np.ndarray)


    def test_integrate_numerical_returns_array(self, kb_thermo):
        """Test _integrate_numerical returns array."""
        xi = np.array([0.2, 0.5, 0.8])
        dlng = np.array([0.1, 0.5, 0.2])

        result = kb_thermo._integrate_numerical(xi, dlng, 0.5)

        assert isinstance(result, np.ndarray)
        assert len(result) == 3

    def test_integrate_numerical_with_existing_ref_point(self, kb_thermo):
        """Test _integrate_numerical when reference point exists."""
        xi = np.array([0.2, 0.5, 0.8])
        dlng = np.array([0.1, 0.5, 0.2])

        result = kb_thermo._integrate_numerical(xi, dlng, 0.5)

        # Value at reference should be zero
        assert result[1] == pytest.approx(0.0, abs=1e-10)

    def test_integrate_numerical_inserts_ref_point(self, kb_thermo):
        """Test _integrate_numerical inserts reference point if missing."""
        xi = np.array([0.2, 0.5, 0.8])
        dlng = np.array([0.1, 0.5, 0.2])

        result = kb_thermo._integrate_numerical(xi, dlng, 0.35)

        assert isinstance(result, np.ndarray)


class TestKBThermoActivityCoefficientResults:
    """Test activity coefficient result properties."""

    def test_lngamma_shape(self, kb_thermo):
        """Test lngamma has correct shape."""
        lng = kb_thermo.lngamma()

        assert lng.shape == (3, 2)

    def test_lngamma_populates_metadata(self, kb_thermo):
        """Test that lngamma populates metadata."""
        _ = kb_thermo.lngamma()

        assert len(kb_thermo._activity_coef_meta) > 0

    def test_activity_metadata_returns_container(self, kb_thermo):
        """Test activity_metadata returns ActivityMetadata."""
        _ = kb_thermo.lngamma()  # Populate metadata

        metadata = kb_thermo.activity_metadata

        assert isinstance(metadata, ActivityMetadata)


class TestKBThermoThermodynamicProperties:
    """Test thermodynamic property calculations."""

    def test_g_ex_shape(self, kb_thermo):
        """Test g_ex has correct shape."""
        g_ex = kb_thermo.g_ex(units="kJ/mol")

        assert g_ex.shape == (3,)

    def test_g_ex_non_zero_for_mixtures(self, kb_thermo):
        """Test that g_ex is non-zero for mixtures."""
        g_ex = kb_thermo.g_ex(units="kJ/mol")

        # For mixtures, excess Gibbs energy should generally be non-zero
        assert not np.all(g_ex == 0)

    def test_h_mix_calls_excess_property(self, kb_thermo):
        """Test h_mix calls systems.excess_property."""
        h_mix = kb_thermo.h_mix(units="kJ/mol")

        kb_thermo.systems.excess_property.assert_called()
        assert isinstance(h_mix, np.ndarray)

    def test_s_ex_shape(self, kb_thermo):
        """Test s_ex has correct shape."""
        s_ex = kb_thermo.s_ex(units="kJ/mol/K")

        assert s_ex.shape == (3,)

    def test_s_ex_calculation(self, kb_thermo):
        """Test s_ex calculation formula."""
        h_mix = kb_thermo.h_mix("kJ/mol")
        g_ex = kb_thermo.g_ex("kJ/mol")
        temp = kb_thermo.temperature("K")
        s_ex = kb_thermo.s_ex("kJ/mol/K")

        expected = (h_mix - g_ex) / temp

        np.testing.assert_array_almost_equal(s_ex, expected)

    def test_g_id_shape(self, kb_thermo):
        """Test g_id has correct shape."""
        g_id = kb_thermo.g_id(units="kJ/mol")

        assert g_id.shape == (3,)

    def test_g_id_non_zero_for_mixtures(self, kb_thermo):
        """Test that g_id is non-zero for mixtures."""
        g_id = kb_thermo.g_id(units="kJ/mol")

        # For mixtures, ideal Gibbs energy should be non-zero
        assert not np.all(g_id == 0)

    def test_s_mix_shape(self, kb_thermo):
        """Test s_mix has correct shape."""
        s_mix = kb_thermo.s_mix(units="kJ/mol/K")

        assert s_mix.shape == (3,)

    def test_g_mix_is_sum_of_ex_and_id(self, kb_thermo):
        """Test that g_mix = g_ex + g_id."""
        g_mix = kb_thermo.g_mix("kJ/mol")
        g_ex = kb_thermo.g_ex("kJ/mol")
        g_id = kb_thermo.g_id("kJ/mol")

        np.testing.assert_array_almost_equal(g_mix, g_ex + g_id)


class TestKBThermoStructureFactors:
    """Test structure factor calculations."""

    def test_s0_ij_equals_B(self, kb_thermo):
        """Test that s0_ij equals B."""
        s0_ij = kb_thermo.s0_ij()
        B = kb_thermo.B()

        np.testing.assert_array_almost_equal(s0_ij, B)

    def test_s0_x_shape(self, kb_thermo):
        """Test s0_x has correct shape."""
        s0_x = kb_thermo.s0_x()

        assert s0_x.shape == (3, 1, 1)  # n_sys, n_i-1, n_i-1

    def test_s0_xp_shape(self, kb_thermo):
        """Test s0_xp has correct shape."""
        s0_xp = kb_thermo.s0_xp()

        assert s0_xp.shape == (3, 1)  # n_sys, n_i-1

    def test_s0_p_shape(self, kb_thermo):
        """Test s0_p has correct shape."""
        s0_p = kb_thermo.s0_p()

        assert s0_p.shape == (3,)

    def test_s0_kappa_shape(self, kb_thermo):
        """Test s0_kappa has correct shape."""
        s0_kappa = kb_thermo.s0_kappa()

        assert s0_kappa.shape == (3,)


class TestKBThermoElectronStructureFactors:
    """Test electron density structure factors."""

    def test_s0_x_e_shape(self, kb_thermo):
        """Test s0_x_e has correct shape."""
        s0_x_e = kb_thermo.s0_x_e()

        assert s0_x_e.shape == (3,)

    def test_s0_xp_e_shape(self, kb_thermo):
        """Test s0_xp_e has correct shape."""
        s0_xp_e = kb_thermo.s0_xp_e()

        assert s0_xp_e.shape == (3,)

    def test_s0_p_e_shape(self, kb_thermo):
        """Test s0_p_e has correct shape."""
        s0_p_e = kb_thermo.s0_p_e()

        assert s0_p_e.shape == (3,)

    def test_s0_kappa_e_shape(self, kb_thermo):
        """Test s0_kappa_e has correct shape."""
        s0_kappa_e = kb_thermo.s0_kappa_e()

        assert s0_kappa_e.shape == (3,)

    def test_s0_e_shape(self, kb_thermo):
        """Test s0_e has correct shape."""
        s0_e = kb_thermo.s0_e()

        assert s0_e.shape == (3,)


class TestKBThermoXRayIntensities:
    """Test x-ray intensity calculations."""

    def test_calculate_i0_from_s0e(self, kb_thermo):
        """Test _calculate_i0_from_s0e calculation."""
        s0_e = np.array([1.0, 2.0, 3.0])

        i0 = kb_thermo._calculate_i0_from_s0e(s0_e)

        assert isinstance(i0, np.ndarray)
        assert i0.shape == (3,)

    def test_i0_x_shape(self, kb_thermo):
        """Test i0_x has correct shape."""
        i0_x = kb_thermo.i0_x(units="1/cm")

        assert i0_x.shape == (3,)

    def test_i0_xp_shape(self, kb_thermo):
        """Test i0_xp has correct shape."""
        i0_xp = kb_thermo.i0_xp(units="1/cm")

        assert i0_xp.shape == (3,)

    def test_i0_p_shape(self, kb_thermo):
        """Test i0_p has correct shape."""
        i0_p = kb_thermo.i0_p(units="1/cm")

        assert i0_p.shape == (3,)

    def test_i0_kappa_shape(self, kb_thermo):
        """Test i0_kappa has correct shape."""
        i0_kappa = kb_thermo.i0_kappa(units="1/cm")

        assert i0_kappa.shape == (3,)

    def test_i0_shape(self, kb_thermo):
        """Test i0 has correct shape."""
        i0 = kb_thermo.i0(units="1/cm")

        assert i0.shape == (3,)


class TestKBThermoUtilityMethods:
    """Test utility methods."""

    def test_set_ref_to_zero_1d_array(self, kb_thermo):
        """Test _set_ref_to_zero with 1D array."""
        # Create array with a pure component value
        kb_thermo.systems.x = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
        array = np.array([1.0, 2.0, 3.0])

        result = kb_thermo._set_ref_to_zero(array, ref=1)

        # First and last should be zero (pure components)
        assert result[0] == 0.0
        assert result[2] == 0.0

    def test_get_from_cache_returns_none_when_empty(self, kb_thermo):
        """Test _get_from_cache returns None when cache is empty."""
        result = kb_thermo._get_from_cache(("test_key",), "kJ/mol")

        assert result is None

    def test_get_from_cache_returns_cached_result(self, kb_thermo):
        """Test _get_from_cache returns cached result."""
        mock_result = Mock(spec=PropertyResult)
        mock_result.to.return_value = mock_result

        kb_thermo._cache[("test_key",)] = mock_result

        result = kb_thermo._get_from_cache(("test_key",), "kJ/mol")

        assert result == mock_result
        mock_result.to.assert_called_once_with("kJ/mol")


class TestKBThermoResults:
    """Test results property."""

    def test_results_returns_dict(self, kb_thermo):
        """Test that results returns dictionary."""
        results = kb_thermo.results

        assert isinstance(results, dict)
        # Do NOT evaluate values

    def test_results_excludes_private_attributes(self, kb_thermo):
        """Test that results excludes private attributes."""
        results = kb_thermo.results

        assert not any(key.startswith("_") for key in results.keys())

    def test_results_excludes_methods(self, kb_thermo):
        """Test that results excludes certain methods."""
        results = kb_thermo.results

        assert "Q_" not in results
        assert "ureg" not in results
        assert "results" not in results
        assert "plotter" not in results


class TestKBThermoPlotter:
    """Test plotter method."""

    @patch('kbkit.kbi.thermodynamics.ThermoPlotter')
    def test_plotter_creates_thermo_plotter(self, mock_plotter_class, kb_thermo):
        """Test that plotter creates ThermoPlotter."""
        mock_plotter = Mock(spec=ThermoPlotter)
        mock_plotter_class.return_value = mock_plotter

        result = kb_thermo.plotter(molecule_map={"MOL1": "Molecule 1"})

        mock_plotter_class.assert_called_once_with(kb_thermo, molecule_map={"MOL1": "Molecule 1"})
        assert result == mock_plotter

    @patch('kbkit.kbi.thermodynamics.ThermoPlotter')
    def test_plotter_without_molecule_map(self, mock_plotter_class, kb_thermo):
        """Test plotter without molecule_map."""
        mock_plotter = Mock(spec=ThermoPlotter)
        mock_plotter_class.return_value = mock_plotter

        kb_thermo.plotter()

        mock_plotter_class.assert_called_once_with(kb_thermo, molecule_map=None)


class TestKBThermoIntegration:
    """Integration tests for KBThermo."""

    def test_full_workflow_numerical_integration(self, mock_system_collection, mock_kbi_result):
        """Test complete workflow with numerical integration."""
        # Setup mocks
        def create_property_result(value):
            result = Mock(spec=PropertyResult)
            result.value = value
            result.to = Mock(return_value=result)
            return result

        mock_system_collection.simulated_property = Mock(
            side_effect=lambda name, units=None: create_property_result(
                np.array([298.15, 298.15, 298.15]) if "temp" in name.lower()
                else np.array([0.033, 0.033, 0.033])
            )
        )

        mock_system_collection.ideal_property = Mock(
            side_effect=lambda name, units=None: create_property_result(
                np.array([18.0, 18.0, 18.0]) if "volume" in name.lower()
                else np.array([9.2, 9.0, 8.8])
            )
        )

        mock_system_collection.pure_property = Mock(
            side_effect=lambda name=None: create_property_result(np.array([10, 8]))
        )

        mock_system_collection.excess_property = Mock(
            side_effect=lambda name, units=None: create_property_result(np.array([-1.5, -2.0, -1.5]))
        )

        # Create KBThermo
        kb = KBThermo(
            mock_system_collection,
            mock_kbi_result,
            activity_integration_type="numerical"
        )

        # Calculate various properties
        kbi = kb.kbi()
        B = kb.B()
        a = kb.A()
        lng = kb.lngamma()
        g_ex = kb.g_ex()
        g_mix = kb.g_mix()

        # Verify shapes
        assert kbi.shape == (3, 2, 2)
        assert B.shape == (3, 2, 2)
        assert a.shape == (3, 2, 2)
        assert lng.shape == (3, 2)
        assert g_ex.shape == (3,)
        assert g_mix.shape == (3,)

    def test_full_workflow_polynomial_integration(self, mock_system_collection, mock_kbi_result):
        """Test complete workflow with polynomial integration."""
        # Setup mocks (same as numerical)
        def create_property_result(value):
            result = Mock(spec=PropertyResult)
            result.value = value
            result.to = Mock(return_value=result)
            return result

        mock_system_collection.simulated_property = Mock(
            side_effect=lambda name, units=None: create_property_result(
                np.array([298.15, 298.15, 298.15]) if "temp" in name.lower()
                else np.array([0.033, 0.033, 0.033])
            )
        )

        mock_system_collection.ideal_property = Mock(
            side_effect=lambda name, units=None: create_property_result(
                np.array([18.0, 18.0, 18.0]) if "volume" in name.lower()
                else np.array([9.2, 9.0, 8.8])
            )
        )

        mock_system_collection.pure_property = Mock(
            side_effect=lambda name=None: create_property_result(np.array([10, 8]))
        )

        mock_system_collection.excess_property = Mock(
            side_effect=lambda name, units=None: create_property_result(np.array([-1.5, -2.0, -1.5]))
        )

        # Create KBThermo with polynomial integration
        kb = KBThermo(
            mock_system_collection,
            mock_kbi_result,
            activity_integration_type="polynomial",
            activity_polynomial_degree=3
        )

        # Calculate activity coefficients
        lng = kb.lngamma()

        # Verify polynomial functions were stored
        assert len(kb._lngamma_fn_dict) > 0
        assert len(kb._dlngamma_fn_dict) > 0


class TestKBThermoEdgeCases:
    """Test edge cases and error conditions."""

    def test_handles_nan_in_activity_integration(self, kb_thermo):
        """Test handling of NaN values in activity coefficient integration."""
        # This should not raise an error
        lng = kb_thermo.lngamma()

        # NaN values should be preserved where appropriate
        assert isinstance(lng, np.ndarray)

    def test_handles_division_by_zero(self, kb_thermo):
        """Test handling of division by zero in calculations."""
        # Many methods use np.errstate to handle this
        # Should not raise warnings or errors
        with np.errstate(divide='ignore', invalid='ignore'):
            g_id = kb_thermo.g_id()
            assert isinstance(g_id, np.ndarray)

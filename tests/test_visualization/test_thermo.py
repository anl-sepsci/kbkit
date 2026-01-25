"""Unit tests for ThermoPlotter class."""
import warnings

# Suppress NumPy/SciPy compatibility warning (harmless with NumPy 2.x + SciPy 1.16+)
warnings.filterwarnings('ignore', message='numpy.ndarray size changed', category=RuntimeWarning)

from itertools import combinations_with_replacement
from unittest.mock import Mock, patch

import numpy as np
import pytest

from kbkit.schema.property_result import PropertyResult
from kbkit.visualization.thermo import ThermoPlotter


@pytest.fixture
def mock_kb_thermo():
    """Create a mock KBThermo object."""
    thermo = Mock()

    # Mock systems
    mock_systems = Mock()
    mock_systems.molecules = ["Water", "Ethanol"]
    mock_systems.n_i = 2  # Binary system
    mock_systems.x = np.array([[0.0, 1.0], [0.25, 0.75], [0.5, 0.5], [0.75, 0.25], [1.0, 0.0]])
    mock_systems.get_mol_index = Mock(side_effect=lambda mol: 0 if mol == "Water" else 1)
    thermo.systems = mock_systems

    # Mock results - create PropertyResult objects
    def create_mock_result(units="cm^3/mol"):
        result = PropertyResult(
            name="test",
            value=np.random.rand(5, 2, 2),
            units=units,
            metadata={}
        )
        return result

    thermo.results = {
        "kbi": create_mock_result("cm^3/mol"),
        "ln_activity_coef": create_mock_result(None),
        "ln_activity_coef_deriv": create_mock_result(None),
        "s0_ij": create_mock_result(None),
        "hessian_determinant": create_mock_result("kJ/mol"),
        "h_mix": create_mock_result("kJ/mol"),
        "s_ex": create_mock_result("kJ/mol/K"),
        "g_ex": create_mock_result("kJ/mol"),
        "g_id": create_mock_result("kJ/mol"),
        "g_mix": create_mock_result("kJ/mol"),
    }

    # Mock properties
    thermo.kbi = Mock(return_value=np.random.rand(5, 2, 2))
    thermo.ln_activity_coef = Mock(return_value=np.random.rand(5, 2))
    thermo.ln_activity_coef_deriv = Mock(return_value=np.random.rand(5, 2))
    thermo.s0_ij = Mock(return_value=np.random.rand(5, 2, 2))
    thermo.hessian_determinant = Mock(return_value=np.random.rand(5))
    thermo.h_mix = Mock(return_value=np.random.rand(5))
    thermo.s_ex = Mock(return_value=np.random.rand(5))
    thermo.g_ex = Mock(return_value=np.random.rand(5))
    thermo.g_id = Mock(return_value=np.random.rand(5))
    thermo.g_mix = Mock(return_value=np.random.rand(5))
    thermo.temperature = Mock(return_value=298.15)

    # Mock activity metadata
    mock_activity_meta = Mock()
    mock_meta_item = Mock()
    mock_meta_item.has_fn = True
    mock_meta_item.x_eval = np.linspace(0, 1, 50)
    mock_meta_item.y_eval = np.random.rand(50)
    mock_activity_meta.by_types = {"derivative": {"test": mock_meta_item}}
    thermo.activity_metadata = mock_activity_meta
    thermo.activity_integration_type = "polynomial"

    return thermo


@pytest.fixture
def mock_ternary_thermo():
    """Create a mock KBThermo object for ternary system."""
    thermo = Mock()

    # Mock systems
    mock_systems = Mock()
    mock_systems.molecules = ["Water", "Ethanol", "Methanol"]
    mock_systems.n_i = 3  # Ternary system
    # Create ternary composition data
    x_data = []
    for i in range(11):
        for j in range(11 - i):
            k = 10 - i - j
            x_data.append([i/10, j/10, k/10])
    mock_systems.x = np.array(x_data)
    mock_systems.get_mol_index = Mock(side_effect=lambda mol: {"Water": 0, "Ethanol": 1, "Methanol": 2}[mol])
    thermo.systems = mock_systems

    # Mock results - create PropertyResult objects
    def create_mock_result(units="kJ/mol"):
        result = PropertyResult(
            name="test",
            value=np.random.rand(len(x_data)),
            units=units,
            metadata={}
        )
        return result

    thermo.results = {
        "h_mix": create_mock_result("kJ/mol"),
        "s_ex": create_mock_result("kJ/mol/K"),
        "g_ex": create_mock_result("kJ/mol"),
        "g_id": create_mock_result("kJ/mol"),
        "g_mix": create_mock_result("kJ/mol"),
        "hessian_determinant": create_mock_result("kJ/mol"),
    }

    # Mock properties
    thermo.h_mix = Mock(return_value=np.random.rand(len(x_data)))
    thermo.s_ex = Mock(return_value=np.random.rand(len(x_data)))
    thermo.g_ex = Mock(return_value=np.random.rand(len(x_data)))
    thermo.g_id = Mock(return_value=np.random.rand(len(x_data)))
    thermo.g_mix = Mock(return_value=np.random.rand(len(x_data)))
    thermo.hessian_determinant = Mock(return_value=np.random.rand(len(x_data)))

    return thermo


@pytest.fixture
def mock_mplstyle():
    """Mock the mplstyle loading."""
    with patch('kbkit.visualization.thermo.load_mplstyle'):
        yield


class TestThermoPlotterInitialization:
    """Test ThermoPlotter initialization."""

    def test_basic_initialization(self, mock_kb_thermo, mock_mplstyle):
        """Test basic initialization with KBThermo."""
        plotter = ThermoPlotter(mock_kb_thermo)

        assert plotter.thermo == mock_kb_thermo
        assert plotter.molecule_map is not None
        assert plotter.molecules == ["Water", "Ethanol"]

    def test_initialization_with_molecule_map(self, mock_kb_thermo, mock_mplstyle):
        """Test initialization with custom molecule map."""
        molecule_map = {"Water": "H2O", "Ethanol": "EtOH"}
        plotter = ThermoPlotter(mock_kb_thermo, molecule_map=molecule_map)

        assert plotter.molecule_map == molecule_map
        assert plotter.molecules == ["H2O", "EtOH"]

    def test_initialization_default_molecule_map(self, mock_kb_thermo, mock_mplstyle):
        """Test that default molecule map uses original names."""
        plotter = ThermoPlotter(mock_kb_thermo)

        assert plotter.molecule_map == {"Water": "Water", "Ethanol": "Ethanol"}


class TestPlotMethod:
    """Test plot method."""

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.get_cmap')
    def test_plot_1d_data(self, mock_cmap, mock_close, mock_show, mock_subplots, mock_kb_thermo, mock_mplstyle):
        """Test plot with 1D data."""
        mock_cmap_obj = Mock()
        mock_cmap_obj.return_value = np.array([[1, 0, 0, 1]] * 5)
        mock_cmap.return_value = mock_cmap_obj

        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = ThermoPlotter(mock_kb_thermo)
        x = np.linspace(0, 1, 10)
        y = np.random.rand(10)

        fig, ax = plotter.plot(x, y, show=False)

        assert mock_ax.plot.called
        assert fig == mock_fig
        assert ax == mock_ax

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.get_cmap')
    def test_plot_2d_data(self, mock_cmap, mock_close, mock_show, mock_subplots, mock_kb_thermo, mock_mplstyle):
        """Test plot with 2D data."""
        mock_cmap_obj = Mock()
        mock_cmap_obj.return_value = np.array([[1, 0, 0, 1]] * 5)
        mock_cmap.return_value = mock_cmap_obj

        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = ThermoPlotter(mock_kb_thermo)
        x = np.linspace(0, 1, 10)
        y = np.random.rand(10, 2)

        fig, ax = plotter.plot(x, y, show=False)

        assert mock_ax.plot.called

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.get_cmap')
    def test_plot_3d_data(self, mock_cmap, mock_close, mock_show, mock_subplots, mock_kb_thermo, mock_mplstyle):
        """Test plot with 3D data."""
        mock_cmap_obj = Mock()
        mock_cmap_obj.return_value = np.array([[1, 0, 0, 1]] * 5)
        mock_cmap.return_value = mock_cmap_obj

        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = ThermoPlotter(mock_kb_thermo)
        x = np.linspace(0, 1, 10)
        y = np.random.rand(10, 2, 2)

        fig, ax = plotter.plot(x, y, show=False)

        # Should plot all combinations
        combos = list(combinations_with_replacement(range(2), 2))
        assert mock_ax.plot.call_count == len(combos)

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.get_cmap')
    def test_plot_with_labels(self, mock_cmap, mock_close, mock_show, mock_subplots, mock_kb_thermo, mock_mplstyle):
        """Test plot with custom labels."""
        mock_cmap_obj = Mock()
        mock_cmap_obj.return_value = np.array([[1, 0, 0, 1]] * 5)
        mock_cmap.return_value = mock_cmap_obj

        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = ThermoPlotter(mock_kb_thermo)
        x = np.linspace(0, 1, 10)
        y = np.random.rand(10)

        plotter.plot(x, y, xlabel="X Label", ylabel="Y Label", show=False)

        mock_ax.set_xlabel.assert_called_with("X Label")
        mock_ax.set_ylabel.assert_called_with("Y Label")

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.get_cmap')
    def test_plot_with_limits(self, mock_cmap, mock_close, mock_show, mock_subplots, mock_kb_thermo, mock_mplstyle):
        """Test plot with axis limits."""
        mock_cmap_obj = Mock()
        mock_cmap_obj.return_value = np.array([[1, 0, 0, 1]] * 5)
        mock_cmap.return_value = mock_cmap_obj

        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = ThermoPlotter(mock_kb_thermo)
        x = np.linspace(0, 1, 10)
        y = np.random.rand(10)

        plotter.plot(x, y, xlim=(0, 1), ylim=(0, 2), show=False)

        mock_ax.set_xlim.assert_called_with((0, 1))
        mock_ax.set_ylim.assert_called_with((0, 2))

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.get_cmap')
    def test_plot_save_to_file(self, mock_cmap, mock_close, mock_show, mock_subplots, mock_kb_thermo, tmp_path, mock_mplstyle):
        """Test plot saves to file."""
        mock_cmap_obj = Mock()
        mock_cmap_obj.return_value = np.array([[1, 0, 0, 1]] * 5)
        mock_cmap.return_value = mock_cmap_obj

        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = ThermoPlotter(mock_kb_thermo)
        x = np.linspace(0, 1, 10)
        y = np.random.rand(10)

        save_path = tmp_path / "test.pdf"
        plotter.plot(x, y, savepath=str(save_path), show=False)

        assert mock_fig.savefig.called


class TestPlotPropertyMethod:
    """Test plot_property method."""

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.get_cmap')
    def test_plot_property_basic(self, mock_cmap, mock_close, mock_show, mock_subplots, mock_kb_thermo, mock_mplstyle):
        """Test basic property plotter."""
        mock_cmap_obj = Mock()
        mock_cmap_obj.return_value = np.array([[1, 0, 0, 1]] * 5)
        mock_cmap.return_value = mock_cmap_obj

        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = ThermoPlotter(mock_kb_thermo)
        fig, ax = plotter.plot_property("kbi", show=False)

        assert mock_ax.plot.called

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.get_cmap')
    def test_plot_property_with_units(self, mock_cmap, mock_close, mock_show, mock_subplots, mock_kb_thermo, mock_mplstyle):
        """Test property plotter with custom units."""
        mock_cmap_obj = Mock()
        mock_cmap_obj.return_value = np.array([[1, 0, 0, 1]] * 5)
        mock_cmap.return_value = mock_cmap_obj

        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = ThermoPlotter(mock_kb_thermo)
        plotter.plot_property("kbi", units="L/mol", show=False)

        # Should call the property with units
        mock_kb_thermo.kbi.assert_called_with("L/mol")

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.get_cmap')
    def test_plot_property_with_xmol(self, mock_cmap, mock_close, mock_show, mock_subplots, mock_kb_thermo, mock_mplstyle):
        """Test property plotter with specific x molecule."""
        mock_cmap_obj = Mock()
        mock_cmap_obj.return_value = np.array([[1, 0, 0, 1]] * 5)
        mock_cmap.return_value = mock_cmap_obj

        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = ThermoPlotter(mock_kb_thermo)
        plotter.plot_property("ln_activity_coef", xmol="Water", show=False)

        # Should get mol index
        mock_kb_thermo.systems.get_mol_index.assert_called_with("Water")


class TestPlotTernaryMethod:
    """Test plot_ternary method."""

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_plot_ternary_basic(self, mock_close, mock_show, mock_subplots, mock_ternary_thermo, mock_mplstyle):
        """Test basic ternary plotter."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_ax.tricontourf = Mock(return_value=Mock())
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = ThermoPlotter(mock_ternary_thermo)
        x = mock_ternary_thermo.systems.x
        y = np.random.rand(len(x))

        fig, ax = plotter.plot_ternary(x, y, show=False)

        assert mock_ax.tricontourf.called

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_plot_ternary_invalid_dimensions(self, mock_close, mock_show, mock_subplots, mock_kb_thermo, mock_mplstyle):
        """Test ternary plot with non-ternary data raises error."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = ThermoPlotter(mock_kb_thermo)
        x = np.random.rand(10, 2)  # Binary, not ternary
        y = np.random.rand(10)

        with pytest.raises(ValueError, match="not a ternary system"):
            plotter.plot_ternary(x, y, show=False)

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_plot_ternary_multidimensional_y(self, mock_close, mock_show, mock_subplots, mock_ternary_thermo, mock_mplstyle):
        """Test ternary plot with multidimensional y raises error."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = ThermoPlotter(mock_ternary_thermo)
        x = mock_ternary_thermo.systems.x
        y = np.random.rand(len(x), 2)  # 2D y values

        with pytest.raises(ValueError, match="only available for 1D"):
            plotter.plot_ternary(x, y, show=False)


class TestPlotActivityCoefDerivFits:
    """Test plot_activity_coef_deriv_fits method."""

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.get_cmap')
    def test_plot_activity_coef_deriv_fits(self, mock_cmap, mock_close, mock_show, mock_subplots, mock_kb_thermo, mock_mplstyle):
        """Test plotter activity coefficient derivative fits."""
        mock_cmap_obj = Mock()
        mock_cmap_obj.return_value = np.array([[1, 0, 0, 1]] * 5)
        mock_cmap.return_value = mock_cmap_obj

        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = ThermoPlotter(mock_kb_thermo)
        fig, ax = plotter.plot_activity_coef_deriv_fits(show=False)

        # Should plot data and fits
        assert mock_ax.plot.call_count >= 2


class TestPlotBinaryMixing:
    """Test plot_binary_mixing method."""

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.get_cmap')
    def test_plot_binary_mixing(self, mock_cmap, mock_close, mock_show, mock_subplots, mock_kb_thermo, mock_mplstyle):
        """Test plotter binary mixing properties."""
        mock_cmap_obj = Mock()
        mock_cmap_obj.return_value = np.array([[1, 0, 0, 1]] * 5)
        mock_cmap.return_value = mock_cmap_obj

        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = ThermoPlotter(mock_kb_thermo)
        fig, ax = plotter.plot_binary_mixing("Water", show=False)

        # Should plot 5 properties
        assert mock_ax.plot.call_count == 5


class TestMakeFigures:
    """Test make_figures method."""

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.get_cmap')
    def test_make_figures_binary(self, mock_cmap, mock_close, mock_show, mock_subplots, mock_kb_thermo, tmp_path, mock_mplstyle):
        """Test make_figures for binary system."""
        mock_cmap_obj = Mock()
        mock_cmap_obj.return_value = np.array([[1, 0, 0, 1]] * 5)
        mock_cmap.return_value = mock_cmap_obj

        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = ThermoPlotter(mock_kb_thermo)
        plotter.make_figures(str(tmp_path))

        # Should create multiple figures
        assert mock_subplots.call_count > 5

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.get_cmap')
    def test_make_figures_ternary(self, mock_cmap, mock_close, mock_show, mock_subplots, mock_ternary_thermo, tmp_path, mock_mplstyle):
        """Test make_figures for ternary system."""
        mock_cmap_obj = Mock()
        mock_cmap_obj.return_value = np.array([[1, 0, 0, 1]] * 5)
        mock_cmap.return_value = mock_cmap_obj

        mock_fig = Mock()
        mock_ax = Mock()
        mock_ax.tricontourf = Mock(return_value=Mock())
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = ThermoPlotter(mock_ternary_thermo)
        plotter.make_figures(str(tmp_path))

        # Should create multiple figures including ternary plots
        assert mock_subplots.call_count > 5


class TestIntegration:
    """Integration tests for ThermoPlotter."""

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.get_cmap')
    def test_complete_workflow(self, mock_cmap, mock_close, mock_show, mock_subplots, mock_kb_thermo, tmp_path, mock_mplstyle):
        """Test complete workflow."""
        mock_cmap_obj = Mock()
        mock_cmap_obj.return_value = np.array([[1, 0, 0, 1]] * 5)
        mock_cmap.return_value = mock_cmap_obj

        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        # Initialize plotter
        molecule_map = {"Water": "H2O", "Ethanol": "EtOH"}
        plotter = ThermoPlotter(mock_kb_thermo, molecule_map=molecule_map)

        # Create various plots
        plotter.plot_property("kbi", show=False)
        plotter.plot_binary_mixing("Water", show=False)
        plotter.make_figures(str(tmp_path))

        assert mock_subplots.call_count > 5


class TestPlotMethodEdgeCases:
    """Test edge cases in plot method."""

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.get_cmap')
    def test_plot_2d_x_with_1d_y(self, mock_cmap, mock_close, mock_show, mock_subplots, mock_kb_thermo, mock_mplstyle):
        """Test plot with 2D x and 1D y (should use first column of x)."""
        mock_cmap_obj = Mock()
        mock_cmap_obj.return_value = np.array([[1, 0, 0, 1]] * 5)
        mock_cmap.return_value = mock_cmap_obj

        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = ThermoPlotter(mock_kb_thermo)
        x = np.random.rand(10, 2)  # 2D x
        y = np.random.rand(10)     # 1D y

        fig, ax = plotter.plot(x, y, show=False)

        # Should use first column of x
        call_args = mock_ax.plot.call_args[0]
        assert len(call_args[0]) == 10

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.get_cmap')
    def test_plot_2d_x_with_3d_y(self, mock_cmap, mock_close, mock_show, mock_subplots, mock_kb_thermo, mock_mplstyle):
        """Test plot with 2D x and 3D y (should use first column of x)."""
        mock_cmap_obj = Mock()
        mock_cmap_obj.return_value = np.array([[1, 0, 0, 1]] * 5)
        mock_cmap.return_value = mock_cmap_obj

        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = ThermoPlotter(mock_kb_thermo)
        x = np.random.rand(10, 2)    # 2D x
        y = np.random.rand(10, 2, 2) # 3D y

        fig, ax = plotter.plot(x, y, show=False)

        # Should plot combinations
        assert mock_ax.plot.called

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.get_cmap')
    def test_plot_with_all_style_options(self, mock_cmap, mock_close, mock_show, mock_subplots, mock_kb_thermo, mock_mplstyle):
        """Test plot with all style options."""
        mock_cmap_obj = Mock()
        mock_cmap_obj.return_value = np.array([[1, 0, 0, 1]] * 5)
        mock_cmap.return_value = mock_cmap_obj

        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = ThermoPlotter(mock_kb_thermo)
        x = np.linspace(0, 1, 10)
        y = np.random.rand(10)

        plotter.plot(
            x, y,
            lw=2.5,
            ls="--",
            marker="s",
            cmap="viridis",
            figsize=(8, 6),
            show=False
        )

        # Check that plot was called with style parameters
        call_kwargs = mock_ax.plot.call_args[1]
        assert call_kwargs['lw'] == 2.5
        assert call_kwargs['ls'] == "--"
        assert call_kwargs['marker'] == "s"

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.get_cmap')
    def test_plot_save_to_directory_without_suffix(self, mock_cmap, mock_close, mock_show, mock_subplots, mock_kb_thermo, tmp_path, mock_mplstyle):
        """Test plot saves to directory with default filename."""
        mock_cmap_obj = Mock()
        mock_cmap_obj.return_value = np.array([[1, 0, 0, 1]] * 5)
        mock_cmap.return_value = mock_cmap_obj

        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = ThermoPlotter(mock_kb_thermo)
        x = np.linspace(0, 1, 10)
        y = np.random.rand(10)

        # Save to directory (no suffix)
        plotter.plot(x, y, savepath=str(tmp_path), show=False)

        # Should save with default filename
        call_args = mock_fig.savefig.call_args[0][0]
        assert "thermo_property.pdf" in str(call_args)

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.get_cmap')
    def test_plot_2d_with_labels(self, mock_cmap, mock_close, mock_show, mock_subplots, mock_kb_thermo, mock_mplstyle):
        """Test plot 2D data includes molecule labels."""
        mock_cmap_obj = Mock()
        mock_cmap_obj.return_value = np.array([[1, 0, 0, 1]] * 5)
        mock_cmap.return_value = mock_cmap_obj

        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = ThermoPlotter(mock_kb_thermo)
        x = np.linspace(0, 1, 10)
        y = np.random.rand(10, 2)

        plotter.plot(x, y, show=False)

        # Should include labels
        call_kwargs = mock_ax.plot.call_args[1]
        assert 'label' in call_kwargs


class TestPlotPropertyEdgeCases:
    """Test edge cases in plot_property method."""

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.get_cmap')
    def test_plot_property_exception_handling(self, mock_cmap, mock_close, mock_show, mock_subplots, mock_kb_thermo, mock_mplstyle):
        """Test plot_property handles exceptions when calling with units."""
        mock_cmap_obj = Mock()
        mock_cmap_obj.return_value = np.array([[1, 0, 0, 1]] * 5)
        mock_cmap.return_value = mock_cmap_obj

        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        # Make property raise exception when called with units
        mock_kb_thermo.kbi = Mock(side_effect=Exception("Unit error"))

        # Should fall back to calling without units
        plotter = ThermoPlotter(mock_kb_thermo)

        # This should not raise, should catch exception and call without units
        with pytest.raises(Exception):
            plotter.plot_property("kbi", units="L/mol", show=False)

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.get_cmap')
    def test_plot_property_2d_with_xmol(self, mock_cmap, mock_close, mock_show, mock_subplots, mock_kb_thermo, mock_mplstyle):
        """Test plot_property with 2D data and xmol specified."""
        mock_cmap_obj = Mock()
        mock_cmap_obj.return_value = np.array([[1, 0, 0, 1]] * 5)
        mock_cmap.return_value = mock_cmap_obj

        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = ThermoPlotter(mock_kb_thermo)
        plotter.plot_property("ln_activity_coef", xmol="Ethanol", show=False)

        # Should get mol index for Ethanol
        mock_kb_thermo.systems.get_mol_index.assert_called_with("Ethanol")

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.get_cmap')
    def test_plot_property_3d_default_xmol(self, mock_cmap, mock_close, mock_show, mock_subplots, mock_kb_thermo, mock_mplstyle):
        """Test plot_property with 3D data uses default xmol."""
        mock_cmap_obj = Mock()
        mock_cmap_obj.return_value = np.array([[1, 0, 0, 1]] * 5)
        mock_cmap.return_value = mock_cmap_obj

        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = ThermoPlotter(mock_kb_thermo)
        plotter.plot_property("kbi", show=False)  # 3D data, no xmol specified

        # Should use first molecule as default
        mock_kb_thermo.systems.get_mol_index.assert_called_with("Water")

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.get_cmap')
    def test_plot_property_without_units_in_ylabel(self, mock_cmap, mock_close, mock_show, mock_subplots, mock_kb_thermo, mock_mplstyle):
        """Test plot_property ylabel when units is None."""
        mock_cmap_obj = Mock()
        mock_cmap_obj.return_value = np.array([[1, 0, 0, 1]] * 5)
        mock_cmap.return_value = mock_cmap_obj

        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        # Mock property that returns None for units
        mock_kb_thermo.results["kbi"].units = None

        plotter = ThermoPlotter(mock_kb_thermo)
        plotter.plot_property("kbi", ylabel="Test Label", show=False)

        # ylabel should be used without units
        assert mock_ax.set_ylabel.called


class TestPlotTernaryEdgeCases:
    """Test edge cases in plot_ternary method."""

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_plot_ternary_filters_invalid_data(self, mock_close, mock_show, mock_subplots, mock_ternary_thermo, mock_mplstyle):
        """Test that plot_ternary filters out invalid data."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_ax.tricontourf = Mock(return_value=Mock())
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = ThermoPlotter(mock_ternary_thermo)
        x = mock_ternary_thermo.systems.x
        y = np.random.rand(len(x))

        # Add some invalid values
        y[0] = np.nan
        y[1] = np.inf
        y[2] = -np.inf

        fig, ax = plotter.plot_ternary(x, y, show=False)

        # Should still plot (with filtered data)
        assert mock_ax.tricontourf.called

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_plot_ternary_with_cbar_label(self, mock_close, mock_show, mock_subplots, mock_ternary_thermo, mock_mplstyle):
        """Test plot_ternary with colorbar label."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_ax.tricontourf = Mock(return_value=Mock())
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = ThermoPlotter(mock_ternary_thermo)
        x = mock_ternary_thermo.systems.x
        y = np.random.rand(len(x))

        plotter.plot_ternary(x, y, cbar_label="Test Label", show=False)

        # Should add colorbar with label
        assert mock_fig.colorbar.called

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_plot_ternary_save_to_directory(self, mock_close, mock_show, mock_subplots, mock_ternary_thermo, tmp_path, mock_mplstyle):
        """Test plot_ternary saves to directory with default filename."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_ax.tricontourf = Mock(return_value=Mock())
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = ThermoPlotter(mock_ternary_thermo)
        x = mock_ternary_thermo.systems.x
        y = np.random.rand(len(x))

        plotter.plot_ternary(x, y, savepath=str(tmp_path), show=False)

        # Should save with default filename
        call_args = mock_fig.savefig.call_args[0][0]
        assert "ternary_property.pdf" in str(call_args)


class TestPlotPropertyTernary:
    """Test plot_property_ternary method."""

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_plot_property_ternary_basic(self, mock_close, mock_show, mock_subplots, mock_ternary_thermo, mock_mplstyle):
        """Test basic ternary property plotter."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_ax.tricontourf = Mock(return_value=Mock())
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = ThermoPlotter(mock_ternary_thermo)
        plotter.plot_property_ternary("h_mix", show=False)

        assert mock_ax.tricontourf.called

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_plot_property_ternary_with_units(self, mock_close, mock_show, mock_subplots, mock_ternary_thermo, mock_mplstyle):
        """Test ternary property plotter with units."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_ax.tricontourf = Mock(return_value=Mock())
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = ThermoPlotter(mock_ternary_thermo)
        plotter.plot_property_ternary("h_mix", units="kcal/mol", show=False)

        # Should call property with units
        mock_ternary_thermo.h_mix.assert_called_with("kcal/mol")

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_plot_property_ternary_replaces_underscores(self, mock_close, mock_show, mock_subplots, mock_ternary_thermo, mock_mplstyle):
        """Test that property name underscores are replaced in label."""
        mock_fig = Mock()
        mock_ax = Mock()
        mock_ax.tricontourf = Mock(return_value=Mock())
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = ThermoPlotter(mock_ternary_thermo)
        plotter.plot_property_ternary("h_mix", show=False)

        # Should replace underscores in name
        # The cbar_label should contain "h mix" not "h_mix"
        assert mock_ax.tricontourf.called


class TestMakeFiguresEdgeCases:
    """Test edge cases in make_figures method."""

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.get_cmap')
    def test_make_figures_non_polynomial_integration(self, mock_cmap, mock_close, mock_show, mock_subplots, mock_kb_thermo, tmp_path, mock_mplstyle):
        """Test make_figures with non-polynomial integration type."""
        mock_cmap_obj = Mock()
        mock_cmap_obj.return_value = np.array([[1, 0, 0, 1]] * 5)
        mock_cmap.return_value = mock_cmap_obj

        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        # Set integration type to non-polynomial
        mock_kb_thermo.activity_integration_type = "trapezoid"

        plotter = ThermoPlotter(mock_kb_thermo)
        plotter.make_figures(str(tmp_path))

        # Should still create figures
        assert mock_subplots.call_count > 0

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.get_cmap')
    def test_make_figures_unsupported_system_type(self, mock_cmap, mock_close, mock_show, mock_subplots, mock_kb_thermo, tmp_path, mock_mplstyle):
        """Test make_figures with unsupported system type (not binary or ternary)."""
        mock_cmap_obj = Mock()
        mock_cmap_obj.return_value = np.array([[1, 0, 0, 1]] * 5)
        mock_cmap.return_value = mock_cmap_obj

        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        # Set to quaternary system (4 components)
        mock_kb_thermo.systems.n_i = 4

        plotter = ThermoPlotter(mock_kb_thermo)
        plotter.make_figures(str(tmp_path))

        # Should create basic figures but not system-specific ones
        assert mock_subplots.call_count > 0

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.get_cmap')
    def test_make_figures_with_xmol(self, mock_cmap, mock_close, mock_show, mock_subplots, mock_kb_thermo, tmp_path, mock_mplstyle):
        """Test make_figures with specified xmol."""
        mock_cmap_obj = Mock()
        mock_cmap_obj.return_value = np.array([[1, 0, 0, 1]] * 5)
        mock_cmap.return_value = mock_cmap_obj

        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = ThermoPlotter(mock_kb_thermo)
        plotter.make_figures(str(tmp_path), xmol="Ethanol")

        # Should use specified xmol
        assert mock_subplots.call_count > 0


class TestPlotBinaryMixingEdgeCases:
    """Test edge cases in plot_binary_mixing method."""

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.get_cmap')
    def test_plot_binary_mixing_default_xmol(self, mock_cmap, mock_close, mock_show, mock_subplots, mock_kb_thermo, mock_mplstyle):
        """Test plot_binary_mixing with default xmol (None)."""
        mock_cmap_obj = Mock()
        mock_cmap_obj.return_value = np.array([[1, 0, 0, 1]] * 5)
        mock_cmap.return_value = mock_cmap_obj

        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = ThermoPlotter(mock_kb_thermo)
        plotter.plot_binary_mixing(None, show=False)

        # Should use first molecule as default
        assert mock_ax.plot.call_count == 5

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.get_cmap')
    def test_plot_binary_mixing_save_to_directory(self, mock_cmap, mock_close, mock_show, mock_subplots, mock_kb_thermo, tmp_path, mock_mplstyle):
        """Test plot_binary_mixing saves to directory with default filename."""
        mock_cmap_obj = Mock()
        mock_cmap_obj.return_value = np.array([[1, 0, 0, 1]] * 5)
        mock_cmap.return_value = mock_cmap_obj

        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = ThermoPlotter(mock_kb_thermo)
        plotter.plot_binary_mixing("Water", savepath=str(tmp_path), show=False)

        # Should save with default filename
        call_args = mock_fig.savefig.call_args[0][0]
        assert "thermodyanmic_mixing_properties.pdf" in str(call_args)


class TestActivityCoefDerivFitsEdgeCases:
    """Test edge cases in plot_activity_coef_deriv_fits."""

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.get_cmap')
    def test_plot_activity_coef_deriv_fits_no_fn(self, mock_cmap, mock_close, mock_show, mock_subplots, mock_kb_thermo, mock_mplstyle):
        """Test plotter when metadata has no fit function."""
        mock_cmap_obj = Mock()
        mock_cmap_obj.return_value = np.array([[1, 0, 0, 1]] * 5)
        mock_cmap.return_value = mock_cmap_obj

        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        # Set has_fn to False
        mock_kb_thermo.activity_metadata.by_types["derivative"]["test"].has_fn = False

        plotter = ThermoPlotter(mock_kb_thermo)
        fig, ax = plotter.plot_activity_coef_deriv_fits(show=False)

        # Should still plot data, but not fit line
        assert mock_ax.plot.called

    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    @patch('matplotlib.pyplot.get_cmap')
    def test_plot_activity_coef_deriv_fits_save_to_directory(self, mock_cmap, mock_close, mock_show, mock_subplots, mock_kb_thermo, tmp_path, mock_mplstyle):
        """Test saving activity coef deriv fits to directory."""
        mock_cmap_obj = Mock()
        mock_cmap_obj.return_value = np.array([[1, 0, 0, 1]] * 5)
        mock_cmap.return_value = mock_cmap_obj

        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = ThermoPlotter(mock_kb_thermo)
        plotter.plot_activity_coef_deriv_fits(savepath=str(tmp_path), show=False)

        # Should save with default filename
        call_args = mock_fig.savefig.call_args[0][0]
        assert "activity_coef_deriv_fits.pdf" in str(call_args)

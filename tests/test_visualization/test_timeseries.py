"""Unit tests for TimeseriesPlotter class."""
import warnings

# Suppress NumPy/SciPy compatibility warning (harmless with NumPy 2.x + SciPy 1.16+)
warnings.filterwarnings('ignore', message='numpy.ndarray size changed', category=RuntimeWarning)

from unittest.mock import Mock, patch

import numpy as np
import pytest

from kbkit.visualization.timeseries import TimeseriesPlotter


@pytest.fixture
def mock_system_properties():
    """Create a mock SystemProperties object."""
    props = Mock()
    props.start_time = 0

    # Mock energy property with both lowercase and capitalized keys
    mock_energy = Mock()
    mock_energy.units = {
        "potential": "kJ/mol",
        "Potential": "kJ/mol",
        "temperature": "K",
        "Temperature": "K",
        "pressure": "kPa",
        "Pressure": "kPa",
        "volume": "nm^3",
        "Volume": "nm^3",
        "test": "units",
        "Time": "ps"
    }
    props.energy = [mock_energy]

    # Mock get method
    def get_side_effect(name, units=None, avg=False, time_series=False):
        time = np.linspace(0, 10000, 100)  # ps
        name_lower = name.lower()

        if name_lower == "potential":
            values = -1000 + np.random.normal(0, 10, 100)
        elif name_lower == "temperature":
            values = 298 + np.random.normal(0, 2, 100)
        elif name_lower == "pressure":
            values = 100 + np.random.normal(0, 5, 100)
        else:
            values = np.random.normal(100, 10, 100)

        if time_series:
            return time, values
        elif avg:
            return values.mean()
        else:
            return values

    props.get = Mock(side_effect=get_side_effect)

    return props


@pytest.fixture
def mock_system_collection(mock_system_properties):
    """Create a mock SystemCollection object."""
    collection = Mock()

    # Mock system metadata
    mock_system = Mock()
    mock_system.props = mock_system_properties

    # Make collection subscriptable
    collection.__getitem__ = Mock(return_value=mock_system)

    return collection


@pytest.fixture
def mock_mplstyle():
    """Mock the mplstyle loading."""
    with patch('kbkit.visualization.timeseries.load_mplstyle'):
        yield


class TestTimeseriesPlotterInitialization:
    """Test TimeseriesPlotter initialization."""

    def test_basic_initialization(self, mock_system_properties, mock_mplstyle):
        """Test basic initialization with SystemProperties."""
        plotter = TimeseriesPlotter(mock_system_properties)

        assert plotter.props is not None
        assert plotter.props.start_time == 0

    def test_initialization_with_start_time(self, mock_system_properties, mock_mplstyle):
        """Test initialization with custom start_time."""
        plotter = TimeseriesPlotter(mock_system_properties, start_time=5000)

        assert plotter.props.start_time == 5000

    def test_initialization_copies_props(self, mock_system_properties, mock_mplstyle):
        """Test that initialization creates a copy of props."""
        original_start_time = mock_system_properties.start_time

        plotter = TimeseriesPlotter(mock_system_properties, start_time=5000)

        # Original should not be modified
        assert mock_system_properties.start_time == original_start_time
        # Plotter's copy should be modified
        assert plotter.props.start_time == 5000

    def test_initialization_preserves_props_attributes(self, mock_system_properties, mock_mplstyle):
        """Test that initialization preserves props attributes."""
        plotter = TimeseriesPlotter(mock_system_properties)

        assert hasattr(plotter.props, 'energy')
        assert hasattr(plotter.props, 'get')


class TestFromCollectionClassMethod:
    """Test from_collection class method."""

    def test_from_collection_with_string_name(self, mock_system_collection, mock_mplstyle):
        """Test from_collection with string system name."""
        plotter = TimeseriesPlotter.from_collection(
            mock_system_collection,
            "water",
            start_time=1000
        )

        assert isinstance(plotter, TimeseriesPlotter)
        assert plotter.props.start_time == 1000

    def test_from_collection_with_int_index(self, mock_system_collection, mock_mplstyle):
        """Test from_collection with integer system index."""
        plotter = TimeseriesPlotter.from_collection(
            mock_system_collection,
            0,
            start_time=2000
        )

        assert isinstance(plotter, TimeseriesPlotter)
        assert plotter.props.start_time == 2000

    def test_from_collection_default_start_time(self, mock_system_collection, mock_mplstyle):
        """Test from_collection with default start_time."""
        plotter = TimeseriesPlotter.from_collection(
            mock_system_collection,
            "water"
        )

        assert plotter.props.start_time == 0

    def test_from_collection_accesses_correct_system(self, mock_system_collection, mock_mplstyle):
        """Test that from_collection accesses the correct system."""
        plotter = TimeseriesPlotter.from_collection(
            mock_system_collection,
            "test_system"
        )

        # Verify the collection was accessed with correct key
        mock_system_collection.__getitem__.assert_called_with("test_system")


class TestPlotMethod:
    """Test plot method."""

    @patch('kbkit.visualization.timeseries.resolve_attr_key')
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_plot_basic(self, mock_close, mock_show, mock_subplots, mock_resolve, mock_system_properties, mock_mplstyle):
        """Test basic plot creation."""
        mock_resolve.return_value = "potential"
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = TimeseriesPlotter(mock_system_properties)
        fig, ax = plotter.plot("potential", show=False)

        assert mock_ax.plot.called
        assert mock_ax.set_xlabel.called
        assert mock_ax.set_ylabel.called

    @patch('kbkit.visualization.timeseries.resolve_attr_key')
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_plot_resolves_property_name(self, mock_close, mock_show, mock_subplots, mock_resolve, mock_system_properties, mock_mplstyle):
        """Test that plot resolves property name aliases."""
        mock_resolve.return_value = "potential"
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = TimeseriesPlotter(mock_system_properties)
        plotter.plot("U", show=False)  # Alias for potential

        mock_resolve.assert_called_once()

    @patch('kbkit.visualization.timeseries.resolve_attr_key')
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_plot_with_custom_units(self, mock_close, mock_show, mock_subplots, mock_resolve, mock_system_properties, mock_mplstyle):
        """Test plot with custom units."""
        mock_resolve.return_value = "potential"
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = TimeseriesPlotter(mock_system_properties)
        plotter.plot("potential", units="kcal/mol", show=False)

        # Should call get with custom units
        assert mock_system_properties.get.called

    @patch('kbkit.visualization.timeseries.resolve_attr_key')
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_plot_with_running_average(self, mock_close, mock_show, mock_subplots, mock_resolve, mock_system_properties, mock_mplstyle):
        """Test plot with running average."""
        mock_resolve.return_value = "potential"
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = TimeseriesPlotter(mock_system_properties)
        plotter.plot("potential", show_avg=True, show=False)

        # Should plot twice: once for data, once for running average
        assert mock_ax.plot.call_count == 2
        assert mock_ax.legend.called

    @patch('kbkit.visualization.timeseries.resolve_attr_key')
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_plot_without_running_average(self, mock_close, mock_show, mock_subplots, mock_resolve, mock_system_properties, mock_mplstyle):
        """Test plot without running average."""
        mock_resolve.return_value = "potential"
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = TimeseriesPlotter(mock_system_properties)
        plotter.plot("potential", show_avg=False, show=False)

        # Should plot once: only for data
        assert mock_ax.plot.call_count == 1
        assert not mock_ax.legend.called

    @patch('kbkit.visualization.timeseries.resolve_attr_key')
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_plot_with_custom_figsize(self, mock_close, mock_show, mock_subplots, mock_resolve, mock_system_properties, mock_mplstyle):
        """Test plot with custom figure size."""
        mock_resolve.return_value = "potential"
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = TimeseriesPlotter(mock_system_properties)
        plotter.plot("potential", figsize=(12, 6), show=False)

        # Check that subplots was called with correct figsize
        mock_subplots.assert_called_with(figsize=(12, 6))

    @patch('kbkit.visualization.timeseries.resolve_attr_key')
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_plot_with_custom_labels(self, mock_close, mock_show, mock_subplots, mock_resolve, mock_system_properties, mock_mplstyle):
        """Test plot with custom labels."""
        mock_resolve.return_value = "potential"
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = TimeseriesPlotter(mock_system_properties)
        plotter.plot(
            "potential",
            xlabel="Custom X",
            ylabel="Custom Y",
            title="Custom Title",
            show=False
        )

        mock_ax.set_xlabel.assert_called_with("Custom X")
        mock_ax.set_ylabel.assert_called_with("Custom Y")
        mock_ax.set_title.assert_called_with("Custom Title")

    @patch('kbkit.visualization.timeseries.resolve_attr_key')
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_plot_with_axis_limits(self, mock_close, mock_show, mock_subplots, mock_resolve, mock_system_properties, mock_mplstyle):
        """Test plot with custom axis limits."""
        mock_resolve.return_value = "potential"
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = TimeseriesPlotter(mock_system_properties)
        plotter.plot(
            "potential",
            xlim=(0, 10),
            ylim=(-1100, -900),
            show=False
        )

        mock_ax.set_xlim.assert_called_with((0, 10))
        mock_ax.set_ylim.assert_called_with((-1100, -900))

    @patch('kbkit.visualization.timeseries.resolve_attr_key')
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_plot_with_custom_style(self, mock_close, mock_show, mock_subplots, mock_resolve, mock_system_properties, mock_mplstyle):
        """Test plot with custom line style."""
        mock_resolve.return_value = "potential"
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = TimeseriesPlotter(mock_system_properties)
        plotter.plot(
            "potential",
            color="red",
            show_avg=False,
            alpha=0.5,
            ls="--",
            lw=2,
            marker="o",
            show=False
        )

        # Check that plot was called with correct style parameters
        call_args = mock_ax.plot.call_args
        assert call_args[1]['c'] == "red"
        assert call_args[1]['alpha'] == 0.5
        assert call_args[1]['ls'] == "--"
        assert call_args[1]['lw'] == 2
        assert call_args[1]['marker'] == "o"

    @patch('kbkit.visualization.timeseries.resolve_attr_key')
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_plot_save_to_file(self, mock_close, mock_show, mock_subplots, mock_resolve, mock_system_properties, tmp_path, mock_mplstyle):
        """Test plot saves to file."""
        mock_resolve.return_value = "potential"
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        save_path = tmp_path / "test_plot.pdf"

        plotter = TimeseriesPlotter(mock_system_properties)
        plotter.plot("potential", savepath=str(save_path), show=False)

        # Check that savefig was called
        assert mock_fig.savefig.called

    @patch('kbkit.visualization.timeseries.resolve_attr_key')
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_plot_save_to_directory(self, mock_close, mock_show, mock_subplots, mock_resolve, mock_system_properties, tmp_path, mock_mplstyle):
        """Test plot saves to directory with default filename."""
        mock_resolve.return_value = "potential"
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = TimeseriesPlotter(mock_system_properties)
        plotter.plot("potential", savepath=str(tmp_path), show=False)

        # Should save with default filename
        assert mock_fig.savefig.called
        call_args = mock_fig.savefig.call_args[0][0]
        assert "energy.pdf" in str(call_args)

    @patch('kbkit.visualization.timeseries.resolve_attr_key')
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_plot_show_true(self, mock_close, mock_show, mock_subplots, mock_resolve, mock_system_properties, mock_mplstyle):
        """Test plot with show=True."""
        mock_resolve.return_value = "potential"
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = TimeseriesPlotter(mock_system_properties)
        plotter.plot("potential", show=True)

        assert mock_show.called
        assert not mock_close.called

    @patch('kbkit.visualization.timeseries.resolve_attr_key')
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_plot_show_false(self, mock_close, mock_show, mock_subplots, mock_resolve, mock_system_properties, mock_mplstyle):
        """Test plot with show=False."""
        mock_resolve.return_value = "potential"
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = TimeseriesPlotter(mock_system_properties)
        plotter.plot("potential", show=False)

        assert not mock_show.called
        assert mock_close.called

    @patch('kbkit.visualization.timeseries.resolve_attr_key')
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_plot_returns_fig_ax(self, mock_close, mock_show, mock_subplots, mock_resolve, mock_system_properties, mock_mplstyle):
        """Test that plot returns figure and axes."""
        mock_resolve.return_value = "potential"
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = TimeseriesPlotter(mock_system_properties)
        fig, ax = plotter.plot("potential", show=False)

        assert fig == mock_fig
        assert ax == mock_ax

    @patch('kbkit.visualization.timeseries.resolve_attr_key')
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_plot_time_conversion(self, mock_close, mock_show, mock_subplots, mock_resolve, mock_system_properties, mock_mplstyle):
        """Test that time is converted from ps to ns."""
        mock_resolve.return_value = "potential"
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = TimeseriesPlotter(mock_system_properties)
        plotter.plot("potential", show=False)

        # Check that plot was called with time/1000
        call_args = mock_ax.plot.call_args[0]
        time_values = call_args[0]
        # Time should be divided by 1000 (ps to ns)
        assert np.max(time_values) < 100  # Original was 10000 ps

    @patch('kbkit.visualization.timeseries.resolve_attr_key')
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    @patch('kbkit.visualization.timeseries.format_unit_str')
    def test_plot_formats_units(self, mock_format, mock_close, mock_show, mock_subplots, mock_resolve, mock_system_properties, mock_mplstyle):
        """Test that units are formatted in ylabel."""
        mock_resolve.return_value = "potential"
        mock_format.return_value = "kJ mol^{-1}"
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = TimeseriesPlotter(mock_system_properties)
        plotter.plot("potential", show=False)

        # format_unit_str should be called
        assert mock_format.called


class TestRunningAverage:
    """Test running average calculation and display."""

    @patch('kbkit.visualization.timeseries.resolve_attr_key')
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_running_average_calculation(self, mock_close, mock_show, mock_subplots, mock_resolve, mock_system_properties, mock_mplstyle):
        """Test running average is calculated correctly."""
        mock_resolve.return_value = "potential"
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = TimeseriesPlotter(mock_system_properties)
        plotter.plot("potential", show_avg=True, show=False)

        # Should have two plot calls
        assert mock_ax.plot.call_count == 2

        # Second call should be for running average
        second_call = mock_ax.plot.call_args_list[1]
        assert second_call[1]['c'] == 'k'
        assert second_call[1]['ls'] == '-'

    @patch('kbkit.visualization.timeseries.resolve_attr_key')
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_running_average_label_format_small(self, mock_close, mock_show, mock_subplots, mock_resolve, mock_mplstyle):
        """Test running average label format for small values."""
        mock_resolve.return_value = "test"

        props = Mock()
        props.start_time = 0
        mock_energy = Mock()
        mock_energy.units = {"test": "units"}
        props.energy = [mock_energy]

        # Small values (< 1)
        time = np.linspace(0, 10000, 100)
        values = np.ones(100) * 0.5
        props.get = Mock(return_value=(time, values))

        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = TimeseriesPlotter(props)
        plotter.plot("test", show_avg=True, show=False)

        # Check label format (should use .3f for small values)
        second_call = mock_ax.plot.call_args_list[1]
        label = second_call[1]['label']
        assert ".500" in label or "0.500" in label

    @patch('kbkit.visualization.timeseries.resolve_attr_key')
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_running_average_label_format_large(self, mock_close, mock_show, mock_subplots, mock_resolve, mock_mplstyle):
        """Test running average label format for large values."""
        mock_resolve.return_value = "test"

        props = Mock()
        props.start_time = 0
        mock_energy = Mock()
        mock_energy.units = {"test": "units"}
        props.energy = [mock_energy]

        # Large values (>= 1)
        time = np.linspace(0, 10000, 100)
        values = np.ones(100) * 100
        props.get = Mock(return_value=(time, values))

        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = TimeseriesPlotter(props)
        plotter.plot("test", show_avg=True, show=False)

        # Check label format (should use .0f for large values)
        second_call = mock_ax.plot.call_args_list[1]
        label = second_call[1]['label']
        assert "100" in label


class TestEdgeCases:
    """Test edge cases and error conditions."""

    @patch('kbkit.visualization.timeseries.resolve_attr_key')
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_plot_with_empty_data(self, mock_close, mock_show, mock_subplots, mock_resolve, mock_mplstyle):
        """Test plot with empty data arrays."""
        mock_resolve.return_value = "test"

        props = Mock()
        props.start_time = 0
        mock_energy = Mock()
        mock_energy.units = {"test": "units"}
        props.energy = [mock_energy]
        props.get = Mock(return_value=(np.array([]), np.array([])))

        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = TimeseriesPlotter(props)
        # Should handle gracefully
        plotter.plot("test", show=False)

    @patch('kbkit.visualization.timeseries.resolve_attr_key')
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_plot_with_single_point(self, mock_close, mock_show, mock_subplots, mock_resolve, mock_mplstyle):
        """Test plot with single data point."""
        mock_resolve.return_value = "test"

        props = Mock()
        props.start_time = 0
        mock_energy = Mock()
        mock_energy.units = {"test": "units"}
        props.energy = [mock_energy]
        props.get = Mock(return_value=(np.array([1000]), np.array([100])))

        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = TimeseriesPlotter(props)
        plotter.plot("test", show=False)

    @patch('kbkit.visualization.timeseries.resolve_attr_key')
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_plot_with_nan_values(self, mock_close, mock_show, mock_subplots, mock_resolve, mock_mplstyle):
        """Test plot with NaN values in data."""
        mock_resolve.return_value = "test"

        props = Mock()
        props.start_time = 0
        mock_energy = Mock()
        mock_energy.units = {"test": "units"}
        props.energy = [mock_energy]

        time = np.linspace(0, 10000, 100)
        values = np.ones(100) * 100
        values[50] = np.nan
        props.get = Mock(return_value=(time, values))

        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = TimeseriesPlotter(props)
        # Should handle NaN gracefully
        plotter.plot("test", show_avg=True, show=False)

    @patch('kbkit.visualization.timeseries.resolve_attr_key')
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_plot_with_inf_values(self, mock_close, mock_show, mock_subplots, mock_resolve, mock_mplstyle):
        """Test plot with infinite values in data."""
        mock_resolve.return_value = "test"

        props = Mock()
        props.start_time = 0
        mock_energy = Mock()
        mock_energy.units = {"test": "units"}
        props.energy = [mock_energy]

        time = np.linspace(0, 10000, 100)
        values = np.ones(100) * 100
        values[50] = np.inf
        props.get = Mock(return_value=(time, values))

        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = TimeseriesPlotter(props)
        # Should handle inf gracefully
        plotter.plot("test", show_avg=True, show=False)


class TestIntegration:
    """Integration tests for TimeseriesPlotter."""

    @patch('kbkit.visualization.timeseries.resolve_attr_key')
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_complete_workflow(self, mock_close, mock_show, mock_subplots, mock_resolve, mock_system_properties, mock_mplstyle):
        """Test complete workflow from initialization to plotting."""
        mock_resolve.return_value = "potential"
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        # Initialize plotter
        plotter = TimeseriesPlotter(mock_system_properties, start_time=1000)

        # Create plot
        fig, ax = plotter.plot(
            "potential",
            units="kJ/mol",
            show_avg=True,
            figsize=(10, 5),
            xlabel="Time (ns)",
            ylabel="Potential Energy",
            title="Energy vs Time",
            show=False
        )

        # Verify all components
        assert fig == mock_fig
        assert ax == mock_ax
        assert mock_ax.plot.called
        assert mock_ax.set_xlabel.called
        assert mock_ax.set_ylabel.called
        assert mock_ax.set_title.called

    @patch('kbkit.visualization.timeseries.resolve_attr_key')
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_from_collection_to_plot(self, mock_close, mock_show, mock_subplots, mock_resolve, mock_system_collection, mock_mplstyle):
        """Test workflow from collection to plot."""
        mock_resolve.return_value = "temperature"
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        # Create plotter from collection
        plotter = TimeseriesPlotter.from_collection(
            mock_system_collection,
            "water",
            start_time=2000
        )

        # Create plot
        fig, ax = plotter.plot("temperature", show=False)

        assert fig == mock_fig
        assert ax == mock_ax

    @patch('kbkit.visualization.timeseries.resolve_attr_key')
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_multiple_plots(self, mock_close, mock_show, mock_subplots, mock_resolve, mock_system_properties, mock_mplstyle):
        """Test creating multiple plots from same plotter."""
        mock_resolve.side_effect = ["potential", "temperature", "pressure"]
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = TimeseriesPlotter(mock_system_properties)

        # Create multiple plots
        plotter.plot("potential", show=False)
        plotter.plot("temperature", show=False)
        plotter.plot("pressure", show=False)

        # Should have created 3 plots
        assert mock_subplots.call_count == 3

    @patch('kbkit.visualization.timeseries.resolve_attr_key')
    @patch('matplotlib.pyplot.subplots')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_plot_with_all_options(self, mock_close, mock_show, mock_subplots, mock_resolve, mock_system_properties, tmp_path, mock_mplstyle):
        """Test plot with all available options."""
        mock_resolve.return_value = "potential"
        mock_fig = Mock()
        mock_ax = Mock()
        mock_subplots.return_value = (mock_fig, mock_ax)

        plotter = TimeseriesPlotter(mock_system_properties, start_time=1000)

        fig, ax = plotter.plot(
            "potential",
            units="kJ/mol",
            show_avg=True,
            figsize=(12, 6),
            xlabel="Custom X",
            ylabel="Custom Y",
            title="Custom Title",
            ylim=(-1100, -900),
            xlim=(0, 10),
            color="red",
            alpha=0.5,
            ls="--",
            lw=2,
            marker="o",
            savepath=str(tmp_path / "test.pdf"),
            show=False
        )

        # Verify all options were applied
        assert mock_subplots.called
        assert mock_ax.plot.called
        assert mock_ax.set_xlabel.called
        assert mock_ax.set_ylabel.called
        assert mock_ax.set_title.called
        assert mock_ax.set_xlim.called
        assert mock_ax.set_ylim.called
        assert mock_fig.savefig.called

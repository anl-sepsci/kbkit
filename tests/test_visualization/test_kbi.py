"""
Unit tests for the KBIAnalysisPlotter module.

This test suite provides comprehensive coverage of the KBIAnalysisPlotter class,
including initialization, plotting methods, and unit conversions.
"""
import warnings

# Suppress NumPy/SciPy compatibility warning
warnings.filterwarnings('ignore', message='numpy.ndarray size changed', category=RuntimeWarning)

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

from kbkit.schema.kbi_metadata import KBIMetadata
from kbkit.schema.property_result import PropertyResult
from kbkit.visualization.kbi import KBIAnalysisPlotter


@pytest.fixture
def mock_kbi_metadata():
    """Create mock KBIMetadata objects."""
    meta1 = Mock(spec=KBIMetadata)
    meta1.mols = ("MOL1", "MOL2")
    meta1.r = np.linspace(0, 3, 100)
    meta1.g = np.ones(100)
    meta1.rkbi = np.linspace(0, 1.5, 100)
    meta1.scaled_rkbi = np.linspace(0, 4.5, 100)
    meta1.r_fit = np.linspace(2, 3, 50)
    meta1.scaled_rkbi_fit = np.linspace(3, 4.5, 50)
    meta1.scaled_rkbi_est = np.linspace(3, 4.5, 50)
    meta1.kbi_limit = 1.5

    meta2 = Mock(spec=KBIMetadata)
    meta2.mols = ("MOL1", "MOL1")
    meta2.r = np.linspace(0, 3, 100)
    meta2.g = np.ones(100)
    meta2.rkbi = np.linspace(0, 1.2, 100)
    meta2.scaled_rkbi = np.linspace(0, 3.6, 100)
    meta2.r_fit = np.linspace(2, 3, 50)
    meta2.scaled_rkbi_fit = np.linspace(2.4, 3.6, 50)
    meta2.scaled_rkbi_est = np.linspace(2.4, 3.6, 50)
    meta2.kbi_limit = 1.2

    return {
        "system_1": {
            "MOL1.MOL2": meta1,
            "MOL1.MOL1": meta2
        },
        "system_2": {
            "MOL1.MOL2": meta1
        }
    }


@pytest.fixture
def mock_kbi_result(mock_kbi_metadata):
    """Create a mock KBI PropertyResult."""
    kbi_values = np.array([
        [[1.2, 1.5], [1.5, 1.2]],
        [[1.0, 1.3], [1.3, 1.0]]
    ])

    result = PropertyResult(
        name="kbi",
        value=kbi_values,
        units="nm^3/molecule",
        metadata=mock_kbi_metadata
    )

    # Mock the to() method to return a PropertyResult with metadata preserved
    def mock_to(units=None):
        new_result = PropertyResult(
            name="kbi",
            value=kbi_values,
            units=units or "nm^3/molecule",
            metadata=mock_kbi_metadata
        )
        return new_result

    result.to = mock_to
    return result


class TestKBIAnalysisPlotterInitialization:
    """Test KBIAnalysisPlotter initialization."""

    def test_init_with_minimal_parameters(self, mock_kbi_result):
        """Test initialization with minimal parameters."""
        plotter = KBIAnalysisPlotter(mock_kbi_result)

        assert plotter.result == mock_kbi_result
        assert plotter.metadata == mock_kbi_result.metadata
        assert plotter.molecule_map is None
        assert plotter.ureg is not None
        assert plotter.Q_ is not None

    def test_init_with_molecule_map(self, mock_kbi_result):
        """Test initialization with molecule_map."""
        molecule_map = {"MOL1": "Molecule 1", "MOL2": "Molecule 2"}
        plotter = KBIAnalysisPlotter(mock_kbi_result, molecule_map=molecule_map)

        assert plotter.molecule_map == molecule_map

    def test_init_stores_metadata(self, mock_kbi_result):
        """Test that initialization stores metadata."""
        plotter = KBIAnalysisPlotter(mock_kbi_result)

        assert "system_1" in plotter.metadata
        assert "system_2" in plotter.metadata


class TestKBIAnalysisPlotterPlot:
    """Test the plot method."""

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_plot_creates_figure(self, mock_close, mock_show, mock_savefig, mock_kbi_result):
        """Test that plot creates a figure."""
        plotter = KBIAnalysisPlotter(mock_kbi_result)

        plotter.plot("system_1", show=False)

        # Should create 1x3 subplot
        fig = plt.gcf()
        assert len(fig.axes) == 3

        mock_close.assert_called_once()
        mock_show.assert_not_called()

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_plot_shows_figure_when_requested(self, mock_show, mock_savefig, mock_kbi_result):
        """Test that plot shows figure when show=True."""
        plotter = KBIAnalysisPlotter(mock_kbi_result)

        plotter.plot("system_1", show=True)

        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_plot_saves_figure_when_savepath_provided(self, mock_close, mock_show, mock_savefig,
                                                       mock_kbi_result, tmp_path):
        """Test that plot saves figure when savepath is provided."""
        plotter = KBIAnalysisPlotter(mock_kbi_result)

        savepath = tmp_path / "test_plot.pdf"
        plotter.plot("system_1", savepath=str(savepath), show=False)

        mock_savefig.assert_called_once()
        call_args = mock_savefig.call_args
        assert str(savepath) in str(call_args[0][0])

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_plot_with_custom_units(self, mock_close, mock_show, mock_savefig, mock_kbi_result):
        """Test plot with custom units."""
        plotter = KBIAnalysisPlotter(mock_kbi_result)

        plotter.plot("system_1", units="cm^3/mol", show=False)

        # Should call PropertyResult.to() for unit conversion
        assert mock_close.called

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_plot_with_molecule_map(self, mock_close, mock_show, mock_savefig, mock_kbi_result):
        """Test plot with molecule_map."""
        molecule_map = {"MOL1": "Molecule 1", "MOL2": "Molecule 2"}
        plotter = KBIAnalysisPlotter(mock_kbi_result, molecule_map=molecule_map)

        plotter.plot("system_1", show=False)

        # Check that legend was created (would use mapped names)
        fig = plt.gcf()
        ax = fig.axes[0]
        legend = ax.get_legend()
        assert legend is not None

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_plot_without_molecule_map_uses_original_names(self, mock_close, mock_show,
                                                            mock_savefig, mock_kbi_result):
        """Test plot without molecule_map uses original molecule names."""
        plotter = KBIAnalysisPlotter(mock_kbi_result)

        plotter.plot("system_1", show=False)

        # Should still create legend with original names
        fig = plt.gcf()
        ax = fig.axes[0]
        legend = ax.get_legend()
        assert legend is not None

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_plot_sets_axis_labels(self, mock_close, mock_show, mock_savefig, mock_kbi_result):
        """Test that plot sets correct axis labels."""
        plotter = KBIAnalysisPlotter(mock_kbi_result)

        plotter.plot("system_1", show=False)

        fig = plt.gcf()
        axes = fig.axes

        # Check x-labels
        assert "r" in axes[0].get_xlabel().lower()
        assert "r" in axes[1].get_xlabel().lower()
        assert "r" in axes[2].get_xlabel().lower()

        # Check y-labels
        assert "g(r)" in axes[0].get_ylabel().lower()

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_plot_plots_all_pairs(self, mock_close, mock_show, mock_savefig, mock_kbi_result):
        """Test that plot plots all molecular pairs."""
        plotter = KBIAnalysisPlotter(mock_kbi_result)

        plotter.plot("system_1", show=False)

        fig = plt.gcf()
        # system_1 has 2 pairs (MOL1.MOL2 and MOL1.MOL1)
        # Each subplot should have 2 lines
        for ax in fig.axes:
            assert len(ax.lines) >= 2

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_plot_includes_extrapolation_line(self, mock_close, mock_show, mock_savefig, mock_kbi_result):
        """Test that plot includes extrapolation line in third subplot."""
        plotter = KBIAnalysisPlotter(mock_kbi_result)

        plotter.plot("system_1", show=False)

        fig = plt.gcf()
        ax2 = fig.axes[2]  # Third subplot

        # Should have dashed black line for extrapolation
        dashed_lines = [line for line in ax2.lines if line.get_linestyle() == '--']
        assert len(dashed_lines) > 0


class TestKBIAnalysisPlotterPlotAll:
    """Test the plot_all method."""

    @patch.object(KBIAnalysisPlotter, 'plot')
    def test_plot_all_calls_plot_for_each_system(self, mock_plot, mock_kbi_result, tmp_path):
        """Test that plot_all calls plot for each system."""
        plotter = KBIAnalysisPlotter(mock_kbi_result)

        plotter.plot_all(savepath=str(tmp_path), show=False)

        # Should call plot for each system in metadata
        assert mock_plot.call_count == 2  # system_1 and system_2

        # Check that it was called with correct system names
        call_args_list = [call[1]['system_name'] for call in mock_plot.call_args_list]
        assert "system_1" in call_args_list
        assert "system_2" in call_args_list

    @patch.object(KBIAnalysisPlotter, 'plot')
    def test_plot_all_passes_units(self, mock_plot, mock_kbi_result, tmp_path):
        """Test that plot_all passes units to plot."""
        plotter = KBIAnalysisPlotter(mock_kbi_result)

        plotter.plot_all(units="cm^3/mol", savepath=str(tmp_path), show=False)

        # Check that units were passed
        for call_obj in mock_plot.call_args_list:
            assert call_obj[1]['units'] == "cm^3/mol"

    @patch.object(KBIAnalysisPlotter, 'plot')
    def test_plot_all_passes_show(self, mock_plot, mock_kbi_result, tmp_path):
        """Test that plot_all passes show parameter to plot."""
        plotter = KBIAnalysisPlotter(mock_kbi_result)

        plotter.plot_all(savepath=str(tmp_path), show=True)

        # Check that show was passed
        for call_obj in mock_plot.call_args_list:
            assert call_obj[1]['show'] is True

    @patch.object(KBIAnalysisPlotter, 'plot')
    def test_plot_all_creates_savepaths(self, mock_plot, mock_kbi_result, tmp_path):
        """Test that plot_all creates correct savepaths."""
        plotter = KBIAnalysisPlotter(mock_kbi_result)

        plotter.plot_all(savepath=str(tmp_path), show=False)

        # Check that savepaths were created for each system
        for call_obj in mock_plot.call_args_list:
            savepath = call_obj[1]['savepath']
            assert savepath is not None
            assert ".pdf" in str(savepath)

    @patch.object(KBIAnalysisPlotter, 'plot')
    def test_plot_all_handles_directory_savepath(self, mock_plot, mock_kbi_result, tmp_path):
        """Test that plot_all handles directory as savepath."""
        plotter = KBIAnalysisPlotter(mock_kbi_result)

        plotter.plot_all(savepath=str(tmp_path), show=False)

        # Should use directory and create filenames
        for call_obj in mock_plot.call_args_list:
            savepath = Path(call_obj[1]['savepath'])
            assert savepath.parent == tmp_path

    @patch.object(KBIAnalysisPlotter, 'plot')
    def test_plot_all_handles_file_savepath(self, mock_plot, mock_kbi_result, tmp_path):
        """Test that plot_all handles file path as savepath."""
        plotter = KBIAnalysisPlotter(mock_kbi_result)

        file_path = tmp_path / "test.pdf"
        plotter.plot_all(savepath=str(file_path), show=False)

        # Should use parent directory
        for call_obj in mock_plot.call_args_list:
            savepath = Path(call_obj[1]['savepath'])
            assert savepath.parent == tmp_path


class TestKBIAnalysisPlotterIntegration:
    """Integration tests for KBIAnalysisPlotter."""

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_full_workflow_single_plot(self, mock_close, mock_show, mock_savefig,
                                       mock_kbi_result, tmp_path):
        """Test complete workflow for single plot."""
        molecule_map = {"MOL1": "Molecule 1", "MOL2": "Molecule 2"}
        plotter = KBIAnalysisPlotter(mock_kbi_result, molecule_map=molecule_map)

        savepath = tmp_path / "test_plot.pdf"
        plotter.plot(
            "system_1",
            units="cm^3/mol",
            savepath=str(savepath),
            show=False
        )

        # Verify all components
        assert mock_savefig.called
        assert mock_close.called
        assert not mock_show.called

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_full_workflow_plot_all(self, mock_close, mock_show, mock_savefig,
                                    mock_kbi_result, tmp_path):
        """Test complete workflow for plot_all."""
        plotter = KBIAnalysisPlotter(mock_kbi_result)

        plotter.plot_all(
            units="cm^3/mol",
            savepath=str(tmp_path),
            show=False
        )

        # Should create plots for all systems
        assert mock_savefig.call_count == 2  # system_1 and system_2
        assert mock_close.call_count == 2

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    def test_plot_with_show_true(self, mock_show, mock_savefig, mock_kbi_result):
        """Test plot with show=True displays figure."""
        plotter = KBIAnalysisPlotter(mock_kbi_result)

        plotter.plot("system_1", show=True)

        assert mock_show.called
        # Should not close when showing
        assert plt.get_fignums()  # Figure still exists

    def test_unit_conversion_in_plot(self, mock_kbi_result):
        """Test that unit conversion works in plot."""
        plotter = KBIAnalysisPlotter(mock_kbi_result)

        # Track calls to to()
        call_tracker = []
        original_to = mock_kbi_result.to

        def tracked_to(units=None):
            call_tracker.append(units)
            return original_to(units)

        mock_kbi_result.to = tracked_to

        with patch('matplotlib.pyplot.show'), patch('matplotlib.pyplot.close'):
            plotter.plot("system_1", units="cm^3/mol", show=False)

        # Should have called to() with cm^3/mol
        assert "cm^3/mol" in call_tracker


class TestKBIAnalysisPlotterEdgeCases:
    """Test edge cases and error conditions."""

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_plot_with_empty_metadata(self, mock_close, mock_show, mock_savefig):
        """Test plot with system that has no metadata."""
        # Create result with empty metadata for a system
        result = PropertyResult(
            name="kbi",
            value=np.array([[[1.0, 1.0], [1.0, 1.0]]]),
            units="nm^3/molecule",
            metadata={"system_1": {}}
        )

        # Mock the to() method to preserve metadata
        def mock_to(units=None):
            return PropertyResult(
                name="kbi",
                value=np.array([[[1.0, 1.0], [1.0, 1.0]]]),
                units=units or "nm^3/molecule",
                metadata={"system_1": {}}
            )
        result.to = mock_to

        plotter = KBIAnalysisPlotter(result)

        # Should not crash, just create empty plot
        plotter.plot("system_1", show=False)

        assert mock_close.called

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.close')
    def test_plot_with_single_pair(self, mock_close, mock_show, mock_savefig):
        """Test plot with system that has only one molecular pair."""
        meta = Mock(spec=KBIMetadata)
        meta.mols = ("MOL1", "MOL2")
        meta.r = np.linspace(0, 3, 100)
        meta.g = np.ones(100)
        meta.rkbi = np.linspace(0, 1.5, 100)
        meta.scaled_rkbi = np.linspace(0, 4.5, 100)
        meta.r_fit = np.linspace(2, 3, 50)
        meta.scaled_rkbi_fit = np.linspace(3, 4.5, 50)
        meta.scaled_rkbi_est = np.linspace(3, 4.5, 50)
        meta.kbi_limit = 4.2

        metadata = {"system_1": {"MOL1.MOL2": meta}}

        result = PropertyResult(
            name="kbi",
            value=np.array([[[1.5, 1.5], [1.5, 1.5]]]),
            units="nm^3/molecule",
            metadata=metadata
        )

        # Mock the to() method to preserve metadata
        def mock_to(units=None):
            return PropertyResult(
                name="kbi",
                value=np.array([[[1.5, 1.5], [1.5, 1.5]]]),
                units=units or "nm^3/molecule",
                metadata=metadata
            )
        result.to = mock_to

        plotter = KBIAnalysisPlotter(result)
        plotter.plot("system_1", show=False)

        assert mock_close.called

    @patch.object(KBIAnalysisPlotter, 'plot')
    def test_plot_all_with_no_systems(self, mock_plot, tmp_path):
        """Test plot_all with no systems in metadata."""
        result = PropertyResult(
            name="kbi",
            value=np.array([]),
            units="nm^3/molecule",
            metadata={}
        )

        # Mock the to() method
        def mock_to(units=None):
            return PropertyResult(
                name="kbi",
                value=np.array([]),
                units=units or "nm^3/molecule",
                metadata={}
            )
        result.to = mock_to

        plotter = KBIAnalysisPlotter(result)
        plotter.plot_all(savepath=str(tmp_path), show=False)

        # Should not call plot
        mock_plot.assert_not_called()

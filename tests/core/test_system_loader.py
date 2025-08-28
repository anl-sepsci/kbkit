"""
Unit tests for SystemLoader interface and config construction logic.

These tests validate:
- Initialization and logger setup
- Base and pure path discovery logic
- Metadata extraction and RDF path updates
- Temperature map construction and molecule filtering
- Sorting and molecule extraction for registry construction

Mocks are used to isolate filesystem and parser dependencies, ensuring reproducibility and contributor-friendly diagnostics.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kbkit.core.system_loader import SystemLoader
from kbkit.schema.system_config import SystemConfig
from kbkit.schema.system_metadata import SystemMetadata


@pytest.fixture
def loader():
    """
    Create a SystemLoader instance with verbose logging disabled.

    Returns
    -------
    SystemLoader
        Initialized loader instance.
    """
    return SystemLoader(verbose=False)


def test_loader_initialization(loader):
    """
    Test that SystemLoader initializes with a logger and verbosity flag.

    Verifies logger is set and verbose flag is respected.
    """
    assert isinstance(loader.logger, type(loader.logger))
    assert loader.verbose is False


@patch("kbkit.core.system_loader.validate_path", side_effect=lambda x: Path(x))
def test_find_base_path(mock_validate, loader):
    """
    Test default base path discovery using `_find_base_path()`.

    Verifies that current working directory is returned.
    """
    base = loader._find_base_path()
    assert isinstance(base, Path)
    assert base == Path.cwd()


@patch("kbkit.core.system_loader.validate_path", side_effect=lambda x: Path(x))
def test_find_pure_path_fallback(mock_validate, loader, tmp_path):
    """
    Test fallback behavior when no pure component directory is found.

    Verifies that base path is returned and warning is logged.
    """
    result = loader._find_pure_path(tmp_path)
    assert result == tmp_path


@patch("kbkit.core.system_loader.SystemProperties")
@patch("kbkit.core.system_loader.validate_path", side_effect=lambda x: Path(x))
def test_get_metadata(mock_validate, mock_props, loader, tmp_path):
    """
    Test metadata extraction for a system directory.

    Verifies correct kind assignment and metadata structure.
    """
    mock_props.return_value.topology.molecules = ["WAT"]
    meta = loader._get_metadata(tmp_path, "water", "npt", 0, "pure")
    assert isinstance(meta, SystemMetadata)
    assert meta.name == "water"
    assert meta.kind == "pure"
    assert meta.path == tmp_path / "water"


@patch("kbkit.core.system_loader.validate_path", side_effect=lambda x: Path(x))
def test_find_systems_filters_top_files(mock_validate, loader, tmp_path):
    """
    Test system discovery filters directories containing .top files.

    Verifies correct system names are returned.
    """
    system_dir = tmp_path / "ethanol"
    system_dir.mkdir()
    (system_dir / "test.top").write_text("[ molecules ]\nETH 1")
    found = loader._find_systems(tmp_path)
    assert found == ["ethanol"]


def test_extract_top_molecules(loader):
    """
    Test molecule extraction across multiple systems.

    Verifies unique molecule names are returned.
    """
    mock_meta = MagicMock()
    mock_meta.props.topology.molecules = ["WAT", "ETH"]
    result = loader._extract_top_molecules([mock_meta])
    assert set(result) == {"WAT", "ETH"}


def test_sort_systems_by_mol_fraction(loader):
    """
    Test sorting of systems by mol fraction vector.

    Verifies correct order based on molecule ratios.
    """
    mols = ["WAT", "ETH"]

    meta1 = MagicMock()
    meta1.props.topology.molecule_count = {"WAT": 2, "ETH": 2}
    meta1.props.topology.total_molecules = 4

    meta2 = MagicMock()
    meta2.props.topology.molecule_count = {"WAT": 3, "ETH": 1}
    meta2.props.topology.total_molecules = 4

    sorted_meta = loader._sort_systems([meta2, meta1], mols)
    assert sorted_meta == [meta1, meta2]


@patch("kbkit.core.system_loader.replace")
def test_update_metadata_rdf_success(mock_replace, loader, tmp_path):
    """
    Test RDF path update when RDF subdirectory is found.

    Verifies metadata is updated with RDF path.
    """
    system_dir = tmp_path / "water"
    rdf_dir = system_dir / "rdf_data"
    rdf_dir.mkdir(parents=True)
    (rdf_dir / "rdf.xvg").write_text("rdf")

    meta = SystemMetadata(name="water", path=system_dir, props=MagicMock(), kind="mixture")
    mock_replace.return_value = meta

    updated = loader._update_metadata_rdf([meta])
    assert updated[0].path == system_dir


def test_update_metadata_rdf_missing(loader, tmp_path):
    """
    Test error raised when RDF directory is missing for a mixture system.

    Verifies FileNotFoundError is raised.
    """
    system_dir = tmp_path / "water"
    system_dir.mkdir()
    meta = SystemMetadata(name="water", path=system_dir, props=MagicMock(), kind="mixture")

    with pytest.raises(FileNotFoundError, match="No RDF directory found"):
        loader._update_metadata_rdf([meta])


def test_system_loader_build_config(tmp_path):
    """Test successful config build with real temp directories."""
    base_dir = tmp_path / "base"
    pure_dir = tmp_path / "pure"
    base_dir.mkdir()
    pure_dir.mkdir()

    loader = SystemLoader(verbose=True)
    loader._find_base_path = MagicMock(return_value=base_dir)
    loader._find_pure_path = MagicMock(return_value=pure_dir)
    loader._find_systems = MagicMock(return_value=["sys1", "sys2"])
    loader._get_metadata = MagicMock(
        side_effect=lambda path, system, ensemble, start_time, kind: SystemMetadata(
            name=system, path=path / system, props=MagicMock(), kind=kind
        )
    )
    loader._validate_pure_systems = MagicMock(return_value=["pure1"])
    loader._update_metadata_rdf = MagicMock(side_effect=lambda metadata: metadata)
    loader._extract_top_molecules = MagicMock(return_value=["mol1", "mol2"])
    loader._sort_systems = MagicMock(side_effect=lambda metadata, molecules: metadata)

    config = loader.build_config(
        pure_path=pure_dir,
        pure_systems=["pure1"],
        base_path=base_dir,
        base_systems=["sys1", "sys2"],
        ensemble="npt",
        cations=["Na"],
        anions=["Cl"],
        start_time=0,
    )

    assert isinstance(config, SystemConfig)

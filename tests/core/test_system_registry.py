"""
Unit tests for SystemRegistry interface and semantic access patterns.

These tests validate:
- Retrieval of systems by name via `get()`
- Filtering by kind (e.g., "pure", "mixture") using `filter_by_kind()`
- Index lookup and error handling via `get_idx()`
- Iteration and length semantics for registry traversal
- Preservation of input order and metadata integrity

Mocked data includes:
- Three SystemMetadata objects with distinct names, kinds, and minimal props

Tests focus on reproducibility, discoverability, and contributor-friendly diagnostics.
"""

import pytest

from kbkit.core.system_registry import SystemRegistry
from kbkit.schema.system_metadata import SystemMetadata

# Constants used to avoid magic values in assertions
NUM_PURE_SYSTEMS = 2
NUM_TOTAL_SYSTEMS = 3
INDEX_WATER_ETHANOL = 2
DENSITY_ETHANOL = 0.789


@pytest.fixture
def sample_systems():
    """
    Create a list of mock SystemMetadata objects for testing.

    Returns
    -------
    list[SystemMetadata]
        Three systems with distinct names, kinds, and minimal props.
    """
    return [
        SystemMetadata(name="water", kind="pure", path="/mock/water", props={"density": 1.0}),
        SystemMetadata(name="ethanol", kind="pure", path="/mock/ethanol", props={"density": DENSITY_ETHANOL}),
        SystemMetadata(name="water_ethanol", kind="mixture", path="/mock/water_ethanol", props={"ratio": "50:50"}),
    ]


@pytest.fixture
def registry(sample_systems):
    """
    Initialize a SystemRegistry with sample systems.

    Returns
    -------
    SystemRegistry
        Registry containing three mock systems.
    """
    return SystemRegistry(systems=sample_systems)


def test_get_by_name(registry):
    """
    Test retrieval of a system by name using `get()`.

    Verifies correct object is returned and attributes match.
    """
    system = registry.get("ethanol")
    assert isinstance(system, SystemMetadata)
    assert system.name == "ethanol"
    assert system.kind == "pure"
    assert system.props["density"] == DENSITY_ETHANOL


def test_filter_by_kind(registry):
    """
    Test filtering systems by kind using `filter_by_kind()`.

    Verifies correct grouping and list contents.
    """
    pure_systems = registry.filter_by_kind("pure")
    mixture_systems = registry.filter_by_kind("mixture")

    assert len(pure_systems) == NUM_PURE_SYSTEMS
    assert all(s.kind == "pure" for s in pure_systems)

    assert len(mixture_systems) == 1
    assert mixture_systems[0].name == "water_ethanol"


def test_all_systems(registry):
    """
    Test retrieval of all systems using `all()`.

    Verifies full list is returned and matches input.
    """
    all_systems = registry.all()
    assert len(all_systems) == NUM_TOTAL_SYSTEMS
    assert all(isinstance(s, SystemMetadata) for s in all_systems)


def test_get_idx_valid(registry):
    """
    Test index lookup of a system by name using `get_idx()`.

    Verifies correct index is returned.
    """
    idx = registry.get_idx("water_ethanol")
    assert isinstance(idx, int)
    assert idx == INDEX_WATER_ETHANOL


def test_get_idx_invalid(registry):
    """
    Test error raised when `get_idx()` is called with unknown name.

    Verifies ValueError is raised with appropriate message.
    """
    with pytest.raises(ValueError, match="'nonexistent' is not in list"):
        registry.get_idx("nonexistent")


def test_iteration(registry):
    """
    Test iteration over registry using `__iter__()`.

    Verifies all systems are yielded in order.
    """
    names = [s.name for s in registry]
    assert names == ["water", "ethanol", "water_ethanol"]


def test_len(registry):
    """
    Test length of registry using `__len__()`.

    Verifies correct count is returned.
    """
    assert len(registry) == NUM_TOTAL_SYSTEMS

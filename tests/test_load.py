"""
Smoke tests for kbkit.

These are lightweight tests designed to quickly verify that the package
can be imported and that core functionality runs without error. They are
not intended to be exhaustive or cover edge cases, but to serve as a
sanity check in CI and local development environments.
"""

import kbkit


def test_import() -> None:
    """Ensure the kbkit package can be imported."""
    assert kbkit is not None


def test_version() -> None:
    """Check that the package defines a version string."""
    assert hasattr(kbkit, "__version__")
    assert isinstance(kbkit.__version__, str)

"""Version check."""

import kbkit


def test_import() -> None:
    """Check that kbkit can be imported and has a __version__ attribute."""
    assert hasattr(kbkit, "__version__")

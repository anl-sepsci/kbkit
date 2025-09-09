"""
Update the package version in _version.py and pyproject.toml.

This script centralizes version management for the kbkit package, ensuring
that the version number is consistent across all relevant files.
"""

import re
import sys as sys_arg
from pathlib import Path

VERSION = sys_arg.argv[1]

# Paths
BASE_DIR = Path(__file__).parent.parent
VERSION_FILE = BASE_DIR / "src" / "kbkit" / "_version.py"
PYPROJECT_FILE = BASE_DIR / "pyproject.toml"
CONDA_FILE = BASE_DIR / ".conda" / "meta.yaml"

# update _version.py
VERSION_FILE.write_text(f'__version__ = "{VERSION}"\n')

# Update pyproject.toml
content = PYPROJECT_FILE.read_text()
new_content = re.sub(r'^(version\s*=\s*)".*?"', rf'\1"{VERSION}"', content, flags=re.MULTILINE)
PYPROJECT_FILE.write_text(new_content)


# Update .conda/meta.yaml
conda_content = CONDA_FILE.read_text()


def replace_version_line(match):
    """Helper function to replace the version line in meta.yaml while preserving quote styles."""
    prefix = match.group(1)
    quote1 = match.group(2)
    quote2 = match.group(3)
    # Use the same quote style if present, else no quotes
    if quote1 or quote2:
        return f"{prefix}{quote1}{VERSION}{quote2}"
    else:
        return f"{prefix}{VERSION}"


new_conda_content = re.sub(
    r'^\s*(version\s*:\s*)(["\']?).*?([\'"]?)$', replace_version_line, conda_content, flags=re.MULTILINE
)
CONDA_FILE.write_text(new_conda_content)

print(f"Version updated successfully to {VERSION}")

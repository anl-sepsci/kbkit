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

# update _version.py
VERSION_FILE.write_text(f'__version__ = "{VERSION}"\n')

# Update pyproject.toml
content = PYPROJECT_FILE.read_text()

# Replace version line in [project] section
new_content = re.sub(r'^(version\s*=\s*)".*?"', rf'\1"{VERSION}"', content, flags=re.MULTILINE)

PYPROJECT_FILE.write_text(new_content)

print(f"Version updated successfully to {VERSION}")

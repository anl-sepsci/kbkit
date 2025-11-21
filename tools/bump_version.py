"""
Update the package version in _version.py and pyproject.toml.

This script centralizes version management for the kbkit package, ensuring
that the version number is consistent across all relevant files.
"""

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path

import requests

# --- CLI Arguments ---
parser = argparse.ArgumentParser(description="Bump version for kbkit package.")
parser.add_argument("version", type=str, help="New version number (e.g., 1.0.1)")
parser.add_argument("-p", "--publish", action="store_true", help="Update meta.yaml and publish to PyPI")
args = parser.parse_args()

VERSION = args.version
PUBLISH = args.publish

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
DIST_DIR = BASE_DIR / "dist"
VERSION_FILE = BASE_DIR / "src" / "kbkit" / "_version.py"
PYPROJECT_FILE = BASE_DIR / "pyproject.toml"
CONDA_META = BASE_DIR / ".conda" / "meta.yaml"

# --- Validate version format ---
if not re.match(r"^\d+\.\d+\.\d+([a-zA-Z0-9\.\-]*)?$", VERSION):
    print("Error: Version must be in format X.Y.Z (e.g., 1.0.1)", file=sys.stderr)
    sys.exit(1)

# --- Update _version.py ---
VERSION_FILE.write_text(f'__version__ = "{VERSION}"\n')
print(f"Updated _version.py => {VERSION}")

# --- Update pyproject.toml ---
pyproject_content = PYPROJECT_FILE.read_text()
pyproject_content = re.sub(r'^(version\s*=\s*)".*?"', rf'\1"{VERSION}"', pyproject_content, flags=re.MULTILINE)
PYPROJECT_FILE.write_text(pyproject_content)
print(f"Updated pyproject.toml => {VERSION}")


# --- Helper: Replace version line in meta.yaml ---
def replace_version_line(match):
    """Helper function to replace the version line in meta.yaml while preserving quote styles."""
    prefix, quote1, quote2 = match.group(1), match.group(2), match.group(3)
    quote = quote1 or quote2 or ""
    return f"{prefix}{quote}{VERSION}{quote}"


# --- Helper: Check if conda env exists ---
def conda_env_exists(env_name):
    """Check if a conda environment with the given name exists."""
    result = subprocess.run(["conda", "env", "list"], check=False, capture_output=True, text=True)
    return any(env_name in line.split() for line in result.stdout.splitlines())


# --- Publish to PyPI ---
if PUBLISH:
    if not conda_env_exists("kbkit-dev"):
        print("Conda environment 'kbkit-dev' not found.", file=sys.stderr)
        sys.exit(1)

    # Update meta.yaml version
    conda_content = CONDA_META.read_text()
    conda_content = re.sub(
        r'^\s*(version\s*:\s*)(["\']?).*?([\'"]?)$', replace_version_line, conda_content, flags=re.MULTILINE
    )

    # Build if .tar.gz doesn't exist
    dist_file = DIST_DIR / f"kbkit-{VERSION}.tar.gz"
    if not dist_file.exists():
        print("Building source distribution...")
        try:
            subprocess.run(
                f"conda run -n kbkit-dev python -m build --sdist --outdir {DIST_DIR}",
                shell=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Build failed with error: {e}")

    if not dist_file.exists():
        print(f"Distribution file not found: {dist_file}", file=sys.stderr)
        sys.exit(1)

    print(f"Uploading {dist_file.name} to PyPI...")
    subprocess.run([f"conda run -n kbkit-dev python -m twine upload {dist_file!s}"], check=True, shell=True)

    # Add a delay to allow PyPI to process the upload
    print("Waiting 10 seconds for PyPI to process the upload...")
    import time

    time.sleep(10)

    # Update URL and SHA256 in meta.yaml
    new_url = f"https://files.pythonhosted.org/packages/source/k/kbkit/kbkit-{VERSION}.tar.gz"
    conda_content = re.sub(
        r'^\s*(  url\s*:\s*)(["\']?).*?([\'\"]?)$', rf'\1"{new_url}"', conda_content, flags=re.MULTILINE
    )

    print("Fetching tarball from PyPI to compute SHA256...")
    try:
        response = requests.get(new_url, timeout=30)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching tarball: {e}", file=sys.stderr)
        sys.exit(1)

    digest = hashlib.sha256(response.content).hexdigest()
    conda_content = re.sub(
        r'^\s*(  sha256\s*:\s*)(["\']?).*?([\'\"]?)$', rf'\1"{digest}"', conda_content, flags=re.MULTILINE
    )

    CONDA_META.write_text(conda_content)
    print("Updated meta.yaml => version, URL, and SHA256")

    # --- Push to version updated files to main branch ---
    try:
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", f"'Updated version to v{VERSION}'"], check=True)
        subprocess.run(["git", "push", "origin", "main"], check=True)
        print(f"Pushed updates to version: v{VERSION}")
    except subprocess.CalledProcessError:
        print("Skipped GitHub push to main.")

    # --- Git Tag and Push ---
    tag_name = f"v{VERSION}"
    # list existing tags
    existing = subprocess.run(["git", "tag"], capture_output=True, text=True, check=True)
    if tag_name in existing.stdout.split():
        print(f"Tag {tag_name} already exists, skipping creation.")
    else:
        subprocess.run(["git", "tag", tag_name], check=True)
        subprocess.run(["git", "push", "origin", tag_name], check=True)
        print(f"Tagged and pushed release: {tag_name}")

print(f"Version bump complete => {VERSION}")

"""
Update the package version in _version.py and pyproject.toml.

This script centralizes version management for the kbkit package, ensuring
that the version number is consistent across all relevant files.
"""

import re
import os
import sys as sys_arg
from pathlib import Path
import subprocess
import argparse


def main():
    parser = argparse.ArgumentParser(description="Bump version for kbkit package.")
    parser.add_argument("version", type=str, help="New version number (e.g., 1.0.1)")
    parser.add_argument("-p", "--publish", action="store_true", help="Update .conda/meta.yaml and publish to PyPI")
    args = parser.parse_args()

    VERSION = args.version
    publish = args.publish

    # Paths
    BASE_DIR = Path(__file__).parent.parent
    DIST_PATH = BASE_DIR / "dist" / f"kbkit-{VERSION}.tar.gz"
    VERSION_FILE = BASE_DIR / "src" / "kbkit" / "_version.py"
    PYPROJECT_FILE = BASE_DIR / "pyproject.toml"
    CONDA_FILE = BASE_DIR / ".conda" / "meta.yaml"

    # Validate version format
    if not re.match(r"^\d+\.\d+\.\d+([a-z-z0-9\.\-]*)?$", VERSION):
        print("Error: Version must be in format X.Y.Z (e.g., 1.0.1)")
        sys_arg.exit(1)

    # update _version.py
    VERSION_FILE.write_text(f'__version__ = "{VERSION}"\n')

    # Update pyproject.toml
    content = PYPROJECT_FILE.read_text()
    
    new_version = re.sub(r'^(version\s*=\s*)".*?"', rf'\1"{VERSION}"', content, flags=re.MULTILINE)
    PYPROJECT_FILE.write_text(new_version)


    # Update .conda/meta.yaml
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

    if publish:
        conda_content = CONDA_FILE.read_text()

        # Update version
        conda_content = re.sub(
            r'^\s*(version\s*:\s*)(["\']?).*?([\'"]?)$', replace_version_line, conda_content, flags=re.MULTILINE
        )

        # if version doesn't have tar.gz, run python build
        tar_gz_files = list((BASE_DIR / "dist").glob("kbkit-*.tar.gz"))
        if not any(re.match(rf'kbkit-{re.escape(VERSION)}\.tar\.gz', file.name) for file in tar_gz_files):
            # build source distribution
            print("Building source distribution...")
            os.system(f"{sys_arg.executable} -m build --sdist --outdir {BASE_DIR / 'dist'}")
            # upload to pypi
            print("Uploading to PyPI...")
            subprocess.run(
                [sys_arg.executable, "-m", "twine", "upload", str(DIST_PATH)],
                check=True
            )


        # Update url
        new_url = f"https://files.pythonhosted.org/packages/source/k/kbkit/kbkit-{VERSION}.tar.gz"
        conda_content = re.sub(
            r'^\s*(  url\s*:\s*)(["\']?).*?([\'\"]?)$', rf'\1"{new_url}"', conda_content, flags=re.MULTILINE
        )

        # Update sha256
        cmd = f"curl -L {new_url} | shasum -a 256"
        output = os.popen(cmd).read()
        sha256 = output.split()[0]
        conda_content = re.sub(
            r'^\s*(  sha256\s*:\s*)(["\']?).*?([\'\"]?)$', rf'\1"{sha256}"', conda_content, flags=re.MULTILINE
        )

        CONDA_FILE.write_text(conda_content)
        
    print(f"Version updated successfully to {VERSION}")

if __name__ == "__main__":
    main()
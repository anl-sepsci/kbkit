"""Handles parsing and validation of RDF files."""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from kbkit.config.mplstyle import load_mplstyle
from kbkit.utils.validation import validate_path

load_mplstyle()  # load mpl figure configuration


class RdfParser:
    """
    Class to handle RDF (Radial Distribution Function) data.

    Reads RDF data from a file provides methods to plot the RDF and extract molecular information.

    Parameters
    ----------
    filepath : str
        Path to the RDF text file.

    Attributes
    ----------
    fname: str
        File name.
    r: np.ndarray
        Array of radial distances (nm).
    gr: np.ndarray
        Array of radial distribution function values.
    """

    def __init__(
        self,
        path: str | Path,
    ) -> None:
        self.filepath = self._validate_file(path)
        self.r, self.gr = self._read(self.filepath)

    @staticmethod
    def _validate_file(filepath: str | Path) -> Path:
        """Validate file and check that it can be read."""
        filepath = validate_path(filepath, suffix=Path(filepath).suffix)
        if not filepath.is_file():
            raise FileNotFoundError(f"RDF file '{filepath}' not found.")
        try:
            with open(filepath) as f:
                f.readline()
                f.close()
        except IOError as ioe:
            raise IOError(f"Error reading file '{filepath}': {ioe}.") from ioe
        except ValueError as ve:
            raise ValueError(f"Failed to parse RDF data from '{filepath}': {ve}.") from ve
        except Exception as e:
            raise RuntimeError(f"Unexpected error reading '{filepath}': {e}") from e
        return filepath

    def _read(self, filepath: Path) -> tuple[np.ndarray, np.ndarray]:
        """Read RDF file and extracts radial distances (r) and g(r) values.

        The file is expected to have two columns: r and g(r).
        It filters out noise from the tail of the RDF curve.
        """
        if filepath.suffix in (".xvg", ".txt"):
            r, gr = np.loadtxt(filepath, comments=["@", "#", ";"], unpack=True)
        elif filepath.suffix in (".csv", ".xlsx"):
            if filepath.suffix == ".csv":
                arr = pd.read_csv(filepath, delimiter=",").to_numpy(dtype=float)
            else:
                arr = pd.read_excel(filepath).to_numpy(dtype=float)
            r, gr = arr[:, 0], arr[:, 1]
        else:
            raise ValueError("Filetype not supported! Supported file types: ('.xvg','.txt','.csv','.xlsx')")

        return r[:-1], gr[:-1]

    @staticmethod
    def extract_molecules(text: str, mol_list: list[str]) -> list[str]:
        """
        Extract molecule names used in RDF from the RDF file name.

        Parameters
        ----------
        rdf_file : str
            Path to the RDF file.
        mol_list : list of str
            List of molecule names to search for in the file name.

        Returns
        -------
        list[str]
            List of molecule names found in the RDF file name.
        """
        if not isinstance(text, str):
            try:
                text = str(text)
            except TypeError as e:
                raise TypeError("Could not convert filename to type str.") from e

        if not mol_list:
            raise ValueError("Unable to match molecules to an empty list.")

        # define pattern for mol in mol_list
        pattern = r"(" + "|".join(re.escape(mol) for mol in mol_list) + r")"

        # find matches of pattern in filename
        matches = re.findall(pattern, text)
        return matches

    def plotRDF(self, axhandle, **kwargs) -> None:
        """Add rdf to plt.axes."""
        axhandle.plot(self.r, self.gr, **kwargs)

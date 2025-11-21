# KBKit: Kirkwood-Buff Analysis Toolkit

[![License](https://img.shields.io/github/license/aperoutka/kbkit)](https://github.com/aperoutka/kbkit/blob/master/LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/kbkit.svg)](https://pypi.org/project/kbkit/)
[![Powered by: Pixi](https://img.shields.io/badge/Powered_by-Pixi-facc15)](https://pixi.sh)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Coverage Status](https://coveralls.io/repos/github/aperoutka/kbkit/badge.svg?branch=main)](https://coveralls.io/github/aperoutka/kbkit?branch=main)
[![docs](http://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://kbkit.readthedocs.io/)
![python 3.12](https://img.shields.io/badge/Python-3.12%2B-blue)

**KBKit** is a Python package for automated Kirkwood-Buff (KB) analysis of molecular simulation data. It provides tools to parse simulation outputs, compute Kirkwood-Buff integrals, and extract thermodynamic properties for binary and multicomponent systems. **KBKit** supports flexible workflows, including:

* Parsing and processing of simulation data (e.g., RDFs, densities)
* Calculation of KB integrals and related thermodynamic quantities
* Integration of activity coefficient derivatives (numerical or polynomial)
* Automated pipelines for batch analysis
* Calculation of static structure factor and X-ray intensities in the limit of q &rarr; 0
* Visualization tools for KB integrals, thermodynamic properties, and static structure factors

**KBKit** is designed for researchers in computational chemistry, soft matter, and statistical mechanics who need robust, reproducible KB analysis from simulation data. The package is modular, extensible, and integrates easily with Jupyter notebooks and Python scripts.

## Installation

### Quick install via PyPI

```python
pip install kbkit
```

### Developer install (recommended for contributors or conda users)

Clone the GitHub repository and use the provided Makefile to set up your development environment:

```python
git clone https://github.com/aperoutka/kbkit.git
cd kbkit
make setup-dev
```

This one-liner creates the `kbkit-dev` conda environment, installs `kbkit` in editable mode, and runs the test suite.

To install without running tests:

```python
make dev-install
```

To build and install the package into a clean user environment:

```python
make setup-user
```

For a full list of available commands:

```python
make help
```

## File Organization

For running `kbkit.Pipeline` or its dependencies, the following file structure is required: a structured directory layout that separates mixed systems from pure components. 
This organization enables automated parsing, reproducible KB integrals, and scalable analysis across chemical systems.

* NOTE: **KBKit** currently only supports parsing for *GROMACS* files.

An example of file structure:
```python
kbi_dir/
├── project/
│   └── system/
│       ├── rdf_dir/
│       │   ├── mol1_mol1.xvg
│       │   ├── mol1_mol2.xvg
│       │   └── mol1_mol2.xvg
│       ├── system_npt.edr
│       ├── system_npt.gro
│       └── system.top
└── pure_components/
    └── molecule1/
        ├── molecule1_npt.edr
        └── molecule1.top
```

**Requirements:**

* Each system to be analyzed must include:
    * rdf_dir/ containing .xvg RDF files for all pairwise interactions
        * Both molecule IDs in RDF calculation *MUST BE* in filename
    * either .top topology file or .gro structure file (.gro is recommended)
    * .edr energy file
* Each pure component must include:
    * either .top topology file or .gro structure file (.gro is recommended)
    * .edr energy file
    * all other files (optional)

## Examples

Below are several examples on various ways to implement **KBKit**.
See examples for a more complete example on the ethanol/water binary system.

### Calculating Kirkwood-Buff integrals on a single RDF

```python
import os
from kbkit.analysis import KBIntegrator
from kbkit.systems import SystemProperties

syspath = "./examples/test_data/ethanol_water_26C/sys_405"
rdf_path = os.path.join(sys_path, "kbi_rdf_files_gmx25", "rdf_ETHOL_SPCEW.xvg")

# create integrator object from single RDF file
integrator = KBIntegrator(
    rdf_file=rdf_path,
    system_properties=SystemProperties(sys_path),
    use_fixed_rmin=False,
)

# calculate KBI in thermodynamic limit
kbi = integrator.kbi_limit(mol_j="SPCEW")
```

### Run an automated pipeline for batch analysis

```python
from kbkit import Pipeline

# Set up and run the pipeline
pipe = Pipeline(
    base_path="./examples/test_data/ethanol_water_26C", # directory with system data
    pure_path="./examples/test_data/pure_components",   # directory with pure component data
    pure_systems=["ETHOL_300", "SPCEW_300"],            # list of pure systems
    ensemble="npt",                                     # ensemble type: npt or nvt
    gamma_integration_type="numerical",                 # integration method
    verbose=False                                       # logging verbosity
)

# run kbkit pipeline
pipe.run()

# Access the results properties
# stored in dataclass (ThermoProperty); attributes: name, value, units
# example for excess energy
ge_obj = pipe.get("ge")
print("GE summary: ", ge_array.shape)

# Convert units from kJ/mol -> kcal/mol
# default units will be those from GROMACS
pipe.convert_units("ge", "kcal/mol")

# make figures for select thermodynamic properties
pipe.plot(
    molecule_map={"ETHOL": "ethanol", "SPCEW": "water"}, # dictionary mapping MD names to label names
    x_mol="ETHOL"                                        # MD name of molecule for x-axis
)
```

### Parse GROMACS files

```python
import os
from kbkit.parsers import TopFileParser, EdrFileParser, GroFileParser

syspath = "./examples/test_data/ethanol_water_26C/sys_405"
top_file = os.path.join(syspath, "sys_405.top")
gro_file = os.path.join(syspath, "sys_405.gro")
edr_file = os.path.join(syspath, "sys_405_npt.edr")

# determines molecules present in simulation and their counts
top_parser = TopFileParser(top_file)
print("molecule dict: ", top_parser.molecule_counts)
print("molecule names: ", top_parser.molecules)
print("total molecule number: ", top_parser.total_molecules)

# determines electron count for each molecule type
gro_parser = GroFileParser(gro_file)
print("electron dict: ", gro_parser.electron_count)
print("box volume: ", gro_parser.compute_box_volume())

# computes energy properties by calling gmx energy
edr_parser = EdrFileParser(edr_file)
print("List of available properties: ", edr_parser.available_properties())
print("Density array over simulation time: ", edr_parser.extract_timeseries("density"))
print("Average density with std deviation: ", edr_parser.average_property("density", return_std=True))
```

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [jevandezande/pixi-cookiecutter](https://github.com/jevandezande/pixi-cookiecutter) project template.

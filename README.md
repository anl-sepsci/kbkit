# KBKit: Kirkwood-Buff Analysis Toolkit

[![License](https://img.shields.io/github/license/anl-sepsci/kbkit)](https://github.com/anl-sepsci/kbkit/blob/master/LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/kbkit.svg)](https://pypi.org/project/kbkit/)
[![Powered by: Pixi](https://img.shields.io/badge/Powered_by-Pixi-facc15)](https://pixi.sh)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![codecov](https://codecov.io/gh/anl-sepsci/kbkit/graph/badge.svg?token=XJ5LXJYP76)](https://codecov.io/gh/anl-sepsci/kbkit)
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
git clone https://github.com/anl-sepsci/kbkit.git
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
from kbkit.kbi import KBIntegrator
from kbkit.systems import SystemProperties
from kbkit.io import RdfParser

syspath = "./examples/test_data/ethanol_water_26C/sys_405"
rdf_path = os.path.join(sys_path, "kbi_rdf_files_gmx25", "rdf_ETHOL_SPCEW.xvg")

# create integrator object from single RDF file
rdf = RDFParser(path=rdf_path)
integrator = KBIntegrator.from_system_properties(
    rdf=rdf,
    system_properties=SystemProperties(sys_path),
)

# calculate KBI in thermodynamic limit
kbi = integrator.compute_kbi(mol_j="SPCEW")
```

### Run an automated pipeline for batch analysis

```python
from kbkit.api import Pipeline

# Set up and run the pipeline
pipe = Pipeline(
    base_path="./examples/test_data/ethanol_water_26C", 
    pure_path="./examples/test_data/pure_components",   
    pure_systems=["ETHOL_300", "SPCEW_300"],            
    include_mode="npt",                                 
    activity_integration_type="numerical",                 
)

# Access the properties in PropertyResults objects
res = pipe.results

# Convert units from kJ/mol -> kcal/mol
# current units will be read from existing PropertyResult object
g_ex_res = res["g_ex"].to("kJ/mol")

# make figures for KBI analysis and select thermodynamic properties
pipe.make_figures(xmol="ETHOL")
```

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [jevandezande/pixi-cookiecutter](https://github.com/jevandezande/pixi-cookiecutter) project template.

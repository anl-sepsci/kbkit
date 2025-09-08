# KBKit: Kirkwood-Buff Analysis Toolkit

[![License](https://img.shields.io/github/license/aperoutka/kbkit)](https://github.com/aperoutka/kbkit/blob/master/LICENSE)
[![Powered by: Pixi](https://img.shields.io/badge/Powered_by-Pixi-facc15)](https://pixi.sh)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/aperoutka/kbkit/build-and-test.yml?branch=main&logo=github-actions)](https://github.com/aperoutka/kbkit/actions/workflows/build-and-test.yml)
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

**KBKit** can be installed from cloning its github repository and creating an anaconda environment with dependencies.

```python
git clone https://github.com/aperoutka/kbkit.git
cd kbkit
conda create --name kbkit python=3.12
conda activate kbkit
pip install .
```

## Examples

Below are several examples on various ways to implement **KBKit**. 

### Calculating Kirkwood-Buff integrals on a single RDF

```python
from kbkit.analysis import KBIntegrator

# create integrator object from single RDF file
integrator = KBIntegrator(rdf_file)

# calculate running-KBI
rkbi = integrator.rkbi()

# calculate KBI in thermodynamic limit
kbi = integrator.integrate()

# visualize KBI integration and extrapolation
integrator.plot()
```

### Run an automated pipeline for batch analysis

```python
from kbkit.core import KBPipeline

# Set up and run the pipeline
pipe = KBPipeline(
    base_path="/path/to/systems",                # directory with system data
    pure_path="/path/to/pure_components",        # directory with pure component data
    pure_systems=["acetone_300", "water_300"],   # list of pure systems
    ensemble="npt",                              # ensemble type: npt or nvt
    gamma_integration_type="numerical",          # integration method
    verbose=False                                # logging verbosity
)

# run kbkit pipeline
results = pipe.run()

# Access the results properties
# stored in dataclass (ThermoProperty); attributes: name, value, units
# example for excess energy
ge_obj = results.ge
ge_array = ge_obj.value
ge_units = ge_obj.units
print("GE summary: ", ge_array.shape, ge_units)

# Convert units from kJ/mol -> kcal/mol
# default units will be those from GROMACS
pipe.convert_units("ge", "kcal/mol")
```

### Create plots for thermodynamic properties from pipeline

```python
from kbkit.viz import Plotter

# Map molecule IDs (as present in .top files) to names for figures
molecule_map = {
    "ACETO": "Acetone",
    "TIP4P": "Water",
}
x_mol = "ACETO"  # molecule for x-axis labels

plotter = Plotter(pipeline=pipe, x_mol=x_mol, molecule_map=molecule_map)

# Plot Kirkwood-Buff integrals
plotter.plot("kbi")

# Plot log of activity coefficients
plotter.plot("lngammas")

# Generate all figures (saved to /path/to/systems/kb_analysis)
plotter.make_figures()
```

### Parse GROMACS files

```python
from kbkit.parsers import TopFileParser, EdrFileParser, GroFileParser

# determines molecules present in simulation and their counts
top_parser = TopFileParser(top_file.top)
print("molecule dict: ", top_parser.molecule_counts)
print("molecule names: ", top_parser.molecules)
print("total molecule number: ", top_parser.total_molecules)

# determines electron count for each molecule type
gro_parser = GroFileParser(gro_file.gro)
print("electron dict: ", gro_parser.electron_count)
print("box volume: ", gro_parser.compute_box_volume())

# computes energy properties by calling gmx energy
edr_parser = EdrFileParser(edr_file.edr)
print("List of available properties: ", edr_parser.available_properties())
print("Density array over simulation time: ", edr_parser.extract_timeseries("density"))
print("Average density with std deviation: ", edr_parser.average_property("density", return_std=True))
```

### Calculate static structure factors and x-ray intensities as q &rarr; 0

```python
from kbkit.calculators import StaticStructureCalculator

"""
Requires:
    - mol fraction matrix: shape(num compositions, num components)
    - molar_volume: shape(num components), units cm^3/mol
    - n_electrons: shape(num components)
"""
calculator = StaticStructureCalculator(mol_fr, molar_volume, n_electrons)

# update conditions
# if conditions are not updated all calculations will use previous values
calculator.update_conditions(T, Hessian, isothermal_compressibility)

# calculate x-ray intensity
calculator.i0()
```

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [jevandezande/pixi-cookiecutter](https://github.com/jevandezande/pixi-cookiecutter) project template.

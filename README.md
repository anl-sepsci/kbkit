# KBKit: Kirkwood-Buff Analysis Toolkit

[![License](https://img.shields.io/github/license/aperoutka/kbkit)](https://github.com/aperoutka/kbkit/blob/master/LICENSE)
[![Powered by: Pixi](https://img.shields.io/badge/Powered_by-Pixi-facc15)](https://pixi.sh)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/aperoutka/kbkit/build-and-test.yml?branch=main&logo=github-actions)](https://github.com/aperoutka/kbkit/actions/workflows/build-and-test.yml)
[![docs](http://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://kbkit.readthedocs.io/)
![python 3.12](https://img.shields.io/badge/Python-3.12%2B-blue)

[![Examples](https://img.shields.io/badge/Examples-Jupyter-orange?style=flat-square&logo=jupyter)](https://nbviewer.org/github/aperoutka/kbkit/blob/main/docs/examples/kbkit_example.ipynb)


`kbkit` is a `Python` library designed to streamline the analysis of molecular dynamics simulations, focusing on the application of Kirkwood-Buff (KB) theory for the calculation of activity coefficients and excess thermodynamic properties.

## Installation

`kbkit` can be installed from cloning github repository.

```python
git clone https://github.com/aperoutka/kbkit.git
```

Creating an anaconda environment with dependencies and install `kbkit`.

```python
cd kbkit
conda create --name kbkit python=3.12 --file requirements.txt
conda activate kbkit
pip install .
```

## Examples

A Jupyter notebook example for a binary system is available and provided in documentation.
This notebook demonstrates how to apply `kbkit` to simulation data.
An example of file structure is provided in `docs/test_data`.

```python
docs/examples/kbkit_example.ipynb
```


## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [jevandezande/pixi-cookiecutter](https://github.com/jevandezande/pixi-cookiecutter) project template.

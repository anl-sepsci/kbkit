KBKit: Kirkwood-Buff Analysis Toolkit
======================================

.. image:: https://img.shields.io/github/v/release/aperoutka/kbkit?label=Latest%20Release&logo=github
    :target: https://github.com/aperoutka/kbkit/releases
    :alt: Latest GitHub Release

Check out our `latest release <https://github.com/aperoutka/kbkit/releases>`_ on GitHub for updates, changelogs, and tagged versions.

.. toctree::
    :maxdepth: 1
    :caption: API Reference:
    :titlesonly:

    kbkit.analysis
    kbkit.calculators
    kbkit.core
    kbkit.parsers
    kbkit.viz.plotter


Installation
-------------
.. image:: http://img.shields.io/badge/License-MIT-blue.svg
    :target: https://tldrlegal.com/license/mit-license
    :alt: license
.. image:: https://img.shields.io/pypi/v/kbkit.svg
    :target: https://pypi.org/project/kbkit/
    :alt: PyPI version
.. image:: https://img.shields.io/badge/Powered_by-Pixi-facc15
    :target: https://pixi.sh
    :alt: Powered by: Pixi
.. image:: https://img.shields.io/badge/code%20style-ruff-000000.svg
    :target: https://github.com/astral-sh/ruff
    :alt: Code style: ruff
.. image:: https://img.shields.io/github/actions/workflow/status/aperoutka/kbkit/build-and-test.yml?branch=main&logo=github-actions
    :target: https://github.com/aperoutka/kbkit/actions/workflows/build-and-test.yml
    :alt: GitHub Workflow Status
.. image:: https://coveralls.io/repos/github/aperoutka/kbkit/badge.svg?branch=main
    :target: https://coveralls.io/github/aperoutka/kbkit?branch=main
    :alt: Coverage Status
.. image:: http://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat
    :target: https://kbkit.readthedocs.io/
    :alt: docs
.. image:: https://img.shields.io/badge/Python-3.12%2B-blue

**Quick install via PyPI:**

.. code-block:: bash

    pip install kbkit

**Developer install (recommended for contributors or conda users):**

Clone the GitHub repository and use the provided Makefile to set up your development environment:

.. code-block:: bash

    git clone https://github.com/aperoutka/kbkit.git
    cd kbkit
    make setup-dev

This one-liner creates the `kbkit-dev` conda environment, installs `kbkit` in editable mode, and runs the test suite.

To install without running tests:

.. code-block:: bash

    make dev-install

To build and install the package into a clean user environment:

.. code-block:: bash

    make setup-user

For a full list of available commands:

.. code-block:: bash

    make help


File Organization
------------------

KBKit expects a structured directory layout that separates mixed systems from pure components. This organization enables automated parsing, reproducible KB integrals, and scalable analysis across chemical systems.

.. code-block:: text
    :caption: KB Analysis File Structure

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

**Requirements:**

- Each **system** to be analyzed must include:
  - `rdf_dir/` containing `.xvg` RDF files for all pairwise interactions
  - `.top` topology file
  - `.edr` energy file
  - `.gro` structure file *(optional but recommended)*

- Each **pure component** must include:
  - `.top` topology file
  - `.edr` energy file
  - All other files *(optional)*

These inputs are parsed by `kbkit.parsers` and fed into `kbkit.calculators` for Kirkwood-Buff integrals and thermodynamic predictions. The layout is designed for:

- **Discoverability**: Clear separation of mixed vs. pure systems
- **Reproducibility**: Consistent file naming and structure
- **Extensibility**: Easy to add new molecules or systems

For contributors, this structure ensures semantic clarity and minimizes onboarding friction. New systems can be added by replicating the folder pattern and updating input paths—no need to modify core logic.

Indices and tables
===================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

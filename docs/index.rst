KBKit: Kirkwood-Buff Analysis Toolkit
======================================

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

``kbkit`` can be installed from cloning `github repository <https://github.com/aperoutka/kbkit>`_.

.. code-block:: text

    git clone https://github.com/aperoutka/kbkit.git

Creating an anaconda environment with package dependencies and install ``kbkit``.

.. code-block:: text

    cd kbkit
    conda create --name kbkit python=3.12
    conda activate kbkit
    pip install .


File Organization
------------------

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

Indices and tables
===================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

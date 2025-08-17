"""Creates file tree for documentation."""

import sys as sys_arg
from operator import itemgetter

from tree_format import format_tree


def print_tree(tree):
    """Print file tree."""
    print(format_tree(tree, format_node=itemgetter(0), get_children=itemgetter(1)))


kb_ex_tree = (
    "kbi_dir/",
    [
        (
            "project/",
            [
                (
                    "system/",
                    [
                        (
                            "rdf_dir/",
                            [
                                ("mol1_mol1.xvg", []),
                                ("mol1_mol2.xvg", []),
                                ("mol1_mol2.xvg", []),
                            ],
                        ),
                        ("system_npt.edr", []),
                        ("system_npt.gro", []),
                        ("system.top", []),
                    ],
                )
            ],
        ),
        (
            "pure_components/",
            [
                (
                    "molecule1/",
                    [
                        ("molecule1_npt.edr", []),
                        ("molecule1.top", []),
                    ],
                )
            ],
        ),
    ],
)


kbkit_tree = (
    "kbkit/",
    [
        (
            ".github/",
            [
                (
                    "workflows/",
                    [
                        ("build-and-test.yml", []),
                        ("publish.yml", []),
                    ],
                ),
            ],
        ),
        (
            "docs/",
            [
                ("index.rst", []),
                (
                    "Examples/",
                    [
                        ("test_data/", []),
                        ("kbkit_example.ipynb", []),
                    ],
                ),
            ],
        ),
        (
            "tests/",
            [
                ("__init__.py", []),
                ("smoketest.py", []),
            ],
        ),
        (
            "src/",
            [
                (
                    "kbkit/",
                    [
                        ("__init__.py", []),
                        ("_version.py", []),
                        ("kb_pipeline.py", []),
                        ("mapped.py", []),
                        ("plotter.py", []),
                        ("presentation.mplstyle", []),
                        ("unit_registry.py", []),
                        ("utils.py", []),
                        (
                            "kb/",
                            [
                                ("__init__.py", []),
                                ("kb_thermo.py", []),
                                ("kbi.py", []),
                                ("rdf.py", []),
                                ("system_set.py", []),
                            ],
                        ),
                        (
                            "properties/",
                            [
                                ("__init__.py", []),
                                ("energy_reader.py", []),
                                ("system_properties.py", []),
                                ("topology.py", []),
                            ],
                        ),
                    ],
                ),
            ],
        ),
        (".coveragerc", []),
        (".editorconfig", []),
        (".gitattributes", []),
        (".gitconfig", []),
        (".gitignore", []),
        (".pre-commit-config.yaml", []),
        (".readthedocs.yaml", []),
        ("LICENSE", []),
        ("pixi.lock", []),
        ("pyproject.toml", []),
        ("README.md", []),
        ("requirements.txt", []),
    ],
)


if __name__ == "__main__":
    tree_name = sys_arg.argv[1] if len(sys_arg.argv) > 1 else "ex"
    trees_mapped: dict[str, tuple] = {"ex": kb_ex_tree, "kbkit": kbkit_tree}
    tree = trees_mapped.get(tree_name, kb_ex_tree)
    print(f"\nPrinting tree: {tree_name}\n")
    print_tree(tree)

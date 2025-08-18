from pathlib import Path
from operator import itemgetter
from tree_format import format_tree
import pathspec


def load_gitignore(root: Path) -> pathspec.PathSpec:
    """Load .gitignore patterns from the root directory."""
    gitignore_path = root / ".gitignore"
    if gitignore_path.exists():
        patterns = gitignore_path.read_text().splitlines()
        return pathspec.PathSpec.from_lines("gitwildmatch", patterns)
    return pathspec.PathSpec.from_lines("gitwildmatch", [])


def build_tree(path: Path, spec: pathspec.PathSpec, root: Path, exclude_names: set[str] = {"__pycache__"}) -> tuple[str, list]:
    """Recursively build a tree-format-compatible tuple, excluding .gitignore and manual names."""
    children = []
    for entry in sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower())):
        rel_path = entry.relative_to(root)
        if entry.name in exclude_names or spec.match_file(str(rel_path)):
            continue
        if entry.is_dir():
            children.append(build_tree(entry, spec, root, exclude_names))
        else:
            children.append((entry.name, []))
    return (path.name + "/", children)


def print_tree(tree):
    """Print file tree using tree_format."""
    print(format_tree(tree, format_node=itemgetter(0), get_children=itemgetter(1)))


if __name__ == "__main__":
    import sys
    root_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent.parent
    spec = load_gitignore(root_path)
    exclude_names = {"__pycache__", ".git", "docs", "junit", ".pytest_cache", ".ruff_cache"}
    tree = build_tree(root_path, spec, root_path, exclude_names)
    print(f"\nPrinting tree for: {root_path}\n")
    print_tree(tree)

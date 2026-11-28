"""hands starts its children through SpawnedChildren, and only a composition root names it, binding it as what makes
every command's children: so every other module, and every test of one, runs on whatever children it is given
(tests/fakechildren.py)."""

import ast
from pathlib import Path

import hands
from imports import imported, qualified
from test_processes_boundary import EXCEPTION, marked

SPAWNED = "hands.sessions.child.SpawnedChildren"
# Where SpawnedChildren is named: the root every command goes through.
ROOT = "daemon/cli.py"


def names(root: Path) -> dict[str, list[int]]:
    """Each module under the package `root` that names SpawnedChildren, but in a type or where it is marked as the
    exception it is, by its path from `root`, with the lines it names it on: a module that builds one, or hands the class
    on to build them, makes children of its own."""

    def found(path: Path) -> list[int]:
        source = path.read_text()
        tree = ast.parse(source)
        named = imported(tree, ".".join([root.name, *path.relative_to(root).parent.parts]))
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        # A type names what a value is, and makes nothing.
        typed = {id(node) for annotation in annotations(tree) for node in ast.walk(annotation)}
        uses = (node for node in ast.walk(tree) if isinstance(node, ast.Name | ast.Attribute) and id(node) not in typed and qualified(node, named) == SPAWNED)
        return sorted(use.lineno for use in uses if not marked(use, parents, source.splitlines()))

    return {path.relative_to(root).as_posix(): lines for path in sorted(root.rglob("*.py")) if (lines := found(path))}


def annotations(tree: ast.Module) -> list[ast.expr]:
    """Every annotation in `tree`: of a parameter, a return, or a variable."""
    found: list[ast.expr] = []
    for node in ast.walk(tree):
        match node:
            case ast.arg(annotation=ast.expr() as annotation) | ast.AnnAssign(annotation=annotation):
                found.append(annotation)
            case ast.FunctionDef(returns=ast.expr() as returns) | ast.AsyncFunctionDef(returns=ast.expr() as returns):
                found.append(returns)
            case _:
                pass
    return found


def test_only_the_root_names_spawned_children() -> None:
    assert list(names(Path(hands.__file__).parent)) == [ROOT]


def test_the_check_finds_each_unmarked_naming_that_makes_children(tmp_path: Path) -> None:
    root = tmp_path / "hands"
    (root / "sessions").mkdir(parents=True)
    for name, source in {
        "bare.py": "from hands.sessions.child import SpawnedChildren\nSpawnedChildren(clock, emit)\n",
        "aliased.py": "from hands.sessions import child as c\nif c.SpawnedChildren(clock, emit):\n    pass\n",
        "relative.py": "from .child import SpawnedChildren as S\nasync def f():\n    return await g(S(clock, emit))\n",
        "handed.py": "from hands.sessions.child import SpawnedChildren\nmain(children_on=SpawnedChildren)\n",
        "marked.py": f"from hands.sessions.child import SpawnedChildren\n# {EXCEPTION}: why.\nSpawnedChildren(clock, emit)\n",
        "typed.py": "from hands.sessions.child import SpawnedChildren\ndef f(spawned: SpawnedChildren) -> SpawnedChildren: ...\nkept: SpawnedChildren\n",
    }.items():
        (root / "sessions" / name).write_text(source)
    assert names(root) == {"sessions/aliased.py": [2], "sessions/bare.py": [2], "sessions/handed.py": [2], "sessions/relative.py": [3]}

"""hands starts its children through the Runner, and only a composition root builds one, handing it on as Children: so
every other module, and every test of one, runs on whatever children it is given (tests/fakechildren.py)."""

import ast
from pathlib import Path

import hands
from imports import imported, qualified
from test_processes_boundary import EXCEPTION, marked

RUNNER = "hands.sessions.child.Runner"
# Where a Runner is built: the root every command goes through.
ROOT = "daemon/cli.py"


def builds(root: Path) -> dict[str, list[int]]:
    """Each module under the package `root` that builds a Runner not marked as the exception it is, by its path from
    `root`, with the lines it builds one on."""

    def found(path: Path) -> list[int]:
        source = path.read_text()
        tree = ast.parse(source)
        named = imported(tree, ".".join([root.name, *path.relative_to(root).parent.parts]))
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        calls = (node for node in ast.walk(tree) if isinstance(node, ast.Call) and qualified(node.func, named) == RUNNER)
        return sorted(call.lineno for call in calls if not marked(call, parents, source.splitlines()))

    return {path.relative_to(root).as_posix(): lines for path in sorted(root.rglob("*.py")) if (lines := found(path))}


def test_only_the_root_builds_a_runner() -> None:
    assert list(builds(Path(hands.__file__).parent)) == [ROOT]


def test_the_build_check_finds_each_runner_built_unmarked(tmp_path: Path) -> None:
    root = tmp_path / "hands"
    (root / "sessions").mkdir(parents=True)
    for name, source in {
        "bare.py": "from hands.sessions.child import Runner\nRunner(clock, emit)\n",
        "aliased.py": "from hands.sessions import child as c\nif c.Runner(clock, emit):\n    pass\n",
        "relative.py": "from .child import Runner as R\nasync def f():\n    return await g(R(clock, emit))\n",
        "marked.py": f"from hands.sessions.child import Runner\n# {EXCEPTION}: why.\nRunner(clock, emit)\n",
        "typed.py": "from hands.sessions.child import Runner\ndef f(runner: Runner) -> None: ...\n",
    }.items():
        (root / "sessions" / name).write_text(source)
    assert builds(root) == {"sessions/aliased.py": [2], "sessions/bare.py": [2], "sessions/relative.py": [3]}

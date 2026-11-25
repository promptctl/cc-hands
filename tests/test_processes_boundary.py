"""hands reaches the Mac's processes through `hands.sessions.kernel`, and only a composition root reaches that, handing it on
as `Processes`: so every other module, and every test of one, runs on whatever table it is given. A call that reaches
another process past it is in an adapter, or marked as the exception it is."""

import ast
from pathlib import Path

import hands
from imports import imported, qualified
from test_clock_boundary import ROOTS

KERNEL = "hands.sessions.kernel"
# The calls that reach another process past Processes: a signal to it, the uid whose processes are the user's, and the
# children the kernel lists of one; and anything read of the bindings to the kernel's process libraries, whose every
# call is a word on some process.
REACHES = frozenset({"os.kill", "os.killpg", "os.getuid", "os.geteuid", "hands.darwin.child_pids"})
BINDINGS = ("hands.darwin.libc", "hands.darwin.libproc")
# Where those calls belong: the bindings themselves, the kernel's adapter, for any process, and hands.sessions.child,
# for the children hands starts and waits on.
ADAPTERS = frozenset({"darwin.py", "sessions/kernel.py", "sessions/child.py"})
EXCEPTION = "[LAW:effects-at-boundaries] exception"


def importers(root: Path, adapter: str = KERNEL) -> list[str]:
    """Each module under the package `root` that reaches the module `adapter`, by its path from `root`: by an import of
    it, a relative one among them, or a use of it by the package it was imported under."""

    def reaches(path: Path) -> bool:
        tree = ast.parse(path.read_text())
        named = imported(tree, ".".join([root.name, *path.relative_to(root).parent.parts]))
        used = (qualified(node, named) for node in ast.walk(tree) if isinstance(node, ast.Attribute))
        return any(name == adapter or name.startswith(f"{adapter}.") for name in (*named.values(), *used))

    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*.py") if reaches(path))


def unmarked_reaches(root: Path) -> list[str]:
    """Each call under the package `root`, outside the adapters, that reaches another process past Processes and is not
    marked as the exception it is, by its path from `root` and its line: marked in the comment just above it, or just
    above a statement it is part of."""

    def found(path: Path) -> list[str]:
        source = path.read_text()
        tree = ast.parse(source)
        named = imported(tree, ".".join([root.name, *path.relative_to(root).parent.parts]))
        parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
        reaching = (node for node in ast.walk(tree) if isinstance(node, ast.Name | ast.Attribute) and _reaches(qualified(node, named)))
        return sorted({f"{path.relative_to(root).as_posix()}:{node.lineno}" for node in reaching if not _marked(node, parents, source.splitlines())})

    return sorted(each for path in root.rglob("*.py") if path.relative_to(root).as_posix() not in ADAPTERS for each in found(path))


def _reaches(name: str) -> bool:
    return name in REACHES or any(name == binding or name.startswith(f"{binding}.") for binding in BINDINGS)


def _marked(node: ast.AST, parents: dict[ast.AST, ast.AST], lines: list[str]) -> bool:
    while True:
        if isinstance(node, ast.stmt) and EXCEPTION in _comment_above(node.lineno, lines):
            return True
        if node not in parents:
            return False
        node = parents[node]


def _comment_above(line: int, lines: list[str]) -> str:
    above = lines[: line - 1]
    start = len(above)
    while start and above[start - 1].strip().startswith("#"):
        start -= 1
    return "\n".join(above[start:])


def test_only_a_composition_root_reaches_the_kernels_processes() -> None:
    assert importers(Path(hands.__file__).parent) == sorted(ROOTS)


def test_every_other_reach_of_another_process_is_in_an_adapter_or_marked_as_an_exception() -> None:
    assert unmarked_reaches(Path(hands.__file__).parent) == []


def test_the_reach_check_finds_each_reach_left_unmarked(tmp_path: Path) -> None:
    root = tmp_path / "hands"
    (root / "sessions").mkdir(parents=True)
    for name, source in {
        "bare.py": "import os\nos.kill(1, 0)\n",
        "named.py": "from hands.darwin import child_pids\nchild_pids(1)\n",
        "aliased.py": "from hands.darwin import libproc as lp\nlp.proc_pidinfo(1, 3, 0, None, 0)\n",
        "marked.py": f"import os\n# {EXCEPTION}: why.\nos.getuid()\n",
        "enclosing.py": f"import os\n# {EXCEPTION}: why.\nwith open('x'):\n    os.killpg(1, 9)\n",
        "table.py": "from hands.darwin import libc\nlibc.sysctl(None, 0, None, None, None, 0)\n",
        "kernel.py": "import os\nos.kill(1, 0)\n",
        "own.py": "import os\nos.getpid()\n",
    }.items():
        (root / "sessions" / name).write_text(source)
    (root / "darwin.py").write_text("import ctypes\nlibproc = ctypes.CDLL('libproc')\nlibproc.proc_listchildpids(1, None, 0)\n")
    assert unmarked_reaches(root) == ["sessions/aliased.py:2", "sessions/bare.py:2", "sessions/named.py:2", "sessions/table.py:2"]


def test_the_check_finds_each_way_of_reaching_the_kernel(tmp_path: Path) -> None:
    root = tmp_path / "hands"
    (root / "sessions").mkdir(parents=True)
    for name, source in {
        "named.py": "from hands.sessions.kernel import KERNEL\n",
        "module.py": "from hands.sessions import kernel\n",
        "whole.py": "import hands.sessions.kernel\nhands.sessions.kernel.KERNEL.table()\n",
        "relative.py": "from .kernel import KERNEL\n",
        "unrelated.py": "from hands.sessions.processes import Processes\n",
    }.items():
        (root / "sessions" / name).write_text(source)
    assert importers(root) == ["sessions/module.py", "sessions/named.py", "sessions/relative.py", "sessions/whole.py"]

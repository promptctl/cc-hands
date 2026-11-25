"""hands reaches the Mac's processes through `hands.sessions.kernel`, and only a composition root reaches that, handing it on
as `Processes`: so every other module, and every test of one, runs on whatever table it is given."""

import ast
from pathlib import Path

import hands
from imports import imported, qualified
from test_clock_boundary import ROOTS

KERNEL = "hands.sessions.kernel"


def kernel_importers(root: Path) -> list[str]:
    """Each module under the package `root` that reaches hands.sessions.kernel, by its path from `root`: by an import of
    it, a relative one among them, or a use of it by the package it was imported under."""

    def reaches(path: Path) -> bool:
        tree = ast.parse(path.read_text())
        named = imported(tree, ".".join([root.name, *path.relative_to(root).parent.parts]))
        used = (qualified(node, named) for node in ast.walk(tree) if isinstance(node, ast.Attribute))
        return any(name == KERNEL or name.startswith(f"{KERNEL}.") for name in (*named.values(), *used))

    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*.py") if reaches(path))


def test_only_a_composition_root_reaches_the_kernels_processes() -> None:
    assert kernel_importers(Path(hands.__file__).parent) == sorted(ROOTS)


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
    assert kernel_importers(root) == ["sessions/module.py", "sessions/named.py", "sessions/relative.py", "sessions/whole.py"]

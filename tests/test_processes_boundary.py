"""hands reaches the Mac's processes through `hands.sessions.kernel`, and only a composition root reaches that, handing it on
as `Processes`: so every other module, and every test of one, runs on whatever table it is given."""

import ast
from pathlib import Path

import hands
from test_clock_boundary import ROOTS


def kernel_importers(root: Path) -> list[str]:
    """Each module under `root` that imports hands.sessions.kernel, by its path from `root`."""

    def imports(tree: ast.AST) -> bool:
        return any(
            (isinstance(node, ast.ImportFrom) and node.module == "hands.sessions.kernel")
            or (isinstance(node, ast.ImportFrom) and node.module == "hands.sessions" and any(alias.name == "kernel" for alias in node.names))
            or (isinstance(node, ast.Import) and any(alias.name == "hands.sessions.kernel" for alias in node.names))
            for node in ast.walk(tree)
        )

    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*.py") if imports(ast.parse(path.read_text())))


def test_only_a_composition_root_reaches_the_kernels_processes() -> None:
    assert kernel_importers(Path(hands.__file__).parent) == sorted(ROOTS)


def test_the_check_finds_each_way_of_importing_the_kernel(tmp_path: Path) -> None:
    for name, source in {
        "named.py": "from hands.sessions.kernel import KERNEL\n",
        "module.py": "from hands.sessions import kernel\n",
        "whole.py": "import hands.sessions.kernel\n",
        "unrelated.py": "from hands.sessions.processes import Processes\n",
    }.items():
        (tmp_path / name).write_text(source)
    assert kernel_importers(tmp_path) == ["module.py", "named.py", "whole.py"]

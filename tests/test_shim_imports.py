"""The hook shim starts on every hook, in Claude Code's critical path: what it imports, every hook pays for."""

import ast
from pathlib import Path

import hands

SOURCE = Path(hands.__file__).parent.parent
# What hands.clock imports where a sleep, a wait, or an id needs it, and never with a module, for the shim's sake.
SLOW = {"asyncio", "secrets", "uuid"}


def _file(module: str) -> Path | None:
    base = SOURCE.joinpath(*module.split("."))
    return next((path for path in (base.with_suffix(".py"), base / "__init__.py") if path.exists()), None)


def _imported(path: Path, package: str) -> set[str]:
    """What `path` imports as it is imported: its module-level statements, not an import a function makes as it runs."""
    names: set[str] = set()
    for node in ast.parse(path.read_text()).body:
        match node:
            case ast.Import(names=aliases):
                names.update(alias.name for alias in aliases)
            case ast.ImportFrom(module=str(module), level=0):
                names.add(module)
                names.update(f"{module}.{alias.name}" for alias in node.names)
            case ast.ImportFrom(module=module, level=level):
                names.add(".".join([*package.split(".")[: len(package.split(".")) - level + 1], *([module] if module else [])]))
            case _:
                pass
    return names


def imported_with(root: str) -> set[str]:
    """Every module outside hands that importing `root` imports, through each hands module it imports in turn."""
    seen: set[str] = set()
    outside: set[str] = set()
    todo = [root]
    while todo:
        module = todo.pop()
        if module in seen or (path := _file(module)) is None:
            continue
        seen.add(module)
        parts = module.split(".")
        # A package's __init__ runs on the import of anything under it.
        todo += [".".join(parts[:at]) for at in range(1, len(parts))]
        for name in _imported(path, module if path.name == "__init__.py" else ".".join(parts[:-1])):
            if name.split(".")[0] == "hands":
                todo.append(name)
            else:
                outside.add(name.split(".")[0])
    return outside


def test_the_hook_shim_imports_none_of_what_hands_imports_only_where_it_is_needed() -> None:
    assert imported_with("hands.sessions.shim") & SLOW == set()

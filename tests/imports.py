"""[LAW:one-source-of-truth] the names a module's code reads, with the imports that bound them written out: the one reading
the source scans share, so an alias or a relative import is resolved the same way by each."""

import ast


def imported(tree: ast.Module, package: str) -> dict[str, str]:
    """Each name the module binds by an import, to the dotted name it was imported as."""
    named: dict[str, str] = {}
    for node in ast.walk(tree):
        match node:
            case ast.Import(names=aliases):
                named.update({alias.asname: alias.name for alias in aliases if alias.asname})
                named.update({alias.name.split(".")[0]: alias.name.split(".")[0] for alias in aliases if not alias.asname})
            case ast.ImportFrom(module=str(module), names=aliases, level=0):
                named.update({alias.asname or alias.name: f"{module}.{alias.name}" for alias in aliases})
            # A relative import climbs from the package the module is in, one package per dot after the first.
            case ast.ImportFrom(module=module, names=aliases, level=int(level)) if level > 0:
                base = ".".join([*package.split(".")[: len(package.split(".")) - (level - 1)], *([module] if module else [])])
                named.update({alias.asname or alias.name: f"{base}.{alias.name}" for alias in aliases})
            case _:
                pass
    return named


def qualified(node: ast.expr, named: dict[str, str]) -> str:
    """The dotted name `node` reads, with the import its first part was bound by written out."""
    match node:
        case ast.Name(id=name):
            return named.get(name, name)
        case ast.Attribute(value=value, attr=attr):
            return f"{qualified(value, named)}.{attr}"
        case _:
            return ""

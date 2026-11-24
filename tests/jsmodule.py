"""A page's script module (`src/hands/voice/*.js`) run in QuickJS: the one loader the tests of those scripts share, so they
run the page's code under the same rules."""

import json
from importlib import resources
from typing import cast

import quickjs


class JsModule:
    """`script`, a module of hands.voice, in a context of its own with what the test set up in `before`. QuickJS here
    loads no module a module imports, so the names the module exports (`exports`, as a page's import would name them) are
    handed to the test as globals."""

    def __init__(self, script: str, exports: str, before: str = "") -> None:
        self._js = quickjs.Context()
        if before:
            self._js.eval(before)
        module = resources.files("hands.voice").joinpath(script).read_text()
        self._js.module(f"{module}\nObject.assign(globalThis, {{{exports}}});")

    def eval(self, code: str) -> object:
        return self._js.eval(code)

    def settle(self) -> None:
        """Every job the script has to run before it next waits on something outside it."""
        while self._js.execute_pending_job():
            pass

    def run(self, code: str) -> None:
        self._js.eval(code)
        self.settle()

    def value(self, expression: str) -> object:
        return json.loads(cast(str, self._js.eval(f"JSON.stringify({expression})")))

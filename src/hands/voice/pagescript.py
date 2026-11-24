"""The scripts of hands' package that the phone's pages run, each served at its own name."""

from importlib import resources

from aiohttp import web


def script_route(name: str) -> web.RouteDef:
    """The script `name` of this package, served at `/{name}` as a module a page imports, to anyone: it is the page's own
    code, and the key it asks hands with is in the page's address.

    [LAW:single-enforcer] the one place a script is served. Never kept by the browser, as no page is: a page and the
    scripts it runs are always one run's.
    """
    script = resources.files("hands.voice").joinpath(name).read_text()

    async def scripted(_request: web.Request) -> web.Response:
        return web.Response(text=script, content_type="text/javascript", headers={"Cache-Control": "no-store"})

    return web.get(f"/{name}", scripted)

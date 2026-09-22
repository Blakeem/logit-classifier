"""The `logit-classifier` command.

`serve` is the front door for anyone who installed from PyPI rather than cloning,
since it needs no uvicorn invocation and no import path.
"""

from __future__ import annotations

import argparse

from . import __version__
from .config import Config
from .deps import require


def _serve(host: str, port: int, reload: bool) -> int:
    # uvicorn imports the app from a string, so a missing fastapi surfaces as a raw
    # traceback out of its importer rather than as the line that installs the extra.
    require("fastapi", "service")
    uvicorn = require("uvicorn", "service")
    uvicorn.run("logit_classifier.service:app", host=host, port=port, reload=reload)
    return 0


def _show_config() -> int:
    config = Config.from_env()
    for name, value in vars(config).items():
        print(f"{name}={value}")
    return 0


def _parser(prog: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=prog, description=__doc__.splitlines()[0])
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="run the HTTP service and the browser page")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8077)
    serve.add_argument("--reload", action="store_true", help="restart on a source change")

    commands.add_parser("config", help="print the configuration the LOGIT_ variables produce")
    return parser


def main(argv: list[str] | None = None, prog: str = "logit-classifier") -> int:
    args = _parser(prog).parse_args(argv)

    if args.command == "serve":
        return _serve(args.host, args.port, args.reload)
    return _show_config()

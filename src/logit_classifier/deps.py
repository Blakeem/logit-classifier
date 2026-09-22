"""Loading the packages that live behind an extra.

The core install names numpy alone so a ComfyUI pack never asks pip to resolve
torch beside ComfyUI's pinned CUDA build. Everything outside that core is
imported here, on first use, and a missing one fails with its install line.
"""

from __future__ import annotations

import importlib
from types import ModuleType

from .errors import LogitClassifierError

DISTRIBUTION = "logit-classifier"


class MissingDependencyError(LogitClassifierError, ImportError):
    """A path needs an optional extra that is not installed."""


def require(module: str, extra: str) -> ModuleType:
    """Import a module from an extra, or fail naming the command that installs it."""
    try:
        return importlib.import_module(module)
    except ImportError as error:
        raise MissingDependencyError(
            f"{module} is needed here and is not installed. "
            f'Install it with: pip install "{DISTRIBUTION}[{extra}]"'
        ) from error

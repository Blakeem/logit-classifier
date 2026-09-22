"""The one base every error this library raises inherits.

A host catching one name covers the library. An error the caller can fix by
changing its input also inherits ValueError. An internal invariant failure also
inherits RuntimeError. A missing optional dependency also inherits ImportError.
"""

from __future__ import annotations


class LogitClassifierError(Exception):
    """Any failure this library raises on purpose."""


class ConfigError(LogitClassifierError, ValueError):
    """A Config field or a LOGIT_ environment variable holds a value it cannot take."""

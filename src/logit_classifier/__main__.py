"""`python -m logit_classifier`, for a venv whose scripts directory is not on PATH."""

from __future__ import annotations

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main(prog="python -m logit_classifier"))

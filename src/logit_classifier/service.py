"""HTTP service exposing the Jev-shaped System One endpoint."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

from fastapi import Body, FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .backends.base import VisionUnsupportedError
from .classifier import Classifier
from .config import Config
from .errors import LogitClassifierError
from .schema import ErrorBody, SchemaError, parse_request
from .vision import ImageError

WEB_DIR = Path(__file__).parent / "web"

_log = logging.getLogger(__name__)

EXAMPLE_REQUEST = {
    "state": "The integration keeps failing and I am losing sales.",
    "questions": {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this",
            "criteria": {"billing": "Payment issues", "technical": "Integration problems"},
        }
    },
}


@dataclass(frozen=True)
class Loaded:
    """What the lifespan built, present only once the model is resident."""

    config: Config
    classifier: Classifier
    load_seconds: float


_loaded: Loaded | None = None
# One GPU, and batch composition must stay a pure function of the request, so
# requests are served one at a time.
_gpu_lock = threading.Lock()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    global _loaded
    config = Config.from_env()
    started = time.time()
    classifier = Classifier(config)
    _loaded = Loaded(config=config, classifier=classifier,
                     load_seconds=round(time.time() - started, 2))
    yield
    classifier.persist_prior()


app = FastAPI(title="Logit Classifier", version=__version__, lifespan=lifespan)


def _rejected(message: str, field: str | None = None) -> JSONResponse:
    body = ErrorBody(error="unprocessable_entity", message=message, field=field)
    return JSONResponse(status_code=422, content=body.to_dict())


def _failed(message: str) -> JSONResponse:
    body = ErrorBody(error="internal_error", message=message)
    return JSONResponse(status_code=500, content=body.to_dict())


def _unavailable() -> JSONResponse:
    body = ErrorBody(error="service_unavailable", message="model is still loading")
    return JSONResponse(status_code=503, content=body.to_dict())


@app.exception_handler(RequestValidationError)
async def on_validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
    first = exc.errors()[0] if exc.errors() else {}
    location = ".".join(str(p) for p in first.get("loc", []) if p != "body")
    return _rejected(first.get("msg", "request failed validation"), location or None)


@app.post("/v1/systemone")
def system_one(
    body: Annotated[dict[str, Any], Body(openapi_examples={"jev": {"value": EXAMPLE_REQUEST}})],
    diagnostics: Annotated[bool, Query()] = False,
) -> Any:
    loaded = _loaded

    if loaded is None:
        return _unavailable()
    try:
        request = parse_request(body)
    except SchemaError as error:
        return _rejected(error.message, error.field)

    started = time.time()
    try:
        with _gpu_lock:
            response, diag = loaded.classifier.classify(request, allow_image_paths=False)
    except (ImageError, VisionUnsupportedError) as error:
        return _rejected(str(error), "state")
    except LogitClassifierError as error:
        # A backend contract or label failure is ours, not the caller's, so it must
        # not come back as a rejected request they could fix. The traceback is the
        # operator's only diagnostic, and returning a body swallows it.
        _log.exception("classify failed")
        return _failed(str(error))
    except Exception as error:
        # A CUDA out of memory or a kernel error must still reach a Jev caller as an
        # ErrorBody, never as a plain-text 500.
        _log.exception("classify failed")
        return _failed(f"{type(error).__name__}: {error}")
    payload = response.to_dict()

    if diagnostics:
        payload["diagnostics"] = {
            "candidate_mass": diag.candidate_mass,
            "prior_applied": diag.prior_applied,
            "branch_counts": diag.branch_counts,
            "latency_ms": round((time.time() - started) * 1000, 1),
            "temperature": loaded.classifier.temperature,
            "score_method": loaded.config.score_method,
            "prior_observations": loaded.classifier.priors.stats(),
        }
    return payload


@app.get("/v1/models")
def models() -> Any:
    loaded = _loaded

    if loaded is None:
        return _unavailable()
    return {
        "models": [
            {
                "name": loaded.config.served_model_id,
                "description": f"Restricted-label scorer over {loaded.config.model_id}",
                "aliases": list(loaded.config.model_aliases),
            }
        ]
    }


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    loaded = _loaded
    return {
        "ready": loaded is not None,
        "model_id": loaded.config.model_id if loaded else None,
        "sees_images": loaded.classifier.backend.sees_images if loaded else None,
        "load_seconds": loaded.load_seconds if loaded else None,
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

"""GET /v1/info — driver metadata."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, status

from .. import __version__
from .._generated.models import BackendKind, DriverInfo, Problem
from ..config import ConfigStore

router = APIRouter(tags=["meta"])


@router.get("/v1/info", response_model=DriverInfo)
async def info(request: Request) -> DriverInfo:
    store: ConfigStore = request.app.state.config_store
    backend_value = str(store.get("adapter") or "")

    # Tolerate malformed config — the whole point of degraded-mode is that
    # the driver stays reachable even when config is bad. Return 503 with a
    # clear message instead of raising 500.
    try:
        backend = BackendKind(backend_value)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=Problem(
                type="https://github.com/eugene-plexus/hemisphere-driver#config-invalid",
                title="Configuration invalid",
                status=503,
                detail=f"adapter config value is invalid: {e}",
                component="hemisphere-driver:degraded",
            ).model_dump(exclude_none=True),
        ) from e

    return DriverInfo(
        backend=backend,
        modelId=store.get("modelId"),
        version=__version__,
    )

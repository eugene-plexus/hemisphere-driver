"""GET /v1/info — driver metadata."""

from __future__ import annotations

from fastapi import APIRouter, Request

from .. import __version__
from .._generated.models import BackendKind, DriverInfo, Hemisphere
from ..config import ConfigStore

router = APIRouter(tags=["meta"])


@router.get("/v1/info", response_model=DriverInfo)
async def info(request: Request) -> DriverInfo:
    store: ConfigStore = request.app.state.config_store
    backend_value = str(store.get("adapter"))
    hemisphere_value = str(store.get("hemisphere"))
    return DriverInfo(
        backend=BackendKind(backend_value),
        modelId=store.get("modelId"),
        hemisphere=Hemisphere(hemisphere_value),
        version=__version__,
    )

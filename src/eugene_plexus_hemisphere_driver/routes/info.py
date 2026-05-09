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
    # Spec requires modelId; surface "default" when the config hasn't pinned one,
    # meaning the adapter falls back to its built-in default. Will become a
    # nullable field in a follow-up specs PR.
    model_id = store.get("modelId") or "default"
    return DriverInfo(
        backend=BackendKind(backend_value),
        modelId=model_id,
        # TODO(specs): Message.hemisphere uses an inline enum so codegen
        # produces Hemisphere1 alongside the schema-level Hemisphere; both
        # have identical members. Make Message.hemisphere a $ref in a
        # follow-up specs PR to deduplicate.
        hemisphere=Hemisphere(hemisphere_value),  # type: ignore[arg-type]
        version=__version__,
    )

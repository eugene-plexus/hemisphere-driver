"""POST /v1/generate (real) and POST /v1/generate/stream (still 501)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request, status

from .._generated.models import GenerateRequest, GenerateResponse, Problem
from ..adapters._subprocess import CliError

if TYPE_CHECKING:
    from ..app import _Adapter

router = APIRouter(tags=["inference"])

log = logging.getLogger(__name__)


@router.post("/v1/generate", response_model=GenerateResponse)
async def generate(request: Request, body: GenerateRequest) -> GenerateResponse:
    adapter: _Adapter = request.app.state.adapter
    try:
        return await adapter.generate(body)
    except CliError as e:
        log.warning("CLI invocation failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=Problem(
                type="https://github.com/eugene-plexus/hemisphere-driver#cli-error",
                title="Backend CLI error",
                status=502,
                detail=str(e),
                component=f"hemisphere-driver:{adapter.backend_kind}",
            ).model_dump(exclude_none=True),
        ) from e


@router.post("/v1/generate/stream")
async def generate_stream(body: GenerateRequest) -> None:
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=Problem(
            type="https://github.com/eugene-plexus/hemisphere-driver#not-implemented",
            title="Not Implemented",
            status=501,
            detail=(
                "POST /v1/generate/stream is not yet wired up; will land "
                "alongside the orchestrator + UI consumers in v0.2."
            ),
            component="hemisphere-driver",
        ).model_dump(exclude_none=True),
    )

"""POST /v1/generate and POST /v1/generate/stream — stubs returning 501.

Real subprocess plumbing for the CLI adapters lands in the next commit.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from .._generated.models import GenerateRequest, GenerateResponse, Problem

router = APIRouter(tags=["inference"])


def _not_implemented(operation: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=Problem(
            # TODO(specs): Problem.type is generated as AnyUrl because the spec
            # uses `format: uri`. Pydantic accepts strings here at runtime;
            # plain string is more ergonomic for callers. Either drop the
            # format hint or model it as `string` in a follow-up specs PR.
            type="https://github.com/eugene-plexus/hemisphere-driver#not-implemented",  # type: ignore[arg-type]
            title="Not Implemented",
            status=501,
            detail=f"{operation} is not yet wired up; v0.1 scaffolding only.",
            component="hemisphere-driver",
        ).model_dump(),
    )


@router.post("/v1/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    raise _not_implemented("POST /v1/generate")


@router.post("/v1/generate/stream")
async def generate_stream(request: GenerateRequest) -> None:
    raise _not_implemented("POST /v1/generate/stream")

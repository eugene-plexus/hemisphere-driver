"""GET /healthz — liveness / readiness probe."""

from __future__ import annotations

from fastapi import APIRouter, Request

from .. import __version__
from .._generated.models import Health, Status

router = APIRouter(tags=["meta"])


@router.get("/healthz", response_model=Health)
async def healthz(request: Request) -> Health:
    # The driver always serves /healthz so config endpoints stay reachable.
    # When adapter init failed at startup, surface that as `degraded` so
    # consumers (UI, orchestrator) can tell the driver is alive but can't
    # generate anything until the operator fixes config.
    adapter = getattr(request.app.state, "adapter", None)
    adapter_error = getattr(request.app.state, "adapter_error", None)

    if adapter is None:
        return Health(
            status=Status.degraded,
            version=__version__,
            component="hemisphere-driver",
            details={"adapter_error": adapter_error},
        )

    return Health(
        status=Status.ok,
        version=__version__,
        component="hemisphere-driver",
    )

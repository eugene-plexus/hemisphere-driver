"""FastAPI app factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Protocol

from fastapi import FastAPI

from . import __version__
from ._generated.models import GenerateRequest, GenerateResponse
from .adapters.claude_code_cli import ClaudeCodeCliAdapter
from .adapters.codex_cli import CodexCliAdapter
from .config import ConfigStore
from .routes import config as config_routes
from .routes import generate as generate_routes
from .routes import health as health_routes
from .routes import info as info_routes
from .settings import Settings, load_settings


class _Adapter(Protocol):
    backend_kind: str

    async def generate(self, request: GenerateRequest) -> GenerateResponse: ...


def build_adapter(store: ConfigStore) -> _Adapter:
    """Construct the configured adapter from the runtime config store."""
    adapter_kind = str(store.get("adapter") or "")
    model_id = store.get("modelId") or None
    timeout = float(store.get("requestTimeoutSeconds") or 120)

    if adapter_kind == "claude_code_cli":
        return ClaudeCodeCliAdapter(
            binary_path=str(store.get("claudeCodeCliPath") or "claude"),
            model_id=model_id,
            timeout_seconds=timeout,
        )
    if adapter_kind == "codex_cli":
        return CodexCliAdapter(
            binary_path=str(store.get("codexCliPath") or "codex"),
            model_id=model_id,
            timeout_seconds=timeout,
        )
    raise ValueError(f"unknown adapter {adapter_kind!r}; valid: claude_code_cli, codex_cli")


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    store = ConfigStore(settings.config_file)
    store.load()
    app.state.config_store = store
    app.state.adapter = build_adapter(store)
    yield


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a FastAPI app with all routers mounted."""
    settings = settings or load_settings()

    app = FastAPI(
        title="Eugene Plexus — hemisphere-driver",
        description="One half of an Eugene Plexus bicameral pair.",
        version=__version__,
        lifespan=_lifespan,
    )
    app.state.settings = settings

    app.include_router(health_routes.router)
    app.include_router(info_routes.router)
    app.include_router(config_routes.router)
    app.include_router(generate_routes.router)

    return app

"""FastAPI app factory."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Protocol

from fastapi import FastAPI

from . import __version__
from ._generated.models import GenerateRequest, GenerateResponse
from .adapters.claude_code_cli import ClaudeCodeCliAdapter
from .adapters.codex_cli import CodexCliAdapter
from .adapters.openai_api import OpenAiApiAdapter
from .config import ConfigStore
from .routes import config as config_routes
from .routes import generate as generate_routes
from .routes import health as health_routes
from .routes import info as info_routes
from .settings import Settings, load_settings

log = logging.getLogger(__name__)


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
    if adapter_kind == "openai_api":
        return OpenAiApiAdapter(
            api_key=str(store.get("openaiApiKey") or "") or None,
            base_url=str(store.get("openaiBaseUrl") or "https://api.openai.com"),
            model_id=model_id or "gpt-5",
            timeout_seconds=timeout,
        )
    raise ValueError(
        f"unknown adapter {adapter_kind!r}; valid: claude_code_cli, codex_cli, openai_api"
    )


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    store = ConfigStore(settings.config_file)
    store.load()
    app.state.config_store = store

    # Adapter construction can fail (missing API key, unknown adapter kind,
    # bad binary path, etc). The driver MUST come up anyway so its
    # /v1/config endpoints stay reachable — otherwise a broken config locks
    # operators out of fixing it through the UI, exactly the OpenClaw
    # failure mode this project exists to avoid. We record the error on
    # app.state and let /v1/generate surface it as a 503 until the config
    # is fixed and the driver restarted.
    try:
        app.state.adapter = build_adapter(store)
        app.state.adapter_error = None
        log.info("adapter ready: %s", app.state.adapter.backend_kind)
    except Exception as e:
        app.state.adapter = None
        app.state.adapter_error = str(e)
        log.error(
            "adapter initialization failed (%s); driver running in degraded "
            "mode — fix config via /v1/config and restart",
            e,
        )

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

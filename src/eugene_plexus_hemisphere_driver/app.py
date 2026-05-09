"""FastAPI app factory."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from . import __version__
from .config import ConfigStore
from .routes import config as config_routes
from .routes import generate as generate_routes
from .routes import health as health_routes
from .routes import info as info_routes
from .settings import Settings, load_settings


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    store = ConfigStore(settings.config_file)
    store.load()
    app.state.config_store = store
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

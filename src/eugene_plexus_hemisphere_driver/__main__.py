"""Entrypoint: `python -m eugene_plexus_hemisphere_driver`."""

from __future__ import annotations

import uvicorn

from .app import create_app
from .config import ConfigStore
from .settings import load_settings


def main() -> None:
    settings = load_settings()

    # Read the persisted port from config without needing the FastAPI lifespan.
    bootstrap_store = ConfigStore(settings.config_file)
    bootstrap_store.load()
    port = int(bootstrap_store.get("port") or 8081)

    app = create_app(settings)
    uvicorn.run(
        app,
        host=settings.bind_host,
        port=port,
        log_level=str(bootstrap_store.get("logLevel") or "INFO").lower(),
    )


if __name__ == "__main__":
    main()

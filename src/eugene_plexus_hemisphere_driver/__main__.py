"""Entrypoint: `python -m eugene_plexus_hemisphere_driver`."""

from __future__ import annotations

import os

import uvicorn

from .app import create_app
from .config import ConfigStore
from .settings import load_settings

# Default bind port when neither the watchdog (via env var) nor the
# operator (via standalone launch) overrides it. Matches the smoke-test
# convention for the canonical bicameral pair.
_DEFAULT_PORT = 8081


def _resolve_port(bootstrap_store: ConfigStore) -> int:
    """Resolution order, highest precedence first:

    1. `EUGENE_PLEXUS_HD_BIND_PORT` env var — the watchdog sets this when
       it spawns the driver, parsed from the topology's component URL.
       Watchdog-supervised installs always hit this branch.
    2. Built-in default (8081). Used when running the driver standalone
       outside the watchdog. The `port` field used to live in the
       per-driver config file; it's gone now (one source of truth: the
       watchdog topology owns ports).
    """
    env_port = os.environ.get("EUGENE_PLEXUS_HD_BIND_PORT")
    if env_port:
        return int(env_port)
    return _DEFAULT_PORT


def main() -> None:
    settings = load_settings()

    # Bootstrap the config store just to discover log_level. Ports are
    # owned by the watchdog now (or the default for standalone launch).
    bootstrap_store = ConfigStore(settings.config_file)
    if not settings.safe_mode:
        bootstrap_store.load()

    port = _resolve_port(bootstrap_store)

    app = create_app(settings)
    uvicorn.run(
        app,
        host=settings.bind_host,
        port=port,
        log_level=str(bootstrap_store.get("logLevel") or "INFO").lower(),
    )


if __name__ == "__main__":
    main()

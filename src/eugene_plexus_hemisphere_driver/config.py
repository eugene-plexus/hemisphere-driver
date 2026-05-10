"""Runtime configuration: schema declaration + file-backed state + PATCH apply.

Implements the shared Eugene Plexus config protocol:

* `GET /v1/config/schema` -> field metadata for UI rendering (`as_schema()`)
* `GET /v1/config` -> current effective values, secrets redacted (`as_document()`)
* `PATCH /v1/config` -> partial update, per-key validation (`apply_patch()`)

Storage backend in v0.1 is a flat YAML file. Sensitive values are stored
plain on disk for now; at-rest encryption is future work.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import yaml

from ._generated.models import (
    ConfigDocument,
    ConfigField,
    ConfigFieldError,
    ConfigFieldShowWhen,
    ConfigSchema,
    ConfigUpdateRequest,
    ConfigUpdateResult,
    ConfigValueType,
)

REDACTED = "<redacted>"

CATEGORY_LABELS: dict[str, str] = {
    "adapter": "Backend Adapter",
    "network": "Network",
    "logging": "Logging",
}

# Schema for hemisphere-driver's config surface. The order here is the
# order the UI will render the fields in.
FIELDS: list[ConfigField] = [
    ConfigField(
        key="adapter",
        label="Backend",
        description="Which model backend this hemisphere wraps.",
        category="adapter",
        valueType=ConfigValueType.enum,
        default="claude_code_cli",
        enumValues=["claude_code_cli", "codex_cli", "openai_api"],
        required=True,
        requiresRestart=True,
    ),
    ConfigField(
        key="hemisphere",
        label="Hemisphere",
        description="Which side of the bicameral pair this instance is.",
        category="adapter",
        valueType=ConfigValueType.enum,
        default="left",
        enumValues=["left", "right"],
        required=True,
    ),
    ConfigField(
        key="modelId",
        label="Model ID",
        description=(
            'Backend-specific model identifier (e.g. "claude-opus-4-7"). '
            "Optional — adapter uses its built-in default if unset."
        ),
        category="adapter",
        valueType=ConfigValueType.string,
        requiresRestart=True,
    ),
    ConfigField(
        key="claudeCodeCliPath",
        label="Claude Code CLI binary",
        description="Path to the `claude` executable. Used when adapter is claude_code_cli.",
        category="adapter",
        valueType=ConfigValueType.file_path,
        default="claude",
        requiresRestart=True,
        showWhen=ConfigFieldShowWhen(key="adapter", equals="claude_code_cli"),
    ),
    ConfigField(
        key="codexCliPath",
        label="Codex CLI binary",
        description="Path to the `codex` executable. Used when adapter is codex_cli.",
        category="adapter",
        valueType=ConfigValueType.file_path,
        default="codex",
        requiresRestart=True,
        showWhen=ConfigFieldShowWhen(key="adapter", equals="codex_cli"),
    ),
    ConfigField(
        key="openaiApiKey",
        label="OpenAI API key",
        description=(
            "API key sent as `Authorization: Bearer ...`. Used when adapter is "
            "openai_api. Falls back to the OPENAI_API_KEY env var when unset."
        ),
        category="adapter",
        valueType=ConfigValueType.secret,
        sensitive=True,
        requiresRestart=True,
        showWhen=ConfigFieldShowWhen(key="adapter", equals="openai_api"),
    ),
    ConfigField(
        key="openaiBaseUrl",
        label="OpenAI base URL",
        description=(
            "Base URL of the OpenAI-compatible HTTP API. Default targets OpenAI; "
            "override for OpenAI-compatible providers (Together, Groq, MiniMax, "
            "vLLM, LM Studio, etc)."
        ),
        category="adapter",
        valueType=ConfigValueType.url,
        default="https://api.openai.com",
        requiresRestart=True,
        showWhen=ConfigFieldShowWhen(key="adapter", equals="openai_api"),
    ),
    ConfigField(
        key="port",
        label="HTTP Port",
        description="Port to listen on.",
        category="network",
        valueType=ConfigValueType.integer,
        default=8081,
        minimum=1,
        maximum=65535,
        requiresRestart=True,
    ),
    ConfigField(
        key="logLevel",
        label="Log Level",
        description=(
            "Logging verbosity. Read by uvicorn at startup; restart required "
            "for the new level to take effect."
        ),
        category="logging",
        valueType=ConfigValueType.enum,
        default="INFO",
        enumValues=["DEBUG", "INFO", "WARNING", "ERROR"],
        requiresRestart=True,
    ),
    ConfigField(
        key="requestTimeoutSeconds",
        label="Request Timeout",
        description=(
            "Maximum seconds the driver will wait for one backend call "
            "(HTTP request or CLI invocation) before aborting and returning "
            "an error. Baked into the adapter at startup; restart required."
        ),
        category="network",
        valueType=ConfigValueType.duration,
        default=120,
        minimum=5,
        maximum=900,
        requiresRestart=True,
    ),
]
# NOTE on what's NOT in this schema:
# Temperature, max-tokens, stop sequences and other parameters that alter
# LLM output are owned by the *caller* (the orchestrator) and arrive on
# every `GenerateRequest`. The driver applies what it's given and never
# substitutes a local default. In v0.2+ the orchestrator's NT system will
# modulate these per-request — placing defaults here would make that
# layering invisible and a future NT signal trivially overridable from a
# config file.

_FIELDS_BY_KEY: dict[str, ConfigField] = {f.key: f for f in FIELDS}


def as_schema() -> ConfigSchema:
    return ConfigSchema(
        component="hemisphere-driver",
        fields=FIELDS,
        categories=CATEGORY_LABELS,
    )


def _defaults() -> dict[str, Any]:
    return {f.key: f.default for f in FIELDS if f.default is not None}


def _validate_value(field: ConfigField, value: Any) -> str | None:
    """Return None if valid, otherwise an error message."""
    if value is None:
        return None  # null clears to default

    vt = field.valueType

    if vt == ConfigValueType.string or vt == ConfigValueType.url or vt == ConfigValueType.file_path:
        if not isinstance(value, str):
            return f"expected string, got {type(value).__name__}"
        if field.pattern is not None:
            import re

            if re.search(field.pattern, value) is None:
                return f"value does not match pattern {field.pattern!r}"
        return None

    if vt == ConfigValueType.secret:
        if not isinstance(value, str):
            return f"expected string, got {type(value).__name__}"
        if value == REDACTED:
            return "refusing to write the literal redacted value back"
        return None

    if vt == ConfigValueType.integer:
        if isinstance(value, bool) or not isinstance(value, int):
            return f"expected integer, got {type(value).__name__}"
        if field.minimum is not None and value < field.minimum:
            return f"must be >= {field.minimum}"
        if field.maximum is not None and value > field.maximum:
            return f"must be <= {field.maximum}"
        return None

    if vt == ConfigValueType.number or vt == ConfigValueType.duration:
        if isinstance(value, bool) or not isinstance(value, int | float):
            return f"expected number, got {type(value).__name__}"
        if field.minimum is not None and value < field.minimum:
            return f"must be >= {field.minimum}"
        if field.maximum is not None and value > field.maximum:
            return f"must be <= {field.maximum}"
        return None

    if vt == ConfigValueType.boolean:
        if not isinstance(value, bool):
            return f"expected boolean, got {type(value).__name__}"
        return None

    if vt == ConfigValueType.enum:
        if not isinstance(value, str):
            return f"expected string, got {type(value).__name__}"
        allowed = field.enumValues or []
        if value not in allowed:
            return f"must be one of {allowed}"
        return None

    return f"unsupported valueType: {vt}"


class ConfigStore:
    """File-backed config state. Thread-safe for the simple read/write pattern."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._values: dict[str, Any] = _defaults()
        self._pending_restart: set[str] = set()

    def load(self) -> None:
        """Load from the configured file, creating it with defaults if absent."""
        with self._lock:
            if self._path.exists():
                raw = yaml.safe_load(self._path.read_text(encoding="utf-8")) or {}
                if not isinstance(raw, dict):
                    raise ValueError(f"config file {self._path} must be a YAML mapping at the root")
                merged = _defaults()
                for k, v in raw.items():
                    if k in _FIELDS_BY_KEY:
                        merged[k] = v
                self._values = merged
            else:
                self._values = _defaults()
                self._write_locked()

    def as_document(self) -> ConfigDocument:
        with self._lock:
            out: dict[str, Any] = {}
            for key, value in self._values.items():
                field = _FIELDS_BY_KEY.get(key)
                if field is not None and field.sensitive and value is not None:
                    out[key] = REDACTED
                else:
                    out[key] = value
            return ConfigDocument.model_validate(out)

    def apply_patch(self, request: ConfigUpdateRequest) -> ConfigUpdateResult:
        applied: list[str] = []
        rejected: list[ConfigFieldError] = []
        pending_restart: list[str] = []

        # ConfigUpdateRequest is a free-form mapping; iterate its raw dict form.
        patch: dict[str, Any] = request.model_dump()

        with self._lock:
            for key, new_value in patch.items():
                field = _FIELDS_BY_KEY.get(key)
                if field is None:
                    rejected.append(ConfigFieldError(key=key, message="unknown field"))
                    continue

                err = _validate_value(field, new_value)
                if err is not None:
                    rejected.append(ConfigFieldError(key=key, message=err))
                    continue

                if new_value is None and field.default is not None:
                    self._values[key] = field.default
                else:
                    self._values[key] = new_value

                applied.append(key)
                if field.requiresRestart:
                    self._pending_restart.add(key)
                    pending_restart.append(key)

            if applied:
                self._write_locked()

            requires_restart = bool(self._pending_restart)
            return ConfigUpdateResult(
                applied=applied,
                rejected=rejected,
                requiresRestart=requires_restart,
                pendingRestart=sorted(self._pending_restart),
            )

    def get(self, key: str) -> Any:
        with self._lock:
            return self._values.get(key)

    def _write_locked(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self._values, f, sort_keys=True, default_flow_style=False)

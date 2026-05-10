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
        description=(
            "Which kind of LLM this driver talks to. "
            "`claude_code_cli` shells out to your locally-installed Claude "
            "Code CLI (uses your Claude Pro/Max subscription — no API "
            "billing). `codex_cli` shells out to OpenAI's Codex CLI "
            "(same idea, ChatGPT subscription). `openai_api` calls any "
            "OpenAI-compatible HTTP API directly with an API key — "
            "OpenAI itself, or self-hosted servers like Ollama, vLLM, "
            "or LM Studio. Switching this changes which other fields "
            "below are relevant."
        ),
        category="adapter",
        valueType=ConfigValueType.enum,
        default="claude_code_cli",
        enumValues=["claude_code_cli", "codex_cli", "openai_api"],
        required=True,
        requiresRestart=True,
    ),
    ConfigField(
        key="modelId",
        label="Model ID",
        description=(
            "Specific model name to ask the backend for (e.g. "
            "\"claude-opus-4-7\", \"gpt-5\", \"llama3.1:70b\"). Leave "
            "blank to use the adapter's built-in default — fine for "
            "personal-subscription CLIs, but you'll usually want to pin "
            "a specific model when calling an API."
        ),
        category="adapter",
        valueType=ConfigValueType.string,
        requiresRestart=True,
    ),
    ConfigField(
        key="claudeCodeCliPath",
        label="Claude Code CLI binary",
        description=(
            "Where to find the `claude` command. Just `claude` works if "
            "the binary is on your `PATH`; otherwise give the full path "
            "(e.g. `/usr/local/bin/claude` or `C:\\Users\\you\\AppData"
            "\\Local\\claude\\claude.exe`)."
        ),
        category="adapter",
        valueType=ConfigValueType.file_path,
        default="claude",
        requiresRestart=True,
        showWhen=ConfigFieldShowWhen(key="adapter", equals="claude_code_cli"),
    ),
    ConfigField(
        key="codexCliPath",
        label="Codex CLI binary",
        description=(
            "Where to find the `codex` command. Just `codex` works if "
            "the binary is on your `PATH`; otherwise give the full path."
        ),
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
            "Secret key sent as the `Authorization: Bearer ...` header on "
            "every API call. Get one from the provider you're using "
            "(OpenAI, Groq, Together, your self-hosted server, etc.). "
            "If you leave this blank, the driver will fall back to the "
            "`OPENAI_API_KEY` environment variable in the shell that "
            "started it. Stored on disk in plain text in v0.1; "
            "at-rest encryption is on the v0.2 list."
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
            "Where to send the API calls. The default points at OpenAI's "
            "servers. Change it to use any OpenAI-compatible provider — "
            "for example: `https://api.groq.com/openai` for Groq, "
            "`https://api.together.xyz` for Together, "
            "`http://127.0.0.1:11434` for a local Ollama, "
            "`http://127.0.0.1:8000` for a local vLLM. The "
            "`/v1/chat/completions` path is appended automatically."
        ),
        category="adapter",
        valueType=ConfigValueType.url,
        default="https://api.openai.com",
        requiresRestart=True,
        showWhen=ConfigFieldShowWhen(key="adapter", equals="openai_api"),
    ),
    ConfigField(
        key="port",
        label="HTTP port",
        description=(
            "Port the orchestrator (or any other client) connects to "
            "*this driver* on. The driver listens on it; nothing outside "
            "this process. v0.1 default leaves the canonical bicameral "
            "pair on 8081 (left) and 8082 (right). Change only if those "
            "are taken on your machine."
        ),
        category="network",
        valueType=ConfigValueType.integer,
        default=8081,
        minimum=1,
        maximum=65535,
        requiresRestart=True,
    ),
    ConfigField(
        key="logLevel",
        label="Log level",
        description=(
            "How chatty the driver's terminal output is. `DEBUG` prints "
            "every backend call (useful when something's broken); "
            "`INFO` is the normal operating level; `WARNING` and "
            "`ERROR` go progressively quieter."
        ),
        category="logging",
        valueType=ConfigValueType.enum,
        default="INFO",
        enumValues=["DEBUG", "INFO", "WARNING", "ERROR"],
        requiresRestart=True,
    ),
    ConfigField(
        key="requestTimeoutSeconds",
        label="Backend timeout",
        description=(
            "How long the driver will wait on a single LLM call before "
            "giving up and returning an error. Long-running CLI "
            "invocations and large reasoning models can take a while; "
            "120s is the v0.1 default. Bump it if you see timeouts on "
            "complex prompts."
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


def as_schema(*, available_models: list[str] | None = None) -> ConfigSchema:
    """Return the driver's schema, with `modelId` upgraded to an enum
    dropdown when the caller supplies a discovered model list.

    The list comes from the adapter's `list_models()` (live for
    openai_api, hardcoded for the CLIs) and arrives at the schema
    endpoint via `app.state.available_models`. When it's missing or
    empty (degraded mode, unreachable upstream, etc.), `modelId`
    stays as a free-text `string` input so the operator can still
    type a value by hand.
    """
    fields = list(FIELDS)
    if available_models:
        fields = [
            _with_model_dropdown(f, available_models) if f.key == "modelId" else f
            for f in fields
        ]
    return ConfigSchema(
        component="hemisphere-driver",
        fields=fields,
        categories=CATEGORY_LABELS,
    )


def _with_model_dropdown(model_field: ConfigField, models: list[str]) -> ConfigField:
    """Return a copy of `modelId` re-typed as an enum with the given
    models as `enumValues`. An empty-string entry is prepended so the
    UI can offer "(use adapter default)" — preserving the current
    behavior where leaving modelId unset falls back to the adapter's
    built-in default model."""
    return model_field.model_copy(
        update={
            "valueType": ConfigValueType.enum,
            "enumValues": ["", *models],
        }
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

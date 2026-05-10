"""HemisphereAdapter — uniform contract every backend implementation honors.

Adapters wrap a single LLM backend (Anthropic API, OpenAI API, Claude Code
CLI, Codex CLI, OpenAI-compatible HTTP). They are stateless: the orchestrator
owns conversation state and passes the full prompt every call. The adapter's
job is shape adaptation between Eugene Plexus's wire types and whatever the
backend speaks.

v0.1 ships with two CLI adapters (`claude_code_cli`, `codex_cli`); both are
stubs in the scaffolding commit and will be filled in next.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from .._generated.models import GenerateRequest, GenerateResponse


class StreamChunk(Protocol):
    """One event in the SSE stream emitted by `HemisphereAdapter.stream`."""

    text: str
    """The newly-generated text fragment for this event. Empty for non-token events."""

    done: bool
    """If True, this is the final event; `result` is set."""

    result: GenerateResponse | None
    """Set on the final event, capturing the assembled response and usage."""


class HemisphereAdapter(Protocol):
    """The interface every backend implementation honors.

    Adapters are constructed once at process startup based on the configured
    `adapter` field. They are stateless across requests.
    """

    backend_kind: str
    """One of the values in `BackendKind` (see specs/common.yaml)."""

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        """Single-shot generation. Returns the full response."""
        ...

    async def stream(self, request: GenerateRequest) -> AsyncIterator[StreamChunk]:
        """Streamed generation. Yields chunks ending with one where `done` is True."""
        ...

    async def list_models(self) -> list[str]:
        """Return the model IDs this backend offers, post-policy-filter.

        Used by `GET /v1/config/schema` to populate the `modelId` field's
        `enumValues` so the UI can render a dropdown instead of a free-text
        input. The list is **already filtered** for Eugene Plexus
        compatibility (no temperature-uncontrollable models, etc.) — the
        UI takes it as authoritative.

        For adapters that talk to a discoverable API (openai_api), this
        SHOULD live-fetch from the backend. For CLI adapters where the
        backend doesn't expose a list endpoint, return a hardcoded set
        of known-good model IDs.

        Failure is non-fatal: return `[]` if the backend can't be
        reached. Callers fall back to a free-text input so the operator
        can still type a model ID by hand.
        """
        ...

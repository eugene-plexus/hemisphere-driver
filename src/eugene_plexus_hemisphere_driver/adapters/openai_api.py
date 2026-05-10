"""Adapter for OpenAI-compatible HTTP APIs.

Speaks the well-known `POST /v1/chat/completions` shape, so it works
against:

  - OpenAI itself (`https://api.openai.com`)
  - Together, Groq, Fireworks, DeepInfra, MiniMax, etc — set `base_url`
    to the provider's OpenAI-compatible endpoint
  - Local OpenAI-compat servers (vLLM, LM Studio, Ollama, llama.cpp's
    server) — set `base_url` to the local address

Unlike the CLI adapters, this one talks to a raw model API, so the
system prompt actually controls model behavior. No agentic identity
to subvert; no working-directory injection; no subprocess argv quoting
games. This is the right side of the bicameral pair when paired with
Claude Code on the left.

Auth: an `Authorization: Bearer <api_key>` header. The API key is read
from config (`openaiApiKey`, sensitive) with a fallback to the
`OPENAI_API_KEY` env var.
"""

from __future__ import annotations

import os
import re
from collections.abc import AsyncIterator
from typing import Any

import httpx

from .._generated.models import (
    BackendKind,
    FinishReason,
    GenerateRequest,
    GenerateResponse,
    Role,
    Usage,
)
from ._subprocess import CliError

# Models that can't be used as a Eugene Plexus hemisphere because they
# don't actually support a tunable `temperature`:
#
# - OpenAI o-series (o1, o3, and their *-mini / *-preview variants):
#   reject the `temperature` parameter outright.
# - gpt-5 family: schema accepts the parameter but only the default
#   value (1) — any other value 400s with "Unsupported value:
#   'temperature' does not support X with this model. Only the default
#   (1) value is supported."
#
# Eugene Plexus uses temperature as the per-pass divergence knob
# between hemispheres, and v0.2's NT system will modulate it per-pass
# per-driver. A model that won't move with temperature has no place in
# the loop. The adapter refuses to construct against one of these,
# dropping the driver into degraded mode with a clear message.
_TEMPERATURE_UNCONTROLLABLE_PATTERN = re.compile(
    # Prefix-anchored: matches anything in the gpt-5 family
    # (gpt-5, gpt-5-mini, gpt-5.1, gpt-5.5-pro-2026-04-23, gpt-5.1-codex, …)
    # or the o-series (o1, o3, o1-preview-2024-09, …). The trailing
    # `(?:[-.]|$)` insists the match ends at a version separator or
    # end-of-string so we don't accidentally hit hypothetical
    # `gpt-50` / `gpt-5o` style names that aren't actually 5.x.
    r"^(?:o\d+|gpt-5)(?:[-.]|$)",
    re.IGNORECASE,
)


def _model_rejects_temperature_control(model_id: str) -> bool:
    return _TEMPERATURE_UNCONTROLLABLE_PATTERN.match(model_id) is not None


# OpenAI's `/v1/models` returns every model on the account — embeddings,
# image, audio, moderation, retired completion-only models, etc. — and
# the API doesn't tag them by capability. Heuristic-filter to chat-likely
# IDs so the dropdown isn't 80 entries of `text-embedding-3-large`.
_NON_CHAT_PREFIXES = (
    "text-embedding-",
    "text-similarity-",
    "text-search-",
    "code-search-",
    "dall-e-",
    "whisper-",
    "tts-",
    "omni-moderation-",
    "babbage",
    "davinci",
    "ada-",
    "curie-",
    "computer-use-",
    "codex-",  # the standalone Codex models — different surface from `codex_cli`
)

_CHAT_MODEL_PREFIXES = (
    "gpt-",
    "chatgpt-",
    "claude-",
    "llama",
    "mistral",
    "qwen",
    "deepseek",
)


def _is_plausible_chat_model(model_id: str) -> bool:
    lowered = model_id.lower()
    if lowered.startswith(_NON_CHAT_PREFIXES):
        return False
    if "embedding" in lowered or "moderation" in lowered:
        return False
    if "audio" in lowered or "realtime" in lowered or "transcribe" in lowered or "tts" in lowered:
        return False
    if "image" in lowered or "vision-preview" in lowered:
        # Pure image-gen models. Vision-capable chat models (e.g. gpt-4o)
        # don't carry "image" in the id, so they're unaffected.
        return False
    return lowered.startswith(_CHAT_MODEL_PREFIXES)

_FINISH_REASON_MAP = {
    "stop": FinishReason.stop,
    "length": FinishReason.length,
    "content_filter": FinishReason.error,
    "tool_calls": FinishReason.stop,
    "function_call": FinishReason.stop,
}


def _max_tokens_field_for(base_url: str) -> str:
    """Pick the right output-cap field name for this base URL.

    OpenAI's chat-completions API now requires `max_completion_tokens`
    for newer models (gpt-5, o1, o3, etc.) and explicitly rejects
    `max_tokens`:

        Unsupported parameter: 'max_tokens' is not supported with this
        model. Use 'max_completion_tokens' instead.

    Self-hosted OpenAI-compatible servers (Ollama, vLLM, LM Studio,
    llama.cpp) implement the older spec and only understand the
    legacy `max_tokens` field. Pick by base URL: openai.com → new
    field; anything else → legacy. If you point this adapter at a
    third-party provider that has also migrated to
    `max_completion_tokens` and your requests start 400'ing on the
    legacy field, log a bug — we'll either expand the URL match list
    or move to a try/fallback strategy.
    """
    return "max_completion_tokens" if "openai.com" in base_url.lower() else "max_tokens"


class OpenAiApiAdapter:
    backend_kind = "openai_api"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com",
        model_id: str = "gpt-4o",
        timeout_seconds: float = 120.0,
    ) -> None:
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise CliError(
                "openai_api adapter has no API key — set `openaiApiKey` in "
                "config or export OPENAI_API_KEY."
            )
        if _model_rejects_temperature_control(model_id):
            raise CliError(
                f"openai_api: model {model_id!r} doesn't support a tunable "
                "`temperature` parameter (OpenAI's o-series rejects it "
                "outright; the gpt-5 family schema-accepts it but only the "
                "default value of 1 — any other value 400s). Eugene Plexus "
                "uses temperature as the per-pass divergence knob between "
                "hemispheres, and v0.2's NT system modulates it per-pass "
                "per-driver — a model that won't move with temperature has "
                "no place in the bicameral loop. Pick a non-reasoning chat "
                "model with controllable temperature: gpt-4o, gpt-4o-mini, "
                "gpt-4.1, gpt-4-turbo."
            )
        self._api_key = resolved_key
        self._base_url = base_url.rstrip("/")
        self._model_id = model_id
        self._timeout_seconds = timeout_seconds

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        payload: dict[str, Any] = {
            "model": self._model_id,
            "messages": _to_openai_messages(list(request.messages)),
        }
        if request.maxTokens is not None:
            payload[_max_tokens_field_for(self._base_url)] = request.maxTokens
        if request.temperature is not None:
            payload["temperature"] = float(request.temperature)
        if request.stop:
            payload["stop"] = list(request.stop)

        async with httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(self._timeout_seconds, connect=10.0),
        ) as client:
            try:
                response = await client.post(
                    "/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Accept": "application/json",
                    },
                    json=payload,
                )
            except httpx.HTTPError as e:
                raise CliError(f"openai_api request failed: {e}") from e

        if response.status_code >= 400:
            raise CliError(
                f"openai_api returned {response.status_code}: {_redact(response.text[:500])}"
            )

        try:
            body = response.json()
        except ValueError as e:
            raise CliError(f"openai_api returned non-JSON: {response.text[:200]!r}") from e

        choices = body.get("choices") or []
        if not choices:
            raise CliError(f"openai_api returned no choices: {body!r}")

        first = choices[0] or {}
        message = first.get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            raise CliError(f"openai_api response missing string content: {body!r}")

        return GenerateResponse(
            content=content,
            finishReason=_FINISH_REASON_MAP.get(
                str(first.get("finish_reason") or "stop"), FinishReason.stop
            ),
            usage=_usage_from_envelope(body.get("usage") or {}),
            requestId=request.requestId,
            backend=BackendKind.openai_api,
            modelId=str(body.get("model") or self._model_id),
            latencyMs=int(response.elapsed.total_seconds() * 1000),
        )

    async def stream(self, request: GenerateRequest) -> AsyncIterator[object]:
        # OpenAI Chat Completions supports SSE streaming via stream=true,
        # but no consumer of hemisphere-driver streaming exists yet.
        raise NotImplementedError("OpenAiApiAdapter.stream not implemented in v0.1")
        yield  # pragma: no cover

    async def list_models(self) -> list[str]:
        """Live-fetch the model catalog from the configured base URL.

        OpenAI proper and most OpenAI-compatible providers expose
        `GET /v1/models` returning `{"data": [{"id": "<model>", ...}]}`.
        We filter out:

        - non-chat models (embeddings, dall-e, whisper, tts, moderation,
          codex-only, audio) — heuristic by id prefix / substring
        - models we'd reject at adapter construction anyway
          (`_TEMPERATURE_UNCONTROLLABLE_PATTERN`)

        Returns `[]` on transport / parse failure so the schema endpoint
        can fall back to free-text input. Driver stays up either way.
        """
        try:
            async with httpx.AsyncClient(
                base_url=self._base_url,
                timeout=httpx.Timeout(15.0, connect=5.0),
            ) as client:
                response = await client.get(
                    "/v1/models",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Accept": "application/json",
                    },
                )
            if response.status_code >= 400:
                return []
            body = response.json()
        except (httpx.HTTPError, ValueError):
            return []
        data = body.get("data") if isinstance(body, dict) else None
        if not isinstance(data, list):
            return []
        ids: list[str] = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            mid = entry.get("id")
            if not isinstance(mid, str):
                continue
            if not _is_plausible_chat_model(mid):
                continue
            if _model_rejects_temperature_control(mid):
                continue
            ids.append(mid)
        ids.sort()
        return ids


def _to_openai_messages(messages: list[Any]) -> list[dict[str, str]]:
    """Map our Message[] to OpenAI chat-completions messages.

    OpenAI's role taxonomy is system / user / assistant. We collapse our
    `hemisphere` role into `assistant` (those messages came from a
    hemisphere on a previous pass and act as prior assistant turns).
    """
    out: list[dict[str, str]] = []
    for m in messages:
        if m.role == Role.system:
            out.append({"role": "system", "content": m.content})
        elif m.role == Role.user:
            out.append({"role": "user", "content": m.content})
        elif m.role == Role.assistant or m.role == Role.hemisphere:
            out.append({"role": "assistant", "content": m.content})
        else:
            out.append({"role": "user", "content": m.content})
    return out


def _usage_from_envelope(usage: dict[str, Any]) -> Usage | None:
    if not usage:
        return None
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    total = usage.get("total_tokens")
    if prompt is None and completion is None:
        return None
    return Usage(
        promptTokens=prompt or 0,
        completionTokens=completion or 0,
        totalTokens=total if total is not None else (prompt or 0) + (completion or 0),
    )


def _redact(text: str) -> str:
    """Best-effort redaction of API keys that might appear in error bodies."""
    return text.replace("Bearer ", "Bearer <redacted>")

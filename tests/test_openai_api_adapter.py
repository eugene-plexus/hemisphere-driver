"""Unit tests for the OpenAI-compatible HTTP adapter.

Mocks the upstream HTTP via `respx` so these tests don't hit OpenAI.
End-to-end verification against a real key is gated behind
EUGENE_PLEXUS_HD_LIVE_API=1 and skipped by default.
"""

from __future__ import annotations

import os

import httpx
import pytest
import respx

from eugene_plexus_hemisphere_driver._generated.models import (
    BackendKind,
    FinishReason,
    GenerateRequest,
    Message,
    Role,
)
from eugene_plexus_hemisphere_driver.adapters._subprocess import CliError
from eugene_plexus_hemisphere_driver.adapters.openai_api import OpenAiApiAdapter


def _request(prompt: str = "say PING", system: str | None = None) -> GenerateRequest:
    messages = []
    if system is not None:
        messages.append(Message(role=Role.system, content=system))
    messages.append(Message(role=Role.user, content=prompt))
    return GenerateRequest(messages=messages)


OK_BODY = {
    "id": "chatcmpl-abc",
    "object": "chat.completion",
    "created": 1234567890,
    "model": "gpt-5",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "PING"},
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 12,
        "completion_tokens": 1,
        "total_tokens": 13,
    },
}


@respx.mock
async def test_openai_adapter_parses_chat_completion() -> None:
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=OK_BODY)
    )
    adapter = OpenAiApiAdapter(api_key="sk-test", model_id="gpt-5", timeout_seconds=30.0)

    response = await adapter.generate(_request(system="You are helpful."))

    assert response.content == "PING"
    assert response.finishReason == FinishReason.stop
    assert response.backend == BackendKind.openai_api
    assert response.modelId == "gpt-5"
    assert response.usage is not None
    assert response.usage.promptTokens == 12
    assert response.usage.completionTokens == 1

    assert route.called
    call = route.calls[0]
    sent = call.request
    assert sent.headers.get("authorization") == "Bearer sk-test"
    body = httpx._utils.URLPattern  # noqa: F841 — silencing httpx unused-import patterns
    payload = call.request.read()
    import json as _json

    parsed = _json.loads(payload)
    assert parsed["model"] == "gpt-5"
    assert parsed["messages"] == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "say PING"},
    ]


@respx.mock
async def test_openai_adapter_uses_configured_base_url() -> None:
    route = respx.post("https://api.minimax.io/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=OK_BODY)
    )
    adapter = OpenAiApiAdapter(
        api_key="sk-test",
        base_url="https://api.minimax.io",
        model_id="abab6.5-chat",
    )
    await adapter.generate(_request())
    assert route.called


@respx.mock
async def test_openai_adapter_raises_on_http_error() -> None:
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(401, json={"error": {"message": "invalid key"}})
    )
    adapter = OpenAiApiAdapter(api_key="sk-bad")
    with pytest.raises(CliError, match="401"):
        await adapter.generate(_request())


@respx.mock
async def test_openai_adapter_raises_when_no_choices() -> None:
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"choices": []})
    )
    adapter = OpenAiApiAdapter(api_key="sk-test")
    with pytest.raises(CliError, match="no choices"):
        await adapter.generate(_request())


@respx.mock
async def test_openai_adapter_collapses_hemisphere_role_to_assistant() -> None:
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=OK_BODY)
    )
    adapter = OpenAiApiAdapter(api_key="sk-test")
    await adapter.generate(
        GenerateRequest(
            messages=[
                Message(role=Role.system, content="You are Eugene."),
                Message(role=Role.user, content="hi"),
                Message(role=Role.hemisphere, content="(left side wisdom)"),
                Message(role=Role.user, content="reconsider"),
            ]
        )
    )
    import json as _json

    sent_payload = _json.loads(route.calls[0].request.read())
    roles = [m["role"] for m in sent_payload["messages"]]
    assert roles == ["system", "user", "assistant", "user"]


def test_openai_adapter_rejects_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(CliError, match="no API key"):
        OpenAiApiAdapter()


def test_openai_adapter_falls_back_to_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env")
    adapter = OpenAiApiAdapter()
    assert adapter is not None  # constructor accepts the env-var key without raising


# ---------------------------------------------------------------------------
# Live test — opt-in via env var (uses real OpenAI quota)
# ---------------------------------------------------------------------------

LIVE = os.environ.get("EUGENE_PLEXUS_HD_LIVE_API") == "1"


@pytest.mark.skipif(not LIVE, reason="set EUGENE_PLEXUS_HD_LIVE_API=1 to run live")
async def test_openai_live_call() -> None:
    adapter = OpenAiApiAdapter(timeout_seconds=60.0)
    response = await adapter.generate(
        _request(prompt="Reply with exactly the four characters: PING", system="You are concise.")
    )
    assert "PING" in response.content

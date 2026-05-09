"""Unit tests for the two CLI adapters.

Subprocess invocation is mocked at the `run_cli` boundary so these tests
don't require `claude` or `codex` to be installed. End-to-end verification
against real binaries is gated behind an env var (see the bottom of this
file) and skipped by default.
"""

from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from eugene_plexus_hemisphere_driver._generated.models import (
    BackendKind,
    FinishReason,
    GenerateRequest,
    Message,
    Role,
)
from eugene_plexus_hemisphere_driver.adapters import _subprocess
from eugene_plexus_hemisphere_driver.adapters._subprocess import CliError, CliResult
from eugene_plexus_hemisphere_driver.adapters.claude_code_cli import ClaudeCodeCliAdapter
from eugene_plexus_hemisphere_driver.adapters.codex_cli import CodexCliAdapter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _request(prompt: str = "say PING") -> GenerateRequest:
    return GenerateRequest(
        messages=[
            Message(role=Role.user, content=prompt),
        ],
    )


def _patch_run_cli(
    monkeypatch: pytest.MonkeyPatch,
    fn: Callable[[list[str]], Awaitable[CliResult] | CliResult],
) -> dict[str, Any]:
    """Replace adapters._subprocess.run_cli with `fn` and capture argv calls."""
    captured: dict[str, Any] = {"argv": None, "timeout": None}

    async def _fake_run_cli(argv: list[str], *, timeout_seconds: float) -> CliResult:
        captured["argv"] = argv
        captured["timeout"] = timeout_seconds
        result = fn(argv)
        if hasattr(result, "__await__"):
            return await result  # type: ignore[no-any-return]
        return result  # type: ignore[return-value]

    # Patch on every importing module's symbol table because we use
    # `from ._subprocess import run_cli` style imports.
    monkeypatch.setattr(_subprocess, "run_cli", _fake_run_cli)
    monkeypatch.setattr(
        "eugene_plexus_hemisphere_driver.adapters.claude_code_cli.run_cli",
        _fake_run_cli,
    )
    monkeypatch.setattr(
        "eugene_plexus_hemisphere_driver.adapters.codex_cli.run_cli",
        _fake_run_cli,
    )
    return captured


# ---------------------------------------------------------------------------
# Claude Code CLI
# ---------------------------------------------------------------------------

CLAUDE_OK_ENVELOPE = {
    "type": "result",
    "subtype": "success",
    "is_error": False,
    "result": "PING",
    "stop_reason": "end_turn",
    "duration_ms": 1942,
    "usage": {
        "input_tokens": 6,
        "output_tokens": 6,
        "cache_read_input_tokens": 33349,
        "cache_creation_input_tokens": 0,
    },
}


async def test_claude_adapter_parses_success_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_run_cli(
        monkeypatch,
        lambda argv: CliResult(
            stdout=json.dumps(CLAUDE_OK_ENVELOPE).encode(),
            stderr=b"",
            returncode=0,
            elapsed_ms=2000,
        ),
    )
    adapter = ClaudeCodeCliAdapter(model_id="claude-opus-4-7", timeout_seconds=30.0)

    response = await adapter.generate(_request())

    assert response.content == "PING"
    assert response.finishReason == FinishReason.stop
    assert response.backend == BackendKind.claude_code_cli
    assert response.modelId == "claude-opus-4-7"
    assert response.latencyMs == 2000
    assert response.usage is not None
    assert response.usage.completionTokens == 6
    # Reflects cache reads as part of prompt accounting.
    assert response.usage.promptTokens == 6 + 33349

    # argv shape: claude --print --output-format json --model <id> "<prompt>"
    assert captured["argv"][:5] == [
        "claude",
        "--print",
        "--output-format",
        "json",
        "--model",
    ]
    assert captured["argv"][5] == "claude-opus-4-7"
    assert captured["argv"][-1].startswith("[USER]")
    assert captured["timeout"] == 30.0


async def test_claude_adapter_omits_model_when_not_pinned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_run_cli(
        monkeypatch,
        lambda argv: CliResult(
            stdout=json.dumps(CLAUDE_OK_ENVELOPE).encode(),
            stderr=b"",
            returncode=0,
            elapsed_ms=1000,
        ),
    )
    adapter = ClaudeCodeCliAdapter()  # model_id=None
    await adapter.generate(_request())
    assert "--model" not in captured["argv"]


async def test_claude_adapter_raises_on_is_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    envelope = {**CLAUDE_OK_ENVELOPE, "is_error": True, "result": "Not logged in"}
    _patch_run_cli(
        monkeypatch,
        lambda argv: CliResult(
            stdout=json.dumps(envelope).encode(),
            stderr=b"",
            returncode=0,
            elapsed_ms=100,
        ),
    )
    adapter = ClaudeCodeCliAdapter()
    with pytest.raises(CliError, match="Not logged in"):
        await adapter.generate(_request())


async def test_claude_adapter_raises_on_nonzero_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_run_cli(
        monkeypatch,
        lambda argv: CliResult(
            stdout=b"",
            stderr=b"some error",
            returncode=2,
            elapsed_ms=50,
        ),
    )
    adapter = ClaudeCodeCliAdapter()
    with pytest.raises(CliError, match="exited 2"):
        await adapter.generate(_request())


# ---------------------------------------------------------------------------
# Codex CLI
# ---------------------------------------------------------------------------

CODEX_OK_STREAM = b"""
{"type":"thread.started","thread_id":"abc"}
{"type":"turn.started"}
{"type":"item.completed","item":{"type":"reasoning","summary":"thinking"}}
{"type":"item.completed","item":{"id":"item_0","type":"agent_message","text":"PING"}}
{"type":"turn.completed","usage":{"input_tokens":26820,"cached_input_tokens":6528,"output_tokens":21,"reasoning_output_tokens":14}}
""".strip()


async def test_codex_adapter_parses_jsonl_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _patch_run_cli(
        monkeypatch,
        lambda argv: CliResult(stdout=CODEX_OK_STREAM, stderr=b"", returncode=0, elapsed_ms=1500),
    )
    adapter = CodexCliAdapter(model_id="gpt-5", timeout_seconds=60.0)

    response = await adapter.generate(_request())

    assert response.content == "PING"
    assert response.finishReason == FinishReason.stop
    assert response.backend == BackendKind.codex_cli
    assert response.modelId == "gpt-5"
    assert response.latencyMs == 1500
    assert response.usage is not None
    assert response.usage.promptTokens == 26820 + 6528
    assert response.usage.completionTokens == 21 + 14

    argv = captured["argv"]
    assert argv[0] == "codex"
    assert argv[1] == "exec"
    assert "--json" in argv
    assert "--ephemeral" in argv
    assert "--skip-git-repo-check" in argv
    assert "read-only" in argv
    assert "--model" in argv and argv[argv.index("--model") + 1] == "gpt-5"


async def test_codex_adapter_raises_when_no_agent_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = b'{"type":"turn.started"}\n{"type":"turn.completed","usage":{}}\n'
    _patch_run_cli(
        monkeypatch,
        lambda argv: CliResult(stdout=stream, stderr=b"", returncode=0, elapsed_ms=10),
    )
    adapter = CodexCliAdapter()
    with pytest.raises(CliError, match="without producing an agent_message"):
        await adapter.generate(_request())


async def test_codex_adapter_concatenates_multi_message_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = (
        b'{"type":"item.completed","item":{"type":"agent_message","text":"Hello"}}\n'
        b'{"type":"item.completed","item":{"type":"agent_message","text":" world"}}\n'
        b'{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":2}}\n'
    )
    _patch_run_cli(
        monkeypatch,
        lambda argv: CliResult(stdout=stream, stderr=b"", returncode=0, elapsed_ms=10),
    )
    adapter = CodexCliAdapter()
    response = await adapter.generate(_request())
    assert response.content == "Hello world"


# ---------------------------------------------------------------------------
# End-to-end against real binaries — opt-in via env var
# ---------------------------------------------------------------------------

LIVE = os.environ.get("EUGENE_PLEXUS_HD_LIVE_CLI") == "1"


@pytest.mark.skipif(not LIVE, reason="set EUGENE_PLEXUS_HD_LIVE_CLI=1 to run live")
async def test_claude_live_call() -> None:
    adapter = ClaudeCodeCliAdapter(timeout_seconds=180.0)
    response = await adapter.generate(_request("Reply with exactly the four characters: PING"))
    assert "PING" in response.content


@pytest.mark.skipif(not LIVE, reason="set EUGENE_PLEXUS_HD_LIVE_CLI=1 to run live")
async def test_codex_live_call() -> None:
    adapter = CodexCliAdapter(timeout_seconds=180.0)
    response = await adapter.generate(_request("Reply with exactly the four characters: PING"))
    assert "PING" in response.content

"""Adapter that wraps the Anthropic Claude Code CLI as a subprocess.

Invocation pattern, verified by hand against `claude` v2.1.138:

    claude --print --output-format json [--model <id>] "<prompt>"

The CLI emits a single JSON envelope on stdout:

    {
      "type": "result",
      "subtype": "success" | "error_*",
      "is_error": false,
      "result": "<assistant response text>",
      "stop_reason": "end_turn" | "stop_sequence" | "max_tokens" | ...,
      "duration_ms": 1942,
      "duration_api_ms": 2622,
      "usage": { "input_tokens": ..., "output_tokens": ..., ... },
      ...
    }

Auth is handled by the CLI itself (OAuth / keychain / ANTHROPIC_API_KEY).
Do NOT pass `--bare`: it disables OAuth + keychain and forces API-key auth,
which breaks personal-subscription deployments — Eugene Plexus's primary
production mode.

Internally Claude Code is an *agent* (it routes through haiku for cheap
classification before sending to the user-facing model). The reported
`usage` aggregates tokens across all internal model calls. We surface
those totals as-is.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from .._generated.models import (
    BackendKind,
    FinishReason,
    GenerateRequest,
    GenerateResponse,
    Usage,
)
from ._prompt import messages_to_prompt
from ._subprocess import CliError, run_cli

_STOP_REASON_MAP = {
    "end_turn": FinishReason.stop,
    "stop_sequence": FinishReason.stop_sequence,
    "max_tokens": FinishReason.length,
}


class ClaudeCodeCliAdapter:
    backend_kind = "claude_code_cli"

    def __init__(
        self,
        *,
        binary_path: str = "claude",
        model_id: str | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        self._binary_path = binary_path
        self._model_id = model_id
        self._timeout_seconds = timeout_seconds

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        argv = self._build_argv(messages_to_prompt(list(request.messages)))
        result = await run_cli(argv, timeout_seconds=self._timeout_seconds)

        if result.returncode != 0:
            raise CliError(
                f"claude exited {result.returncode}: "
                f"{result.stderr.decode(errors='replace').strip() or '<no stderr>'}"
            )

        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            raise CliError(f"claude stdout was not valid JSON: {result.stdout[:200]!r}") from e

        if not isinstance(data, dict):
            raise CliError(f"claude JSON envelope was not an object: {data!r}")

        if data.get("is_error"):
            raise CliError(f"claude reported error: {data.get('result')!r}")

        content = data.get("result")
        if not isinstance(content, str):
            raise CliError(f"claude JSON missing string `result`: {data!r}")

        return GenerateResponse(
            content=content,
            finishReason=_STOP_REASON_MAP.get(
                str(data.get("stop_reason") or ""), FinishReason.stop
            ),
            usage=_usage_from_envelope(data.get("usage") or {}),
            requestId=request.requestId,
            backend=BackendKind.claude_code_cli,
            modelId=self._model_id,
            latencyMs=result.elapsed_ms,
        )

    async def stream(self, request: GenerateRequest) -> AsyncIterator[object]:
        # Claude Code CLI does support streaming via
        # --output-format stream-json --include-partial-messages, but no
        # consumer of hemisphere-driver streaming exists yet (orchestrator,
        # ui not implemented). Wire it up when the consumer lands.
        raise NotImplementedError("ClaudeCodeCliAdapter.stream not implemented in v0.1")
        yield  # pragma: no cover

    def _build_argv(self, prompt: str) -> list[str]:
        argv = [
            self._binary_path,
            "--print",
            "--output-format",
            "json",
        ]
        if self._model_id:
            argv += ["--model", self._model_id]
        argv.append(prompt)
        return argv


def _usage_from_envelope(usage: dict[str, Any]) -> Usage | None:
    """Best-effort mapping of claude's usage block to our Usage schema."""
    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")
    cache_read = usage.get("cache_read_input_tokens", 0) or 0
    cache_creation = usage.get("cache_creation_input_tokens", 0) or 0

    if input_tokens is None and output_tokens is None:
        return None

    prompt = (input_tokens or 0) + cache_read + cache_creation
    completion = output_tokens or 0
    return Usage(
        promptTokens=prompt,
        completionTokens=completion,
        totalTokens=prompt + completion,
    )

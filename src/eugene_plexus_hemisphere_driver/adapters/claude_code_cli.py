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
    Role,
    Usage,
)
from ._prompt import messages_to_prompt
from ._subprocess import CliError, run_cli

_STOP_REASON_MAP = {
    "end_turn": FinishReason.stop,
    "stop_sequence": FinishReason.stop_sequence,
    "max_tokens": FinishReason.length,
}

# Hardcoded list of known-good Claude models supported by the Claude
# Code CLI. The CLI doesn't expose a `--list-models` so we can't
# discover this live the way openai_api can. Update by hand when
# Anthropic ships new chat models. Extended-thinking variants are
# excluded — Eugene Plexus IS the synthesis layer (see o-series
# rejection in openai_api).
_KNOWN_CLAUDE_MODELS: list[str] = [
    "claude-opus-4-7",
    "claude-sonnet-4-7",
    "claude-haiku-4-5",
    "claude-opus-4",
    "claude-sonnet-4",
    "claude-3-5-sonnet-latest",
    "claude-3-5-haiku-latest",
]


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
        # Split system messages from the rest. Claude Code's --system-prompt
        # *replaces* its default system prompt entirely (which also disables
        # the "current working directory: ..." injection per the CLI's own
        # docs), giving us closer-to-raw-LLM behavior than passing system
        # text inside the user-message argv.
        system_messages = [m for m in request.messages if m.role == Role.system]
        other_messages = [m for m in request.messages if m.role != Role.system]
        system_prompt = "\n\n".join(m.content for m in system_messages).strip()
        user_prompt = messages_to_prompt(other_messages)

        # The user-prompt transcript can contain newlines (paragraph breaks
        # in prior assistant messages, multi-line user input, etc). On
        # Windows, putting that on argv breaks: cmd.exe — which wraps
        # `claude.cmd` — treats a literal newline inside a quoted arg as a
        # command separator. Pipe via stdin instead. Claude Code reads
        # stdin under --print when no positional prompt is given.
        argv = self._build_argv(system_prompt=system_prompt)
        result = await run_cli(
            argv,
            timeout_seconds=self._timeout_seconds,
            stdin_input=user_prompt.encode("utf-8"),
        )

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

    async def list_models(self) -> list[str]:
        # Claude Code CLI doesn't expose a list endpoint — return a
        # hardcoded set of currently-shipping chat models. All listed
        # models support tunable temperature.
        return list(_KNOWN_CLAUDE_MODELS)

    def _build_argv(self, *, system_prompt: str) -> list[str]:
        argv = [
            self._binary_path,
            "--print",
            "--output-format",
            "json",
        ]
        if system_prompt:
            argv += ["--system-prompt", system_prompt]
        if self._model_id:
            argv += ["--model", self._model_id]
        # No positional prompt; user prompt is piped via stdin.
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

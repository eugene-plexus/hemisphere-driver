"""Adapter that wraps the Anthropic Claude Code CLI as a subprocess.

Stub for the scaffolding commit. Real subprocess plumbing — argv assembly,
stdout streaming, ANSI stripping, error mapping — lands in the next commit
once the invocation pattern has been verified by hand.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from .._generated.models import GenerateRequest, GenerateResponse


class ClaudeCodeCliAdapter:
    backend_kind = "claude_code_cli"

    def __init__(self, *, binary_path: str = "claude") -> None:
        self._binary_path = binary_path

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        raise NotImplementedError("ClaudeCodeCliAdapter.generate is a stub for v0.1 scaffolding")

    async def stream(self, request: GenerateRequest) -> AsyncIterator[object]:
        raise NotImplementedError("ClaudeCodeCliAdapter.stream is a stub for v0.1 scaffolding")
        yield  # unreachable; satisfies the async-generator type

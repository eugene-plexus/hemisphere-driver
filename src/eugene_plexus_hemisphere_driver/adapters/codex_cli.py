"""Adapter that wraps the OpenAI Codex CLI as a subprocess.

Stub for the scaffolding commit. Real subprocess plumbing lands in the next
commit once the invocation pattern has been verified by hand.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from .._generated.models import GenerateRequest, GenerateResponse


class CodexCliAdapter:
    backend_kind = "codex_cli"

    def __init__(self, *, binary_path: str = "codex") -> None:
        self._binary_path = binary_path

    async def generate(self, request: GenerateRequest) -> GenerateResponse:
        raise NotImplementedError("CodexCliAdapter.generate is a stub for v0.1 scaffolding")

    async def stream(self, request: GenerateRequest) -> AsyncIterator[object]:
        raise NotImplementedError("CodexCliAdapter.stream is a stub for v0.1 scaffolding")
        yield

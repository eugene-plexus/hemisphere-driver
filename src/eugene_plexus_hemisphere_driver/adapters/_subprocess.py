"""Tiny shared helper for invoking a CLI subprocess from an adapter.

Centralizes the asyncio plumbing so each adapter only has to write its argv
builder and its output parser.
"""

from __future__ import annotations

import asyncio
import shutil
import time
from dataclasses import dataclass


class CliError(RuntimeError):
    """Raised when a CLI invocation exits non-zero or times out."""


@dataclass
class CliResult:
    stdout: bytes
    stderr: bytes
    returncode: int
    elapsed_ms: int


async def run_cli(argv: list[str], *, timeout_seconds: float) -> CliResult:
    """Run argv as a subprocess. Returns stdout/stderr/returncode + elapsed time.

    Stdin is closed (DEVNULL) so the CLI never blocks waiting for input it
    won't get. The caller is expected to encode the prompt directly into
    argv. Raises `CliError` on timeout; non-zero exit is returned to the
    caller for inspection.
    """
    if not argv:
        raise ValueError("argv is empty")

    # Resolve the binary up-front so the error message is clearer than
    # asyncio's default "FileNotFoundError" if it isn't on PATH.
    binary = argv[0]
    resolved = shutil.which(binary)
    if resolved is None:
        raise CliError(f"binary {binary!r} not found on PATH")
    argv = [resolved, *argv[1:]]

    start = time.perf_counter()
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_seconds)
    except TimeoutError as e:
        proc.kill()
        await proc.wait()
        raise CliError(f"CLI {argv[0]!r} did not respond within {timeout_seconds}s; killed") from e

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    return CliResult(
        stdout=stdout, stderr=stderr, returncode=proc.returncode or 0, elapsed_ms=elapsed_ms
    )

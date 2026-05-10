# Eugene Plexus — `hemisphere-driver`

[![CI](https://github.com/eugene-plexus/hemisphere-driver/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/eugene-plexus/hemisphere-driver/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB.svg)](https://www.python.org)

A uniform HTTP wrapper around a single LLM backend, used by the [Eugene Plexus](https://github.com/eugene-plexus) orchestrator to drive bicameral chat.

The orchestrator runs **at least two** instances of this service side-by-side (v0.1 pairs them; v0.2+ generalizes to N with backup/failover), typically configured with different model families on different instances — Claude on one side, GPT on the other, a local OSS model as a third — so the bicameral pass produces genuine cross-vendor disagreement rather than two echoes of the same RLHF distribution.

## What this service is — and what it isn't

A hemisphere-driver is **anonymous and stateless**. It wraps one LLM backend and serves `POST /v1/generate`. It doesn't know its position in any topology — no "left" / "right", no "primary" / "backup". The orchestrator owns the topology: it has a `drivers` config listing each driver's URL and operator-supplied name, and stamps that name onto every message a driver produces. Driver instances are interchangeable from outside; only the orchestrator knows which one it just labelled "left".

This means a hemisphere-driver also doesn't decide LLM-output-affecting parameters (temperature, max tokens, etc.). The orchestrator owns those and supplies them on every request — in v0.2+ they'll be NT-modulated per-pass-per-driver. The driver applies what it's given and never substitutes a local default.

## Status

**v0.1, working.** The HTTP surface, config protocol with `/v1/config/test`, and three adapters (`claude_code_cli`, `codex_cli`, `openai_api`) are wired up end-to-end. Streaming (`/v1/generate/stream`) is still a 501 stub — it lands alongside the orchestrator + UI consumers.

## Wire contract

This service implements the [`hemisphere-driver.yaml`](https://github.com/eugene-plexus/specs/blob/main/openapi/hemisphere-driver.yaml) OpenAPI 3.1 spec from the [`eugene-plexus/specs`](https://github.com/eugene-plexus/specs) repo. Pydantic models in `src/eugene_plexus_hemisphere_driver/_generated/` are produced via codegen (see [Codegen](#codegen)).

Endpoints:

| Method | Path                  | Status     |
|--------|-----------------------|------------|
| GET    | `/healthz`            | ✅          |
| GET    | `/v1/info`            | ✅          |
| GET    | `/v1/config`          | ✅          |
| GET    | `/v1/config/schema`   | ✅          |
| PATCH  | `/v1/config`          | ✅          |
| POST   | `/v1/config/test`     | ✅          |
| POST   | `/v1/generate`        | ✅          |
| POST   | `/v1/generate/stream` | stub (501) |

## Backends (adapters)

v0.1 ships with three adapters. The two remaining ones (`anthropic_api`, `openai_compat_http`) land in v0.2.

| Adapter              | Status   | Notes |
|----------------------|----------|-------|
| `claude_code_cli`    | ✅ wired | Wraps the `claude` CLI. Uses your Claude Pro/Max subscription — no API billing. System prompts are passed via `--system-prompt` so persona control is preserved. |
| `codex_cli`          | ✅ wired | Wraps `codex-cli`. Uses your ChatGPT subscription. ⚠️ Codex CLI has no `--system-prompt` equivalent; persona override and cwd-injection cannot be suppressed from the CLI surface. Use `openai_api` if you need full persona control on the OpenAI side. |
| `openai_api`         | ✅ wired | Direct HTTP to OpenAI's `/v1/chat/completions` API or any OpenAI-compatible provider (Together, Groq, Fireworks, MiniMax, vLLM, LM Studio, etc.). Pay-per-token when the provider charges. |
| `anthropic_api`      | v0.2+    | Direct HTTP to Anthropic. Pay-per-token. |
| `openai_compat_http` | v0.2+    | Convenience wrapper around `openai_api` for known local server profiles (Ollama, vLLM, LM Studio). |

The CLI adapters are **primary production mode for personal installations** — they run on the AI subscription you already pay for, no separate API bill.

## Running

### From source

```bash
pip install -e ".[dev]"
python -m eugene_plexus_hemisphere_driver
```

By default it listens on `http://127.0.0.1:8081`. Configure via env vars (12-factor) or by editing `config.yaml` (auto-created in the working directory on first run).

### Pairing with the orchestrator

Run two driver instances on different ports — typically 8081 and 8082 — each with a different `adapter` config. Then point the orchestrator's `drivers` config at both:

```yaml
drivers:
  - name: left
    url: http://127.0.0.1:8081
  - name: right
    url: http://127.0.0.1:8082
```

The orchestrator's UI exposes a per-driver Test button that calls each driver's `/v1/info` so you can verify the URLs are reachable before saving.

### Configuration

Every config field is editable at runtime via `PATCH /v1/config`. `GET /v1/config/schema` returns UI-renderable metadata for every field — a generic UI renders the editor with no per-component code. The `eugene-plexus/ui` repo's config tab does this.

The driver also implements `POST /v1/config/test` — given an optional `overrides` body, it builds a temporary adapter from saved-merged-with-overrides config and runs a minimal `generate("Reply with PING")` round-trip. The UI's Test button on each driver's config tab uses this.

#### Degraded mode

If adapter construction fails at startup (missing API key, missing CLI binary, malformed config), the driver does **not** crash. It comes up in degraded mode with a working `/v1/config` and `/v1/config/schema` so the operator can fix the broken field via PATCH and restart. `/v1/generate` returns a 503 with a clear `Problem` message until the config is corrected.

#### v0.1 auth

> **v0.1 has no application auth.** Deployment assumption: behind a [Tailscale](https://tailscale.com/) tailnet or equivalent network boundary. Anyone reachable on the network can read and modify config — including secrets like API keys via PATCH. Auth lands in v0.2.

## Codegen

Pydantic models for the wire contract are generated from the pinned commit of `eugene-plexus/specs` recorded in [`SPECS_REF`](SPECS_REF):

```bash
python scripts/codegen.py
```

The script downloads the specs at the pinned SHA, runs `datamodel-code-generator` against them, and writes Pydantic v2 models to `src/eugene_plexus_hemisphere_driver/_generated/`. **The generated files are committed** so builds are reproducible without network access. CI re-runs codegen and fails the build if the working tree differs.

To bump to a newer specs commit:

```bash
echo "<new-sha>" > SPECS_REF
python scripts/codegen.py
```

…then commit `SPECS_REF` and the regenerated `_generated/` directory together.

## Development

```bash
pip install -e ".[dev]"

# Lint + format
ruff check .
ruff format --check .

# Type-check
mypy src/

# Test
pytest

# Codegen freshness
python scripts/codegen.py && git diff --exit-code src/eugene_plexus_hemisphere_driver/_generated/
```

The test suite (35 tests, ~1s) covers the HTTP surface, the config protocol, all three adapters' shape adaptation, degraded-mode startup, and a UTF-8 round-trip pinning the subprocess encoding fix on Windows. CLI live-fire tests are gated behind `EUGENE_PLEXUS_HD_LIVE_CLI=1` and the API live-fire test behind `EUGENE_PLEXUS_HD_LIVE_API=1`.

## License

Apache 2.0 — see [`LICENSE`](LICENSE).

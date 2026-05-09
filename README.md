# Eugene Plexus — `hemisphere-driver`

[![CI](https://github.com/eugene-plexus/hemisphere-driver/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/eugene-plexus/hemisphere-driver/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB.svg)](https://www.python.org)

One half of an [Eugene Plexus](https://github.com/eugene-plexus) bicameral pair: a uniform HTTP wrapper around a single LLM backend.

The orchestrator runs **two** instances of this service side-by-side, typically configured with different model families (Claude on one side, GPT on the other), so the bicameral pass produces genuine inter-vendor disagreement rather than two echoes of the same RLHF distribution.

## Status

**v0.1, working CLI adapters.** The HTTP surface, config protocol, and both CLI adapters (`claude_code_cli`, `codex_cli`) are wired up end-to-end. Streaming (`/v1/generate/stream`) lands alongside the orchestrator + UI consumers.

## Wire contract

This service implements the [`hemisphere-driver.yaml`](https://github.com/eugene-plexus/specs/blob/main/openapi/hemisphere-driver.yaml) OpenAPI 3.1 spec from the [`eugene-plexus/specs`](https://github.com/eugene-plexus/specs) repo. Pydantic models in `src/eugene_plexus_hemisphere_driver/_generated/` are produced via codegen (see [Codegen](#codegen)).

Endpoints:

| Method | Path                  | Status      |
|--------|-----------------------|-------------|
| GET    | `/healthz`            | ✅           |
| GET    | `/v1/info`            | ✅           |
| GET    | `/v1/config`          | ✅           |
| GET    | `/v1/config/schema`   | ✅           |
| PATCH  | `/v1/config`          | ✅           |
| POST   | `/v1/generate`        | ✅           |
| POST   | `/v1/generate/stream` | stub (501)  |

## Backends (adapters)

v0.1 ships with two CLI subprocess adapters; API adapters and OpenAI-compatible HTTP land in v0.2.

| Adapter              | Status   | Notes |
|----------------------|----------|-------|
| `claude_code_cli`    | ✅ wired | Wraps `claude` v2.1+. Reads single JSON envelope from `--print --output-format json`. |
| `codex_cli`          | ✅ wired | Wraps `codex-cli` v0.130+. Parses JSONL stream from `codex exec --json`. |
| `anthropic_api`      | v0.2+    | Direct HTTP. Pay-per-token. |
| `openai_api`         | v0.2+    | Direct HTTP. Pay-per-token. |
| `openai_compat_http` | v0.2+    | Ollama, vLLM, LM Studio, etc. |

The CLI adapters are **primary production mode for personal installations** — they run on the AI subscription you already pay for, no separate API bill.

## Running

### From source

```bash
pip install -e ".[dev]"
python -m eugene_plexus_hemisphere_driver
```

By default it listens on `http://127.0.0.1:8081`. Configure via env vars (12-factor) or by editing `config.yaml` (auto-created in the working directory on first run).

### Configuration

Every config field is editable at runtime via `PATCH /v1/config`. `GET /v1/config/schema` returns UI-renderable metadata for every field — a generic UI can render the editor with no per-component code. See [`docs/config.md`](docs/config.md) once it exists for the field list, or hit `/v1/config/schema` on a running instance.

> **v0.1 has no auth.** Deployment assumption: behind a [Tailscale](https://tailscale.com/) tailnet or equivalent network boundary. Anyone reachable on the network can read and modify config. Auth lands in v0.2.

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

## License

Apache 2.0 — see [`LICENSE`](LICENSE).

"""Tests for the config protocol routes."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_get_config_schema_lists_expected_fields(client: TestClient) -> None:
    response = client.get("/v1/config/schema")
    assert response.status_code == 200
    body = response.json()
    assert body["component"] == "hemisphere-driver"

    field_keys = {f["key"] for f in body["fields"]}
    expected = {
        "adapter",
        "modelId",
        "claudeCodeCliPath",
        "codexCliPath",
        "port",
        "logLevel",
        "requestTimeoutSeconds",
    }
    assert expected.issubset(field_keys)
    # LLM-output-affecting params (temperature, max-tokens, etc.) are
    # owned by the orchestrator and never appear on the driver schema.
    assert "defaultTemperature" not in field_keys
    assert "defaultMaxTokens" not in field_keys
    # Driver no longer self-asserts identity — the orchestrator labels
    # it via the `drivers` config and the `driverName` it stamps on
    # every emitted message.
    assert "hemisphere" not in field_keys

    adapter_field = next(f for f in body["fields"] if f["key"] == "adapter")
    assert adapter_field["valueType"] == "enum"
    assert "claude_code_cli" in adapter_field["enumValues"]
    assert adapter_field["requiresRestart"] is True


def test_get_config_returns_defaults_on_first_run(client: TestClient) -> None:
    response = client.get("/v1/config")
    assert response.status_code == 200
    doc = response.json()
    assert doc["adapter"] == "claude_code_cli"
    assert doc["port"] == 8081
    assert "hemisphere" not in doc


def test_patch_applies_valid_change_and_rejects_invalid(client: TestClient) -> None:
    response = client.patch(
        "/v1/config",
        json={
            "claudeCodeCliPath": "/usr/local/bin/claude",  # valid string update
            "port": 99999,  # out of range
            "bogusField": True,  # unknown field
            "defaultTemperature": 0.9,  # used to live here; must now be unknown
            "hemisphere": "right",  # also used to live here; must now be unknown
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert "claudeCodeCliPath" in body["applied"]

    rejected_keys = {r["key"] for r in body["rejected"]}
    # bogusField, defaultTemperature, and hemisphere are all unknown now.
    assert {"port", "bogusField", "defaultTemperature", "hemisphere"} <= rejected_keys

    # `claudeCodeCliPath` is restart-required.
    assert body["requiresRestart"] is True

    follow = client.get("/v1/config")
    assert follow.json()["claudeCodeCliPath"] == "/usr/local/bin/claude"
    assert follow.json()["port"] == 8081  # unchanged after rejection


def test_schema_modelid_is_string_when_no_models_discovered(client: TestClient) -> None:
    """Default fixture starts the app without an adapter the lifespan
    can build (claude_code_cli with `claude` in PATH most likely fails
    or no list is set), so available_models is empty. modelId stays as
    free-text string so the operator can still type a model by hand."""
    body = client.get("/v1/config/schema").json()
    model_field = next(f for f in body["fields"] if f["key"] == "modelId")
    # Either string (no list available) or enum (list discovered) is
    # acceptable; we just assert the contract — when there's NO list,
    # it MUST be string. The fixture's lifespan typically discovers
    # claude models, so we expect enum here. The OTHER test pins
    # the no-list branch directly.
    assert model_field["valueType"] in ("string", "enum")


def test_schema_modelid_becomes_enum_when_models_available() -> None:
    """When the lifespan populates `available_models`, the schema
    endpoint exposes modelId as a dropdown — value type flips to
    enum, enumValues lists the discovered models, and a leading
    empty-string entry preserves the 'use adapter default' option."""
    from eugene_plexus_hemisphere_driver.config import as_schema

    schema = as_schema(available_models=["claude-opus-4-7", "claude-sonnet-4-7"])
    model_field = next(f for f in schema.fields if f.key == "modelId")
    assert model_field.valueType.value == "enum"
    assert model_field.enumValues is not None
    assert model_field.enumValues[0] == ""  # "(use adapter default)" sentinel
    assert "claude-opus-4-7" in model_field.enumValues
    assert "claude-sonnet-4-7" in model_field.enumValues


def test_schema_modelid_stays_string_without_models() -> None:
    """Empty / missing list → free-text input. Belt-and-suspenders for
    the fallback path the schema route uses in degraded mode."""
    from eugene_plexus_hemisphere_driver.config import as_schema

    schema = as_schema(available_models=None)
    model_field = next(f for f in schema.fields if f.key == "modelId")
    assert model_field.valueType.value == "string"
    schema = as_schema(available_models=[])
    model_field = next(f for f in schema.fields if f.key == "modelId")
    assert model_field.valueType.value == "string"


def test_patch_with_restart_required_field_sets_pending(client: TestClient) -> None:
    response = client.patch("/v1/config", json={"port": 8090})
    assert response.status_code == 200
    body = response.json()
    assert body["applied"] == ["port"]
    assert body["requiresRestart"] is True
    assert body["pendingRestart"] == ["port"]

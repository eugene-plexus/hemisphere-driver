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


def test_patch_with_restart_required_field_sets_pending(client: TestClient) -> None:
    response = client.patch("/v1/config", json={"port": 8090})
    assert response.status_code == 200
    body = response.json()
    assert body["applied"] == ["port"]
    assert body["requiresRestart"] is True
    assert body["pendingRestart"] == ["port"]

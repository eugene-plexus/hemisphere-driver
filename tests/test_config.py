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
        "hemisphere",
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

    adapter_field = next(f for f in body["fields"] if f["key"] == "adapter")
    assert adapter_field["valueType"] == "enum"
    assert "claude_code_cli" in adapter_field["enumValues"]
    assert adapter_field["requiresRestart"] is True


def test_get_config_returns_defaults_on_first_run(client: TestClient) -> None:
    response = client.get("/v1/config")
    assert response.status_code == 200
    doc = response.json()
    assert doc["adapter"] == "claude_code_cli"
    assert doc["hemisphere"] == "left"
    assert doc["port"] == 8081


def test_patch_applies_valid_change_and_rejects_invalid(client: TestClient) -> None:
    response = client.patch(
        "/v1/config",
        json={
            "hemisphere": "right",  # valid enum change
            "port": 99999,  # out of range
            "bogusField": True,  # unknown field
            "defaultTemperature": 0.9,  # used to live here; must now be unknown
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert "hemisphere" in body["applied"]

    rejected_keys = {r["key"] for r in body["rejected"]}
    # bogusField AND defaultTemperature are both unknown to the driver now.
    assert {"port", "bogusField", "defaultTemperature"} <= rejected_keys

    # `hemisphere` is hot-swappable.
    assert body["requiresRestart"] is False

    follow = client.get("/v1/config")
    assert follow.json()["hemisphere"] == "right"
    assert follow.json()["port"] == 8081  # unchanged after rejection


def test_patch_with_restart_required_field_sets_pending(client: TestClient) -> None:
    response = client.patch("/v1/config", json={"port": 8090})
    assert response.status_code == 200
    body = response.json()
    assert body["applied"] == ["port"]
    assert body["requiresRestart"] is True
    assert body["pendingRestart"] == ["port"]

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
        "defaultMaxTokens",
        "defaultTemperature",
    }
    assert expected.issubset(field_keys)

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
            "defaultTemperature": 0.9,  # valid, hot-swappable
            "port": 99999,  # out of range
            "bogusField": True,  # unknown field
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert "defaultTemperature" in body["applied"]

    rejected_keys = {r["key"] for r in body["rejected"]}
    assert {"port", "bogusField"} <= rejected_keys

    # defaultTemperature is read live at request time, so no restart needed.
    assert body["requiresRestart"] is False

    follow = client.get("/v1/config")
    assert follow.json()["defaultTemperature"] == 0.9
    assert follow.json()["port"] == 8081  # unchanged after rejection


def test_patch_with_restart_required_field_sets_pending(client: TestClient) -> None:
    response = client.patch("/v1/config", json={"port": 8090})
    assert response.status_code == 200
    body = response.json()
    assert body["applied"] == ["port"]
    assert body["requiresRestart"] is True
    assert body["pendingRestart"] == ["port"]

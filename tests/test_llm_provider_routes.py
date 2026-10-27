from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from open_llm_vtuber.config_manager.llm_provider import LLMProviderStore, LLMProviderType
from open_llm_vtuber.llm_provider_routes import init_llm_provider_routes
import open_llm_vtuber.llm_provider_routes as provider_routes


API_KEY = "olv_test_server_key"


def _build_app(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "providers.sqlite3"
    secret_path = tmp_path / ".secret.key"
    context = SimpleNamespace(
        system_config=SimpleNamespace(api_key=API_KEY),
        character_config=SimpleNamespace(
            agent_config=SimpleNamespace(),
            persona_prompt="persona",
        ),
    )
    # Route tests use the same persistence path as the production store.
    import os

    monkeypatch.setenv("LLM_PROVIDER_STORE_PATH", str(db_path))
    monkeypatch.setenv("LLM_PROVIDER_SECRET_KEY_PATH", str(secret_path))

    async def refresh():
        context.refresh_count = getattr(context, "refresh_count", 0) + 1
        return True

    context.refresh_managed_llm_provider = refresh
    app = FastAPI()
    app.include_router(init_llm_provider_routes(context))
    return app, context, db_path


def _auth():
    return {"Authorization": f"Bearer {API_KEY}"}


def test_provider_crud_masks_secret_and_preserves_omitted_key(tmp_path, monkeypatch):
    app, context, _db = _build_app(tmp_path, monkeypatch)
    client = TestClient(app)

    create = client.post(
        "/llm/providers",
        headers=_auth(),
        json={
            "name": "Local Ollama",
            "provider_type": "ollama",
            "base_url": "http://127.0.0.1:11434",
            "api_key": "super-secret-value",
            "model": "mistral:latest",
        },
    )
    assert create.status_code == 200
    body = create.json()["provider"]
    assert body["api_key_configured"] is True
    assert body["api_key_masked"]
    assert "super-secret-value" not in create.text
    provider_id = body["provider_id"]

    listed = client.get("/llm/providers", headers=_auth())
    assert listed.status_code == 200
    assert "super-secret-value" not in listed.text

    updated = client.patch(
        f"/llm/providers/{provider_id}",
        headers=_auth(),
        json={"model": "qwen3:latest"},
    )
    assert updated.status_code == 200

    with LLMProviderStore(_db) as store:
        provider = store.require(provider_id)
        assert provider.secret_value() == "super-secret-value"
        assert provider.model == "qwen3:latest"

    reveal = client.post(
        f"/llm/providers/{provider_id}/reveal-secret",
        headers=_auth(),
    )
    assert reveal.status_code == 200
    assert reveal.json()["api_key"] == "super-secret-value"
    assert reveal.headers["cache-control"] == "no-store, no-cache, must-revalidate, private"


def test_provider_reveal_requires_auth_and_never_appears_in_list(tmp_path, monkeypatch):
    app, _context, _db = _build_app(tmp_path, monkeypatch)
    client = TestClient(app)

    create = client.post(
        "/llm/providers",
        headers=_auth(),
        json={"name": "Cloud", "provider_type": "openai", "base_url": "https://example.com/v1", "api_key": "secret"},
    )
    provider_id = create.json()["provider"]["provider_id"]

    assert client.get("/llm/providers").status_code == 401
    assert client.post(f"/llm/providers/{provider_id}/reveal-secret").status_code == 401
    assert "secret" not in client.get("/llm/providers", headers=_auth()).text


def test_probe_reuses_stored_secret_without_returning_it(tmp_path, monkeypatch):
    app, _context, _db = _build_app(tmp_path, monkeypatch)
    client = TestClient(app)

    created = client.post(
        "/llm/providers",
        headers=_auth(),
        json={
            "name": "Compatible",
            "provider_type": "openai_compatible",
            "base_url": "http://127.0.0.1:9000/v1",
            "api_key": "stored-secret",
            "model": "model-a",
        },
    ).json()["provider"]

    seen = {}

    async def fake_test(self, provider):
        seen["secret"] = provider.secret_value()
        return provider_routes.LLMConnectionResult(
            ok=True,
            provider_id=provider.provider_id,
            provider_name=provider.name,
            provider_type=provider.provider_type.value,
            message="Connection successful",
            latency_ms=1.0,
            model_count=1,
        )

    monkeypatch.setattr(provider_routes.LLMProviderConnectionManager, "test_connection", fake_test)

    response = client.post(
        "/llm/providers/test",
        headers=_auth(),
        json={
            "provider_id": created["provider_id"],
            "name": "Compatible draft",
            "provider_type": "openai_compatible",
            "base_url": "http://127.0.0.1:9001/v1",
            "model": "model-b",
        },
    )
    assert response.status_code == 200
    assert response.json()["ok"] is True
    assert seen["secret"] == "stored-secret"
    assert "stored-secret" not in response.text


def test_activate_and_deactivate_refresh_runtime(tmp_path, monkeypatch):
    app, context, _db = _build_app(tmp_path, monkeypatch)
    client = TestClient(app)

    first = client.post(
        "/llm/providers",
        headers=_auth(),
        json={"name": "One", "provider_type": "ollama", "base_url": "http://127.0.0.1:11434", "model": "a"},
    ).json()["provider"]
    second = client.post(
        "/llm/providers",
        headers=_auth(),
        json={"name": "Two", "provider_type": "ollama", "base_url": "http://127.0.0.1:11435", "model": "b"},
    ).json()["provider"]

    activated = client.post(f"/llm/providers/{first['provider_id']}/activate", headers=_auth())
    assert activated.status_code == 200
    assert activated.json()["active_provider_id"] == first["provider_id"]
    assert context.refresh_count == 1

    client.post(f"/llm/providers/{second['provider_id']}/activate", headers=_auth())
    assert context.refresh_count == 2

    deactivated = client.post("/llm/providers/deactivate", headers=_auth())
    assert deactivated.status_code == 200
    assert deactivated.json()["active_provider_id"] is None
    assert context.refresh_count == 3


def test_update_null_api_key_clears_it(tmp_path, monkeypatch):
    app, _context, _db = _build_app(tmp_path, monkeypatch)
    client = TestClient(app)

    created = client.post(
        "/llm/providers",
        headers=_auth(),
        json={"name": "Cloud", "provider_type": "openai", "base_url": "https://example.com/v1", "api_key": "secret"},
    ).json()["provider"]

    updated = client.patch(
        f"/llm/providers/{created['provider_id']}",
        headers=_auth(),
        json={"api_key": None},
    )
    assert updated.status_code == 200
    assert updated.json()["provider"]["api_key_configured"] is False


# Ensure imported enum remains exercised so a future route refactor does not accidentally
# turn provider types into free-form strings.
def test_provider_type_enum_is_available():
    assert LLMProviderType.OLLAMA.value == "ollama"


def test_create_validation_does_not_echo_api_key(tmp_path, monkeypatch):
    app, _context, _db = _build_app(tmp_path, monkeypatch)
    client = TestClient(app)
    response = client.post(
        "/llm/providers",
        headers=_auth(),
        json={
            "name": "Bad URL",
            "provider_type": "openai",
            "base_url": "https://user:secret@example.com/v1",
            "api_key": "validation-secret",
        },
    )
    assert response.status_code == 422
    assert "validation-secret" not in response.text
    assert "user:secret" not in response.text


def test_probe_backend_error_is_generic_and_does_not_echo_secret(tmp_path, monkeypatch):
    app, _context, _db = _build_app(tmp_path, monkeypatch)

    async def fake_test(self, provider):
        raise RuntimeError(f"unexpected backend detail {provider.secret_value()}")

    monkeypatch.setattr(provider_routes.LLMProviderConnectionManager, "test_connection", fake_test)
    client = TestClient(app)
    response = client.post(
        "/llm/providers/test",
        headers=_auth(),
        json={
            "name": "Probe",
            "provider_type": "openai_compatible",
            "base_url": "https://example.com/v1",
            "api_key": "probe-secret",
        },
    )
    assert response.status_code == 502
    assert "probe-secret" not in response.text
    assert "Could not contact the LLM provider" in response.text

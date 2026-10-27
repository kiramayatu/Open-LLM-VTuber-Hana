import sqlite3
from pathlib import Path

import pytest

from open_llm_vtuber.config_manager.llm_provider import (
    LLMProviderCreate,
    LLMProviderStore,
)
from open_llm_vtuber.config_manager.llm_provider_secrets import (
    LLMProviderSecretError,
    SECRET_CIPHERTEXT_PREFIX,
    SETTINGS_CIPHERTEXT_PREFIX,
    encrypt_text,
    decrypt_text,
    load_encryption_key,
)


def test_aes_gcm_roundtrip_and_tamper_detection(tmp_path: Path):
    key = load_encryption_key(tmp_path / "providers.sqlite3")
    ciphertext = encrypt_text("super-secret", key)

    assert ciphertext.startswith(SECRET_CIPHERTEXT_PREFIX)
    assert "super-secret" not in ciphertext
    assert decrypt_text(ciphertext, key) == "super-secret"

    tampered = ciphertext[:-1] + ("A" if ciphertext[-1] != "A" else "B")
    with pytest.raises(LLMProviderSecretError):
        decrypt_text(tampered, key)


def test_provider_db_stores_api_key_and_sensitive_settings_encrypted(tmp_path: Path):
    db_path = tmp_path / "providers.sqlite3"

    with LLMProviderStore(db_path) as store:
        provider = store.create(
            LLMProviderCreate(
                name="Encrypted Provider",
                base_url="https://api.example.com/v1",
                api_key="API-SECRET-123",
                model="model-a",
                settings={
                    "organization_id": "org-visible",
                    "token": "TOKEN-SECRET-456",
                    "nested": {"password": "PASSWORD-SECRET-789"},
                },
            )
        )

        loaded = store.require(provider.provider_id)
        assert loaded.secret_value() == "API-SECRET-123"
        assert loaded.settings["token"] == "TOKEN-SECRET-456"
        assert loaded.settings["nested"]["password"] == "PASSWORD-SECRET-789"

        public = loaded.to_public_dict()
        assert public["api_key_masked"] == "••••••••"
        assert public["settings"]["organization_id"] == "org-visible"
        assert public["settings"]["token"] == "***configured***"
        assert public["settings"]["nested"]["password"] == "***configured***"
        assert "API-SECRET-123" not in str(public)
        assert "TOKEN-SECRET-456" not in str(public)
        assert "PASSWORD-SECRET-789" not in str(public)

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT api_key, settings_json FROM llm_providers WHERE provider_id = ?",
            (provider.provider_id,),
        ).fetchone()

    assert row is not None
    stored_api_key, settings_json = row
    assert stored_api_key.startswith(SECRET_CIPHERTEXT_PREFIX)
    assert stored_api_key != "API-SECRET-123"
    assert "API-SECRET-123" not in stored_api_key
    assert SETTINGS_CIPHERTEXT_PREFIX in settings_json
    assert "TOKEN-SECRET-456" not in settings_json
    assert "PASSWORD-SECRET-789" not in settings_json


def test_legacy_plaintext_provider_is_migrated_on_open(tmp_path: Path):
    db_path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE llm_providers (
                provider_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                provider_type TEXT NOT NULL,
                base_url TEXT,
                api_key TEXT,
                model TEXT,
                enabled INTEGER NOT NULL,
                settings_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE llm_provider_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            INSERT INTO llm_providers
            (provider_id, name, provider_type, base_url, api_key, model, enabled,
             settings_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "legacy-provider",
                "Legacy",
                "openai_compatible",
                "https://example.com/v1",
                "LEGACY-API-SECRET",
                "model-a",
                1,
                '{"token":"LEGACY-TOKEN-SECRET","safe":"ok"}',
                "2026-09-22T00:00:00+00:00",
                "2026-09-22T00:00:00+00:00",
            ),
        )
        conn.commit()

    with LLMProviderStore(db_path) as store:
        loaded = store.require("legacy-provider")
        assert loaded.secret_value() == "LEGACY-API-SECRET"
        assert loaded.settings["token"] == "LEGACY-TOKEN-SECRET"
        assert loaded.settings["safe"] == "ok"

    with sqlite3.connect(db_path) as conn:
        api_key, settings_json = conn.execute(
            "SELECT api_key, settings_json FROM llm_providers WHERE provider_id = ?",
            ("legacy-provider",),
        ).fetchone()

    assert api_key.startswith(SECRET_CIPHERTEXT_PREFIX)
    assert "LEGACY-API-SECRET" not in api_key
    assert SETTINGS_CIPHERTEXT_PREFIX in settings_json
    assert "LEGACY-TOKEN-SECRET" not in settings_json


def test_env_encryption_key_is_recoverable_and_db_has_no_plaintext(tmp_path: Path, monkeypatch):
    db_path = tmp_path / "env-key.sqlite3"
    monkeypatch.setenv("LLM_PROVIDER_ENCRYPTION_KEY", "local-development-master-secret")

    with LLMProviderStore(db_path) as store:
        provider = store.create(
            LLMProviderCreate(
                name="Env Key",
                base_url="https://example.com/v1",
                api_key="ENV-SECRET",
                model="model-a",
            )
        )
        assert store.require(provider.provider_id).secret_value() == "ENV-SECRET"

    # A different master secret must not be able to decrypt the saved provider.
    monkeypatch.setenv("LLM_PROVIDER_ENCRYPTION_KEY", "different-master-secret")
    with LLMProviderStore(db_path) as store:
        with pytest.raises(LLMProviderSecretError):
            store.require(provider.provider_id)

@pytest.mark.skipif(__import__("os").name == "nt", reason="symlink semantics differ on Windows")
def test_key_file_symlink_is_rejected(tmp_path: Path):
    key_path = tmp_path / ".llm_provider_secret.key"
    target = tmp_path / "target.key"
    target.write_bytes(b"x" * 32)
    key_path.symlink_to(target)

    with pytest.raises(LLMProviderSecretError):
        load_encryption_key(tmp_path / "providers.sqlite3")

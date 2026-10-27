from pathlib import Path

import pytest
from pydantic import SecretStr, ValidationError

from open_llm_vtuber.config_manager.llm_provider import (
    LLMProviderCreate,
    LLMProviderStore,
    LLMProviderType,
)


def test_provider_crud_persists_and_masks_api_key(tmp_path: Path):
    db_path = tmp_path / "providers.sqlite3"

    with LLMProviderStore(db_path) as store:
        created = store.create(
            LLMProviderCreate(
                name="Local Ollama",
                provider_type=LLMProviderType.OLLAMA,
                base_url="http://127.0.0.1:11434/v1",
                api_key=SecretStr("secret-value"),
                model="qwen3:14b",
            )
        )

        assert created.provider_id
        assert created.secret_value() == "secret-value"
        public = created.to_public_dict()
        assert public["api_key_configured"] is True
        assert "secret-value" not in str(public)

        loaded = store.require(created.provider_id)
        assert loaded.model == "qwen3:14b"
        assert loaded.secret_value() == "secret-value"

        updated = store.update(
            created.provider_id,
            {"model": "qwen3:8b", "enabled": False},
        )
        assert updated.model == "qwen3:8b"
        assert updated.enabled is False
        assert updated.secret_value() == "secret-value"

        assert store.delete(created.provider_id) is True
        assert store.get(created.provider_id) is None
        assert store.delete(created.provider_id) is False

    with LLMProviderStore(db_path) as store:
        assert store.list() == []


def test_provider_requires_http_or_https_base_url():
    with pytest.raises(ValidationError):
        LLMProviderCreate(
            name="Bad URL",
            base_url="ftp://example.com/api",
        )

    with pytest.raises(ValidationError):
        LLMProviderCreate(
            name="Bad URL",
            base_url="example.com/api",
        )


def test_provider_allows_local_and_cloud_urls():
    local = LLMProviderCreate(
        name="Local",
        provider_type=LLMProviderType.LM_STUDIO,
        base_url="http://localhost:1234/v1/",
    )
    cloud = LLMProviderCreate(
        name="Cloud",
        provider_type=LLMProviderType.OPENAI_COMPATIBLE,
        base_url="https://api.example.com/v1/",
    )

    assert local.base_url == "http://localhost:1234/v1"
    assert cloud.base_url == "https://api.example.com/v1"

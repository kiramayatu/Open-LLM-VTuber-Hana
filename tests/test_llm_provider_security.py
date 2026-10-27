import os
import stat
from pathlib import Path

import pytest
from pydantic import ValidationError

from open_llm_vtuber.config_manager.llm_provider import (
    LLMProvider,
    LLMProviderCreate,
    LLMProviderStore,
)


def test_base_url_rejects_embedded_credentials_and_query_components():
    bad_urls = [
        "https://user:pass@example.com/v1",
        "https://example.com/v1?api_key=secret",
        "https://example.com/v1#secret",
    ]
    for url in bad_urls:
        with pytest.raises(ValidationError):
            LLMProviderCreate(name="Bad", base_url=url)


def test_public_settings_redact_nested_credential_like_keys():
    provider = LLMProvider.model_validate(
        {
            "provider_id": "safe-test",
            "name": "Safe",
            "base_url": "https://example.com/v1",
            "settings": {
                "safe_value": "ok",
                "api_key": "TOP_SECRET",
                "nested": {"token": "TOKEN_SECRET", "password": "PWD_SECRET"},
            },
        }
    )
    public = provider.to_public_dict()
    assert public["settings"]["safe_value"] == "ok"
    assert public["settings"]["api_key"] == "***configured***"
    assert public["settings"]["nested"]["token"] == "***configured***"
    assert public["settings"]["nested"]["password"] == "***configured***"
    assert "TOP_SECRET" not in str(public)
    assert "TOKEN_SECRET" not in str(public)
    assert "PWD_SECRET" not in str(public)


@pytest.mark.skipif(os.name == "nt", reason="chmod/ACL semantics differ on Windows")
def test_provider_database_permissions_are_owner_only(tmp_path: Path):
    db_path = tmp_path / "providers.sqlite3"
    with LLMProviderStore(db_path) as store:
        store.create(
            LLMProviderCreate(
                name="Secret Provider",
                base_url="https://example.com/v1",
                api_key="TOP_SECRET",
                model="model",
            )
        )
        mode = stat.S_IMODE(db_path.stat().st_mode)
        assert mode & 0o077 == 0

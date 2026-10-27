from pathlib import Path

from pydantic import SecretStr

from open_llm_vtuber.config_manager.agent import AgentConfig
from open_llm_vtuber.config_manager.llm_provider import (
    LLMProvider,
    LLMProviderCreate,
    LLMProviderStore,
    LLMProviderType,
)
from open_llm_vtuber.config_manager.llm_provider_runtime import (
    import_legacy_provider,
    provider_from_legacy_config,
    provider_to_legacy_config,
    resolve_managed_single_llm,
)


def make_provider(**overrides):
    payload = {
        "provider_id": "runtime-test",
        "name": "Runtime Ollama",
        "provider_type": "ollama",
        "base_url": "http://127.0.0.1:11434/v1",
        "api_key": "secret",
        "model": "qwen3:14b",
        "settings": {"temperature": 0.7},
    }
    payload.update(overrides)
    return LLMProvider.model_validate(payload)


def test_provider_maps_to_existing_ollama_runtime():
    runtime_key, config = provider_to_legacy_config(make_provider())
    assert runtime_key == "ollama_llm"
    assert config["base_url"] == "http://127.0.0.1:11434/v1"
    assert config["llm_api_key"] == "secret"
    assert config["model"] == "qwen3:14b"
    assert config["keep_alive"] == -1
    assert config["temperature"] == 0.7


def test_managed_provider_overrides_legacy_runtime_only_when_active(tmp_path: Path):
    config = AgentConfig.model_validate(
        {
            "conversation_agent_choice": "basic_memory_agent",
            "agent_settings": {
                "basic_memory_agent": {
                    "llm_provider": "ollama_llm",
                    "faster_first_response": True,
                    "segment_method": "pysbd",
                    "use_mcpp": False,
                }
            },
            "llm_configs": {
                "ollama_llm": {
                    "base_url": "http://old-host:11434/v1",
                    "llm_api_key": "old-key",
                    "model": "old-model",
                    "temperature": 1.0,
                    "keep_alive": -1,
                    "unload_at_exit": True,
                    "interrupt_method": "system",
                }
            },
        }
    )

    with LLMProviderStore(tmp_path / "providers.sqlite3") as store:
        settings, llm_configs, active = resolve_managed_single_llm(config, store)
        assert active is None
        assert settings["basic_memory_agent"]["llm_provider"] == "ollama_llm"
        assert llm_configs["ollama_llm"]["model"] == "old-model"

        provider = store.create(
            LLMProviderCreate(
                name="Managed Ollama",
                provider_type=LLMProviderType.OLLAMA,
                base_url="http://127.0.0.1:11434/v1",
                api_key=SecretStr("new-key"),
                model="qwen3:14b",
            )
        )
        store.set_active(provider.provider_id)

        settings, llm_configs, active = resolve_managed_single_llm(config, store)
        assert active == provider.provider_id
        assert settings["basic_memory_agent"]["llm_provider"] == "ollama_llm"
        assert llm_configs["ollama_llm"]["model"] == "qwen3:14b"
        assert llm_configs["ollama_llm"]["llm_api_key"] == "new-key"


def test_active_provider_is_cleared_when_deleted(tmp_path: Path):
    with LLMProviderStore(tmp_path / "providers.sqlite3") as store:
        provider = store.create(
            LLMProviderCreate(
                name="Ollama",
                provider_type=LLMProviderType.OLLAMA,
                base_url="http://localhost:11434/v1",
                model="qwen3:14b",
            )
        )
        store.set_active(provider.provider_id)
        assert store.get_active_id() == provider.provider_id
        assert store.delete(provider.provider_id) is True
        assert store.get_active_id() is None


def test_legacy_config_can_be_imported():
    config = AgentConfig.model_validate(
        {
            "conversation_agent_choice": "basic_memory_agent",
            "agent_settings": {
                "basic_memory_agent": {
                    "llm_provider": "openai_compatible_llm",
                    "faster_first_response": True,
                    "segment_method": "pysbd",
                    "use_mcpp": False,
                }
            },
            "llm_configs": {
                "openai_compatible_llm": {
                    "base_url": "https://api.example.com/v1",
                    "llm_api_key": "legacy-key",
                    "model": "model-a",
                    "temperature": 0.5,
                    "interrupt_method": "user",
                }
            },
        }
    )

    provider = provider_from_legacy_config(config, provider_name="Imported Cloud")
    assert provider is not None
    assert provider.name == "Imported Cloud"
    assert provider.base_url == "https://api.example.com/v1"
    assert provider.model == "model-a"
    assert provider.api_key.get_secret_value() == "legacy-key"

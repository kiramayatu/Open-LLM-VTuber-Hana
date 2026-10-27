import asyncio
from pathlib import Path

import httpx

_REAL_ASYNC_CLIENT = httpx.AsyncClient
from pydantic import SecretStr

from open_llm_vtuber.config_manager.llm_provider import (
    LLMProvider,
    LLMProviderCreate,
    LLMProviderType,
)
from open_llm_vtuber.config_manager.llm_provider_connection import (
    LLMProviderConnectionManager,
    LLMProviderConnectionError,
)


def _provider(**overrides):
    payload = {
        "name": "Local Ollama",
        "provider_type": LLMProviderType.OLLAMA,
        "base_url": "http://127.0.0.1:11434/v1",
        "api_key": SecretStr("secret"),
        "model": "qwen3:14b",
    }
    payload.update(overrides)
    payload["provider_id"] = "test-provider"
    return LLMProvider.model_validate(payload)


def test_ollama_discovery_and_connection_result(monkeypatch):
    requests = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert str(request.url) == "http://127.0.0.1:11434/api/tags"
        assert request.headers["authorization"] == "Bearer secret"
        return httpx.Response(
            200,
            json={
                "models": [
                    {
                        "name": "qwen3:14b",
                        "details": {
                            "parameter_size": "14B",
                            "quantization_level": "Q4_K_M",
                        },
                    },
                    {"name": "mistral:latest"},
                ]
            },
        )

    class MockClient:
        def __init__(self, *args, **kwargs):
            self._client = _REAL_ASYNC_CLIENT(transport=httpx.MockTransport(handler))

        async def __aenter__(self):
            await self._client.__aenter__()
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return await self._client.__aexit__(exc_type, exc, tb)

        async def get(self, *args, **kwargs):
            return await self._client.get(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", MockClient)

    async def run():
        manager = LLMProviderConnectionManager(timeout=5)
        provider = _provider()
        models = await manager.discover_models(provider)
        result = await manager.test_connection(provider)
        return models, result

    models, result = asyncio.run(run())
    assert len(requests) == 2
    assert [model.model_id for model in models] == ["qwen3:14b", "mistral:latest"]
    assert models[0].parameter_size == "14B"
    assert models[0].quantization == "Q4_K_M"
    assert result.ok is True
    assert result.model_count == 2
    assert result.provider_id == "test-provider"
    assert result.latency_ms is not None


def test_openai_compatible_discovery_uses_v1_models(monkeypatch):
    async def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://api.example.com/v1/models"
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {
                        "id": "model-a",
                        "owned_by": "example",
                        "context_window": 32768,
                    }
                ],
            },
        )

    class MockClient:
        def __init__(self, *args, **kwargs):
            self._client = _REAL_ASYNC_CLIENT(transport=httpx.MockTransport(handler))

        async def __aenter__(self):
            await self._client.__aenter__()
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return await self._client.__aexit__(exc_type, exc, tb)

        async def get(self, *args, **kwargs):
            return await self._client.get(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", MockClient)

    async def run():
        manager = LLMProviderConnectionManager()
        provider = LLMProvider.model_validate(
            {
                "provider_id": "openai-test",
                "name": "Cloud",
                "provider_type": "openai_compatible",
                "base_url": "https://api.example.com",
                "api_key": "abc",
                "model": "model-a",
            }
        )
        return await manager.discover_models(provider)

    models = asyncio.run(run())
    assert len(models) == 1
    assert models[0].model_id == "model-a"
    assert models[0].owned_by == "example"
    assert models[0].context_window == 32768


def test_custom_provider_can_select_ollama_protocol(monkeypatch):
    async def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://localhost:11434/api/tags"
        return httpx.Response(200, json={"models": [{"name": "custom-model"}]})

    class MockClient:
        def __init__(self, *args, **kwargs):
            self._client = _REAL_ASYNC_CLIENT(transport=httpx.MockTransport(handler))

        async def __aenter__(self):
            await self._client.__aenter__()
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return await self._client.__aexit__(exc_type, exc, tb)

        async def get(self, *args, **kwargs):
            return await self._client.get(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", MockClient)

    async def run():
        manager = LLMProviderConnectionManager()
        provider = LLMProvider.model_validate(
            {
                "provider_id": "custom-ollama",
                "name": "Custom Ollama",
                "provider_type": "custom",
                "base_url": "http://localhost:11434",
                "settings": {"api_style": "ollama"},
            }
        )
        return await manager.discover_models(provider)

    models = asyncio.run(run())
    assert [model.model_id for model in models] == ["custom-model"]


def test_connection_failure_is_safe_and_does_not_expose_api_key(monkeypatch):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "bad key"})

    class MockClient:
        def __init__(self, *args, **kwargs):
            self._client = _REAL_ASYNC_CLIENT(transport=httpx.MockTransport(handler))

        async def __aenter__(self):
            await self._client.__aenter__()
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return await self._client.__aexit__(exc_type, exc, tb)

        async def get(self, *args, **kwargs):
            return await self._client.get(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", MockClient)

    async def run():
        manager = LLMProviderConnectionManager()
        provider = LLMProvider.model_validate(
            {
                "provider_id": "bad-auth",
                "name": "Bad Auth",
                "provider_type": "openai_compatible",
                "base_url": "https://example.com/v1",
                "api_key": "SUPER_SECRET_KEY",
            }
        )
        return await manager.test_connection(provider)

    result = asyncio.run(run())
    assert result.ok is False
    assert "SUPER_SECRET_KEY" not in result.message
    assert "authentication failure" in result.message


def test_unsupported_provider_returns_clear_error():
    provider = LLMProvider.model_validate(
        {
            "provider_id": "anthropic-test",
            "name": "Anthropic",
            "provider_type": "anthropic",
            "base_url": "https://api.anthropic.com/v1",
            "api_key": "secret",
        }
    )
    manager = LLMProviderConnectionManager()

    try:
        manager.adapter_for(provider)
    except LLMProviderConnectionError as exc:
        assert "anthropic" in str(exc).lower()
    else:
        raise AssertionError("Expected Anthropic to be unsupported in Stage 2")

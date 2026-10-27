"""LLM provider connection and model-discovery layer.

Stage 2 adds a provider-agnostic HTTP layer on top of the Stage 1 provider
registry. It intentionally does not alter the existing LLM inference factory.

Supported discovery protocols in this stage:
- Ollama: GET /api/tags
- OpenAI-compatible: GET /v1/models (or /models when the configured base URL
  already ends in /v1)

The OpenAI-compatible adapter also covers common compatible local/cloud
providers such as LM Studio and llama.cpp servers. Provider-specific adapters
can be added later without changing the UI or storage layer.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from .llm_provider import LLMProvider, LLMProviderType

DEFAULT_CONNECTION_TIMEOUT = 15.0
MAX_CONNECTION_TIMEOUT = 120.0


class LLMProviderConnectionError(RuntimeError):
    """Raised when a provider cannot be contacted or returns an invalid API response."""


@dataclass(frozen=True)
class LLMModelInfo:
    """Normalized model metadata returned by a provider."""

    model_id: str
    provider_model_name: str | None = None
    owned_by: str | None = None
    context_window: int | None = None
    parameter_size: str | None = None
    quantization: str | None = None


@dataclass(frozen=True)
class LLMConnectionResult:
    """Safe result for a connection test."""

    ok: bool
    provider_id: str
    provider_name: str
    provider_type: str
    message: str
    latency_ms: float | None = None
    model_count: int | None = None


class LLMProviderAdapter(ABC):
    """Protocol adapter used by :class:`LLMProviderConnectionManager`."""

    @abstractmethod
    async def discover_models(self, provider: LLMProvider) -> list[LLMModelInfo]:
        """Return models exposed by the configured provider."""

    async def test_connection(self, provider: LLMProvider) -> LLMConnectionResult:
        started = time.perf_counter()
        try:
            models = await self.discover_models(provider)
        except LLMProviderConnectionError as exc:
            return LLMConnectionResult(
                ok=False,
                provider_id=provider.provider_id,
                provider_name=provider.name,
                provider_type=provider.provider_type.value,
                message=str(exc),
            )
        latency_ms = (time.perf_counter() - started) * 1000.0
        return LLMConnectionResult(
            ok=True,
            provider_id=provider.provider_id,
            provider_name=provider.name,
            provider_type=provider.provider_type.value,
            message="Connection successful",
            latency_ms=round(latency_ms, 2),
            model_count=len(models),
        )


class _HTTPAdapter(LLMProviderAdapter):
    """Shared HTTP behavior for discovery adapters."""

    def __init__(self, *, timeout: float = DEFAULT_CONNECTION_TIMEOUT) -> None:
        timeout = float(timeout)
        if timeout <= 0:
            raise ValueError("timeout must be greater than 0")
        self.timeout = min(timeout, MAX_CONNECTION_TIMEOUT)

    @staticmethod
    def _validate_url(provider: LLMProvider) -> str:
        if not provider.base_url:
            raise LLMProviderConnectionError("Provider base_url is required")

        parts = urlsplit(provider.base_url)
        if parts.scheme.lower() not in {"http", "https"} or not parts.netloc:
            raise LLMProviderConnectionError("Provider base_url must use HTTP or HTTPS")
        if parts.username or parts.password:
            raise LLMProviderConnectionError("Credentials in provider URLs are not supported")

        # Preserve path but remove trailing slash for predictable endpoint joining.
        return urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), "", ""))

    @staticmethod
    def _headers(provider: LLMProvider) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        secret = provider.secret_value()
        if secret:
            headers["Authorization"] = f"Bearer {secret}"
        return headers

    async def _get_json(
        self,
        provider: LLMProvider,
        url: str,
    ) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout,
                follow_redirects=False,
            ) as client:
                response = await client.get(url, headers=self._headers(provider))
        except httpx.TimeoutException as exc:
            raise LLMProviderConnectionError(
                "Connection timed out while contacting the LLM provider"
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMProviderConnectionError(
                "Could not connect to the LLM provider"
            ) from exc

        if response.is_redirect:
            raise LLMProviderConnectionError(
                f"Provider returned redirect status {response.status_code}; redirects are disabled for security"
            )
        if response.status_code in {401, 403}:
            raise LLMProviderConnectionError("Provider rejected the credentials (HTTP authentication failure)")
        if response.status_code == 404:
            raise LLMProviderConnectionError("Model-list endpoint was not found on the provider")
        if response.status_code >= 400:
            raise LLMProviderConnectionError(
                f"Provider returned HTTP {response.status_code}"
            )

        content_type = response.headers.get("content-type", "")
        if "json" not in content_type.lower():
            raise LLMProviderConnectionError("Provider returned a non-JSON response")

        try:
            payload = response.json()
        except ValueError as exc:
            raise LLMProviderConnectionError("Provider returned invalid JSON") from exc

        if not isinstance(payload, dict):
            raise LLMProviderConnectionError("Provider returned an unexpected JSON shape")
        return payload


class OllamaProviderAdapter(_HTTPAdapter):
    """Ollama model discovery using its native /api/tags endpoint."""

    @staticmethod
    def _server_root(base_url: str) -> str:
        parts = urlsplit(base_url)
        path = parts.path.rstrip("/")
        if path.endswith("/v1"):
            path = path[:-3]
        elif path.endswith("/api"):
            path = path[:-4]
        return urlunsplit((parts.scheme, parts.netloc, path.rstrip("/"), "", ""))

    async def discover_models(self, provider: LLMProvider) -> list[LLMModelInfo]:
        base_url = self._validate_url(provider)
        url = f"{self._server_root(base_url)}/api/tags"
        payload = await self._get_json(provider, url)
        raw_models = payload.get("models")
        if not isinstance(raw_models, list):
            raise LLMProviderConnectionError("Ollama returned no valid 'models' list")

        models: list[LLMModelInfo] = []
        for item in raw_models:
            if not isinstance(item, dict):
                continue
            model_id = item.get("name") or item.get("model")
            if not isinstance(model_id, str) or not model_id.strip():
                continue
            details = item.get("details")
            if not isinstance(details, dict):
                details = {}
            models.append(
                LLMModelInfo(
                    model_id=model_id.strip(),
                    provider_model_name=model_id.strip(),
                    parameter_size=_optional_str(details.get("parameter_size")),
                    quantization=_optional_str(details.get("quantization_level")),
                )
            )
        return models


class OpenAICompatibleProviderAdapter(_HTTPAdapter):
    """Model discovery for providers exposing an OpenAI-compatible models API."""

    @staticmethod
    def _models_url(base_url: str) -> str:
        parts = urlsplit(base_url)
        path = parts.path.rstrip("/")
        if path.endswith("/v1"):
            models_path = f"{path}/models"
        else:
            models_path = f"{path}/v1/models" if path else "/v1/models"
        return urlunsplit((parts.scheme, parts.netloc, models_path, "", ""))

    async def discover_models(self, provider: LLMProvider) -> list[LLMModelInfo]:
        base_url = self._validate_url(provider)
        custom_path = provider.settings.get("models_endpoint")
        if custom_path is not None:
            if not isinstance(custom_path, str) or not custom_path.startswith("/"):
                raise LLMProviderConnectionError(
                    "settings.models_endpoint must be an absolute API path beginning with '/'")
            if "?" in custom_path or "#" in custom_path:
                raise LLMProviderConnectionError(
                    "settings.models_endpoint must not contain query parameters or fragments")
            parts = urlsplit(base_url)
            url = urlunsplit((parts.scheme, parts.netloc, custom_path, "", ""))
        else:
            url = self._models_url(base_url)

        payload = await self._get_json(provider, url)
        raw_models = payload.get("data")
        if not isinstance(raw_models, list):
            raise LLMProviderConnectionError("Provider returned no valid 'data' model list")

        models: list[LLMModelInfo] = []
        for item in raw_models:
            if not isinstance(item, dict):
                continue
            model_id = item.get("id")
            if not isinstance(model_id, str) or not model_id.strip():
                continue
            models.append(
                LLMModelInfo(
                    model_id=model_id.strip(),
                    provider_model_name=model_id.strip(),
                    owned_by=_optional_str(item.get("owned_by")),
                    context_window=_optional_int(
                        item.get("context_window") or item.get("max_context_length")
                    ),
                    parameter_size=_optional_str(item.get("parameter_size")),
                    quantization=_optional_str(item.get("quantization")),
                )
            )
        return models


class LLMProviderConnectionManager:
    """Resolve a configured provider to an adapter and expose safe operations."""

    _OPENAI_COMPATIBLE_TYPES = {
        LLMProviderType.OPENAI_COMPATIBLE,
        LLMProviderType.OPENAI,
        LLMProviderType.LM_STUDIO,
        LLMProviderType.LLAMA_CPP,
        LLMProviderType.MISTRAL,
        LLMProviderType.DEEPSEEK,
        LLMProviderType.GROQ,
    }

    def __init__(self, *, timeout: float = DEFAULT_CONNECTION_TIMEOUT) -> None:
        self.timeout = min(max(float(timeout), 0.1), MAX_CONNECTION_TIMEOUT)
        self._ollama = OllamaProviderAdapter(timeout=self.timeout)
        self._openai_compatible = OpenAICompatibleProviderAdapter(timeout=self.timeout)

    def adapter_for(self, provider: LLMProvider) -> LLMProviderAdapter:
        if provider.provider_type == LLMProviderType.OLLAMA:
            return self._ollama
        if provider.provider_type in self._OPENAI_COMPATIBLE_TYPES:
            return self._openai_compatible
        if provider.provider_type == LLMProviderType.CUSTOM:
            protocol = str(provider.settings.get("api_style", "openai_compatible")).strip().lower()
            if protocol in {"openai", "openai_compatible", "openai-compatible"}:
                return self._openai_compatible
            if protocol == "ollama":
                return self._ollama
        raise LLMProviderConnectionError(
            f"Provider type '{provider.provider_type.value}' does not have a Stage 2 connection adapter yet"
        )

    async def discover_models(self, provider: LLMProvider) -> list[LLMModelInfo]:
        return await self.adapter_for(provider).discover_models(provider)

    async def test_connection(self, provider: LLMProvider) -> LLMConnectionResult:
        try:
            return await self.adapter_for(provider).test_connection(provider)
        except LLMProviderConnectionError as exc:
            return LLMConnectionResult(
                ok=False,
                provider_id=provider.provider_id,
                provider_name=provider.name,
                provider_type=provider.provider_type.value,
                message=str(exc),
            )


def _optional_str(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None

"""Bridge managed LLM providers into the existing single-LLM runtime.

Stage 3 deliberately keeps the existing LLM engine and AgentFactory intact.
When a user explicitly selects a managed provider, this module translates that
provider record into the legacy ``llm_provider`` + ``llm_configs`` shape already
consumed by the current agent stack. With no active managed provider, callers
continue using the YAML configuration unchanged.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .agent import AgentConfig
from .llm_provider import (
    DEFAULT_LLM_PROVIDER_DB,
    LLMProvider,
    LLMProviderCreate,
    LLMProviderStore,
    LLMProviderType,
)

_RUNTIME_PROVIDER_BY_TYPE: dict[LLMProviderType, str] = {
    LLMProviderType.OLLAMA: "ollama_llm",
    LLMProviderType.LM_STUDIO: "lmstudio_llm",
    LLMProviderType.OPENAI_COMPATIBLE: "openai_compatible_llm",
    LLMProviderType.OPENAI: "openai_llm",
    LLMProviderType.GEMINI: "gemini_llm",
    LLMProviderType.MISTRAL: "mistral_llm",
    LLMProviderType.ZHIPU: "zhipu_llm",
    LLMProviderType.DEEPSEEK: "deepseek_llm",
    LLMProviderType.GROQ: "groq_llm",
    LLMProviderType.ANTHROPIC: "claude_llm",
    # llama.cpp HTTP servers expose an OpenAI-compatible API in this project.
    LLMProviderType.LLAMA_CPP: "openai_compatible_llm",
}

_LEGACY_TYPE_BY_KEY: dict[str, LLMProviderType] = {
    "ollama_llm": LLMProviderType.OLLAMA,
    "lmstudio_llm": LLMProviderType.LM_STUDIO,
    "openai_compatible_llm": LLMProviderType.OPENAI_COMPATIBLE,
    "openai_llm": LLMProviderType.OPENAI,
    "gemini_llm": LLMProviderType.GEMINI,
    "mistral_llm": LLMProviderType.MISTRAL,
    "zhipu_llm": LLMProviderType.ZHIPU,
    "deepseek_llm": LLMProviderType.DEEPSEEK,
    "groq_llm": LLMProviderType.GROQ,
    "claude_llm": LLMProviderType.ANTHROPIC,
    "llama_cpp_llm": LLMProviderType.LLAMA_CPP,
}

_ALLOWED_RUNTIME_PROVIDERS = {
    "ollama_llm",
    "lmstudio_llm",
    "openai_compatible_llm",
    "openai_llm",
    "gemini_llm",
    "mistral_llm",
    "zhipu_llm",
    "deepseek_llm",
    "groq_llm",
    "claude_llm",
}


def provider_runtime_key(provider: LLMProvider) -> str:
    """Return the existing AgentFactory provider key for a managed provider."""
    if provider.provider_type is LLMProviderType.CUSTOM:
        runtime_provider = str(
            provider.settings.get("runtime_provider", "openai_compatible_llm")
        ).strip()
        if runtime_provider not in _ALLOWED_RUNTIME_PROVIDERS:
            raise ValueError(
                f"Unsupported custom runtime_provider '{runtime_provider}'"
            )
        return runtime_provider

    runtime_provider = _RUNTIME_PROVIDER_BY_TYPE.get(provider.provider_type)
    if runtime_provider is None:
        raise ValueError(
            f"Provider type '{provider.provider_type.value}' cannot be mapped to the current LLM runtime"
        )
    return runtime_provider


def provider_to_legacy_config(provider: LLMProvider) -> tuple[str, dict[str, Any]]:
    """Translate a managed provider to the current AgentFactory config shape."""
    runtime_provider = provider_runtime_key(provider)
    settings = dict(provider.settings)
    config: dict[str, Any] = {
        "base_url": provider.base_url,
        "llm_api_key": provider.secret_value() or "",
        "model": provider.model or "",
        "temperature": _number_setting(settings, "temperature", 1.0),
        "interrupt_method": _interrupt_method(settings),
    }

    if runtime_provider == "ollama_llm":
        config["llm_api_key"] = provider.secret_value() or "default_api_key"
        config["keep_alive"] = _number_setting(settings, "keep_alive", -1)
        config["unload_at_exit"] = bool(settings.get("unload_at_exit", True))
        config["interrupt_method"] = settings.get("interrupt_method", "system")
    elif runtime_provider == "lmstudio_llm":
        config["base_url"] = provider.base_url or "http://localhost:1234/v1"
        config["llm_api_key"] = provider.secret_value() or "default_api_key"
        config["interrupt_method"] = settings.get("interrupt_method", "system")
    elif runtime_provider == "claude_llm":
        config["base_url"] = provider.base_url or "https://api.anthropic.com"
        config["interrupt_method"] = settings.get("interrupt_method", "user")
    elif runtime_provider in {"openai_llm", "gemini_llm", "mistral_llm"}:
        config["interrupt_method"] = settings.get("interrupt_method", "system" if runtime_provider == "openai_llm" else "user")

    for field in ("organization_id", "project_id", "template"):
        if field in settings:
            config[field] = settings[field]

    if runtime_provider == "openai_compatible_llm" and not provider.base_url:
        raise ValueError("Managed provider requires base_url for OpenAI-compatible runtime")
    if not provider.model:
        raise ValueError("Managed provider requires a selected model")

    return runtime_provider, config


def resolve_managed_single_llm(
    agent_config: AgentConfig,
    store: LLMProviderStore,
) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    """Apply the explicitly selected managed provider to the legacy runtime shape.

    Returns ``(agent_settings, llm_configs, provider_id)``. If there is no active
    managed provider, the returned dictionaries are equivalent to the existing
    ``model_dump()`` values and ``provider_id`` is ``None``.
    """
    agent_settings = agent_config.agent_settings.model_dump()
    llm_configs = agent_config.llm_configs.model_dump()
    active = store.get_active()
    if active is None:
        return agent_settings, llm_configs, None

    runtime_provider, runtime_config = provider_to_legacy_config(active)
    basic = dict(agent_settings.get("basic_memory_agent", {}))
    basic["llm_provider"] = runtime_provider
    agent_settings["basic_memory_agent"] = basic
    llm_configs[runtime_provider] = runtime_config
    return agent_settings, llm_configs, active.provider_id


def provider_from_legacy_config(
    agent_config: AgentConfig,
    *,
    provider_name: str | None = None,
) -> LLMProviderCreate | None:
    """Build an importable managed-provider record from the current YAML config.

    Direct local ``llama_cpp_llm`` model-path configurations are intentionally not
    imported because the managed provider system is HTTP/HTTPS based.
    """
    basic = agent_config.agent_settings.model_dump().get("basic_memory_agent", {})
    legacy_key = str(basic.get("llm_provider") or "").strip()
    provider_type = _LEGACY_TYPE_BY_KEY.get(legacy_key)
    if provider_type is None:
        return None

    legacy = agent_config.llm_configs.model_dump().get(legacy_key)
    if not isinstance(legacy, dict):
        return None

    base_url = legacy.get("base_url")
    if provider_type is LLMProviderType.LLAMA_CPP and not base_url:
        return None
    if not isinstance(base_url, str) or not base_url.strip():
        if provider_type is LLMProviderType.LLAMA_CPP:
            return None
        return None

    model = legacy.get("model")
    if not isinstance(model, str) or not model.strip():
        return None

    reserved = {
        "base_url",
        "llm_api_key",
        "model",
        "organization_id",
        "project_id",
        "template",
        "temperature",
        "interrupt_method",
        "keep_alive",
        "unload_at_exit",
    }
    settings = {key: value for key, value in legacy.items() if key not in reserved}
    for key in ("organization_id", "project_id", "template", "temperature", "interrupt_method", "keep_alive", "unload_at_exit"):
        if key in legacy:
            settings[key] = legacy[key]

    return LLMProviderCreate(
        name=provider_name or f"Imported {legacy_key}",
        provider_type=provider_type,
        base_url=base_url.strip().rstrip("/"),
        api_key=legacy.get("llm_api_key") or None,
        model=model.strip(),
        enabled=True,
        settings=settings,
    )


def import_legacy_provider(
    agent_config: AgentConfig,
    store: LLMProviderStore,
    *,
    provider_name: str | None = None,
    activate: bool = False,
) -> LLMProvider | None:
    """Import the current YAML-selected LLM into the managed provider store."""
    provider_create = provider_from_legacy_config(
        agent_config,
        provider_name=provider_name,
    )
    if provider_create is None:
        return None

    for existing in store.list():
        if (
            existing.provider_type == provider_create.provider_type
            and existing.base_url == provider_create.base_url
            and existing.model == provider_create.model
        ):
            provider = existing
            break
    else:
        provider = store.create(provider_create)

    if activate:
        store.set_active(provider.provider_id)
    return provider


def _number_setting(settings: dict[str, Any], key: str, default: float | int) -> float | int:
    value = settings.get(key, default)
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        try:
            number = float(str(value).strip())
        except (TypeError, ValueError):
            return default
    if not __import__("math").isfinite(number):
        return default
    return int(number) if isinstance(default, int) and number.is_integer() else number


def _interrupt_method(settings: dict[str, Any]) -> str:
    value = str(settings.get("interrupt_method", "user")).strip().lower()
    return value if value in {"system", "user"} else "user"


def managed_store_path() -> Path:
    configured = os.getenv("LLM_PROVIDER_STORE_PATH") or DEFAULT_LLM_PROVIDER_DB
    return Path(configured).expanduser().resolve()

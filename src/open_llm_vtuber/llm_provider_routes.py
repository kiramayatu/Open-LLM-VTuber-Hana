"""Authenticated HTTP API for managed LLM providers.

Stage 4 exposes the provider registry and connection/model-discovery operations to
 the existing frontend. Secrets remain encrypted at rest and are never returned by
normal provider reads. An explicit authenticated reveal endpoint exists only for the
user-facing eye/reveal action.
"""
from __future__ import annotations

import json
import re
import secrets as _secrets
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger
import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator

from .config_manager.llm_provider import (
    LLMProvider,
    LLMProviderCreate,
    LLMProviderStore,
    LLMProviderType,
)
from .config_manager.llm_provider_connection import (
    LLMConnectionResult,
    LLMProviderConnectionError,
    LLMProviderConnectionManager,
    LLMModelInfo,
)
from .config_manager.agent import AgentConfig
from .config_manager.llm_provider_runtime import import_legacy_provider


security = HTTPBearer(auto_error=False)
MAX_PROVIDER_SETTINGS_BYTES = 32 * 1024
MAX_PROVIDERS_PER_RESPONSE = 256
MAX_IMPORTED_YAML_BYTES = 512 * 1024


class LLMProviderUpdate(BaseModel):
    """Partial provider update. Omitted API key preserves the current secret."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=120)
    provider_type: LLMProviderType | None = None
    base_url: str | None = Field(default=None, max_length=2048)
    api_key: SecretStr | None = None
    model: str | None = Field(default=None, max_length=512)
    enabled: bool | None = None
    settings: dict[str, Any] | None = None

    @field_validator("name", "model")
    @classmethod
    def normalize_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str | None) -> str | None:
        # The canonical URL validation is performed again by LLMProvider when the
        # store validates the merged record. Keeping this field validator light
        # lets us preserve the distinction between omitted and explicit null.
        return value

    @field_validator("settings")
    @classmethod
    def validate_settings_size(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return None
        _validate_settings_payload(value)
        return value


class LLMProviderProbeRequest(BaseModel):
    """Provider data used for unsaved connection/model discovery.

    ``provider_id`` is optional. When supplied, the backend uses the stored secret
    unless a new ``api_key`` is explicitly included, allowing users to test edited
    URLs/models without first revealing an existing secret to the browser.
    """

    model_config = ConfigDict(extra="forbid")

    provider_id: str | None = Field(default=None, max_length=128)
    name: str | None = Field(default="Connection Test", min_length=1, max_length=120)
    provider_type: LLMProviderType | None = None
    base_url: str | None = Field(default=None, max_length=2048)
    api_key: SecretStr | None = None
    model: str | None = Field(default=None, max_length=512)
    enabled: bool = True
    settings: dict[str, Any] | None = None

    @field_validator("name", "model")
    @classmethod
    def normalize_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator("settings")
    @classmethod
    def validate_settings_size(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        if value is None:
            return None
        _validate_settings_payload(value)
        return value


class LLMProviderImportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider_name: str | None = Field(default=None, max_length=120)
    activate: bool = False

    @field_validator("provider_name")
    @classmethod
    def normalize_provider_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None


class LLMProviderYamlFileImportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str = Field(min_length=1, max_length=255)
    content: str = Field(min_length=1, max_length=MAX_IMPORTED_YAML_BYTES)
    provider_name: str | None = Field(default=None, max_length=120)
    activate: bool = False

    @field_validator("filename")
    @classmethod
    def normalize_filename(cls, value: str) -> str:
        value = value.strip()
        if not value or value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError("filename must be a simple file name")
        return value

    @field_validator("provider_name")
    @classmethod
    def normalize_provider_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None


def _validate_settings_payload(value: dict[str, Any]) -> None:
    """Bound provider-specific settings before accepting them from the UI."""
    try:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError("settings must contain JSON-compatible values") from exc
    if len(encoded.encode("utf-8")) > MAX_PROVIDER_SETTINGS_BYTES:
        raise ValueError(
            f"settings exceeds the {MAX_PROVIDER_SETTINGS_BYTES // 1024} KiB limit"
        )


def _normalize_api_key(value: str | None) -> str:
    if value is None:
        return ""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1].strip()
    return value


def _auth_dependency(context):
    async def verify_api_key(
        auth: HTTPAuthorizationCredentials = Depends(security),
    ):
        expected = _normalize_api_key(context.system_config.api_key)
        provided = _normalize_api_key(auth.credentials if auth else None)
        if not expected:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Server API key is not configured",
            )
        if not auth or not provided or not _secrets.compare_digest(provided, expected):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or Missing API Key",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return provided

    return verify_api_key


def _public_provider(provider: LLMProvider) -> dict[str, Any]:
    return provider.to_public_dict()


def _serialized_models(models: list[LLMModelInfo]) -> list[dict[str, Any]]:
    return [
        {
            "model_id": model.model_id,
            "provider_model_name": model.provider_model_name,
            "owned_by": model.owned_by,
            "context_window": model.context_window,
            "parameter_size": model.parameter_size,
            "quantization": model.quantization,
        }
        for model in models
    ]


async def _validated_body(request: Request, model_type: type[BaseModel]) -> BaseModel:
    """Parse/validate provider JSON without echoing raw credential-bearing input.

    FastAPI's normal body-validation response includes the original invalid input
    value. That is undesirable for secret-bearing provider configuration, because an
    invalid URL can itself contain credentials. Provider routes therefore parse the
    JSON body manually and return a fixed validation message on model errors.
    """
    try:
        raw = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid JSON body") from exc
    if not isinstance(raw, dict):
        raise HTTPException(status_code=400, detail="JSON body must be an object")
    try:
        return model_type.model_validate(raw)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail="Invalid LLM provider configuration") from exc


def _probe_provider(
    store: LLMProviderStore,
    payload: LLMProviderProbeRequest,
) -> LLMProvider:
    """Build an in-memory provider for test/discovery without persisting it."""
    requested = payload.model_dump(exclude_unset=True)
    provider_id = requested.pop("provider_id", None)

    if provider_id:
        existing = store.get(provider_id)
        if existing is None:
            raise HTTPException(status_code=404, detail="LLM provider not found")
        merged = existing.model_dump()
        # Preserve the stored secret when the browser intentionally leaves the
        # API-key field out of a draft probe request.
        if "api_key" not in requested:
            requested["api_key"] = existing.api_key
        merged.update(requested)
        if merged.get("provider_type") is None:
            merged["provider_type"] = existing.provider_type
        if merged.get("name") is None:
            merged["name"] = existing.name
        if merged.get("settings") is None:
            merged["settings"] = existing.settings
        return LLMProvider.model_validate(merged)

    provider_type = requested.pop("provider_type", None) or LLMProviderType.CUSTOM
    name = requested.pop("name", None) or "Connection Test"
    settings = requested.pop("settings", None) or {}
    return LLMProvider(
        provider_id=f"probe-{uuid4().hex}",
        name=name,
        provider_type=provider_type,
        settings=settings,
        **requested,
    )


def _safe_connection_error(exc: Exception) -> str:
    """Return a safe connection error without echoing request secrets."""
    if isinstance(exc, LLMProviderConnectionError):
        return str(exc)
    return "Could not contact the LLM provider"


async def _refresh_runtime(context) -> bool:
    """Apply the active managed provider to the default runtime when possible."""
    refresh = getattr(context, "refresh_managed_llm_provider", None)
    if not callable(refresh):
        return False
    try:
        return bool(await refresh())
    except Exception:
        # Do not log the exception object; an underlying runtime/library error can
        # contain provider configuration or other secret-bearing context.
        logger.error("Managed LLM provider was saved but runtime refresh failed")
        return False


def init_llm_provider_routes(default_context_cache) -> APIRouter:
    """Create authenticated CRUD, test, discovery, and secret-reveal routes."""
    router = APIRouter(prefix="/llm/providers", tags=["llm-providers"])
    verify_api_key = _auth_dependency(default_context_cache)

    def open_store() -> LLMProviderStore:
        return LLMProviderStore()

    @router.get("", dependencies=[Depends(verify_api_key)])
    async def list_providers() -> dict[str, Any]:
        with open_store() as store:
            providers = store.list()
            if len(providers) > MAX_PROVIDERS_PER_RESPONSE:
                raise HTTPException(
                    status_code=413,
                    detail="Too many configured LLM providers",
                )
            return {
                "providers": [_public_provider(item) for item in providers],
                "active_provider_id": store.get_active_id(),
            }

    @router.post("", dependencies=[Depends(verify_api_key)])
    async def create_provider(request: Request) -> dict[str, Any]:
        payload = await _validated_body(request, LLMProviderCreate)
        try:
            _validate_settings_payload(payload.settings)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        with open_store() as store:
            try:
                provider = store.create(payload)
            except (ValueError, ValidationError) as exc:
                raise HTTPException(status_code=409, detail="Could not create LLM provider") from exc
            return {"provider": _public_provider(provider)}

    @router.patch("/{provider_id}", dependencies=[Depends(verify_api_key)])
    async def update_provider(provider_id: str, request: Request) -> dict[str, Any]:
        payload = await _validated_body(request, LLMProviderUpdate)
        with open_store() as store:
            was_active = store.get_active_id() == provider_id
            try:
                changes = payload.model_dump(exclude_unset=True)
                provider = store.update(provider_id, changes)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="LLM provider not found") from exc
            except (ValueError, ValidationError) as exc:
                raise HTTPException(status_code=422, detail="Invalid LLM provider configuration") from exc
            runtime_applied = await _refresh_runtime(default_context_cache) if was_active else False
            return {
                "provider": _public_provider(provider),
                "active_provider_id": store.get_active_id(),
                "runtime_applied": runtime_applied,
                "requires_reconnect": was_active,
            }

    @router.delete("/{provider_id}", dependencies=[Depends(verify_api_key)])
    async def delete_provider(provider_id: str) -> dict[str, Any]:
        with open_store() as store:
            was_active = store.get_active_id() == provider_id
            if not store.delete(provider_id):
                raise HTTPException(status_code=404, detail="LLM provider not found")
            runtime_applied = await _refresh_runtime(default_context_cache) if was_active else False
            return {
                "success": True,
                "provider_id": provider_id,
                "active_provider_id": store.get_active_id(),
                "runtime_applied": runtime_applied,
                "requires_reconnect": was_active,
            }

    @router.post("/{provider_id}/activate", dependencies=[Depends(verify_api_key)])
    async def activate_provider(provider_id: str) -> dict[str, Any]:
        with open_store() as store:
            try:
                provider = store.set_active(provider_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="LLM provider not found") from exc
            except ValueError as exc:
                raise HTTPException(status_code=409, detail="LLM provider is disabled") from exc
            runtime_applied = await _refresh_runtime(default_context_cache)
            return {
                "provider": _public_provider(provider),
                "active_provider_id": provider.provider_id,
                "runtime_applied": runtime_applied,
                "requires_reconnect": True,
            }

    @router.post("/deactivate", dependencies=[Depends(verify_api_key)])
    async def deactivate_provider() -> dict[str, Any]:
        with open_store() as store:
            store.set_active(None)
            runtime_applied = await _refresh_runtime(default_context_cache)
            return {
                "active_provider_id": None,
                "runtime_applied": runtime_applied,
                "requires_reconnect": True,
            }

    @router.post("/test", dependencies=[Depends(verify_api_key)])
    async def test_unsaved_or_draft_provider(request: Request) -> Response:
        payload = await _validated_body(request, LLMProviderProbeRequest)
        with open_store() as store:
            try:
                provider = _probe_provider(store, payload)
            except HTTPException:
                raise
            except (ValueError, ValidationError) as exc:
                raise HTTPException(status_code=422, detail="Invalid LLM provider configuration") from exc

        try:
            result = await LLMProviderConnectionManager().test_connection(provider)
        except Exception:
            # Never log the exception object here: backend/library errors may embed
            # a provider credential or another sensitive request value.
            logger.error("LLM provider connection test failed")
            result = LLMConnectionResult(
                ok=False,
                provider_id=provider.provider_id,
                provider_name=provider.name,
                provider_type=provider.provider_type.value,
                message="Could not contact the LLM provider",
            )
        body = result.__dict__
        return Response(
            content=json.dumps(body, ensure_ascii=False),
            media_type="application/json",
            status_code=200 if result.ok else 502,
            headers={"Cache-Control": "no-store"},
        )

    @router.get("/models/{provider_id}", dependencies=[Depends(verify_api_key)])
    async def discover_saved_provider_models(provider_id: str) -> Response:
        """Fallback discovery route for saved providers.

        Some reverse proxies or stale deployments can reject the JSON POST model
        discovery route with HTTP 405. Saved providers can safely use this GET
        route because the credential remains entirely server-side.
        """
        with open_store() as store:
            try:
                provider = store.require(provider_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="LLM provider not found") from exc
        try:
            models = await LLMProviderConnectionManager().discover_models(provider)
        except Exception as exc:
            return Response(
                content=json.dumps({"error": _safe_connection_error(exc)}, ensure_ascii=False),
                media_type="application/json",
                status_code=502,
                headers={"Cache-Control": "no-store"},
            )
        return Response(
            content=json.dumps({"models": _serialized_models(models)}, ensure_ascii=False),
            media_type="application/json",
            headers={"Cache-Control": "no-store"},
        )

    @router.post("/models", dependencies=[Depends(verify_api_key)])
    async def discover_unsaved_or_draft_models(request: Request) -> Response:
        payload = await _validated_body(request, LLMProviderProbeRequest)
        with open_store() as store:
            try:
                provider = _probe_provider(store, payload)
            except HTTPException:
                raise
            except (ValueError, ValidationError) as exc:
                raise HTTPException(status_code=422, detail="Invalid LLM provider configuration") from exc

        try:
            models = await LLMProviderConnectionManager().discover_models(provider)
        except Exception as exc:
            return Response(
                content=json.dumps({"error": _safe_connection_error(exc)}, ensure_ascii=False),
                media_type="application/json",
                status_code=502,
                headers={"Cache-Control": "no-store"},
            )
        return Response(
            content=json.dumps({"models": _serialized_models(models)}, ensure_ascii=False),
            media_type="application/json",
            headers={"Cache-Control": "no-store"},
        )

    @router.post("/{provider_id}/reveal-secret", dependencies=[Depends(verify_api_key)])
    async def reveal_secret(provider_id: str) -> Response:
        with open_store() as store:
            try:
                provider = store.require(provider_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="LLM provider not found") from exc
            secret = provider.secret_value()

        if not secret:
            raise HTTPException(status_code=404, detail="No API key is configured for this provider")

        # Explicit reveal is the only normal frontend operation that receives the
        # stored plaintext. Prevent intermediary/browser caches from retaining it.
        return Response(
            content=json.dumps({"api_key": secret}, ensure_ascii=False),
            media_type="application/json",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, private",
                "Pragma": "no-cache",
                "Expires": "0",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @router.post("/import-file", dependencies=[Depends(verify_api_key)])
    async def import_provider_from_yaml_file(request: Request) -> dict[str, Any]:
        """Import one LLM provider from a user-selected YAML configuration file.

        The browser sends the file contents as JSON text instead of multipart data,
        keeping the endpoint simple and avoiding any server-side file persistence.
        Only the character agent configuration is parsed and converted to a managed
        provider; unrelated YAML sections are ignored.
        """
        payload = await _validated_body(request, LLMProviderYamlFileImportRequest)
        filename = payload.filename.strip()
        if not re.search(r"\.(?:ya?ml)$", filename, re.IGNORECASE):
            raise HTTPException(status_code=422, detail="Only .yaml or .yml files can be imported")
        encoded_size = len(payload.content.encode("utf-8"))
        if encoded_size > MAX_IMPORTED_YAML_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Configuration file exceeds the {MAX_IMPORTED_YAML_BYTES // 1024} KiB limit",
            )
        try:
            raw = yaml.safe_load(payload.content)
        except yaml.YAMLError as exc:
            raise HTTPException(status_code=422, detail="Invalid YAML configuration") from exc
        if not isinstance(raw, dict):
            raise HTTPException(status_code=422, detail="YAML configuration must contain an object at the top level")

        character = raw.get("character_config")
        if not isinstance(character, dict):
            raise HTTPException(status_code=422, detail="YAML file does not contain character_config")
        agent_raw = character.get("agent_config")
        if not isinstance(agent_raw, dict):
            raise HTTPException(status_code=422, detail="YAML file does not contain character_config.agent_config")
        try:
            agent_config = AgentConfig.model_validate(agent_raw)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail="The uploaded agent_config is invalid") from exc

        with open_store() as store:
            try:
                provider = import_legacy_provider(
                    agent_config,
                    store,
                    provider_name=payload.provider_name or f"Imported {filename}",
                    activate=payload.activate,
                )
            except (ValueError, ValidationError) as exc:
                raise HTTPException(status_code=422, detail="Could not import the LLM configuration") from exc
            if provider is None:
                raise HTTPException(
                    status_code=409,
                    detail="The uploaded YAML LLM configuration cannot be imported into the managed provider system",
                )
            runtime_applied = await _refresh_runtime(default_context_cache) if payload.activate else False
            return {
                "provider": _public_provider(provider),
                "active_provider_id": store.get_active_id(),
                "runtime_applied": runtime_applied,
                "requires_reconnect": payload.activate,
            }

    @router.post("/import-current", dependencies=[Depends(verify_api_key)])
    async def import_current_provider(request: Request) -> dict[str, Any]:
        payload = await _validated_body(request, LLMProviderImportRequest)
        if not getattr(default_context_cache, "character_config", None):
            raise HTTPException(status_code=503, detail="Character configuration is not loaded")
        agent_config = getattr(default_context_cache.character_config, "agent_config", None)
        if agent_config is None:
            raise HTTPException(status_code=503, detail="Agent configuration is not loaded")

        with open_store() as store:
            try:
                provider = import_legacy_provider(
                    agent_config,
                    store,
                    provider_name=payload.provider_name,
                    activate=payload.activate,
                )
            except (ValueError, ValidationError) as exc:
                raise HTTPException(status_code=422, detail="Could not import current LLM configuration") from exc
            if provider is None:
                raise HTTPException(
                    status_code=409,
                    detail="The current LLM configuration cannot be imported into the managed provider system",
                )
            runtime_applied = await _refresh_runtime(default_context_cache) if payload.activate else False
            return {
                "provider": _public_provider(provider),
                "active_provider_id": store.get_active_id(),
                "runtime_applied": runtime_applied,
                "requires_reconnect": payload.activate,
            }

    return router

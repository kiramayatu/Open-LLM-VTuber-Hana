"""Persistent LLM provider configuration foundation.

Stage 1 of the LLM Provider & Model UI work introduces a small provider registry
without changing the existing LLM inference path.  The registry is intentionally
provider-agnostic so later stages can add connection testing, model discovery,
and a frontend UI around the same data model.
"""
from __future__ import annotations

import os
import re
import sqlite3
import threading
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from .llm_provider_secrets import (
    LLMProviderSecretError,
    SETTINGS_CIPHERTEXT_PREFIX,
    SECRET_CIPHERTEXT_PREFIX,
    decrypt_text,
    encrypt_text,
    is_encrypted,
    load_encryption_key,
)


DEFAULT_LLM_PROVIDER_DB = ".llm_providers.sqlite3"
_PROVIDER_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SENSITIVE_SETTING_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "auth_header",
    "token",
    "access_token",
    "refreshtoken",
    "refresh_token",
    "client_secret",
    "secret",
    "password",
    "passphrase",
}


class LLMProviderType(str, Enum):
    """Known provider families plus an explicit custom option."""

    OLLAMA = "ollama"
    LM_STUDIO = "lmstudio"
    LLAMA_CPP = "llama_cpp"
    OPENAI_COMPATIBLE = "openai_compatible"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    MISTRAL = "mistral"
    DEEPSEEK = "deepseek"
    GROQ = "groq"
    ZHIPU = "zhipu"
    CUSTOM = "custom"


class LLMProvider(BaseModel):
    """User-managed LLM connection configuration.

    ``api_key`` is represented as ``SecretStr`` so accidental ``model_dump`` or
    logging of this model does not expose the credential. Provider persistence
    additionally encrypts the secret before writing it to SQLite.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    provider_id: str = Field(default_factory=lambda: uuid4().hex)
    name: str = Field(min_length=1, max_length=120)
    provider_type: LLMProviderType = LLMProviderType.CUSTOM
    base_url: str | None = Field(default=None, max_length=2048)
    api_key: SecretStr | None = None
    model: str | None = Field(default=None, max_length=512)
    enabled: bool = True
    settings: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=lambda: _utc_now())
    updated_at: str = Field(default_factory=lambda: _utc_now())

    @field_validator("provider_id")
    @classmethod
    def validate_provider_id(cls, value: str) -> str:
        value = value.strip()
        if not _PROVIDER_ID_RE.fullmatch(value):
            raise ValueError("provider_id must contain only letters, numbers, '.', '_' or '-'")
        return value

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
        if value is None:
            return None
        value = value.strip()
        if not value:
            return None
        parsed = urlparse(value)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be a valid HTTP or HTTPS URL")
        if parsed.username or parsed.password:
            raise ValueError("base_url must not contain embedded credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("base_url must not contain query parameters or fragments")
        return value.rstrip("/")

    def secret_value(self) -> str | None:
        return self.api_key.get_secret_value() if self.api_key else None

    def to_public_dict(self) -> dict[str, Any]:
        """Return a frontend-safe representation with the secret masked."""
        return {
            "provider_id": self.provider_id,
            "name": self.name,
            "provider_type": self.provider_type.value,
            "base_url": self.base_url,
            "api_key_configured": bool(self.api_key and self.secret_value()),
            "api_key_masked": "••••••••" if self.api_key and self.secret_value() else "",
            "model": self.model,
            "enabled": self.enabled,
            "settings": _redact_sensitive_settings(self.settings),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class LLMProviderCreate(BaseModel):
    """Input model for creating a provider record."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=120)
    provider_type: LLMProviderType = LLMProviderType.CUSTOM
    base_url: str | None = Field(default=None, max_length=2048)
    api_key: SecretStr | None = None
    model: str | None = Field(default=None, max_length=512)
    enabled: bool = True
    settings: dict[str, Any] = Field(default_factory=dict)

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
        if value is None:
            return None
        value = value.strip()
        if not value:
            return None
        parsed = urlparse(value)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be a valid HTTP or HTTPS URL")
        if parsed.username or parsed.password:
            raise ValueError("base_url must not contain embedded credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("base_url must not contain query parameters or fragments")
        return value.rstrip("/")


class LLMProviderStore:
    """SQLite-backed CRUD store for user-managed LLM providers."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        configured_path = db_path or os.getenv("LLM_PROVIDER_STORE_PATH") or DEFAULT_LLM_PROVIDER_DB
        self.db_path = Path(configured_path).expanduser().resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("PRAGMA secure_delete = ON")
        self._harden_file_permissions()
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_providers (
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
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_llm_providers_updated_at "
            "ON llm_providers(updated_at DESC)"
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_provider_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        self._encryption_key = load_encryption_key(self.db_path)
        self._migrate_plaintext_secrets()
        self._conn.commit()
        self._harden_file_permissions()

    def _harden_file_permissions(self) -> None:
        """Best-effort owner-only permissions for the credential database and WAL sidecars.

        SQLite stores provider API keys in the database. On POSIX systems, keep the
        database and any WAL/SHM sidecars unreadable by other local users.
        """
        paths = (
            self.db_path,
            Path(f"{self.db_path}-wal"),
            Path(f"{self.db_path}-shm"),
        )
        for path in paths:
            try:
                if path.exists():
                    path.chmod(0o600)
            except OSError:
                # Windows ACLs are not represented by chmod semantics; the UI/API
                # must still avoid exposing plaintext secrets in responses/logs.
                continue

    def _encrypt_api_key(self, secret: str | None) -> str | None:
        if not secret:
            return None
        return encrypt_text(secret, self._encryption_key)

    def _decrypt_api_key(self, stored: str | None) -> str | None:
        if not stored:
            return None
        if not is_encrypted(stored):
            # Legacy plaintext values are migrated during store initialization.
            raise LLMProviderSecretError(
                "Legacy plaintext LLM provider secret encountered after migration"
            )
        return decrypt_text(stored, self._encryption_key)

    def _migrate_plaintext_secrets(self) -> None:
        """Encrypt legacy plaintext secrets and protected settings in-place.

        Stage 1 stored provider secrets as plaintext. Existing installations are
        migrated automatically so upgrading does not require users to re-enter
        every credential manually. The provider database is tiny, so the one-time
        migration is intentionally performed before the first runtime read.
        """
        import json

        rows = self._conn.execute(
            "SELECT provider_id, api_key, settings_json FROM llm_providers"
        ).fetchall()
        changed = False
        for row in rows:
            provider_id = row["provider_id"]
            stored_api_key = row["api_key"]
            stored_settings = row["settings_json"]

            encrypted_api_key = (
                stored_api_key
                if not stored_api_key or is_encrypted(stored_api_key)
                else self._encrypt_api_key(stored_api_key)
            )

            try:
                settings = json.loads(stored_settings) if stored_settings else {}
            except (TypeError, ValueError) as exc:
                raise LLMProviderSecretError(
                    f"Provider '{provider_id}' has invalid settings data"
                ) from exc
            if not isinstance(settings, dict):
                raise LLMProviderSecretError(
                    f"Provider '{provider_id}' has invalid settings data"
                )
            protected_settings = _protect_sensitive_settings(
                settings, self._encryption_key
            )
            encoded_settings = json.dumps(
                protected_settings, ensure_ascii=False, sort_keys=True
            )

            if encrypted_api_key != stored_api_key or encoded_settings != stored_settings:
                with self._conn:
                    self._conn.execute(
                        "UPDATE llm_providers SET api_key = ?, settings_json = ? "
                        "WHERE provider_id = ?",
                        (encrypted_api_key, encoded_settings, provider_id),
                    )
                changed = True

        if changed:
            # Remove obsolete plaintext pages from SQLite/WAL where practical.
            # VACUUM is deliberately best-effort: a second Open-LLM-VTuber process
            # may temporarily hold a read transaction. The logical data is already
            # encrypted before this point.
            try:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                self._conn.execute("VACUUM")
            except sqlite3.Error:
                pass

    def create(self, provider: LLMProviderCreate | LLMProvider) -> LLMProvider:
        """Create and persist a new provider. IDs are generated by default."""
        import json

        with self._lock:
            if isinstance(provider, LLMProviderCreate):
                now = _utc_now()
                record = LLMProvider(
                    name=provider.name,
                    provider_type=provider.provider_type,
                    base_url=provider.base_url,
                    api_key=provider.api_key,
                    model=provider.model,
                    enabled=provider.enabled,
                    settings=dict(provider.settings),
                    created_at=now,
                    updated_at=now,
                )
            else:
                record = provider

            try:
                with self._conn:
                    self._conn.execute(
                        """
                        INSERT INTO llm_providers
                          (provider_id, name, provider_type, base_url, api_key,
                           model, enabled, settings_json, created_at, updated_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            record.provider_id,
                            record.name,
                            record.provider_type.value,
                            record.base_url,
                            self._encrypt_api_key(record.secret_value()),
                            record.model,
                            int(record.enabled),
                            json.dumps(
                                _protect_sensitive_settings(
                                    record.settings, self._encryption_key
                                ),
                                ensure_ascii=False,
                                sort_keys=True,
                            ),
                            record.created_at,
                            record.updated_at,
                        ),
                    )
            except sqlite3.IntegrityError as exc:
                raise ValueError(f"LLM provider '{record.provider_id}' already exists") from exc
            self._harden_file_permissions()
            return record

    def get(self, provider_id: str) -> LLMProvider | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM llm_providers WHERE provider_id = ?",
                (provider_id,),
            ).fetchone()
        return self._row_to_provider(row) if row else None

    def require(self, provider_id: str) -> LLMProvider:
        provider = self.get(provider_id)
        if provider is None:
            raise KeyError(f"LLM provider '{provider_id}' not found")
        return provider

    def list(self, *, enabled_only: bool = False) -> list[LLMProvider]:
        query = "SELECT * FROM llm_providers"
        params: tuple[Any, ...] = ()
        if enabled_only:
            query += " WHERE enabled = 1"
        query += " ORDER BY updated_at DESC, name COLLATE NOCASE ASC"
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_provider(row) for row in rows]

    def update(self, provider_id: str, changes: Mapping[str, Any]) -> LLMProvider:
        """Apply partial changes and persist them transactionally.

        Omitted fields are preserved. Supplying ``api_key=None`` explicitly clears
        the saved credential.
        """
        import json

        with self._lock:
            existing = self.require(provider_id)
            payload = {
                "provider_id": existing.provider_id,
                "name": existing.name,
                "provider_type": existing.provider_type,
                "base_url": existing.base_url,
                "api_key": existing.api_key,
                "model": existing.model,
                "enabled": existing.enabled,
                "settings": dict(existing.settings),
                "created_at": existing.created_at,
                "updated_at": existing.updated_at,
            }
            payload.update(dict(changes))
            payload["provider_id"] = existing.provider_id
            payload["created_at"] = existing.created_at
            payload["updated_at"] = _utc_now()
            updated = LLMProvider.model_validate(payload)

            with self._conn:
                self._conn.execute(
                    """
                    UPDATE llm_providers
                    SET name = ?, provider_type = ?, base_url = ?, api_key = ?,
                        model = ?, enabled = ?, settings_json = ?, updated_at = ?
                    WHERE provider_id = ?
                    """,
                    (
                        updated.name,
                        updated.provider_type.value,
                        updated.base_url,
                        self._encrypt_api_key(updated.secret_value()),
                        updated.model,
                        int(updated.enabled),
                        json.dumps(
                            _protect_sensitive_settings(
                                updated.settings, self._encryption_key
                            ),
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                        updated.updated_at,
                        updated.provider_id,
                    ),
                )
            self._harden_file_permissions()
            return updated

    def get_active_id(self) -> str | None:
        """Return the explicitly selected active provider ID, if any."""
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM llm_provider_state WHERE key = ?",
                ("active_provider_id",),
            ).fetchone()
        return str(row["value"]) if row and row["value"] else None

    def get_active(self) -> LLMProvider | None:
        """Return the explicitly selected active provider, if it still exists."""
        provider_id = self.get_active_id()
        if not provider_id:
            return None
        provider = self.get(provider_id)
        if provider is None or not provider.enabled:
            return None
        return provider

    def set_active(self, provider_id: str | None) -> LLMProvider | None:
        """Select exactly one provider for managed single-LLM runtime use.

        ``None`` clears the managed selection and returns ``None``. The existing
        YAML configuration remains the runtime fallback when no provider is active.
        """
        with self._lock:
            if provider_id is None:
                with self._conn:
                    self._conn.execute(
                        "DELETE FROM llm_provider_state WHERE key = ?",
                        ("active_provider_id",),
                    )
                self._harden_file_permissions()
                return None

            provider = self.require(provider_id)
            if not provider.enabled:
                raise ValueError(f"LLM provider '{provider_id}' is disabled")
            with self._conn:
                self._conn.execute(
                    "INSERT INTO llm_provider_state(key, value) VALUES(?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    ("active_provider_id", provider.provider_id),
                )
            self._harden_file_permissions()
            return provider

    def delete(self, provider_id: str) -> bool:
        with self._lock:
            with self._conn:
                cursor = self._conn.execute(
                    "DELETE FROM llm_providers WHERE provider_id = ?",
                    (provider_id,),
                )
                if cursor.rowcount > 0:
                    self._conn.execute(
                        "DELETE FROM llm_provider_state WHERE key = ? AND value = ?",
                        ("active_provider_id", provider_id),
                    )
            self._harden_file_permissions()
            return cursor.rowcount > 0

    def close(self) -> None:
        with self._lock:
            if getattr(self, "_conn", None) is not None:
                self._conn.close()
                self._conn = None

    def __enter__(self) -> "LLMProviderStore":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()

    def _row_to_provider(self, row: sqlite3.Row) -> LLMProvider:
        import json

        stored_api_key = row["api_key"]
        api_key_value = self._decrypt_api_key(stored_api_key)
        api_key = SecretStr(api_key_value) if api_key_value is not None else None
        stored_settings = json.loads(row["settings_json"]) if row["settings_json"] else {}
        settings = _unprotect_sensitive_settings(stored_settings, self._encryption_key)
        return LLMProvider(
            provider_id=row["provider_id"],
            name=row["name"],
            provider_type=LLMProviderType(row["provider_type"]),
            base_url=row["base_url"],
            api_key=api_key,
            model=row["model"],
            enabled=bool(row["enabled"]),
            settings=settings,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


def _normalize_setting_key(key: object) -> str:
    return re.sub(r"[^a-z0-9_]", "", str(key).lower())


def _is_sensitive_setting_key(key: object) -> bool:
    normalized = _normalize_setting_key(key)
    if normalized in _SENSITIVE_SETTING_KEYS:
        return True
    return normalized.endswith(
        ("_token", "_secret", "_password", "_passphrase", "_api_key")
    ) or normalized in {
        "credential",
        "credentials",
        "cookie",
        "cookies",
        "headers",
        "private_key",
        "client_certificate",
        "client_cert",
    }


def _protect_sensitive_settings(value: Any, key: bytes, *, _setting_key: object | None = None) -> Any:
    """Recursively encrypt credential-like settings for database storage."""
    if _setting_key is not None and _is_sensitive_setting_key(_setting_key):
        if is_encrypted(value, prefix=SETTINGS_CIPHERTEXT_PREFIX):
            return value
        import json
        try:
            serialized = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError):
            # Non-JSON settings will fail the normal settings_json serialization later.
            return value
        return encrypt_text(
            serialized, key, prefix=SETTINGS_CIPHERTEXT_PREFIX
        )
    if isinstance(value, dict):
        return {
            item_key: _protect_sensitive_settings(
                item, key, _setting_key=item_key
            )
            for item_key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _protect_sensitive_settings(item, key)
            for item in value
        ]
    return value


def _unprotect_sensitive_settings(value: Any, key: bytes) -> Any:
    """Recursively decrypt protected settings after loading from SQLite."""
    if is_encrypted(value, prefix=SETTINGS_CIPHERTEXT_PREFIX):
        import json
        try:
            return json.loads(
                decrypt_text(value, key, prefix=SETTINGS_CIPHERTEXT_PREFIX)
            )
        except (TypeError, ValueError, LLMProviderSecretError) as exc:
            if isinstance(exc, LLMProviderSecretError):
                raise
            raise LLMProviderSecretError(
                "Stored LLM provider setting could not be decrypted"
            ) from exc
    if isinstance(value, dict):
        return {item_key: _unprotect_sensitive_settings(item, key) for item_key, item in value.items()}
    if isinstance(value, list):
        return [_unprotect_sensitive_settings(item, key) for item in value]
    return value


def _redact_sensitive_settings(value: Any, *, _key: str | None = None) -> Any:
    """Recursively redact credential-like setting keys for public serialization."""
    if _key is not None and _is_sensitive_setting_key(_key):
        return "***configured***" if value not in (None, "", [], {}) else value
    if isinstance(value, dict):
        return {
            item_key: _redact_sensitive_settings(item, _key=str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_sensitive_settings(item) for item in value]
    return value


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

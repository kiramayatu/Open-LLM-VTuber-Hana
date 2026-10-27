"""Encryption helpers for managed LLM-provider secrets.

Secrets are encrypted before they are written to the provider SQLite database.
AES-GCM provides confidentiality and integrity; the encryption key is either
provided through ``LLM_PROVIDER_ENCRYPTION_KEY`` or stored in a separate
owner-only key file next to the provider database.
"""
from __future__ import annotations

import base64
import hashlib
import os
import secrets
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

SECRET_CIPHERTEXT_PREFIX = "ENC:v1:"
SETTINGS_CIPHERTEXT_PREFIX = "ENCSET:v1:"
KEY_FILE_ENV = "LLM_PROVIDER_SECRET_KEY_PATH"
KEY_ENV = "LLM_PROVIDER_ENCRYPTION_KEY"
KEY_FILE_NAME = ".llm_provider_secret.key"
_KEY_BYTES = 32
_NONCE_BYTES = 12
_AAD = b"open-llm-vtuber:llm-provider-secret:v1"


class LLMProviderSecretError(RuntimeError):
    """Raised when a provider secret cannot be safely encrypted/decrypted."""


def resolve_key_path(db_path: str | Path) -> Path:
    """Return the configured or default secret-key file path."""
    configured = os.getenv(KEY_FILE_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    database = Path(db_path).expanduser().resolve()
    return database.parent / KEY_FILE_NAME


def _derive_env_key(value: str) -> bytes:
    if not value.strip():
        raise LLMProviderSecretError(f"{KEY_ENV} must not be empty")
    # The environment value is a human-manageable secret rather than raw AES
    # key material.  Domain-separated SHA-256 derives fixed-length AES-256 key
    # material without ever storing the original environment value in SQLite.
    return hashlib.sha256(
        b"open-llm-vtuber:llm-provider-encryption-key:v1:" + value.encode("utf-8")
    ).digest()


def _set_owner_only(path: Path) -> None:
    try:
        path.chmod(0o600)
    except OSError:
        # Windows ACLs are not represented by chmod semantics.  The file remains
        # separate from the provider DB and API responses never expose its bytes.
        pass


def _load_or_create_key_file(path: Path) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise LLMProviderSecretError(
            "LLM provider secret key file must not be a symbolic link"
        )
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        data = secrets.token_bytes(_KEY_BYTES)
        try:
            # Exclusive creation prevents two application processes from
            # generating different keys concurrently.
            with path.open("xb") as handle:
                handle.write(data)
        except FileExistsError:
            data = path.read_bytes()
        _set_owner_only(path)
    if len(data) != _KEY_BYTES:
        raise LLMProviderSecretError(
            f"Invalid LLM provider secret key file: expected {_KEY_BYTES} bytes"
        )
    _set_owner_only(path)
    return data


def load_encryption_key(db_path: str | Path) -> bytes:
    """Load the application encryption key from environment or key file."""
    configured = os.getenv(KEY_ENV)
    if configured is not None:
        return _derive_env_key(configured)
    return _load_or_create_key_file(resolve_key_path(db_path))


def encrypt_text(value: str, key: bytes, *, prefix: str = SECRET_CIPHERTEXT_PREFIX) -> str:
    """Encrypt UTF-8 text using AES-256-GCM and return a versioned string."""
    if not isinstance(value, str):
        raise TypeError("secret value must be a string")
    if len(key) != _KEY_BYTES:
        raise LLMProviderSecretError("Invalid LLM provider encryption key")
    nonce = secrets.token_bytes(_NONCE_BYTES)
    ciphertext = AESGCM(key).encrypt(nonce, value.encode("utf-8"), _AAD)
    token = base64.urlsafe_b64encode(nonce + ciphertext).decode("ascii")
    return f"{prefix}{token}"


def decrypt_text(value: str, key: bytes, *, prefix: str = SECRET_CIPHERTEXT_PREFIX) -> str:
    """Decrypt a versioned AES-256-GCM ciphertext string."""
    if not isinstance(value, str) or not value.startswith(prefix):
        raise LLMProviderSecretError("Secret is not stored in a supported encrypted format")
    if len(key) != _KEY_BYTES:
        raise LLMProviderSecretError("Invalid LLM provider encryption key")
    encoded = value[len(prefix) :]
    try:
        payload = base64.urlsafe_b64decode(encoded.encode("ascii"))
        if len(payload) <= _NONCE_BYTES:
            raise ValueError("ciphertext too short")
        nonce, ciphertext = payload[:_NONCE_BYTES], payload[_NONCE_BYTES:]
        plaintext = AESGCM(key).decrypt(nonce, ciphertext, _AAD)
        return plaintext.decode("utf-8")
    except (ValueError, UnicodeDecodeError) as exc:
        raise LLMProviderSecretError("Stored LLM provider secret could not be decrypted") from exc
    except Exception as exc:
        # Avoid exposing backend cryptographic details or ciphertext material.
        raise LLMProviderSecretError("Stored LLM provider secret could not be decrypted") from exc


def is_encrypted(value: object, *, prefix: str = SECRET_CIPHERTEXT_PREFIX) -> bool:
    return isinstance(value, str) and value.startswith(prefix)

"""Core artifact domain types.

An Artifact is metadata/identity for something the assistant creates or manages.
It is deliberately separate from the physical file on disk; storage is added in
later feature parts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


class ArtifactType(str, Enum):
    """High-level kinds that the artifact system can represent."""

    FILE = "file"
    PROJECT = "project"
    DOCUMENT = "document"
    SPREADSHEET = "spreadsheet"
    PRESENTATION = "presentation"
    IMAGE = "image"


def utc_now_iso() -> str:
    """Return a timezone-aware UTC timestamp in ISO-8601 form."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(slots=True)
class Artifact:
    """Stable identity and metadata for an assistant-managed artifact.

    This object contains no filesystem access and no arbitrary path operations.
    The ``path`` field is only a logical relative path for now; workspace/storage
    validation belongs to the workspace implementation in Part 2.
    """

    name: str
    artifact_type: ArtifactType = ArtifactType.FILE
    mime_type: str | None = None
    workspace_id: str | None = None
    path: str | None = None
    size: int = 0
    artifact_id: str = field(default_factory=lambda: f"artifact_{uuid4().hex}")
    created_at: str = field(default_factory=utc_now_iso)
    updated_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Artifact name cannot be empty")
        if self.size < 0:
            raise ValueError("Artifact size cannot be negative")
        if not self.artifact_id:
            raise ValueError("Artifact ID cannot be empty")

    def update(self, **changes: Any) -> None:
        """Update metadata fields while preserving the stable artifact ID."""

        protected = {"artifact_id", "created_at"}
        for key, value in changes.items():
            if key in protected:
                raise ValueError(f"Cannot modify protected field: {key}")
            if not hasattr(self, key):
                raise ValueError(f"Unknown artifact field: {key}")
            setattr(self, key, value)

        if self.size < 0:
            raise ValueError("Artifact size cannot be negative")
        if not self.name or not self.name.strip():
            raise ValueError("Artifact name cannot be empty")
        self.updated_at = utc_now_iso()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the artifact into JSON-compatible data."""

        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type.value,
            "name": self.name,
            "mime_type": self.mime_type,
            "workspace_id": self.workspace_id,
            "path": self.path,
            "size": self.size,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": dict(self.metadata),
        }

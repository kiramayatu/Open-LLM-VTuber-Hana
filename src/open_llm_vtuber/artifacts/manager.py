"""Persistent artifact metadata manager for Part 3.1."""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .types import Artifact, ArtifactType


class ArtifactManager:
    """Manage persistent artifact metadata, optionally sharing a workspace DB."""

    def __init__(self, db_path: str | Path = "generated_files/.workspaces.sqlite3") -> None:
        self._artifacts: dict[str, Artifact] = {}
        self._db_path = Path(db_path).expanduser().resolve()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id TEXT PRIMARY KEY,
                workspace_id TEXT,
                name TEXT NOT NULL,
                artifact_type TEXT NOT NULL,
                mime_type TEXT,
                path TEXT,
                size INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            )
        """)
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_artifacts_workspace ON artifacts(workspace_id)"
        )
        # At most one managed artifact may represent a logical path in a workspace.
        self._conn.execute(
            "CREATE UNIQUE INDEX IF NOT EXISTS uq_artifacts_workspace_path "
            "ON artifacts(workspace_id, path) WHERE path IS NOT NULL"
        )
        self._conn.commit()
        self._load_persisted()

    def _load_persisted(self) -> None:
        rows = self._conn.execute(
            "SELECT artifact_id,workspace_id,name,artifact_type,mime_type,path,size,created_at,updated_at,metadata_json FROM artifacts"
        ).fetchall()
        for row in rows:
            try:
                artifact_id, workspace_id, name, artifact_type, mime_type, path, size, created_at, updated_at, metadata_json = row
                self._artifacts[artifact_id] = Artifact(
                    artifact_id=artifact_id,
                    workspace_id=workspace_id,
                    name=name,
                    artifact_type=ArtifactType(artifact_type),
                    mime_type=mime_type,
                    path=path,
                    size=int(size),
                    created_at=created_at,
                    updated_at=updated_at,
                    metadata=json.loads(metadata_json) if metadata_json else {},
                )
            except (ValueError, TypeError, json.JSONDecodeError):
                continue

    def _persist(self, artifact: Artifact) -> None:
        self._conn.execute(
            """
            INSERT INTO artifacts
              (artifact_id,workspace_id,name,artifact_type,mime_type,path,size,created_at,updated_at,metadata_json)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(artifact_id) DO UPDATE SET
              workspace_id=excluded.workspace_id, name=excluded.name, artifact_type=excluded.artifact_type,
              mime_type=excluded.mime_type, path=excluded.path, size=excluded.size,
              updated_at=excluded.updated_at, metadata_json=excluded.metadata_json
            """,
            (artifact.artifact_id, artifact.workspace_id, artifact.name, artifact.artifact_type.value,
             artifact.mime_type, artifact.path, artifact.size, artifact.created_at, artifact.updated_at,
             json.dumps(artifact.metadata, ensure_ascii=False, sort_keys=True)),
        )
        self._conn.commit()

    def create(self, **kwargs: Any) -> Artifact:
        artifact = Artifact(**kwargs)
        if artifact.artifact_id in self._artifacts:
            raise RuntimeError("Generated duplicate artifact ID")
        if artifact.workspace_id is not None and artifact.path is not None:
            existing = self._conn.execute(
                "SELECT artifact_id FROM artifacts WHERE workspace_id = ? AND path = ? LIMIT 1",
                (artifact.workspace_id, artifact.path),
            ).fetchone()
            if existing is not None:
                raise FileExistsError(
                    f"An artifact already exists for path '{artifact.path}' in workspace '{artifact.workspace_id}'"
                )
        self._artifacts[artifact.artifact_id] = artifact
        try:
            self._persist(artifact)
        except Exception:
            self._artifacts.pop(artifact.artifact_id, None)
            raise
        return artifact

    def get(self, artifact_id: str) -> Artifact | None:
        return self._artifacts.get(artifact_id)

    def require(self, artifact_id: str) -> Artifact:
        artifact = self.get(artifact_id)
        if artifact is None:
            raise KeyError(f"Artifact not found: {artifact_id}")
        return artifact

    def update(self, artifact_id: str, **changes: Any) -> Artifact:
        artifact = self.require(artifact_id)
        snapshot = artifact.to_dict()
        artifact.update(**changes)
        try:
            self._persist(artifact)
        except Exception:
            artifact.update(**{k: v for k, v in snapshot.items() if k not in {"artifact_id", "created_at", "artifact_type"}})
            raise
        return artifact

    def delete(self, artifact_id: str) -> bool:
        if artifact_id not in self._artifacts:
            return False
        self._conn.execute("DELETE FROM artifacts WHERE artifact_id = ?", (artifact_id,))
        self._conn.commit()
        self._artifacts.pop(artifact_id, None)
        return True

    def list(self, *, workspace_id: str | None = None) -> list[Artifact]:
        artifacts: Iterable[Artifact] = self._artifacts.values()
        if workspace_id is not None:
            artifacts = (a for a in artifacts if a.workspace_id == workspace_id)
        return list(artifacts)

    def clear(self) -> None:
        self._conn.execute("DELETE FROM artifacts")
        self._conn.commit()
        self._artifacts.clear()

    def close(self) -> None:
        if getattr(self, "_conn", None) is not None:
            self._conn.close()

"""Secure persistent workspace storage for the artifact system (Part 2.1)."""

from __future__ import annotations

import os
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from uuid import uuid4


DEFAULT_MAX_TOTAL_SIZE = 5 * 1024 * 1024 * 1024
DEFAULT_MAX_WORKSPACE_SIZE = 500 * 1024 * 1024
DEFAULT_MAX_FILE_SIZE = 100 * 1024 * 1024
DEFAULT_MAX_FILES_PER_WORKSPACE = 1000


@dataclass(slots=True)
class Workspace:
    workspace_id: str
    name: str
    root_path: Path
    conversation_id: str | None = None
    created_at: str = ""
    updated_at: str = ""
    metadata: dict[str, object] = field(default_factory=dict)


class WorkspaceManager:
    """Manage isolated workspaces below one trusted root with SQLite metadata."""

    def __init__(
        self,
        root_dir: str | Path = "generated_files",
        *,
        max_total_size: int = DEFAULT_MAX_TOTAL_SIZE,
        max_workspace_size: int = DEFAULT_MAX_WORKSPACE_SIZE,
        max_file_size: int = DEFAULT_MAX_FILE_SIZE,
        max_files_per_workspace: int = DEFAULT_MAX_FILES_PER_WORKSPACE,
    ) -> None:
        self.root_dir = Path(root_dir).expanduser().resolve()
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.max_total_size = self._validate_limit(max_total_size, "max_total_size")
        self.max_workspace_size = self._validate_limit(max_workspace_size, "max_workspace_size")
        self.max_file_size = self._validate_limit(max_file_size, "max_file_size")
        self.max_files_per_workspace = self._validate_limit(max_files_per_workspace, "max_files_per_workspace")
        self._file_lock = threading.RLock()
        if self.max_workspace_size > self.max_total_size:
            raise ValueError("max_workspace_size cannot exceed max_total_size")
        self._db_path = self.root_dir / ".workspaces.sqlite3"
        self._conn = sqlite3.connect(self._db_path)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS workspaces (
                workspace_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                conversation_id TEXT,
                root_name TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            )
            """
        )
        self._conn.commit()
        self._workspaces: dict[str, Workspace] = {}
        self._load_persisted()

    @staticmethod
    def _validate_limit(value: int, name: str) -> int:
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
        return value

    @staticmethod
    def _utc_now_iso() -> str:
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat(timespec="seconds")

    @staticmethod
    def _validate_workspace_id(workspace_id: str) -> None:
        if (
            not workspace_id
            or workspace_id in {".", ".."}
            or "/" in workspace_id
            or "\\" in workspace_id
            or Path(workspace_id).name != workspace_id
        ):
            raise ValueError("Invalid workspace ID")

    def _workspace_root(self, workspace_id: str) -> Path:
        self._validate_workspace_id(workspace_id)
        candidate = self.root_dir / workspace_id
        # Validate the parent/root relationship without following a malicious
        # workspace symlink. Existing symlinked workspace directories are never
        # accepted as managed roots.
        if candidate.is_symlink():
            raise ValueError("Workspace root cannot be a symlink")
        root = candidate.resolve(strict=False)
        self._ensure_within(root, self.root_dir)
        return root

    @staticmethod
    def _ensure_within(path: Path, root: Path) -> None:
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("Path escapes its allowed workspace") from exc

    @staticmethod
    def _safe_relative_path(relative_path: str | Path) -> Path:
        candidate = Path(relative_path)
        raw = str(relative_path).strip()
        if not raw or candidate.is_absolute() or candidate.anchor:
            raise ValueError("Artifact path must be a non-empty relative path")
        if any(part in ("", ".", "..") for part in candidate.parts):
            raise ValueError("Artifact path contains an unsafe component")
        return candidate

    @staticmethod
    def _assert_no_symlink_components(path: Path, stop_at: Path) -> None:
        current = path
        while current != stop_at:
            if current.is_symlink():
                raise ValueError("Symlink paths are not allowed in workspaces")
            current = current.parent
        if stop_at.is_symlink():
            raise ValueError("Workspace root cannot be a symlink")

    def _resolve_file(self, workspace_id: str, relative_path: str | Path) -> Path:
        workspace_root = self.require(workspace_id).root_path
        rel = self._safe_relative_path(relative_path)
        candidate = workspace_root / rel
        self._assert_no_symlink_components(candidate.parent, workspace_root)
        resolved = candidate.resolve(strict=False)
        self._ensure_within(resolved, workspace_root)
        return candidate

    def _persist_workspace(self, workspace: Workspace) -> None:
        import json
        self._conn.execute(
            """
            INSERT INTO workspaces
              (workspace_id, name, conversation_id, root_name, created_at, updated_at, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(workspace_id) DO UPDATE SET
              name=excluded.name,
              conversation_id=excluded.conversation_id,
              root_name=excluded.root_name,
              updated_at=excluded.updated_at,
              metadata_json=excluded.metadata_json
            """,
            (
                workspace.workspace_id,
                workspace.name,
                workspace.conversation_id,
                workspace.root_path.name,
                workspace.created_at,
                workspace.updated_at,
                json.dumps(workspace.metadata, ensure_ascii=False, sort_keys=True),
            ),
        )
        self._conn.commit()

    def _load_persisted(self) -> None:
        import json
        rows = self._conn.execute(
            "SELECT workspace_id,name,conversation_id,root_name,created_at,updated_at,metadata_json FROM workspaces"
        ).fetchall()
        for workspace_id, name, conversation_id, root_name, created_at, updated_at, metadata_json in rows:
            try:
                self._validate_workspace_id(workspace_id)
                root = self.root_dir / root_name
                if root_name != workspace_id or root.is_symlink() or not root.is_dir():
                    continue
                resolved = root.resolve()
                self._ensure_within(resolved, self.root_dir)
                metadata = json.loads(metadata_json) if metadata_json else {}
                if not isinstance(metadata, dict):
                    metadata = {}
                self._workspaces[workspace_id] = Workspace(
                    workspace_id=workspace_id,
                    name=name,
                    root_path=resolved,
                    conversation_id=conversation_id,
                    created_at=created_at,
                    updated_at=updated_at,
                    metadata=metadata,
                )
            except (ValueError, OSError, json.JSONDecodeError):
                continue

    def create(
        self,
        *,
        name: str,
        conversation_id: str | None = None,
        workspace_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> Workspace:
        name = name.strip()
        if not name:
            raise ValueError("Workspace name cannot be empty")
        workspace_id = workspace_id or f"workspace_{uuid4().hex}"
        self._validate_workspace_id(workspace_id)
        if workspace_id in self._workspaces:
            raise ValueError(f"Workspace already exists: {workspace_id}")

        root = self._workspace_root(workspace_id)
        root.mkdir(parents=True, exist_ok=False)
        now = self._utc_now_iso()
        workspace = Workspace(
            workspace_id=workspace_id,
            name=name,
            root_path=root,
            conversation_id=conversation_id,
            created_at=now,
            updated_at=now,
            metadata=dict(metadata or {}),
        )
        try:
            self._persist_workspace(workspace)
        except Exception:
            root.rmdir()
            raise
        self._workspaces[workspace_id] = workspace
        return workspace

    def get(self, workspace_id: str) -> Workspace | None:
        return self._workspaces.get(workspace_id)

    def rename(self, workspace_id: str, name: str, *, touch_updated: bool = True) -> Workspace:
        """Update the human-readable workspace name without changing its stable ID."""
        name = str(name).strip()
        if not name:
            raise ValueError("Workspace name cannot be empty")
        workspace = self.require(workspace_id)
        workspace.name = name
        if touch_updated:
            workspace.updated_at = self._utc_now_iso()
        self._persist_workspace(workspace)
        return workspace

    def get_by_conversation_id(self, conversation_id: str) -> Workspace | None:
        if not conversation_id:
            return None
        for workspace in self._workspaces.values():
            if workspace.conversation_id == conversation_id:
                return workspace
        return None

    def require(self, workspace_id: str) -> Workspace:
        workspace = self.get(workspace_id)
        if workspace is None:
            raise KeyError(f"Workspace not found: {workspace_id}")
        return workspace

    def list(self) -> list[Workspace]:
        return list(self._workspaces.values())

    def delete(self, workspace_id: str, *, delete_files: bool = True) -> bool:
        workspace = self.require(workspace_id)
        if delete_files:
            root = self.require(workspace_id).root_path
            for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
                if path.is_symlink() or path.is_file():
                    path.unlink(missing_ok=True)
                elif path.is_dir():
                    path.rmdir()
            root.rmdir()
        self._conn.execute("DELETE FROM workspaces WHERE workspace_id = ?", (workspace_id,))
        self._conn.commit()
        self._workspaces.pop(workspace_id, None)
        return True

    def delete_workspace_metadata(self, workspace_id: str) -> list[str]:
        """Atomically remove a workspace row and its artifact metadata.

        Filesystem deletion is staged by the caller before this transaction.
        Returns the deleted artifact IDs so in-memory caches can be updated.
        """
        workspace = self.require(workspace_id)
        rows = self._conn.execute(
            "SELECT artifact_id FROM artifacts WHERE workspace_id = ?", (workspace_id,)
        ).fetchall()
        artifact_ids = [str(row[0]) for row in rows]
        try:
            self._conn.execute("BEGIN IMMEDIATE")
            self._conn.execute("DELETE FROM artifacts WHERE workspace_id = ?", (workspace_id,))
            self._conn.execute("DELETE FROM workspaces WHERE workspace_id = ?", (workspace_id,))
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        self._workspaces.pop(workspace_id, None)
        return artifact_ids

    def size(self, workspace_id: str) -> int:
        root = self.require(workspace_id).root_path
        total = 0
        for path in root.rglob("*"):
            if path.is_symlink():
                continue
            if path.is_file():
                total += path.stat().st_size
        return total

    def total_size(self) -> int:
        return sum(self.size(w.workspace_id) for w in self._workspaces.values())

    def file_count(self, workspace_id: str) -> int:
        """Return the number of regular files currently in a workspace."""
        root = self.require(workspace_id).root_path
        count = 0
        for path in root.rglob("*"):
            if path.is_symlink() or not path.is_file():
                continue
            count += 1
        return count

    def _check_capacity(self, workspace_id: str, incoming_bytes: int, *, replacing: Path | None = None) -> None:
        if incoming_bytes < 0 or incoming_bytes > self.max_file_size:
            raise ValueError(f"File exceeds max_file_size ({self.max_file_size} bytes)")
        current_file_size = replacing.stat().st_size if replacing is not None and replacing.exists() and replacing.is_file() else 0
        workspace_size = self.size(workspace_id) - current_file_size
        if workspace_size + incoming_bytes > self.max_workspace_size:
            raise ValueError("Workspace storage limit exceeded")
        total_size = self.total_size() - current_file_size
        if total_size + incoming_bytes > self.max_total_size:
            raise ValueError("Global workspace storage limit exceeded")

    def _prepare_write(self, workspace_id: str, relative_path: str | Path, data: bytes, *, allow_replace: bool) -> tuple[Path, Path]:
        path = self._resolve_file(workspace_id, relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        workspace_root = self.require(workspace_id).root_path
        self._assert_no_symlink_components(path.parent, workspace_root)
        if not allow_replace and path.exists():
            raise FileExistsError(relative_path)
        if allow_replace:
            replacing = path if path.exists() and path.is_file() and not path.is_symlink() else None
        else:
            replacing = None
        self._check_capacity(workspace_id, len(data), replacing=replacing)
        if not allow_replace and self.file_count(workspace_id) >= self.max_files_per_workspace:
            raise ValueError(f"Workspace file-count limit exceeded ({self.max_files_per_workspace} files)")
        return path, workspace_root

    def write_bytes(self, workspace_id: str, relative_path: str | Path, data: bytes) -> Path:
        """Write bytes, replacing an existing regular file atomically."""
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("data must be bytes-like")
        data = bytes(data)
        with self._file_lock:
            path, workspace_root = self._prepare_write(workspace_id, relative_path, data, allow_replace=True)
            parent_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0))
            try:
                self._assert_no_symlink_components(path.parent, workspace_root)
                with NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
                    tmp.write(data)
                    tmp.flush()
                    os.fsync(tmp.fileno())
                    temp_path = Path(tmp.name)
                os.replace(temp_path, path)
            finally:
                os.close(parent_fd)
                if 'temp_path' in locals():
                    temp_path.unlink(missing_ok=True)
            workspace = self.require(workspace_id)
            workspace.updated_at = self._utc_now_iso()
            self._persist_workspace(workspace)
            return path

    def create_bytes(self, workspace_id: str, relative_path: str | Path, data: bytes) -> Path:
        """Create a new file without overwriting an existing path."""
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("data must be bytes-like")
        data = bytes(data)
        with self._file_lock:
            path, workspace_root = self._prepare_write(workspace_id, relative_path, data, allow_replace=False)
            # Use O_EXCL + O_NOFOLLOW on the final path so create-if-absent is
            # decided atomically by the filesystem rather than by a prior exists() check.
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
            fd = os.open(path, flags, 0o600)
            try:
                self._assert_no_symlink_components(path.parent, workspace_root)
                view = memoryview(data)
                written = 0
                while written < len(view):
                    written += os.write(fd, view[written:])
                os.fsync(fd)
            except Exception:
                try:
                    os.unlink(path)
                except OSError:
                    pass
                raise
            finally:
                os.close(fd)
            workspace = self.require(workspace_id)
            workspace.updated_at = self._utc_now_iso()
            self._persist_workspace(workspace)
            return path

    def write_text(self, workspace_id: str, relative_path: str | Path, text: str, *, encoding: str = "utf-8") -> Path:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return self.write_bytes(workspace_id, relative_path, text.encode(encoding))

    def read_bytes(self, workspace_id: str, relative_path: str | Path) -> bytes:
        path = self._resolve_file(workspace_id, relative_path)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(relative_path)
        return path.read_bytes()

    def read_text(self, workspace_id: str, relative_path: str | Path, *, encoding: str = "utf-8") -> str:
        return self.read_bytes(workspace_id, relative_path).decode(encoding)

    def delete_file(self, workspace_id: str, relative_path: str | Path) -> bool:
        path = self._resolve_file(workspace_id, relative_path)
        if not path.exists():
            return False
        if not path.is_file() or path.is_symlink():
            raise ValueError("Only regular workspace files can be deleted")
        path.unlink()
        workspace = self.require(workspace_id)
        workspace.updated_at = self._utc_now_iso()
        self._persist_workspace(workspace)
        return True

    def list_files(self, workspace_id: str) -> list[str]:
        root = self.require(workspace_id).root_path
        files: list[str] = []
        for path in root.rglob("*"):
            if path.is_symlink() or not path.is_file():
                continue
            files.append(path.relative_to(root).as_posix())
        return sorted(files)

    def clear(self) -> None:
        for workspace_id in list(self._workspaces):
            self.delete(workspace_id)

    def close(self) -> None:
        if getattr(self, "_conn", None) is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "WorkspaceManager":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

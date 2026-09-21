"""LLM-facing artifact generation tools for Part 3.

These tools create and edit files/projects inside the secure workspace layer.
"""
from __future__ import annotations

from typing import Any
from pathlib import Path
from .workspace import WorkspaceManager
from .manager import ArtifactManager
from .types import ArtifactType
from .generators import generate_docx, generate_xlsx, generate_pdf, generate_pptx, extension_for

MAX_PROJECT_FILES = 100
MAX_PROJECT_PAYLOAD_BYTES = 50 * 1024 * 1024

CREATE_ARTIFACT_TOOL = {
    "type": "function",
    "function": {
        "name": "artifact_create_file",
        "description": (
            "Create one new text/code file in the current assistant workspace. "
            "Use this when the user explicitly asks for a text or code file. "
            "For .docx, .xlsx, .pdf, and .pptx use the dedicated document generators. "
            "Do not overwrite an existing file."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Relative filename, e.g. index.html"},
                "content": {"type": "string", "description": "Complete UTF-8 text content of the file"},
                "artifact_type": {
                    "type": "string",
                    "enum": ["file", "document", "spreadsheet", "presentation", "image", "project"],
                    "description": "High-level artifact category; normally use file for code/text files.",
                },
            },
            "required": ["filename", "content"],
            "additionalProperties": False,
        },
    },
}

CREATE_PROJECT_TOOL = {
    "type": "function",
    "function": {
        "name": "artifact_create_project",
        "description": (
            "Create a new multi-file project in the current assistant workspace. "
            "Use this when the user requests a website or other project requiring multiple files."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "project_name": {"type": "string", "description": "Human-readable project name"},
                "files": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["project_name", "files"],
            "additionalProperties": False,
        },
    },
}

EDIT_ARTIFACT_TOOL = {
    "type": "function",
    "function": {
        "name": "artifact_edit_file",
        "description": (
            "Replace the complete UTF-8 text content of an existing artifact file. "
            "Use this only when the user asks to modify a file previously created or managed by the assistant. "
            "Do not change the artifact ID, workspace, or path."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "artifact_id": {"type": "string", "description": "Stable artifact ID of the existing file"},
                "content": {"type": "string", "description": "Complete replacement UTF-8 text content"},
            },
            "required": ["artifact_id", "content"],
            "additionalProperties": False,
        },
    },
}

LIST_ARTIFACTS_TOOL = {
    "type": "function",
    "function": {
        "name": "artifact_list",
        "description": "List existing artifacts in the current assistant workspace so the assistant can identify files to edit.",
        "parameters": {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
    },
}


CREATE_DOCX_TOOL = {
    "type": "function",
    "function": {
        "name": "artifact_create_document",
        "description": "Create a real Microsoft Word DOCX document. Use this for requests mentioning Word, Microsoft Word, or .docx. Do NOT use artifact_create_file for .docx.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Output .docx filename"},
                "title": {"type": "string"},
                "paragraphs": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["filename", "title", "paragraphs"],
            "additionalProperties": False,
        },
    },
}

CREATE_XLSX_TOOL = {
    "type": "function",
    "function": {
        "name": "artifact_create_spreadsheet",
        "description": "Create a real Microsoft Excel XLSX workbook. Use this for requests mentioning Excel, spreadsheet, or .xlsx. Do NOT use artifact_create_file for .xlsx.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Output .xlsx filename"},
                "sheets": {
                    "type": "array", "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "rows": {"type": "array", "items": {"type": "array", "items": {}}},
                        },
                        "required": ["name", "rows"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["filename", "sheets"],
            "additionalProperties": False,
        },
    },
}

CREATE_PDF_TOOL = {
    "type": "function",
    "function": {
        "name": "artifact_create_pdf",
        "description": "Create a real PDF document. Use this for requests mentioning PDF or .pdf. Do NOT use artifact_create_file for .pdf.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Output .pdf filename"},
                "title": {"type": "string"},
                "paragraphs": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["filename", "title", "paragraphs"],
            "additionalProperties": False,
        },
    },
}

CREATE_PPTX_TOOL = {
    "type": "function",
    "function": {
        "name": "artifact_create_presentation",
        "description": "Create a real Microsoft PowerPoint PPTX presentation. Use this for requests mentioning PowerPoint, presentation, or .pptx. Do NOT use artifact_create_file for .pptx.",
        "parameters": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Output .pptx filename"},
                "title": {"type": "string"},
                "slides": {
                    "type": "array", "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "bullets": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["title", "bullets"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["filename", "title", "slides"],
            "additionalProperties": False,
        },
    },
}


CONVERT_ARTIFACT_TOOL = {
    "type": "function",
    "function": {
        "name": "artifact_convert",
        "description": (
            "Convert an existing artifact to another supported format. Use this when the user says convert, change format, export to another format, or similar. The original artifact is always preserved and a new artifact is created. Do NOT call artifact_create_file directly for a format conversion."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "artifact_id": {"type": "string", "description": "Stable ID of the source artifact"},
                "target_format": {
                    "type": "string",
                    "enum": ["txt", "md", "docx", "csv", "xlsx"],
                    "description": "Target format"
                },
                "filename": {"type": "string", "description": "Optional output filename"},
            },
            "required": ["artifact_id", "target_format"],
            "additionalProperties": False,
        },
    },
}

ARTIFACT_TOOL_NAMES = {
    "artifact_create_file",
    "artifact_create_project",
    "artifact_edit_file",
    "artifact_list",
    "artifact_create_document",
    "artifact_create_spreadsheet",
    "artifact_create_pdf",
    "artifact_create_presentation",
    "artifact_convert",
}
ARTIFACT_TOOLS_OPENAI = [CREATE_ARTIFACT_TOOL, CREATE_PROJECT_TOOL, EDIT_ARTIFACT_TOOL, LIST_ARTIFACTS_TOOL, CREATE_DOCX_TOOL, CREATE_XLSX_TOOL, CREATE_PDF_TOOL, CREATE_PPTX_TOOL, CONVERT_ARTIFACT_TOOL]
# Claude's tool schema is compatible with Anthropic's tools format via the existing adapter shape.
ARTIFACT_TOOLS_CLAUDE = [
    {
        "name": CREATE_ARTIFACT_TOOL["function"]["name"],
        "description": CREATE_ARTIFACT_TOOL["function"]["description"],
        "input_schema": CREATE_ARTIFACT_TOOL["function"]["parameters"],
    },
    {
        "name": CREATE_PROJECT_TOOL["function"]["name"],
        "description": CREATE_PROJECT_TOOL["function"]["description"],
        "input_schema": CREATE_PROJECT_TOOL["function"]["parameters"],
    },
    {
        "name": EDIT_ARTIFACT_TOOL["function"]["name"],
        "description": EDIT_ARTIFACT_TOOL["function"]["description"],
        "input_schema": EDIT_ARTIFACT_TOOL["function"]["parameters"],
    },
    {
        "name": LIST_ARTIFACTS_TOOL["function"]["name"],
        "description": LIST_ARTIFACTS_TOOL["function"]["description"],
        "input_schema": LIST_ARTIFACTS_TOOL["function"]["parameters"],
    },
]
for _tool in (CREATE_DOCX_TOOL, CREATE_XLSX_TOOL, CREATE_PDF_TOOL, CREATE_PPTX_TOOL, CONVERT_ARTIFACT_TOOL):
    ARTIFACT_TOOLS_CLAUDE.append({
        "name": _tool["function"]["name"],
        "description": _tool["function"]["description"],
        "input_schema": _tool["function"]["parameters"],
    })


def _artifact_type(value: str | None) -> ArtifactType:
    if not value:
        return ArtifactType.FILE
    try:
        return ArtifactType(value)
    except ValueError:
        return ArtifactType.FILE


class ArtifactToolService:
    """Creates new artifacts using an isolated workspace."""

    def __init__(
        self,
        workspace_manager: WorkspaceManager,
        artifact_manager: ArtifactManager,
        workspace_id: str | None = None,
        conversation_id: str | None = None,
    ) -> None:
        self.workspace_manager = workspace_manager
        self.artifact_manager = artifact_manager
        self.workspace_id = workspace_id
        self.conversation_id = conversation_id

    def _ensure_workspace(self) -> str:
        if self.workspace_id:
            return self.workspace_id
        if self.conversation_id:
            existing = self.workspace_manager.get_by_conversation_id(self.conversation_id)
            if existing:
                self.workspace_id = existing.workspace_id
                return self.workspace_id
        workspace = self.workspace_manager.create(name="VTuber Artifacts", conversation_id=self.conversation_id)
        self.workspace_id = workspace.workspace_id
        return self.workspace_id

    def create_file(self, filename: str, content: str, artifact_type: str | None = None) -> dict[str, Any]:
        workspace_id = self._ensure_workspace()
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("filename must be a non-empty string")
        if not isinstance(content, str):
            raise ValueError("content must be a string")
        suffix = Path(filename.strip()).suffix.lower()
        binary_formats = {".docx": "docx", ".xlsx": "xlsx", ".pdf": "pdf", ".pptx": "pptx"}
        if suffix in binary_formats:
            kind = binary_formats[suffix]
            raise ValueError(f"FORMAT_REQUIRES_GENERATOR:{kind}:Use the dedicated artifact generator for {suffix} files")
        # Creation-only: the workspace layer performs an atomic create-if-absent.
        path = self.workspace_manager.create_bytes(
            workspace_id, filename, content.encode("utf-8")
        )
        rel = path.relative_to(self.workspace_manager.require(workspace_id).root_path).as_posix()
        mime = _guess_mime(filename)
        artifact = self.artifact_manager.create(
            name=path.name,
            artifact_type=_artifact_type(artifact_type),
            mime_type=mime,
            workspace_id=workspace_id,
            path=rel,
            size=len(content.encode("utf-8")),
        )
        return artifact.to_dict()

    def edit_file(self, artifact_id: str, content: str) -> dict[str, Any]:
        """Replace an existing managed artifact's complete text content."""
        workspace_id = self._ensure_workspace()
        if not isinstance(artifact_id, str) or not artifact_id.strip():
            raise ValueError("artifact_id must be a non-empty string")
        if not isinstance(content, str):
            raise ValueError("content must be a string")

        artifact = self.artifact_manager.require(artifact_id)
        if artifact.workspace_id != workspace_id:
            raise PermissionError("Artifact does not belong to the current workspace")
        if not artifact.path:
            raise ValueError("Artifact has no editable file path")

        # Resolve through the secure workspace layer; never trust the stored path blindly.
        old_bytes = self.workspace_manager.read_bytes(workspace_id, artifact.path)
        new_bytes = content.encode("utf-8")
        if new_bytes == old_bytes:
            # Keep metadata consistent without rewriting the file.
            self.artifact_manager.update(artifact_id, size=len(new_bytes))
            return self.artifact_manager.require(artifact_id).to_dict()

        self.workspace_manager.write_bytes(workspace_id, artifact.path, new_bytes)
        try:
            updated = self.artifact_manager.update(artifact_id, size=len(new_bytes))
        except Exception:
            # Best-effort restoration if metadata persistence fails after the file replacement.
            try:
                self.workspace_manager.write_bytes(workspace_id, artifact.path, old_bytes)
            except Exception:
                pass
            raise
        return updated.to_dict()

    def _create_generated_file(self, filename: str, data: bytes, artifact_type: ArtifactType, mime: str) -> dict[str, Any]:
        workspace_id = self._ensure_workspace()
        if len(data) > self.workspace_manager.max_file_size:
            raise ValueError("Generated file exceeds workspace file-size limit")
        path = self.workspace_manager.create_bytes(workspace_id, filename, data)
        rel = path.relative_to(self.workspace_manager.require(workspace_id).root_path).as_posix()
        artifact = self.artifact_manager.create(
            name=path.name, artifact_type=artifact_type, mime_type=mime,
            workspace_id=workspace_id, path=rel, size=len(data),
        )
        return artifact.to_dict()

    def create_generated_file(
        self, filename: str, data: bytes, artifact_type: str, mime: str
    ) -> dict[str, Any]:
        """Persist already-generated bytes using the secure artifact workspace."""
        return self._create_generated_file(
            filename, data, _artifact_type(artifact_type), mime
        )

    def create_document(self, filename: str, title: str, paragraphs: list[str]) -> dict[str, Any]:
        expected, mime = extension_for("docx")
        if not filename.lower().endswith(expected):
            filename = f"{filename}{expected}"
        return self._create_generated_file(filename, generate_docx(title=title, paragraphs=paragraphs), ArtifactType.DOCUMENT, mime)

    def create_spreadsheet(self, filename: str, sheets: list[dict[str, Any]]) -> dict[str, Any]:
        expected, mime = extension_for("xlsx")
        if not filename.lower().endswith(expected):
            filename = f"{filename}{expected}"
        return self._create_generated_file(filename, generate_xlsx(sheets=sheets), ArtifactType.SPREADSHEET, mime)

    def create_pdf(self, filename: str, title: str, paragraphs: list[str]) -> dict[str, Any]:
        expected, mime = extension_for("pdf")
        if not filename.lower().endswith(expected):
            filename = f"{filename}{expected}"
        return self._create_generated_file(filename, generate_pdf(title=title, paragraphs=paragraphs), ArtifactType.DOCUMENT, mime)

    def create_presentation(self, filename: str, title: str, slides: list[dict[str, Any]]) -> dict[str, Any]:
        expected, mime = extension_for("pptx")
        if not filename.lower().endswith(expected):
            filename = f"{filename}{expected}"
        return self._create_generated_file(filename, generate_pptx(title=title, slides=slides), ArtifactType.PRESENTATION, mime)

    def convert_artifact(self, artifact_id: str, target_format: str, filename: str | None = None) -> dict[str, Any]:
        """Convert a managed artifact to a supported target format and create a new artifact."""
        workspace_id = self._ensure_workspace()
        artifact = self.artifact_manager.require(artifact_id)
        if artifact.workspace_id != workspace_id:
            raise PermissionError("Artifact does not belong to the current workspace")
        if not artifact.path or not artifact.name:
            raise ValueError("Artifact has no source file")

        source_path = self.workspace_manager._resolve_file(workspace_id, artifact.path)
        if source_path.is_symlink() or not source_path.is_file():
            raise FileNotFoundError("Source artifact file not found")
        target = target_format.strip().lower().lstrip(".")
        supported = {"txt", "md", "docx", "csv", "xlsx"}
        if target not in supported:
            raise ValueError(f"Unsupported target format: {target}")

        source_ext = Path(artifact.name).suffix.lower()
        if filename:
            output_name = filename.strip()
        else:
            output_name = Path(artifact.name).stem + "." + target

        if not output_name.lower().endswith("." + target):
            output_name += "." + target

        # Conversion always creates a new artifact. If the natural output name
        # already exists (or the chosen converted name collides), choose a
        # deterministic non-destructive name.
        candidate = Path(output_name)
        suffix = candidate.suffix
        stem = candidate.stem
        parent = candidate.parent
        counter = 0
        while True:
            rel_candidate = parent / (
                f"{stem}_converted{suffix}"
                if counter == 0
                else f"{stem}_converted_{counter}{suffix}"
            )
            try:
                resolved_candidate = self.workspace_manager._resolve_file(
                    workspace_id, rel_candidate.as_posix()
                )
                exists = resolved_candidate.exists()
            except Exception:
                exists = False
            if not exists:
                output_name = rel_candidate.as_posix()
                break
            counter += 1
            if counter > 1000:
                raise ValueError("Unable to select a unique converted artifact filename")

        # Read source into bounded memory using the existing file limits.
        if source_path.stat().st_size > self.workspace_manager.max_file_size:
            raise ValueError("Source artifact exceeds workspace file-size limit")

        text = None
        if source_ext in {".txt", ".md", ".csv"}:
            text = source_path.read_text(encoding="utf-8")
        elif source_ext == ".docx":
            from docx import Document
            doc = Document(source_path)
            text = "\n\n".join(p.text for p in doc.paragraphs)
        elif source_ext == ".xlsx":
            from openpyxl import load_workbook
            wb = load_workbook(source_path, read_only=True, data_only=True)
            try:
                chunks = []
                for ws in wb.worksheets:
                    chunks.append(f"# {ws.title}")
                    for row in ws.iter_rows(values_only=True):
                        chunks.append(",".join("" if v is None else str(v) for v in row))
                text = "\n".join(chunks)
            finally:
                wb.close()
        else:
            raise ValueError(f"Conversion from {source_ext or 'unknown'} is not supported")

        if text is None:
            raise ValueError("Unable to read source artifact")
        encoded = text.encode("utf-8")
        if len(encoded) > 10 * 1024 * 1024:
            raise ValueError("Converted text exceeds 10 MiB limit")

        if target in {"txt", "md"}:
            return self.create_file(output_name, text)
        if target == "docx":
            return self.create_document(output_name, Path(output_name).stem, text.splitlines())
        if target == "csv":
            return self.create_file(output_name, text)
        if target == "xlsx":
            rows = [line.split(",") for line in text.splitlines() if line.strip()]
            return self.create_spreadsheet(output_name, [{"name": "Sheet1", "rows": rows}])
        raise ValueError(f"Unsupported target format: {target}")

    def list_artifacts(self) -> list[dict[str, Any]]:
        workspace_id = self._ensure_workspace()
        return [artifact.to_dict() for artifact in self.artifact_manager.list(workspace_id=workspace_id)]

    def create_project(self, project_name: str, files: list[dict[str, Any]]) -> list[dict[str, Any]]:
        workspace_id = self._ensure_workspace()
        if not isinstance(project_name, str) or not project_name.strip():
            raise ValueError("project_name must be a non-empty string")
        if not isinstance(files, list) or not files:
            raise ValueError("files must be a non-empty list")
        if len(files) > MAX_PROJECT_FILES:
            raise ValueError(f"Project contains too many files (max {MAX_PROJECT_FILES})")
        # Validate the entire project before writing anything, so invalid input cannot
        # leave a half-created project.
        normalized: list[tuple[str, str]] = []
        seen: set[str] = set()
        for item in files:
            if not isinstance(item, dict):
                raise ValueError("Each project file must be an object")
            path = item.get("path")
            content = item.get("content")
            if not isinstance(path, str) or not path.strip() or not isinstance(content, str):
                raise ValueError("Each project file requires string path and content")
            # Validate path without creating it.
            candidate = self.workspace_manager._safe_relative_path(path)
            norm = candidate.as_posix()
            if norm in seen:
                raise ValueError(f"Duplicate project path: {norm}")
            seen.add(norm)
            normalized.append((norm, content))

        total_bytes = sum(len(content.encode("utf-8")) for _, content in normalized)
        if total_bytes > MAX_PROJECT_PAYLOAD_BYTES:
            raise ValueError(f"Project payload exceeds {MAX_PROJECT_PAYLOAD_BYTES} bytes")
        if total_bytes > self.workspace_manager.max_workspace_size:
            raise ValueError("Project exceeds workspace capacity")
        current_files = self.workspace_manager.file_count(workspace_id)
        if current_files + len(normalized) > self.workspace_manager.max_files_per_workspace:
            raise ValueError(
                f"Workspace file-count limit exceeded ({self.workspace_manager.max_files_per_workspace} files)"
            )

        created: list[dict[str, Any]] = []
        created_paths: list[str] = []
        created_artifact_ids: list[str] = []
        try:
            for path, content in normalized:
                # Creation-only semantics: never overwrite an existing path.
                # create_text/create_bytes use atomic O_EXCL creation in the workspace layer.
                self.workspace_manager.create_bytes(workspace_id, path, content.encode("utf-8"))
                created_paths.append(path)
                try:
                    artifact = self.artifact_manager.create(
                        name=path.rsplit("/", 1)[-1],
                        artifact_type=ArtifactType.FILE,
                        mime_type=_guess_mime(path),
                        workspace_id=workspace_id,
                        path=path,
                        size=len(content.encode("utf-8")),
                        metadata={"project_name": project_name.strip()},
                    )
                    created_artifact_ids.append(artifact.artifact_id)
                    created.append(artifact.to_dict())
                except Exception:
                    # This file was created by this operation; remove only that file.
                    try:
                        self.workspace_manager.delete_file(workspace_id, path)
                    finally:
                        raise
        except Exception:
            # Roll back only files/artifacts actually created by this operation.
            for artifact_id in reversed(created_artifact_ids):
                try:
                    self.artifact_manager.delete(artifact_id)
                except Exception:
                    pass
            for path in reversed(created_paths):
                try:
                    self.workspace_manager.delete_file(workspace_id, path)
                except Exception:
                    pass
            raise
        return created


def _guess_mime(filename: str) -> str:
    import mimetypes
    mime, _ = mimetypes.guess_type(filename)
    return mime or "application/octet-stream"

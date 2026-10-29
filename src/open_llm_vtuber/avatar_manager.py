"""Multi-Format Avatar Ingestion, Auto-Detection, and Management API (Part 2).

Supports uploading and managing 2D and 3D VTuber avatars directly from the UI:
- 3D VRM (`.vrm`)
- 3D glTF / GLB (`.glb`, `.gltf`)
- Live2D Cubism 3 / 4 / 5 (`*.model3.json` inside `.zip`)
- Live2D Legacy Cubism 2.1 (`*.model.json` inside `.zip`)
- 2D PNGTuber reactive multi-state sprites (`pngtuber.json` or image states inside `.zip` / single image)

Enforces strict security boundaries:
- Zip-Slip path traversal containment (`target.relative_to(dest_root)`)
- Decompression bomb limits (max 2000 files, max 2048 MB uncompressed, max 1024 MB upload)
- Executable / script extension blocklist
"""
import io
import json
import re
import secrets
import shutil
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import APIRouter
    from .service_context import ServiceContext

MAX_UPLOAD_BYTES = 1024 * 1024 * 1024  # 1024 MB (1 GB) compressed upload limit
MAX_EXTRACTED_BYTES = 2 * 1024 * 1024 * 1024  # 2048 MB (2 GB) uncompressed limit
MAX_ARCHIVE_FILES = 2000

FORBIDDEN_EXTENSIONS = {
    ".exe",
    ".bat",
    ".cmd",
    ".sh",
    ".bash",
    ".zsh",
    ".ps1",
    ".vbs",
    ".py",
    ".pyc",
    ".pyo",
    ".dll",
    ".so",
    ".dylib",
    ".wasm",
    ".php",
    ".jsp",
    ".asp",
    ".aspx",
    ".cgi",
    ".pl",
    ".rb",
    ".jar",
    ".msi",
    ".com",
    ".scr",
    ".elf",
    ".app",
    ".svg",
    ".html",
    ".htm",
}

ALLOWED_SINGLE_EXTENSIONS = {".vrm", ".glb", ".mqo", ".png", ".webp", ".gif"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}

DEFAULT_EMOTION_MAP = {
    "neutral": 0,
    "anger": 2,
    "disgust": 2,
    "fear": 1,
    "joy": 3,
    "smirk": 3,
    "sadness": 1,
    "surprise": 3,
}

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_model_name(raw_name: str) -> str:
    """Return a filesystem-safe model identifier."""
    cleaned = _SAFE_NAME_RE.sub("_", (raw_name or "").strip()).strip("._-")
    if not cleaned or cleaned in {".", "..", "undefined", "null"}:
        raise ValueError("Invalid model name")
    return cleaned[:64]


def detect_model_format_from_entry(entry: dict[str, Any]) -> str:
    """Infer avatar format (`cubism3`, `cubism2`, `vrm`, `gltf`, `mqo`, `pngtuber`) from a model_dict entry."""
    explicit = str(entry.get("format") or "").strip().lower()
    if explicit in {"cubism3", "cubism2", "vrm", "gltf", "mqo", "pngtuber"}:
        return explicit
    url = str(entry.get("url") or "").strip().lower()
    if url.endswith(".vrm"):
        return "vrm"
    if url.endswith(".glb") or url.endswith(".gltf"):
        return "gltf"
    if url.endswith(".mqo"):
        return "mqo"
    if url.endswith(".model3.json"):
        return "cubism3"
    if url.endswith(".model.json"):
        return "cubism2"
    if url.endswith("pngtuber.json") or url.endswith(".png") or url.endswith(".webp"):
        return "pngtuber"
    return "cubism3"


def load_model_dict(model_dict_path: str | Path = "model_dict.json") -> list[dict[str, Any]]:
    path = Path(model_dict_path)
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("model_dict.json must contain a JSON array")
    for item in data:
        if isinstance(item, dict) and "format" not in item:
            item["format"] = detect_model_format_from_entry(item)
    return data


def save_model_dict(
    entries: list[dict[str, Any]], model_dict_path: str | Path = "model_dict.json"
) -> None:
    path = Path(model_dict_path)
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(
        json.dumps(entries, indent=4, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tmp_path.replace(path)


def _find_preview_image(model_dir: Path, model_name: str) -> str | None:
    """Locate a thumbnail/preview image inside `model_dir`."""
    candidates: list[Path] = []
    for ext in (".png", ".jpg", ".jpeg", ".webp"):
        direct = model_dir / f"{model_name}{ext}"
        if direct.is_file():
            return "/" + direct.as_posix().lstrip("/")
        for preview_stem in ("preview", "icon", "thumbnail", "avatar", "idle", "closed"):
            p = model_dir / f"{preview_stem}{ext}"
            if p.is_file():
                return "/" + p.as_posix().lstrip("/")

    for p in sorted(model_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            # Skip Live2D texture atlas files when looking for a small preview unless nothing else exists
            if "texture_" not in p.name.lower():
                return "/" + p.as_posix().lstrip("/")
            candidates.append(p)

    if candidates:
        return "/" + candidates[0].as_posix().lstrip("/")
    return None


def _extract_zip_safely(zip_bytes: bytes, dest_dir: Path) -> list[Path]:
    """Extract a ZIP archive into `dest_dir` with Zip-Slip and bomb protection."""
    dest_root = dest_dir.resolve()
    extracted_files: list[Path] = []
    total_uncompressed = 0

    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
            infos = zf.infolist()
            if len(infos) > MAX_ARCHIVE_FILES:
                raise ValueError(
                    f"Archive contains too many entries ({len(infos)} > {MAX_ARCHIVE_FILES})"
                )

            # Check if all files share a single top-level directory so we can flatten it cleanly
            non_dir_parts = [
                Path(info.filename).parts
                for info in infos
                if not info.is_dir()
                and not info.filename.startswith("__MACOSX/")
                and not Path(info.filename).name.startswith("._")
            ]
            strip_prefix = False
            if non_dir_parts:
                first_roots = {parts[0] for parts in non_dir_parts if len(parts) > 1}
                if len(first_roots) == 1 and all(len(parts) > 1 for parts in non_dir_parts):
                    strip_prefix = True

            for info in infos:
                if info.is_dir():
                    continue
                if info.filename.startswith("__MACOSX/") or Path(info.filename).name.startswith("._"):
                    continue

                # Check symlink bit in external_attr (Unix S_IFLNK = 0o120000)
                if (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError(f"Symlinks are forbidden in avatar archives: {info.filename}")

                raw_rel = info.filename.replace("\\", "/")
                parts = [p for p in Path(raw_rel).parts if p not in ("", ".")]
                if not parts or any(p == ".." for p in parts) or Path(raw_rel).is_absolute():
                    raise ValueError(f"Unsafe path in archive: {info.filename}")

                if strip_prefix and len(parts) > 1:
                    parts = parts[1:]

                rel_path = Path(*parts)
                ext = rel_path.suffix.lower()
                if ext in FORBIDDEN_EXTENSIONS:
                    raise ValueError(f"Forbidden file type in archive: {rel_path.name}")

                total_uncompressed += max(0, info.file_size)
                if total_uncompressed > MAX_EXTRACTED_BYTES:
                    raise ValueError("Archive exceeds maximum uncompressed size limit (2048MB / 2GB)")

                target_path = (dest_root / rel_path).resolve()
                try:
                    target_path.relative_to(dest_root)
                except ValueError as exc:
                    raise ValueError(f"Archive path escapes target directory: {info.filename}") from exc

                target_path.parent.mkdir(parents=True, exist_ok=True)
                written = 0
                with zf.open(info, "r") as src, open(target_path, "wb") as dst:
                    while True:
                        chunk = src.read(64 * 1024)
                        if not chunk:
                            break
                        written += len(chunk)
                        if written > MAX_EXTRACTED_BYTES:
                            raise ValueError("Extracted file exceeds uncompressed size limit")
                        dst.write(chunk)
                extracted_files.append(target_path)
    except zipfile.BadZipFile as exc:
        raise ValueError("Uploaded archive is not a valid ZIP file") from exc

    if not extracted_files:
        raise ValueError("Uploaded archive did not contain any valid avatar files")
    return extracted_files


def _detect_and_configure_extracted_model(
    model_dir: Path, model_name: str, files: list[Path]
) -> tuple[str, str, dict[str, Any]]:
    """Inspect extracted files in `model_dir` and return `(format, url, extra_meta)`."""
    resolved_dir = model_dir.resolve()
    rel_files = [f.resolve().relative_to(resolved_dir) for f in files]

    # 1. Check for Live2D Cubism 3/4/5 (*.model3.json)
    model3_files = [f for f in rel_files if f.name.lower().endswith(".model3.json")]
    if model3_files:
        # Prefer top-level or matching name
        model3_files.sort(key=lambda p: (len(p.parts), p.name != f"{model_name}.model3.json"))
        chosen = model_dir / model3_files[0]
        url = "/" + chosen.as_posix().lstrip("/")
        return "cubism3", url, {}

    # 2. Check for Legacy Live2D Cubism 2.1 (*.model.json)
    model2_files = [
        f
        for f in rel_files
        if f.name.lower().endswith(".model.json") and not f.name.lower().endswith(".model3.json")
    ]
    if model2_files:
        model2_files.sort(key=lambda p: len(p.parts))
        chosen = model_dir / model2_files[0]
        url = "/" + chosen.as_posix().lstrip("/")
        return "cubism2", url, {}

    # 3. Check for 3D VRM (*.vrm)
    vrm_files = [f for f in rel_files if f.suffix.lower() == ".vrm"]
    if vrm_files:
        vrm_files.sort(key=lambda p: len(p.parts))
        chosen = model_dir / vrm_files[0]
        url = "/" + chosen.as_posix().lstrip("/")
        return "vrm", url, {}

    # 4. Check for 3D glTF / GLB (*.glb, *.gltf)
    gltf_files = [f for f in rel_files if f.suffix.lower() in {".glb", ".gltf"}]
    if gltf_files:
        gltf_files.sort(key=lambda p: (p.suffix.lower() != ".glb", len(p.parts)))
        chosen = model_dir / gltf_files[0]
        url = "/" + chosen.as_posix().lstrip("/")
        return "gltf", url, {}

    # 5. Check for 3D Metasequoia (*.mqo)
    mqo_files = [f for f in rel_files if f.suffix.lower() == ".mqo"]
    if mqo_files:
        mqo_files.sort(key=lambda p: len(p.parts))
        chosen = model_dir / mqo_files[0]
        url = "/" + chosen.as_posix().lstrip("/")
        return "mqo", url, {}

    # 6. Check for PNGTuber manifest or multi-state sprite images
    manifest_path = model_dir / "pngtuber.json"
    image_files = [f for f in rel_files if f.suffix.lower() in IMAGE_EXTENSIONS]
    if manifest_path.is_file() or image_files:
        if not manifest_path.is_file():
            # Auto-build a pngtuber.json manifest from image names!
            states: dict[str, str] = {}
            emotions: dict[str, dict[str, str]] = {}
            for img in image_files:
                stem = img.stem.lower()
                img_url = "/" + (model_dir / img).as_posix().lstrip("/")
                if stem in {"idle", "closed", "mouth_closed", "default", "normal"}:
                    states["closed"] = img_url
                elif stem in {"open", "talking", "talk", "mouth_open", "speak"}:
                    states["open"] = img_url
                elif stem in {"blink", "eyes_closed", "closed_eyes"}:
                    states["blink"] = img_url
                elif stem in {"blink_open", "talking_blink"}:
                    states["blink_open"] = img_url
                for emo in ("neutral", "joy", "happy", "anger", "angry", "sadness", "sad", "surprise", "surprised", "fear", "disgust", "smirk"):
                    if emo in stem:
                        canon = {
                            "happy": "joy",
                            "angry": "anger",
                            "sad": "sadness",
                            "surprised": "surprise",
                        }.get(emo, emo)
                        emotions.setdefault(canon, {})
                        if "open" in stem or "talk" in stem:
                            emotions[canon]["open"] = img_url
                        else:
                            emotions[canon]["closed"] = img_url

            first_img_url = "/" + (model_dir / image_files[0]).as_posix().lstrip("/")
            if "closed" not in states:
                states["closed"] = first_img_url
            if "open" not in states:
                states["open"] = states["closed"]

            manifest_data = {
                "name": model_name,
                "format": "pngtuber",
                "states": states,
                "emotions": emotions,
                "lipSyncThreshold": 0.08,
                "bounceIntensity": 6.0,
            }
            manifest_path.write_text(
                json.dumps(manifest_data, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        else:
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))

        url = "/" + manifest_path.as_posix().lstrip("/")
        return "pngtuber", url, {"pngtuber": manifest_data}

    raise ValueError(
        "Could not detect a supported avatar format in archive. Expected .model3.json (Live2D), .model.json (Live2D 2.1), .vrm (3D VRM), .glb/.gltf (3D glTF), or PNGTuber sprites."
    )


def ingest_avatar_package(
    filename: str,
    contents: bytes,
    *,
    custom_name: str | None = None,
    description: str = "",
    k_scale: float = 0.5,
    models_root: str | Path = "live2d-models",
    model_dict_path: str | Path = "model_dict.json",
) -> dict[str, Any]:
    """Validate, extract/store, auto-detect, and register a new VTuber avatar."""
    if not contents:
        raise ValueError("Uploaded file is empty")
    if len(contents) > MAX_UPLOAD_BYTES:
        raise ValueError("Uploaded file exceeds 1024MB (1GB) limit")

    orig_path = Path(filename or "avatar")
    ext = orig_path.suffix.lower()
    if ext in FORBIDDEN_EXTENSIONS:
        raise ValueError(f"Forbidden file extension: {ext}")

    base_name = sanitize_model_name(custom_name or orig_path.stem)
    root_dir = Path(models_root)
    root_dir.mkdir(parents=True, exist_ok=True)
    resolved_root = root_dir.resolve()

    target_dir = (resolved_root / base_name).resolve()
    try:
        target_dir.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("Invalid target model directory") from exc

    # Stage inside a temporary sibling folder first so existing models aren't corrupted on error
    staging_dir = resolved_root / f".staging_{base_name}_{secrets.token_hex(4)}"
    staging_dir.mkdir(parents=True, exist_ok=True)

    try:
        if ext == ".zip":
            _extract_zip_safely(contents, staging_dir)
        elif ext in ALLOWED_SINGLE_EXTENSIONS:
            safe_file = staging_dir / f"{base_name}{ext}"
            safe_file.write_bytes(contents)
        else:
            raise ValueError(
                f"Unsupported upload extension '{ext}'. Please upload a .zip archive, .vrm, .glb, or .png/.webp sprite."
            )

        # Move staging_dir into final target_dir
        if target_dir.exists():
            shutil.rmtree(target_dir)
        staging_dir.rename(target_dir)

        final_files = [p for p in target_dir.rglob("*") if p.is_file()]
        rel_target = Path(models_root) / base_name
        rel_final_files = [rel_target / p.relative_to(target_dir) for p in final_files]

        avatar_format, model_url, extra_meta = _detect_and_configure_extracted_model(
            rel_target, base_name, rel_final_files
        )
        preview = _find_preview_image(rel_target, base_name)
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    entry: dict[str, Any] = {
        "name": base_name,
        "description": description or f"{avatar_format.upper()} Avatar ({base_name})",
        "format": avatar_format,
        "url": model_url,
        "avatar": preview,
        "kScale": float(k_scale) if k_scale > 0 else 0.5,
        "initialXshift": 0,
        "initialYshift": 0,
        "kXOffset": 1150,
        "idleMotionGroupName": "Idle",
        "emotionMap": dict(DEFAULT_EMOTION_MAP),
        "tapMotions": {
            "HitAreaHead": {"": 1},
            "HitAreaBody": {"": 1},
        },
        **extra_meta,
    }

    entries = load_model_dict(model_dict_path)
    updated = False
    for idx, existing in enumerate(entries):
        if isinstance(existing, dict) and existing.get("name") == base_name:
            entries[idx] = entry
            updated = True
            break
    if not updated:
        entries.append(entry)
    save_model_dict(entries, model_dict_path)
    return entry


def init_avatar_model_routes(default_context_cache: "ServiceContext") -> "APIRouter":
    """Create and return authenticated routes for listing, uploading, selecting, and deleting avatars."""
    from fastapi import (
        APIRouter,
        Depends,
        File,
        Form,
        HTTPException,
        Query,
        UploadFile,
        status,
    )
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    from loguru import logger
    from pydantic import BaseModel, Field
    from .routes import _normalize_api_key

    security = HTTPBearer(auto_error=False)

    class SelectAvatarRequest(BaseModel):
        name: str = Field(..., min_length=1, max_length=64)

    router = APIRouter()

    async def verify_api_key(
        auth: HTTPAuthorizationCredentials = Depends(security),
        query_api_key: str | None = Query(default=None, alias="api_key"),
    ):
        expected = _normalize_api_key(default_context_cache.system_config.api_key)
        if not expected:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Server API key is not configured",
            )
        provided = _normalize_api_key(auth.credentials if auth else query_api_key)
        if not provided or not secrets.compare_digest(provided, expected):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or Missing API Key",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return provided

    @router.get("/api/models", dependencies=[Depends(verify_api_key)])
    async def list_avatar_models():
        """Return all registered 2D/3D avatar models and the currently active model name."""
        entries = load_model_dict()
        active_name = (
            default_context_cache.character_config.live2d_model_name
            if default_context_cache and default_context_cache.character_config
            else None
        )
        enriched = []
        for item in entries:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "")
            fmt = detect_model_format_from_entry(item)
            avatar = item.get("avatar")
            if not avatar and name:
                candidate_dir = Path("live2d-models") / name
                if candidate_dir.is_dir():
                    avatar = _find_preview_image(candidate_dir, name)
            enriched.append(
                {
                    **item,
                    "format": fmt,
                    "avatar": avatar,
                    "active": name == active_name,
                }
            )
        return {
            "models": enriched,
            "active_model": active_name,
            "count": len(enriched),
        }

    @router.post("/api/models/upload", dependencies=[Depends(verify_api_key)])
    async def upload_avatar_model(
        file: UploadFile = File(...),
        name: str | None = Form(default=None),
        description: str = Form(default=""),
        k_scale: float = Form(default=0.5),
        activate: bool = Form(default=True),
    ):
        """Upload a .zip (Cubism 3/4/5, Cubism 2.1, VRM, glTF, PNGTuber) or single .vrm/.glb/.png avatar."""
        try:
            contents = await file.read(MAX_UPLOAD_BYTES + 1)
            if len(contents) > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="Uploaded avatar exceeds maximum size of 1024MB (1GB)",
                )
            entry = ingest_avatar_package(
                filename=file.filename or "avatar.zip",
                contents=contents,
                custom_name=name,
                description=description,
                k_scale=k_scale,
            )
            if activate and default_context_cache and default_context_cache.character_config:
                default_context_cache.init_live2d(entry["name"])
            return {
                "status": "ok",
                "model": entry,
                "active_model": (
                    default_context_cache.character_config.live2d_model_name
                    if default_context_cache and default_context_cache.character_config
                    else entry["name"]
                ),
            }
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        except Exception as exc:
            logger.error(f"Avatar upload failed: {exc}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process uploaded avatar package",
            ) from exc

    @router.post("/api/models/select", dependencies=[Depends(verify_api_key)])
    async def select_avatar_model(payload: SelectAvatarRequest):
        """Switch the active VTuber avatar model in the server context and return its model_info."""
        try:
            clean_name = sanitize_model_name(payload.name)
            entries = load_model_dict()
            matched = next(
                (m for m in entries if isinstance(m, dict) and m.get("name") == clean_name),
                None,
            )
            if not matched:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Avatar model '{clean_name}' not found",
                )
            if default_context_cache and default_context_cache.character_config:
                default_context_cache.init_live2d(clean_name)
                if default_context_cache.live2d_model:
                    matched = default_context_cache.live2d_model.model_info
            return {
                "status": "ok",
                "active_model": clean_name,
                "model_info": matched,
            }
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from exc

    @router.delete("/api/models/{model_name}", dependencies=[Depends(verify_api_key)])
    async def delete_avatar_model(model_name: str):
        """Remove a registered avatar model from `model_dict.json` and delete its model folder."""
        try:
            clean_name = sanitize_model_name(model_name)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
            ) from exc

        entries = load_model_dict()
        remaining = [
            m for m in entries if not (isinstance(m, dict) and m.get("name") == clean_name)
        ]
        if len(remaining) == len(entries):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Avatar model '{clean_name}' not found",
            )
        if not remaining:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Cannot delete the last remaining avatar model",
            )

        save_model_dict(remaining)

        models_root = Path("live2d-models").resolve()
        target_dir = (models_root / clean_name).resolve()
        try:
            target_dir.relative_to(models_root)
            if target_dir.is_dir() and not target_dir.is_symlink():
                shutil.rmtree(target_dir)
        except ValueError:
            pass

        active_name = (
            default_context_cache.character_config.live2d_model_name
            if default_context_cache and default_context_cache.character_config
            else None
        )
        fallback_info = None
        if active_name == clean_name and remaining:
            fallback_name = remaining[0]["name"]
            default_context_cache.init_live2d(fallback_name)
            active_name = fallback_name
            if default_context_cache.live2d_model:
                fallback_info = default_context_cache.live2d_model.model_info

        return {
            "status": "ok",
            "deleted": clean_name,
            "active_model": active_name,
            "model_info": fallback_info,
        }

    return router

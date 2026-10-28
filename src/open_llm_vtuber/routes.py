import os
import io
import json
import wave
import asyncio
import secrets
from uuid import uuid4
from pathlib import Path
import numpy as np
from datetime import datetime
from fastapi import APIRouter, WebSocket, UploadFile, File, Response, Depends, HTTPException, status, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.responses import JSONResponse, FileResponse
from starlette.websockets import WebSocketDisconnect
from starlette.background import BackgroundTask
from loguru import logger
from .service_context import ServiceContext
from .websocket_handler import WebSocketHandler
from .proxy_handler import ProxyHandler
from .artifacts.manager import ArtifactManager
from .artifacts.workspace import WorkspaceManager

security = HTTPBearer(auto_error=False)


def _normalize_api_key(value: str | None) -> str:
    """Normalize a user-supplied API key without changing its configured value.

    Browsers/settings can preserve surrounding quotes when users paste a YAML-style
    value such as 'olv_...'. Authentication should treat only matching outer quotes
    as formatting noise; embedded quotes remain part of the credential.
    """
    if value is None:
        return ""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1].strip()
    return value

# TTS WebSocket limits to prevent unauthenticated/abusive clients from
# exhausting CPU/GPU resources with huge text or request bursts.
MAX_TTS_TEXT_LENGTH = 16 * 1024
MAX_TTS_SENTENCES = 64
MAX_TTS_REQUESTS_PER_MINUTE = 20
MAX_TTS_WS_MESSAGE_BYTES = 128 * 1024
MAX_TTS_CONNECTIONS = 8
MAX_CLIENT_CONNECTIONS = 32

# Artifact download/export limits. Keep downloads authenticated and bounded.
MAX_ARTIFACT_DOWNLOAD_BYTES = 100 * 1024 * 1024
MAX_WORKSPACE_ZIP_FILES = 1000
MAX_WORKSPACE_ZIP_BYTES = 500 * 1024 * 1024
ARTIFACT_DOWNLOAD_TEMP_DIR = "cache/artifact-downloads"


async def get_current_api_key(
    default_context_cache: ServiceContext,
    auth: HTTPAuthorizationCredentials = Depends(security),
):
    """Dependency to validate the API key from the Authorization header"""
    expected_api_key = _normalize_api_key(default_context_cache.system_config.api_key)
    if not expected_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server API key is not configured",
        )

    provided = _normalize_api_key(auth.credentials if auth else None)
    if not auth or not secrets.compare_digest(provided, expected_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or Missing API Key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return auth.credentials


def init_client_ws_route(default_context_cache: ServiceContext) -> APIRouter:
    """
    Create and return API routes for handling the `/client-ws` WebSocket connections.

    Args:
        default_context_cache: Default service context cache for new sessions.

    Returns:
        APIRouter: Configured router with WebSocket endpoint.
    """

    router = APIRouter()
    ws_handler = WebSocketHandler(default_context_cache)

    @router.websocket("/client-ws")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for client connections"""
        if len(ws_handler.client_connections) >= MAX_CLIENT_CONNECTIONS:
            await websocket.close(code=status.WS_1013_TRY_AGAIN_LATER)
            return

        await websocket.accept()

        # Handle authentication (fail closed; an empty API key never disables auth)
        expected_api_key = _normalize_api_key(default_context_cache.system_config.api_key)
        if not expected_api_key:
            logger.error("WebSocket connection rejected: server API key is not configured")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        try:
            # Prefer the API key supplied in the WebSocket URL. This makes the
            # authentication flow robust against frontend message-ordering issues.
            query_api_key = websocket.query_params.get("api_key")
            if query_api_key:
                supplied_api_key = _normalize_api_key(query_api_key)
            else:
                # Backward-compatible fallback: require the first frame to be auth.
                auth_msg = await asyncio.wait_for(websocket.receive_json(), timeout=5.0)
                supplied_api_key = _normalize_api_key(auth_msg.get("api_key")) if auth_msg.get("type") == "auth" else None

            if not supplied_api_key or not secrets.compare_digest(supplied_api_key, expected_api_key):
                logger.warning("WebSocket authentication failed")
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                return
            logger.info("WebSocket authenticated successfully")
        except asyncio.TimeoutError:
            logger.warning("WebSocket authentication timed out")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        except Exception as e:
            logger.error(f"Error during WebSocket authentication: {e}")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        client_uid = str(uuid4())

        try:
            await ws_handler.handle_new_connection(websocket, client_uid)
            await ws_handler.handle_websocket_communication(websocket, client_uid)
        except WebSocketDisconnect:
            await ws_handler.handle_disconnect(client_uid)
        except Exception as e:
            logger.error(f"Error in WebSocket connection: {e}")
            await ws_handler.handle_disconnect(client_uid)
            raise

    return router


def init_proxy_route(default_context_cache: ServiceContext) -> APIRouter:
    """
    Create and return API routes for handling proxy connections.

    Args:
        default_context_cache: Default service context cache for new sessions.

    Returns:
        APIRouter: Configured router with proxy WebSocket endpoint
    """
    router = APIRouter()
    server_config = default_context_cache.system_config
    server_url = f"ws://{server_config.host}:{server_config.port}/client-ws"
    proxy_handler = ProxyHandler(server_url, api_key=server_config.api_key)

    @router.websocket("/proxy-ws")
    async def proxy_endpoint(websocket: WebSocket):
        """WebSocket endpoint for proxy connections"""
        try:
            await proxy_handler.handle_client_connection(websocket)
        except Exception as e:
            logger.error(f"Error in proxy connection: {e}")
            raise

    return router


def _artifact_managers() -> tuple[WorkspaceManager, ArtifactManager]:
    """Open the shared artifact metadata/storage managers used by artifact agents."""
    workspace_manager = WorkspaceManager("generated_files")
    artifact_manager = ArtifactManager(workspace_manager._db_path)
    return workspace_manager, artifact_manager


def _cleanup_temp_file(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _conversation_created_at(conversation_id: str | None) -> str | None:
    """Best-effort creation time from the app's timestamp-prefixed history UID.

    History IDs are created with the server's local clock and do not include a timezone,
    so keep the value timezone-naive rather than incorrectly labeling it as UTC.
    """
    if not conversation_id or len(conversation_id) < 19:
        return None
    raw = conversation_id[:19]
    try:
        datetime.strptime(raw, "%Y-%m-%d_%H-%M-%S")
    except ValueError:
        return None
    return raw.replace("_", " ")


def _friendly_workspace_name(artifact_manager: ArtifactManager, workspace) -> str:
    artifacts = artifact_manager.list(workspace_id=workspace.workspace_id)
    for artifact in artifacts:
        metadata = artifact.metadata if isinstance(artifact.metadata, dict) else {}
        project_name = metadata.get("project_name")
        if isinstance(project_name, str) and project_name.strip():
            return " ".join(project_name.strip().split())[:60]
    if len(artifacts) == 1:
        stem = Path(artifacts[0].name).stem.replace("_", " ").replace("-", " ").strip()
        return (stem[:1].upper() + stem[1:])[:60] if stem else "General Files"
    exts = {Path(a.name).suffix.lower() for a in artifacts}
    if {".html", ".css", ".js"} & exts:
        return "Web Project"
    if {".xlsx", ".csv"} & exts:
        return "Spreadsheet"
    if {".docx", ".pdf"} & exts:
        return "Documents"
    if ".pptx" in exts:
        return "Presentation"
    return "File Collection" if artifacts else "General Files"


def init_artifact_download_routes(default_context_cache: ServiceContext) -> APIRouter:
    router = APIRouter()

    async def verify_api_key(
        auth: HTTPAuthorizationCredentials = Depends(security),
    ):
        expected = _normalize_api_key(default_context_cache.system_config.api_key)
        if not expected:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Server API key is not configured")
        provided = _normalize_api_key(auth.credentials if auth else None)
        if not provided or not secrets.compare_digest(provided, expected):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or Missing API Key", headers={"WWW-Authenticate": "Bearer"})
        return provided

    @router.get("/artifacts/{artifact_id}/download", dependencies=[Depends(verify_api_key)])
    async def download_artifact(artifact_id: str):
        """Download one managed artifact without exposing its filesystem path."""
        workspace_manager, artifact_manager = _artifact_managers()
        try:
            artifact = artifact_manager.require(artifact_id)
            if artifact.workspace_id is None or not artifact.path:
                raise HTTPException(status_code=404, detail="Artifact file not found")
            path = workspace_manager._resolve_file(artifact.workspace_id, artifact.path)
            if path.is_symlink() or not path.exists() or not path.is_file():
                raise HTTPException(status_code=404, detail="Artifact file not found")
            size = path.stat().st_size
            if size > MAX_ARTIFACT_DOWNLOAD_BYTES:
                raise HTTPException(status_code=413, detail="Artifact exceeds the download size limit")
            return FileResponse(
                path,
                media_type=artifact.mime_type or "application/octet-stream",
                filename=artifact.name,
            )
        finally:
            artifact_manager.close()
            workspace_manager.close()

    @router.get("/workspaces", dependencies=[Depends(verify_api_key)])
    async def list_artifact_workspaces():
        """List managed artifact workspaces with bounded storage metadata."""
        workspace_manager, artifact_manager = _artifact_managers()
        try:
            items = []
            workspaces = sorted(
                workspace_manager.list(),
                key=lambda w: str(w.created_at or ""),
                reverse=True,
            )
            for workspace in workspaces:
                display_name = workspace.name
                if display_name.strip().lower() in {"vtuber artifacts", "untitled project", "general files"}:
                    display_name = _friendly_workspace_name(artifact_manager, workspace)
                    if display_name and display_name != workspace.name:
                        try:
                            workspace_manager.rename(workspace.workspace_id, display_name, touch_updated=False)
                        except Exception:
                            pass
                items.append({
                    "workspace_id": workspace.workspace_id,
                    "name": display_name,
                    "conversation_id": workspace.conversation_id,
                    "conversation_created_at": _conversation_created_at(workspace.conversation_id),
                    "created_at": workspace.created_at,
                    "updated_at": workspace.updated_at,
                    "size": workspace_manager.size(workspace.workspace_id),
                    "file_count": workspace_manager.file_count(workspace.workspace_id),
                    "artifacts": len(artifact_manager.list(workspace_id=workspace.workspace_id)),
                })
            return {
                "workspaces": items,
                "total_size": workspace_manager.total_size(),
                "max_total_size": workspace_manager.max_total_size,
                "max_workspace_size": workspace_manager.max_workspace_size,
                "max_files_per_workspace": workspace_manager.max_files_per_workspace,
            }
        finally:
            artifact_manager.close()
            workspace_manager.close()

    @router.get("/workspaces/{workspace_id}/files", dependencies=[Depends(verify_api_key)])
    async def list_artifact_workspace_files(workspace_id: str):
        """List regular files and managed artifacts belonging to one workspace."""
        workspace_manager, artifact_manager = _artifact_managers()
        try:
            workspace = workspace_manager.require(workspace_id)
            artifacts = {a.path: a for a in artifact_manager.list(workspace_id=workspace_id) if a.path}
            files = []
            for rel in workspace_manager.list_files(workspace_id):
                artifact = artifacts.get(rel)
                files.append({
                    "artifact_id": artifact.artifact_id if artifact else None,
                    "name": artifact.name if artifact else Path(rel).name,
                    "path": rel,
                    "mime_type": artifact.mime_type if artifact else None,
                    "size": Path(workspace.root_path, rel).stat().st_size,
                    "created_at": artifact.created_at if artifact else None,
                    "updated_at": artifact.updated_at if artifact else None,
                })
            files.sort(
                key=lambda f: str(f.get("created_at") or f.get("updated_at") or ""),
                reverse=True,
            )
            return {
                "workspace": {
                    "workspace_id": workspace.workspace_id,
                    "name": workspace.name,
                    "created_at": workspace.created_at,
                    "updated_at": workspace.updated_at,
                    "conversation_id": workspace.conversation_id,
                    "conversation_created_at": _conversation_created_at(workspace.conversation_id),
                    "size": workspace_manager.size(workspace_id),
                    "file_count": workspace_manager.file_count(workspace_id),
                },
                "files": files,
            }
        finally:
            artifact_manager.close()
            workspace_manager.close()

    @router.delete("/artifacts/{artifact_id}", dependencies=[Depends(verify_api_key)])
    async def delete_artifact(artifact_id: str):
        """Delete one managed artifact with rollback-safe file staging."""
        workspace_manager, artifact_manager = _artifact_managers()
        staged_path: Path | None = None
        original_path: Path | None = None
        try:
            artifact = artifact_manager.require(artifact_id)
            if artifact.workspace_id is None or not artifact.path:
                raise HTTPException(status_code=404, detail="Artifact file not found")
            original_path = workspace_manager._resolve_file(artifact.workspace_id, artifact.path)
            if original_path.is_symlink() or not original_path.exists() or not original_path.is_file():
                raise HTTPException(status_code=404, detail="Artifact file not found")
            staged_path = original_path.with_name(f".{original_path.name}.delete-{uuid4().hex}")
            os.replace(original_path, staged_path)
            try:
                artifact_manager.delete(artifact_id)
            except Exception:
                os.replace(staged_path, original_path)
                staged_path = None
                raise
            try:
                staged_path.unlink(missing_ok=True)
            finally:
                staged_path = None
            return {"success": True, "artifact_id": artifact_id}
        except KeyError:
            raise HTTPException(status_code=404, detail="Artifact not found")
        finally:
            if staged_path is not None and original_path is not None:
                try:
                    os.replace(staged_path, original_path)
                except OSError:
                    pass
            artifact_manager.close()
            workspace_manager.close()

    @router.delete("/workspaces/{workspace_id}", dependencies=[Depends(verify_api_key)])
    async def delete_artifact_workspace(workspace_id: str):
        """Delete one workspace transactionally with rollback-safe filesystem staging."""
        workspace_manager, artifact_manager = _artifact_managers()
        staged_root: Path | None = None
        original_root: Path | None = None
        artifact_ids: list[str] = []
        committed = False
        try:
            workspace = workspace_manager.require(workspace_id)
            original_root = workspace.root_path
            staged_root = original_root.with_name(f".{original_root.name}.delete-{uuid4().hex}")
            os.replace(original_root, staged_root)
            try:
                artifact_ids = workspace_manager.delete_workspace_metadata(workspace_id)
                committed = True
            except Exception:
                os.replace(staged_root, original_root)
                staged_root = None
                raise
            # Keep the in-memory artifact cache consistent with the committed DB transaction.
            for artifact_id in artifact_ids:
                artifact_manager._artifacts.pop(artifact_id, None)
            import shutil
            # Cleanup is best-effort after the durable DB transaction commits. If the OS refuses
            # deletion, leave the staged trash directory outside managed workspace metadata; never
            # recreate a workspace that has already been committed as deleted.
            try:
                shutil.rmtree(staged_root, ignore_errors=False)
            except OSError:
                logger.warning(f"Deferred cleanup needed for deleted workspace temp directory: {staged_root}")
            staged_root = None
            return {"success": True, "workspace_id": workspace_id}
        except KeyError:
            raise HTTPException(status_code=404, detail="Workspace not found")
        finally:
            if staged_root is not None and original_root is not None and not committed:
                try:
                    os.replace(staged_root, original_root)
                except OSError:
                    pass
            artifact_manager.close()
            workspace_manager.close()

    @router.get("/workspaces/{workspace_id}/download.zip", dependencies=[Depends(verify_api_key)])
    async def download_workspace_zip(workspace_id: str):
        """Create a bounded temporary ZIP of one managed workspace and delete it after response."""
        import tempfile
        import zipfile

        workspace_manager, artifact_manager = _artifact_managers()
        try:
            workspace_manager.require(workspace_id)
            files = workspace_manager.list_files(workspace_id)
            if len(files) > MAX_WORKSPACE_ZIP_FILES:
                raise HTTPException(status_code=413, detail="Workspace contains too many files to export")
            total = sum(workspace_manager.require(workspace_id).root_path.joinpath(rel).stat().st_size for rel in files)
            if total > MAX_WORKSPACE_ZIP_BYTES:
                raise HTTPException(status_code=413, detail="Workspace exceeds the ZIP export limit")

            temp_dir = Path(ARTIFACT_DOWNLOAD_TEMP_DIR)
            temp_dir.mkdir(parents=True, exist_ok=True)
            fd, temp_name = tempfile.mkstemp(prefix="artifact-", suffix=".zip", dir=temp_dir)
            os.close(fd)
            try:
                with zipfile.ZipFile(temp_name, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=False) as archive:
                    for rel in files:
                        src = workspace_manager._resolve_file(workspace_id, rel)
                        if src.is_symlink() or not src.is_file():
                            raise HTTPException(status_code=404, detail="Workspace contains an invalid file")
                        fd = os.open(src, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
                        try:
                            before = os.fstat(fd)
                            if before.st_size > MAX_WORKSPACE_ZIP_BYTES:
                                raise HTTPException(status_code=413, detail="Workspace file exceeds the ZIP export limit")
                            info = zipfile.ZipInfo(rel)
                            info.compress_type = zipfile.ZIP_DEFLATED
                            with os.fdopen(fd, "rb", closefd=True) as source, archive.open(info, "w") as target:
                                fd = -1
                                import shutil
                                shutil.copyfileobj(source, target, length=1024 * 1024)
                                after = os.fstat(source.fileno())
                            if after.st_size != before.st_size:
                                raise HTTPException(status_code=409, detail=f"File changed during ZIP export: {rel}")
                        finally:
                            if fd != -1:
                                os.close(fd)
            except Exception:
                _cleanup_temp_file(temp_name)
                raise

            return FileResponse(
                temp_name,
                media_type="application/zip",
                filename=f"{workspace_id}.zip",
                background=BackgroundTask(_cleanup_temp_file, temp_name),
            )
        finally:
            artifact_manager.close()
            workspace_manager.close()

    return router


def init_webtool_routes(default_context_cache: ServiceContext) -> APIRouter:
    """
    Create and return API routes for handling web tool interactions.

    Args:
        default_context_cache: Default service context cache for new sessions.

    Returns:
        APIRouter: Configured router with WebSocket endpoint.
    """

    router = APIRouter()
    tts_connections = 0

    # Pass default_context_cache to dependencies
    async def verify_api_key(
        auth: HTTPAuthorizationCredentials = Depends(security),
        query_api_key: str | None = Query(default=None, alias="api_key"),
    ):
        expected = _normalize_api_key(default_context_cache.system_config.api_key)
        if not expected:
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Server API key is not configured")
        provided = _normalize_api_key(auth.credentials if auth else query_api_key)
        if not provided or not secrets.compare_digest(provided, expected):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or Missing API Key", headers={"WWW-Authenticate": "Bearer"})
        return provided

    @router.get("/web-tool")
    async def web_tool_redirect():
        """Redirect /web-tool to /web_tool/index.html"""
        return Response(status_code=302, headers={"Location": "/web-tool/index.html"})

    @router.get("/web_tool")
    async def web_tool_redirect_alt():
        """Redirect /web_tool to /web_tool/index.html"""
        return Response(status_code=302, headers={"Location": "/web-tool/index.html"})

    @router.get("/live2d-models/info", dependencies=[Depends(verify_api_key)])
    async def get_live2d_folder_info():
        """Get information about available Live2D and multi-format VTuber models"""
        from .avatar_manager import load_model_dict, detect_model_format_from_entry, _find_preview_image

        live2d_dir = "live2d-models"
        valid_characters = []
        seen_names = set()

        try:
            for item in load_model_dict():
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or "")
                url = str(item.get("url") or "")
                if not name or not url:
                    continue
                avatar_file = item.get("avatar")
                if not avatar_file:
                    candidate_dir = Path(live2d_dir) / name
                    if candidate_dir.is_dir():
                        avatar_file = _find_preview_image(candidate_dir, name)
                valid_characters.append(
                    {
                        "name": name,
                        "avatar": avatar_file,
                        "model_path": url.lstrip("/"),
                        "url": url,
                        "format": detect_model_format_from_entry(item),
                    }
                )
                seen_names.add(name)
        except Exception as e:
            logger.warning(f"Failed to read model_dict.json in /live2d-models/info: {e}")

        if os.path.exists(live2d_dir):
            supported_extensions = [".png", ".jpg", ".jpeg", ".webp"]
            for entry in os.scandir(live2d_dir):
                if entry.is_dir() and entry.name not in seen_names:
                    folder_name = entry.name.replace("\\", "/")
                    model3_file = os.path.join(
                        live2d_dir, folder_name, f"{folder_name}.model3.json"
                    ).replace("\\", "/")

                    if os.path.isfile(model3_file):
                        avatar_file = None
                        for ext in supported_extensions:
                            avatar_path = os.path.join(
                                live2d_dir, folder_name, f"{folder_name}{ext}"
                            )
                            if os.path.isfile(avatar_path):
                                avatar_file = avatar_path.replace("\\", "/")
                                break

                        valid_characters.append(
                            {
                                "name": folder_name,
                                "avatar": avatar_file,
                                "model_path": model3_file,
                                "url": "/" + model3_file.lstrip("/"),
                                "format": "cubism3",
                            }
                        )
        return JSONResponse(
            {
                "type": "live2d-models/info",
                "count": len(valid_characters),
                "characters": valid_characters,
            }
        )

    @router.post("/asr", dependencies=[Depends(verify_api_key)])
    async def transcribe_audio(file: UploadFile = File(...)):
        """
        Endpoint for transcribing audio using the ASR engine
        """
        logger.info(f"Received audio file for transcription: {file.filename}")

        # Limit file size to 10MB to prevent OOM
        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
        
        try:
            contents = await file.read()
            
            if len(contents) > MAX_FILE_SIZE:
                logger.error(f"File too large: {len(contents)} bytes")
                return Response(
                    content=json.dumps({"error": "File too large. Maximum size is 10MB."}),
                    status_code=413,
                    media_type="application/json",
                )

            # Validate minimum file size
            if len(contents) < 44:  # Minimum WAV header size
                raise ValueError("Invalid WAV file: File too small")

            # Parse WAV chunks safely using the standard wave module, falling back
            # to a 44-byte header offset if the container uses non-standard tags
            # while still carrying raw 16-bit PCM frames.
            try:
                with wave.open(io.BytesIO(contents), "rb") as wf:
                    if wf.getsampwidth() != 2:
                        raise ValueError(
                            f"Audio format error: Unsupported sample width ({wf.getsampwidth()} bytes). Please ensure the file is 16-bit PCM WAV format."
                        )
                    audio_data = wf.readframes(wf.getnframes())
            except wave.Error:
                wav_header_size = 44  # Standard WAV header size fallback
                audio_data = contents[wav_header_size:]

            # Validate audio data size
            if len(audio_data) % 2 != 0:
                raise ValueError("Invalid audio data: Buffer size must be even")

            # Convert to 16-bit PCM samples to float32
            try:
                audio_array = (
                    np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
                    / 32768.0
                )
            except ValueError as e:
                raise ValueError(
                    f"Audio format error: {str(e)}. Please ensure the file is 16-bit PCM WAV format."
                )

            # Validate audio data
            if len(audio_array) == 0:
                raise ValueError("Empty audio data")

            text = await default_context_cache.asr_engine.async_transcribe_np(
                audio_array
            )
            logger.info(f"Transcription result: {text}")
            return {"text": text}

        except ValueError as e:
            logger.error(f"Audio format error: {e}")
            return Response(
                content=json.dumps({"error": str(e)}),
                status_code=400,
                media_type="application/json",
            )
        except Exception as e:
            logger.error(f"Error during transcription: {e}")
            return Response(
                content=json.dumps(
                    {"error": "Internal server error during transcription"}
                ),
                status_code=500,
                media_type="application/json",
            )

    @router.websocket("/tts-ws")
    async def tts_endpoint(websocket: WebSocket):
        """WebSocket endpoint for TTS generation"""
        nonlocal tts_connections
        await websocket.accept()

        # Handle authentication (fail closed; an empty API key never disables auth)
        expected_api_key = _normalize_api_key(default_context_cache.system_config.api_key)
        if not expected_api_key:
            logger.error("TTS WebSocket connection rejected: server API key is not configured")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        try:
            # Prefer the API key supplied in the WebSocket URL.
            query_api_key = websocket.query_params.get("api_key")
            if query_api_key:
                supplied_api_key = _normalize_api_key(query_api_key)
            else:
                auth_msg = await asyncio.wait_for(websocket.receive_json(), timeout=5.0)
                supplied_api_key = _normalize_api_key(auth_msg.get("api_key")) if auth_msg.get("type") == "auth" else None

            if not supplied_api_key or not secrets.compare_digest(supplied_api_key, expected_api_key):
                logger.warning("TTS WebSocket authentication failed")
                await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
                return
            logger.info("TTS WebSocket authenticated successfully")
        except asyncio.TimeoutError:
            logger.warning("TTS WebSocket authentication timed out")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return
        except Exception as e:
            logger.error(f"Error during TTS WebSocket authentication: {e}")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        if tts_connections >= MAX_TTS_CONNECTIONS:
            await websocket.close(code=1013)
            return

        tts_connections += 1
        logger.info("TTS WebSocket connection established")
        request_times = []

        try:
            while True:
                raw_message = await websocket.receive()
                if raw_message.get("type") == "websocket.disconnect":
                    raise WebSocketDisconnect(code=raw_message.get("code", 1000))

                raw_text = raw_message.get("text")
                if raw_text is None:
                    await websocket.close(code=1003)
                    raise WebSocketDisconnect(code=1003)

                if len(raw_text.encode("utf-8")) > MAX_TTS_WS_MESSAGE_BYTES:
                    await websocket.send_json({"status": "error", "message": "TTS request too large"})
                    await websocket.close(code=1009)
                    raise WebSocketDisconnect(code=1009)

                try:
                    data = json.loads(raw_text)
                except json.JSONDecodeError:
                    await websocket.send_json({"status": "error", "message": "Invalid JSON"})
                    continue

                text = data.get("text")
                if not isinstance(text, str) or not text.strip():
                    continue

                if len(text) > MAX_TTS_TEXT_LENGTH:
                    await websocket.send_json({"status": "error", "message": "TTS text too large"})
                    continue

                now = datetime.now().timestamp()
                request_times = [t for t in request_times if now - t < 60]
                if len(request_times) >= MAX_TTS_REQUESTS_PER_MINUTE:
                    await websocket.send_json({"status": "error", "message": "TTS rate limit exceeded"})
                    await websocket.close(code=1013)
                    raise WebSocketDisconnect(code=1013)
                request_times.append(now)

                logger.info(f"Received text for TTS ({len(text)} chars)")

                # Split text into sentences
                sentences = [s.strip() for s in text.split(".") if s.strip()]
                if len(sentences) > MAX_TTS_SENTENCES:
                    await websocket.send_json({"status": "error", "message": "Too many TTS sentences"})
                    continue

                try:
                    # Generate and send audio for each sentence
                    for sentence in sentences:
                        sentence = sentence + "."  # Add back the period
                        file_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid4())[:8]}"
                        audio_path = (
                            await default_context_cache.tts_engine.async_generate_audio(
                                text=sentence, file_name_no_ext=file_name
                            )
                        )
                        logger.info(
                            f"Generated audio for sentence: {sentence} at: {audio_path}"
                        )

                        await websocket.send_json(
                            {
                                "status": "partial",
                                "audioPath": audio_path,
                                "text": sentence,
                            }
                        )

                    # Send completion signal
                    await websocket.send_json({"status": "complete"})

                except Exception as e:
                    logger.error(f"Error generating TTS: {e}")
                    await websocket.send_json({"status": "error", "message": str(e)})

        except WebSocketDisconnect:
            logger.info("TTS WebSocket client disconnected")
        except Exception as e:
            logger.error(f"Error in TTS WebSocket connection: {e}")
            await websocket.close()
        finally:
            tts_connections = max(0, tts_connections - 1)

    return router

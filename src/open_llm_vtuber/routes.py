import os
import json
import asyncio
from uuid import uuid4
import numpy as np
from datetime import datetime
from fastapi import APIRouter, WebSocket, UploadFile, File, Response, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.responses import JSONResponse
from starlette.websockets import WebSocketDisconnect
from loguru import logger
from .service_context import ServiceContext
from .websocket_handler import WebSocketHandler
from .proxy_handler import ProxyHandler

security = HTTPBearer(auto_error=False)

# TTS WebSocket limits to prevent unauthenticated/abusive clients from
# exhausting CPU/GPU resources with huge text or request bursts.
MAX_TTS_TEXT_LENGTH = 16 * 1024
MAX_TTS_SENTENCES = 64
MAX_TTS_REQUESTS_PER_MINUTE = 20
MAX_TTS_WS_MESSAGE_BYTES = 128 * 1024
MAX_TTS_CONNECTIONS = 8
MAX_CLIENT_CONNECTIONS = 32


async def get_current_api_key(
    default_context_cache: ServiceContext,
    auth: HTTPAuthorizationCredentials = Depends(security),
):
    """Dependency to validate the API key from the Authorization header"""
    expected_api_key = default_context_cache.system_config.api_key
    if not expected_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server API key is not configured",
        )

    if not auth or auth.credentials != expected_api_key:
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
        expected_api_key = default_context_cache.system_config.api_key
        if not expected_api_key:
            logger.error("WebSocket connection rejected: server API key is not configured")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        try:
            # Prefer the API key supplied in the WebSocket URL. This makes the
            # authentication flow robust against frontend message-ordering issues.
            query_api_key = websocket.query_params.get("api_key")
            if query_api_key:
                supplied_api_key = query_api_key
            else:
                # Backward-compatible fallback: require the first frame to be auth.
                auth_msg = await asyncio.wait_for(websocket.receive_json(), timeout=5.0)
                supplied_api_key = auth_msg.get("api_key") if auth_msg.get("type") == "auth" else None

            if supplied_api_key != expected_api_key:
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


def init_webtool_routes(default_context_cache: ServiceContext) -> APIRouter:
    """
    Create and return API routes for handling web tool interactions.

    Args:
        default_context_cache: Default service context cache for new sessions.

    Returns:
        APIRouter: Configured router with WebSocket endpoint.
    """

    router = APIRouter()

    # Pass default_context_cache to dependencies
    async def verify_api_key(auth: HTTPAuthorizationCredentials = Depends(security)):
        return await get_current_api_key(default_context_cache, auth)

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
        """Get information about available Live2D models"""
        live2d_dir = "live2d-models"
        if not os.path.exists(live2d_dir):
            return JSONResponse(
                {"error": "Live2D models directory not found"}, status_code=404
            )

        valid_characters = []
        supported_extensions = [".png", ".jpg", ".jpeg"]

        for entry in os.scandir(live2d_dir):
            if entry.is_dir():
                folder_name = entry.name.replace("\\", "/")
                model3_file = os.path.join(
                    live2d_dir, folder_name, f"{folder_name}.model3.json"
                ).replace("\\", "/")

                if os.path.isfile(model3_file):
                    # Find avatar file if it exists
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

            # Decode the WAV header and get actual audio data
            wav_header_size = 44  # Standard WAV header size
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
        await websocket.accept()

        # Handle authentication (fail closed; an empty API key never disables auth)
        expected_api_key = default_context_cache.system_config.api_key
        if not expected_api_key:
            logger.error("TTS WebSocket connection rejected: server API key is not configured")
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return

        try:
            # Prefer the API key supplied in the WebSocket URL.
            query_api_key = websocket.query_params.get("api_key")
            if query_api_key:
                supplied_api_key = query_api_key
            else:
                auth_msg = await asyncio.wait_for(websocket.receive_json(), timeout=5.0)
                supplied_api_key = auth_msg.get("api_key") if auth_msg.get("type") == "auth" else None

            if supplied_api_key != expected_api_key:
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

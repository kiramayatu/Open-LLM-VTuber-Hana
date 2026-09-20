"""Hana VITS TTS client for the standalone FastAPI Hana service."""

from __future__ import annotations

from urllib.parse import urlparse, urlunparse

import requests
from loguru import logger

from .tts_interface import TTSInterface


class HanaTTSTTS(TTSInterface):
    """Generate WAV audio through the modern Hana VITS FastAPI backend.

    The modern Hana service exposes POST /tts and POST /v1/tts and accepts
    JSON with text, speaker, language, and speed. The response is WAV audio.
    """

    def __init__(
        self,
        api_url: str = "http://127.0.0.1:7860/v1/tts",
        speaker: str = "Hana",
        language: str = "日本語",
        speed: float = 1.0,
        timeout: float = 120.0,
    ) -> None:
        self.api_url = self._normalize_api_url(api_url)
        self.speaker = speaker
        self.language = language
        self.speed = float(speed)
        self.timeout = float(timeout)

    @staticmethod
    def _normalize_api_url(api_url: str) -> str:
        """Accept either the full TTS endpoint or the Hana server base URL."""
        value = (api_url or "").strip()
        if not value:
            return "http://127.0.0.1:7860/v1/tts"

        parsed = urlparse(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(
                "Hana TTS api_url must be an absolute HTTP/HTTPS URL, "
                "for example http://127.0.0.1:7860/v1/tts"
            )

        path = parsed.path.rstrip("/")
        if path in {"", "/"}:
            path = "/v1/tts"
        elif path == "/v1":
            path = "/v1/tts"
        elif path != "/v1/tts" and path != "/tts":
            # Preserve a custom endpoint path instead of silently rewriting it.
            pass

        return urlunparse(
            (parsed.scheme, parsed.netloc, path, parsed.params, parsed.query, parsed.fragment)
        )

    def generate_audio(self, text: str, file_name_no_ext=None) -> str:
        """Generate a WAV file through Hana FastAPI and return its cache path."""
        cache_file = self.generate_cache_file_name(file_name_no_ext, file_extension="wav")
        payload = {
            "text": text,
            "speaker": self.speaker,
            "language": self.language,
            "speed": self.speed,
        }

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                headers={"Accept": "audio/wav", "Content-Type": "application/json"},
                timeout=self.timeout,
            )
            response.raise_for_status()

            content_type = response.headers.get("content-type", "").lower()
            audio_bytes = response.content
            if not audio_bytes:
                logger.error("Hana TTS returned an empty response")
                return ""

            # The service contract is a WAV response. Do not silently save an
            # HTML/JSON error page as a .wav if an intermediary returned 2xx.
            if "audio/wav" not in content_type and not audio_bytes.startswith(b"RIFF"):
                logger.error(
                    "Hana TTS returned unexpected content type: {}",
                    content_type or "missing",
                )
                return ""

            with open(cache_file, "wb") as audio_file:
                audio_file.write(audio_bytes)

            logger.info("Hana TTS generated audio: {}", cache_file)
            return cache_file
        except requests.Timeout:
            logger.error(
                "Hana TTS request timed out after {} seconds: {}",
                self.timeout,
                self.api_url,
            )
            return ""
        except requests.RequestException as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status is not None:
                logger.error("Hana TTS request failed with HTTP {}", status)
            else:
                logger.error("Hana TTS request failed: {}", exc)
            return ""
        except OSError as exc:
            logger.error("Failed to save Hana TTS audio: {}", exc)
            return ""

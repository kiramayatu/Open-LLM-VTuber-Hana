# translate/libretranslate.py
import requests

class LibreTranslate:
    def __init__(self, api_endpoint: str, target_lang: str, source_lang: str = "auto", api_key: str = None):
        self.api_endpoint = api_endpoint.rstrip('/') + '/translate'
        self.target_lang = target_lang
        self.source_lang = source_lang
        self.api_key = api_key

    def translate(self, text: str) -> str:
        payload = {
            "q": text,
            "source": self.source_lang,
            "target": self.target_lang,
            "format": "text"
        }
        if self.api_key:
            payload["api_key"] = self.api_key

        try:
            response = requests.post(self.api_endpoint, json=payload)
            response.raise_for_status()
            return response.json().get("translatedText", text)
        except Exception as e:
            print(f"LibreTranslate Error: {e}")
            return text

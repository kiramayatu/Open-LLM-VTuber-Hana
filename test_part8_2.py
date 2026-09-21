from pathlib import Path

from src.open_llm_vtuber.routes import _normalize_api_key
from src.open_llm_vtuber.artifacts.generators import generate_docx

assert _normalize_api_key("  'abc'  ") == "abc"
assert _normalize_api_key('  "abc"  ') == "abc"
assert _normalize_api_key("abc") == "abc"

data = generate_docx(title="Story", paragraphs=["hello"])
assert data[:2] == b"PK"
print("part8.2 smoke OK")

from pathlib import Path


def test_stage4_ui_bundle_contains_responsive_provider_ui():
    bundle = Path(__file__).parents[1] / "frontend" / "assets" / "main-llmprovider-stage4.js"
    text = bundle.read_text(encoding="utf-8")
    assert "Discard your unsaved provider changes" in text
    assert "LLM Provider API is unavailable" in text
    assert "Import YAML" in text
    assert "Saved providers" in text
    assert 'tt==="llm"?null:jsxRuntimeExports.jsxs(DrawerFooter$1' in text

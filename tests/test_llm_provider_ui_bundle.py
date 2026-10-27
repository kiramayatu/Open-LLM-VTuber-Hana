from pathlib import Path


def test_stage4_ui_bundle_contains_responsive_provider_ui():
    bundle = Path(__file__).parents[1] / "frontend" / "assets" / "main-llmprovider-stage4.js"
    text = bundle.read_text(encoding="utf-8")
    assert "Discard your unsaved provider changes" in text
    assert "provider action is not available on the running backend" in text
    assert "Import YAML" in text
    assert "import-file" in text
    assert "Discard your unsaved provider changes and import this YAML configuration?" in text
    assert "/llm/providers/models/" in text
    assert "model not selected" in text
    index = (bundle.parent.parent / "index.html").read_text(encoding="utf-8")
    assert "main-llmprovider-stage4.2.js" in index
    assert "Saved providers" in text
    assert 'tt==="llm"?null:jsxRuntimeExports.jsxs(DrawerFooter$1' in text

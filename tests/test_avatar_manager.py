"""Unit tests for multi-format avatar ingestion, auto-detection, and security (Parts 1–5)."""
from __future__ import annotations

import io
import tempfile
import unittest
import zipfile
from pathlib import Path

from open_llm_vtuber.avatar_manager import (
    detect_model_format_from_entry,
    ingest_avatar_package,
    load_model_dict,
    read_character_persona,
    sanitize_model_name,
    save_character_persona,
)
from open_llm_vtuber.live2d_model import Live2dModel


class TestAvatarManagerAndSecurity(unittest.TestCase):
    def test_sanitize_model_name_blocks_traversal_and_undefined(self):
        self.assertEqual(sanitize_model_name("Hana VRM 1.0"), "Hana_VRM_1.0")
        for invalid in ("", ".", "..", "undefined", "null"):
            with self.assertRaises(ValueError):
                sanitize_model_name(invalid)

    def test_zip_slip_and_executable_rejection(self):
        with tempfile.TemporaryDirectory() as td:
            models_root = Path(td) / "live2d-models"
            dict_path = Path(td) / "model_dict.json"
            dict_path.write_text("[]\n", encoding="utf-8")

            # 1. Zip-Slip path traversal
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as zf:
                zf.writestr("../../outside.txt", "escape")
            with self.assertRaises(ValueError):
                ingest_avatar_package(
                    "evil.zip",
                    buf.getvalue(),
                    models_root=models_root,
                    model_dict_path=dict_path,
                )

            # 2. Forbidden executable inside archive
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as zf:
                zf.writestr("payload.sh", "#!/bin/sh\necho pwn")
            with self.assertRaises(ValueError):
                ingest_avatar_package(
                    "script.zip",
                    buf.getvalue(),
                    models_root=models_root,
                    model_dict_path=dict_path,
                )

    def test_multi_format_ingestion_and_live2d_model_lookup(self):
        with tempfile.TemporaryDirectory() as td:
            models_root = Path(td) / "live2d-models"
            chars_root = Path(td) / "characters"
            dict_path = Path(td) / "model_dict.json"
            dict_path.write_text("[]\n", encoding="utf-8")

            # 1. 3D VRM
            e_vrm = ingest_avatar_package(
                "hana_3d.vrm",
                b"glTF-vrm-mock",
                character_name="Hana 3D",
                human_name="Senpai",
                persona_prompt="You are Hana 3D, a cheerful VTuber.",
                models_root=models_root,
                model_dict_path=dict_path,
                characters_dir=chars_root,
            )
            self.assertEqual(e_vrm["format"], "vrm")
            self.assertTrue(e_vrm["url"].endswith("hana_3d.vrm"))
            self.assertTrue((chars_root / "hana_3d.yaml").is_file())
            p_vrm = read_character_persona("hana_3d", characters_dir=chars_root)
            self.assertEqual(p_vrm["character_name"], "Hana 3D")
            self.assertEqual(p_vrm["human_name"], "Senpai")
            self.assertIn("cheerful VTuber", p_vrm["persona_prompt"])

            # 2. 3D GLB
            e_glb = ingest_avatar_package(
                "robot.glb",
                b"glTF-glb-mock",
                models_root=models_root,
                model_dict_path=dict_path,
                characters_dir=chars_root,
            )
            self.assertEqual(e_glb["format"], "gltf")

            # 3. Live2D Cubism 3/4/5
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as zf:
                zf.writestr("mao_custom/mao_custom.model3.json", "{}")
            e_c3 = ingest_avatar_package(
                "mao_custom.zip",
                buf.getvalue(),
                models_root=models_root,
                model_dict_path=dict_path,
                characters_dir=chars_root,
            )
            self.assertEqual(e_c3["format"], "cubism3")

            # 4. Legacy Live2D Cubism 2.1
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as zf:
                zf.writestr("shizuku/shizuku.model.json", "{}")
            e_c2 = ingest_avatar_package(
                "shizuku.zip",
                buf.getvalue(),
                models_root=models_root,
                model_dict_path=dict_path,
                characters_dir=chars_root,
            )
            self.assertEqual(e_c2["format"], "cubism2")

            # 5. 2D PNGTuber multi-state sprite package
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as zf:
                zf.writestr("closed.png", b"\x89PNG\r\n")
                zf.writestr("open.png", b"\x89PNG\r\n")
                zf.writestr("joy_open.png", b"\x89PNG\r\n")
            e_png = ingest_avatar_package(
                "chibi_png.zip",
                buf.getvalue(),
                models_root=models_root,
                model_dict_path=dict_path,
                characters_dir=chars_root,
            )
            self.assertEqual(e_png["format"], "pngtuber")
            self.assertEqual(detect_model_format_from_entry(e_png), "pngtuber")

            # 6. 3D Metasequoia (.mqo) archive with texture maps
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w") as zf:
                zf.writestr("hanamqo/hana.mqo", "Metasequoia Document\nFormat Text Ver 1.1\n")
                zf.writestr("hanamqo/maps/body.png", b"\x89PNG\r\n")
            e_mqo = ingest_avatar_package(
                "hanamqo.zip",
                buf.getvalue(),
                models_root=models_root,
                model_dict_path=dict_path,
                characters_dir=chars_root,
            )
            self.assertEqual(e_mqo["format"], "mqo")
            self.assertTrue(e_mqo["url"].endswith("hana.mqo"))
            self.assertEqual(detect_model_format_from_entry(e_mqo), "mqo")

            # Verify updating persona via save_character_persona preserves live2d_model_name
            updated_persona = save_character_persona(
                "hanamqo",
                character_name="Hana Fountainstand",
                human_name="Captain",
                persona_prompt="You are Hana, president of the Yosakoi club!",
                characters_dir=chars_root,
            )
            self.assertEqual(updated_persona["character_name"], "Hana Fountainstand")
            self.assertEqual(updated_persona["human_name"], "Captain")
            self.assertEqual(updated_persona["live2d_model_name"], "hanamqo")

            # Verify path traversal attempts in read_character_persona / save_character_persona are sanitized and stay inside chars_root
            traversal_persona = save_character_persona(
                "../../etc/passwd",
                character_name="SafeChar",
                human_name="SafeUser",
                persona_prompt="Safe prompt.",
                characters_dir=chars_root,
            )
            self.assertEqual(traversal_persona["conf_uid"], "etc_passwd_001")
            self.assertTrue((chars_root / "etc_passwd.yaml").is_file())
            loaded_traversal = read_character_persona(
                "../../etc/passwd", characters_dir=chars_root
            )
            self.assertEqual(loaded_traversal["character_name"], "SafeChar")
            self.assertEqual(loaded_traversal["human_name"], "SafeUser")

            # Verify Live2dModel loads each format cleanly and extracts emotions
            for model_name in ("hana_3d", "robot", "mao_custom", "shizuku", "chibi_png", "hanamqo"):
                m = Live2dModel(model_name, model_dict_path=str(dict_path))
                self.assertEqual(m.extract_emotion("Hello [joy] world!"), [3])

            entries = load_model_dict(dict_path)
            self.assertEqual(len(entries), 6)

    def test_frontend_avatar_runtime_and_uploader_bundled(self):
        root = Path(__file__).parents[1]
        index_html = (root / "frontend" / "index.html").read_text(encoding="utf-8")
        self.assertIn("./libs/avatar-runtime.js", index_html)

        runtime_js = (root / "frontend" / "libs" / "avatar-runtime.js").read_text(
            encoding="utf-8"
        )
        for symbol in (
            "AvatarRuntimeManager",
            "VRMAndGLTF3DController",
            "PNGTuberController",
            "Live2DCubism2Controller",
            "AudioLipSyncAnalyzer",
            "ProceduralLifeController",
            "__olvAvatarRuntime",
        ):
            self.assertIn(symbol, runtime_js)

        for bundle_name in ("main-llmprovider-stage4.2.js", "main-llmprovider-stage4.js"):
            bundle_js = (root / "frontend" / "assets" / bundle_name).read_text(
                encoding="utf-8"
            )
            self.assertIn("AvatarModelUploader", bundle_js)
            self.assertIn("/api/models/upload", bundle_js)
            self.assertIn("Edit Persona", bundle_js)
            self.assertIn("/api/characters/", bundle_js)
            self.assertIn("reload-character-persona", bundle_js)


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
import json
import struct
import sys
import tempfile
import unittest
import zlib
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.export_aseprite import (
    export_texturepacker_json,
    main,
    write_aseprite_file,
)


class ExportAsepriteTests(unittest.TestCase):
    def test_write_aseprite_binary_structure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_file = Path(temp_dir) / "test.aseprite"

            # Create two 16x16 frames
            f1 = Image.new("RGBA", (16, 16), (255, 0, 0, 255))
            f2 = Image.new("RGBA", (16, 16), (0, 255, 0, 255))

            write_aseprite_file(
                frames=[f1, f2],
                out_path=out_file,
                frame_duration_ms=150,
                tags=[("walk", 0, 1)],
            )

            data = out_file.read_bytes()
            self.assertGreater(len(data), 128)

            # Unpack 128-byte Header
            filesize, magic, frames_count, w, h, depth, flags, speed = struct.unpack(
                "<IHHHHHIH", data[:20]
            )
            self.assertEqual(filesize, len(data))
            self.assertEqual(magic, 0xA5E0)
            self.assertEqual(frames_count, 2)
            self.assertEqual(w, 16)
            self.assertEqual(h, 16)
            self.assertEqual(depth, 32)
            self.assertEqual(speed, 150)

            # Unpack Frame 1 Header at offset 128
            frame1_size, frame1_magic, old_chunks, duration, _, new_chunks = struct.unpack(
                "<IHHHHI", data[128:144]
            )
            self.assertEqual(frame1_magic, 0xF1FA)
            self.assertEqual(duration, 150)
            self.assertEqual(new_chunks, 3)  # Layer + Tag + Cel

    def test_export_texturepacker_json(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            out_json = Path(temp_dir) / "sheet.json"
            f1 = Image.new("RGBA", (16, 16), (255, 0, 0, 255))
            f2 = Image.new("RGBA", (16, 16), (0, 255, 0, 255))

            export_texturepacker_json(
                frames=[f1, f2],
                frame_names=["frame_0.png", "frame_1.png"],
                sheet_image_name="sheet.png",
                sheet_width=32,
                sheet_height=16,
                frame_coords=[(0, 0, 16, 16), (16, 0, 16, 16)],
                out_json_path=out_json,
                duration_ms=120,
            )

            self.assertTrue(out_json.is_file())
            content = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertIn("frames", content)
            self.assertIn("meta", content)
            self.assertEqual(content["meta"]["image"], "sheet.png")
            self.assertEqual(content["frames"]["frame_0.png"]["duration"], 120)

    def test_cli_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            frames_dir = root / "frames"
            frames_dir.mkdir()
            Image.new("RGBA", (8, 8), (255, 0, 0, 255)).save(frames_dir / "walk-0.png")
            Image.new("RGBA", (8, 8), (0, 255, 0, 255)).save(frames_dir / "walk-1.png")

            out_file = root / "anim.aseprite"
            rc = main([
                "--frames-dir", str(frames_dir),
                "--prefix", "walk-",
                "--out", str(out_file),
                "--duration", "100",
                "--tag", "walk",
                "--force",
            ])
            self.assertEqual(rc, 0)
            self.assertTrue(out_file.is_file())


if __name__ == "__main__":
    unittest.main()

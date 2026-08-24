#!/usr/bin/env python3
import base64
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "skills" / "generate2dsprite" / "scripts"))
import save_imagegen_result as MODULE


class SaveImagegenResultTests(unittest.TestCase):
    def setUp(self) -> None:
        img = Image.new("RGBA", (8, 8), (255, 0, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        self.png_bytes = buf.getvalue()
        self.b64_png = base64.b64encode(self.png_bytes).decode("ascii")

    def test_sniff_extension(self) -> None:
        self.assertEqual(MODULE.sniff_extension(self.png_bytes), "png")
        with self.assertRaises(ValueError):
            MODULE.sniff_extension(b"not-an-image-data-stream")

    def test_decode_base64(self) -> None:
        decoded = MODULE.decode_base64(self.b64_png)
        self.assertEqual(decoded, self.png_bytes)
        self.assertIsNone(MODULE.decode_base64("too-short"))

    def test_extract_from_jsonl(self) -> None:
        line = json.dumps({
            "payload": {
                "type": "image_generation_call",
                "result": self.b64_png,
            }
        })
        data, ext = MODULE.extract_image(line)
        self.assertEqual(data, self.png_bytes)
        self.assertEqual(ext, "png")

    def test_extract_from_data_url(self) -> None:
        data_url = f"data:image/png;base64,{self.b64_png}"
        data, ext = MODULE.extract_image(data_url)
        self.assertEqual(data, self.png_bytes)
        self.assertEqual(ext, "png")

    def test_cli_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            in_file = root / "input.txt"
            in_file.write_text(self.b64_png, encoding="utf-8")
            out_dir = root / "output"

            parser = MODULE.build_parser()
            args = parser.parse_args([
                "--input", str(in_file),
                "--output-dir", str(out_dir),
                "--filename", "test.png",
            ])
            text = in_file.read_text(encoding="utf-8")
            data, ext = MODULE.extract_image(text)
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "test.png"
            out_path.write_bytes(data)

            self.assertTrue(out_path.is_file())
            self.assertEqual(out_path.read_bytes(), self.png_bytes)


if __name__ == "__main__":
    unittest.main()

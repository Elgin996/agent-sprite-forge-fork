#!/usr/bin/env python3
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.pixel_art import (
    BUILTIN_PALETTES,
    main,
    parse_hex_color,
    quantize_to_palette,
    reduce_mixels,
    resolve_palette,
)


class PixelArtTests(unittest.TestCase):
    def test_parse_and_resolve_palette(self) -> None:
        self.assertEqual(parse_hex_color("#123456"), (0x12, 0x34, 0x56))
        self.assertEqual(parse_hex_color("abc"), (0xAA, 0xBB, 0xCC))

        pico = resolve_palette("pico-8")
        self.assertEqual(len(pico), 16)
        self.assertEqual(pico[0], (0, 0, 0))

        endesga = resolve_palette("endesga-32")
        self.assertEqual(len(endesga), 32)

        custom = resolve_palette("#ff0000, #00ff00, #0000ff")
        self.assertEqual(custom, [(255, 0, 0), (0, 255, 0), (0, 0, 255)])

    def test_quantize_to_palette(self) -> None:
        # Create an image with slightly off-red and transparent background
        img = Image.new("RGBA", (4, 4), (0, 0, 0, 0))
        img.putpixel((1, 1), (250, 5, 10, 255))  # Near pure red
        img.putpixel((2, 2), (10, 245, 15, 255)) # Near pure green

        palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        quantized = quantize_to_palette(img, palette)

        res = np.array(quantized)
        # Transparent pixels should remain transparent with alpha=0
        self.assertEqual(res[0, 0, 3], 0)
        # (1, 1) should be mapped exactly to (255, 0, 0)
        self.assertEqual(tuple(res[1, 1, :3]), (255, 0, 0))
        self.assertEqual(res[1, 1, 3], 255)
        # (2, 2) should be mapped exactly to (0, 255, 0)
        self.assertEqual(tuple(res[2, 2, :3]), (0, 255, 0))
        self.assertEqual(res[2, 2, 3], 255)

    def test_reduce_mixels(self) -> None:
        # 64x64 image downsampled to 16x16 and upscaled
        img = Image.new("RGBA", (64, 64), (100, 150, 200, 255))
        downscaled = reduce_mixels(img, target_grid=16, upscale_back=False)
        self.assertEqual(downscaled.size, (16, 16))

        restored = reduce_mixels(img, target_grid=16, upscale_back=True)
        self.assertEqual(restored.size, (64, 64))

    def test_cli_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            in_file = root / "in.png"
            out_file = root / "out.png"

            img = Image.new("RGBA", (32, 32), (200, 20, 30, 255))
            img.save(in_file)

            rc = main([
                "--input", str(in_file),
                "--out", str(out_file),
                "--palette", "pico-8",
                "--pixel-grid", "16",
                "--force",
            ])
            self.assertEqual(rc, 0)
            self.assertTrue(out_file.is_file())


if __name__ == "__main__":
    unittest.main()

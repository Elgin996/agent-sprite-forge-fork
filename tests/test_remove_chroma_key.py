#!/usr/bin/env python3
import io
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.remove_chroma_key import (
    color_distance,
    main,
    parse_hex_color,
    remove_chroma_key,
)


class RemoveChromaKeyTests(unittest.TestCase):
    def test_parse_hex_color(self) -> None:
        self.assertEqual(parse_hex_color("#ff00ff"), (255, 0, 255))
        self.assertEqual(parse_hex_color("FF00FF"), (255, 0, 255))
        self.assertEqual(parse_hex_color("#f0f"), (255, 0, 255))
        self.assertEqual(parse_hex_color("#00ff00"), (0, 255, 0))
        with self.assertRaises(ValueError):
            parse_hex_color("invalid")

    def test_remove_chroma_key_hard(self) -> None:
        # Create 10x10 image, left half magenta, right half white
        arr = np.zeros((10, 10, 3), dtype=np.uint8)
        arr[:, :5] = [255, 0, 255]
        arr[:, 5:] = [255, 255, 255]
        img = Image.fromarray(arr, mode="RGB")

        cleaned = remove_chroma_key(img, key_color=(255, 0, 255), transparent_threshold=35.0)
        res = np.array(cleaned)

        # Left half should be transparent
        self.assertTrue(np.all(res[:, :5, 3] == 0))
        # Right half should be fully opaque
        self.assertTrue(np.all(res[:, 5:, 3] == 255))

    def test_soft_matte_and_despill(self) -> None:
        # Create gradient from pure magenta to white
        arr = np.zeros((1, 4, 3), dtype=np.uint8)
        arr[0, 0] = [255, 0, 255]       # dist = 0 -> alpha = 0
        arr[0, 1] = [255, 50, 255]      # dist = 50 -> semi-transparent
        arr[0, 2] = [255, 150, 255]     # dist = 150 -> higher alpha
        arr[0, 3] = [255, 255, 255]     # dist = 255 -> opaque
        img = Image.fromarray(arr, mode="RGB")

        cleaned = remove_chroma_key(
            img,
            key_color=(255, 0, 255),
            soft_matte=True,
            transparent_threshold=35.0,
            opaque_threshold=160.0,
            despill=True,
        )
        res = np.array(cleaned)
        alphas = res[0, :, 3]
        self.assertEqual(alphas[0], 0)
        self.assertGreater(alphas[1], 0)
        self.assertGreater(alphas[2], alphas[1])
        self.assertEqual(alphas[3], 255)

    def test_edge_contract(self) -> None:
        # 10x10 square in center of 20x20
        arr = np.zeros((20, 20, 3), dtype=np.uint8)
        arr[:, :] = [255, 0, 255]
        arr[5:15, 5:15] = [255, 255, 255]
        img = Image.fromarray(arr, mode="RGB")

        no_contract = remove_chroma_key(img, edge_contract=0)
        with_contract = remove_chroma_key(img, edge_contract=1)

        area_no_contract = np.sum(np.array(no_contract)[:, :, 3] > 0)
        area_with_contract = np.sum(np.array(with_contract)[:, :, 3] > 0)
        self.assertEqual(area_no_contract, 100)
        self.assertLess(area_with_contract, area_no_contract)

    def test_cli_execution(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            in_file = root / "input.png"
            out_file = root / "output.png"

            img = Image.new("RGB", (16, 16), (255, 0, 255))
            img.save(in_file)

            rc = main([
                "--input", str(in_file),
                "--out", str(out_file),
                "--key-color", "#ff00ff",
                "--soft-matte",
                "--despill",
                "--edge-contract", "1",
                "--force",
            ])
            self.assertEqual(rc, 0)
            self.assertTrue(out_file.exists())


if __name__ == "__main__":
    unittest.main()

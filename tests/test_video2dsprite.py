from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "skills"
    / "video2dsprite"
    / "scripts"
    / "video2dsprite.py"
)
SPEC = importlib.util.spec_from_file_location("video2dsprite", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class Video2DSpriteTests(unittest.TestCase):
    @staticmethod
    def make_clean_frame(size: int = 8) -> Image.Image:
        image = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        for y in range(1, size - 1):
            for x in range(1, size - 1):
                image.putpixel((x, y), (40, 80, 120, 255))
        image.putpixel((2, 2), (40, 80, 120, 128))
        return image

    @staticmethod
    def write_clean_frames(directory: Path, count: int) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            image = Video2DSpriteTests.make_clean_frame()
            image.save(directory / f"clean_{index:04d}.png")

    @staticmethod
    def capture_cp1252_stdout(callback) -> str:
        buffer = io.BytesIO()
        stream = io.TextIOWrapper(buffer, encoding="cp1252")
        with contextlib.redirect_stdout(stream):
            callback()
        stream.flush()
        return buffer.getvalue().decode("cp1252")

    def test_normalize_sprite_preserves_semitransparent_alpha(self) -> None:
        source = self.make_clean_frame(size=6)
        normalized = MODULE.normalize_sprite(
            source,
            cell=8,
            body_height=4,
            foot_y=6,
            anchor="center",
        )
        self.assertEqual(normalized.getpixel((3, 3)), (40, 80, 120, 128))

    def test_build_exports_preserves_pixels_in_strip_and_grid(self) -> None:
        sprites = [self.make_clean_frame(size=4), self.make_clean_frame(size=4)]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            info = MODULE.build_exports(sprites, root / "sprite", tag="x2", n_frames=2)

            for index, sprite_path in enumerate(info["sprites"]):
                with Image.open(sprite_path) as individual:
                    individual_pixels = individual.tobytes()
                with Image.open(info["strip"]) as strip:
                    strip_pixels = strip.crop((index * 4, 0, (index + 1) * 4, 4)).tobytes()
                with Image.open(info["grid"]) as grid:
                    grid_pixels = grid.crop((index * 4, 0, (index + 1) * 4, 4)).tobytes()
                self.assertEqual(individual_pixels, strip_pixels)
                self.assertEqual(individual_pixels, grid_pixels)

    def test_clean_frames_removes_stale_generated_files_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_dir = root / "raw"
            clean_dir = root / "clean"
            raw_dir.mkdir()
            clean_dir.mkdir()
            for index in range(10):
                Image.new("RGBA", (4, 4), (255, 0, 255, 255)).save(
                    raw_dir / f"frame_{index:04d}.png"
                )
                Image.new("RGBA", (4, 4), (1, 2, 3, 255)).save(
                    clean_dir / f"clean_{index:04d}.png"
                )
            (clean_dir / "keep.txt").write_text("keep", encoding="utf-8")

            outputs = MODULE.clean_frames(raw_dir, clean_dir)

            self.assertEqual(len(outputs), 10)
            self.assertTrue((clean_dir / "clean_0009.png").exists())
            self.assertTrue((clean_dir / "keep.txt").exists())

            for index in range(3):
                Image.new("RGBA", (4, 4), (255, 0, 255, 255)).save(
                    raw_dir / f"frame_{index:04d}.png"
                )
            for old_raw in raw_dir.glob("frame_*.png"):
                if int(old_raw.stem.split("_")[1]) >= 3:
                    old_raw.unlink()
            outputs = MODULE.clean_frames(raw_dir, clean_dir)

            self.assertEqual(len(outputs), 3)
            self.assertEqual(
                sorted(path.name for path in clean_dir.glob("clean_*.png")),
                ["clean_0000.png", "clean_0001.png", "clean_0002.png"],
            )
            self.assertTrue((clean_dir / "keep.txt").exists())

    def test_missing_raw_frames_fail_before_clean_output_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_dir = root / "raw"
            clean_dir = root / "clean"
            raw_dir.mkdir()
            clean_dir.mkdir()
            sentinel = clean_dir / "clean_0000.png"
            sentinel.write_bytes(b"old")
            (clean_dir / "keep.txt").write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "no raw frames"):
                MODULE.clean_frames(raw_dir, clean_dir)

            self.assertTrue(sentinel.exists())
            self.assertTrue((clean_dir / "keep.txt").exists())

    def test_sample_and_export_cleans_stale_sprite_frames_and_truthful_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clean_dir = root / "clean"
            out_dir = root / "out"
            self.write_clean_frames(clean_dir, 8)

            first = MODULE.sample_and_export(
                clean_dir, out_dir, [8], cell=8, body_height=4, foot_y=6
            )
            self.assertEqual(first["sets"][0]["count"], 8)
            (out_dir / "sprite" / "keep.txt").write_text("keep", encoding="utf-8")
            (out_dir / "sprite" / "x8" / "keep.txt").write_text("keep", encoding="utf-8")
            (out_dir / "sprite" / "x4").mkdir()
            for index in range(1, 9):
                (out_dir / "sprite" / "x4" / f"sprite_{index:02d}.png").write_bytes(b"old")

            second = MODULE.sample_and_export(
                clean_dir, out_dir, [4], cell=8, body_height=4, foot_y=6
            )
            info = second["sets"][0]
            self.assertEqual(info["count"], 4)
            self.assertEqual(len(info["sprites"]), 4)
            self.assertEqual(Path(info["sprites"][-1]).name, "sprite_04.png")
            self.assertFalse((out_dir / "sprite" / "x4" / "sprite_05.png").exists())
            self.assertTrue((out_dir / "sprite" / "sprite_08.png").exists())
            self.assertTrue((out_dir / "sprite" / "x8" / "sprite_08.png").exists())
            self.assertTrue((out_dir / "sprite" / "keep.txt").exists())
            self.assertTrue((out_dir / "sprite" / "x8" / "keep.txt").exists())
            with Image.open(info["strip"]) as strip:
                self.assertEqual(strip.size, (32, 8))

    def test_insufficient_source_frames_fail_before_export_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clean_dir = root / "clean"
            out_dir = root / "out"
            self.write_clean_frames(clean_dir, 3)
            sprite_dir = out_dir / "sprite"
            sprite_dir.mkdir(parents=True)
            sentinel = sprite_dir / "sprite_01.png"
            sentinel.write_bytes(b"old")
            (sprite_dir / "keep.txt").write_text("keep", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "requested 8 frames.*3 clean frames"):
                MODULE.sample_and_export(clean_dir, out_dir, [8])

            self.assertTrue(sentinel.exists())
            self.assertTrue((sprite_dir / "keep.txt").exists())

    def test_sample_indices_rejects_impossible_request(self) -> None:
        with self.assertRaisesRegex(ValueError, "requested 8 frames.*3 clean frames"):
            MODULE.sample_indices(3, 8)

    def test_empty_frame_counts_fail_before_export_cleanup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clean_dir = root / "clean"
            out_dir = root / "out"
            self.write_clean_frames(clean_dir, 1)
            sprite_dir = out_dir / "sprite"
            sprite_dir.mkdir(parents=True)
            sentinel = sprite_dir / "sprite_01.png"
            sentinel.write_bytes(b"old")

            with self.assertRaisesRegex(ValueError, "at least one frame count"):
                MODULE.sample_and_export(clean_dir, out_dir, [])

            self.assertTrue(sentinel.exists())

    def test_cli_help_and_progress_are_cp1252_safe(self) -> None:
        buffer = io.BytesIO()
        stream = io.TextIOWrapper(buffer, encoding="cp1252")
        with contextlib.redirect_stdout(stream):
            with self.assertRaises(SystemExit) as raised:
                MODULE.main(["--help"])
        stream.flush()
        output = buffer.getvalue().decode("cp1252")
        self.assertEqual(raised.exception.code, 0)
        self.assertIn("Video -> dense", output)
        self.assertNotIn("→", output)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            Image.new("RGBA", (4, 4), (255, 0, 255, 255)).save(raw_dir / "frame_0000.png")
            clean_output = root / "clean"
            clean_output_text = self.capture_cp1252_stdout(
                lambda: MODULE.cmd_clean(
                    argparse.Namespace(raw_dir=str(raw_dir), out_dir=str(clean_output), dist=55.0)
                )
            )
            self.assertIn("->", clean_output_text)

            sample_output = root / "sample"
            sample_text = self.capture_cp1252_stdout(
                lambda: MODULE.cmd_sample(
                    argparse.Namespace(
                        clean_dir=str(clean_output),
                        out_dir=str(sample_output),
                        frame_counts="1",
                        cell_size=8,
                        body_height=4,
                        foot_y=6,
                        anchor="feet",
                    )
                )
            )
            self.assertIn("->", sample_text)

    def test_exception_text_is_replaced_for_incompatible_console_encoding(self) -> None:
        buffer = io.BytesIO()
        stream = io.TextIOWrapper(buffer, encoding="cp1252")
        MODULE.safe_print("filename: \u2603.png", file=stream)
        stream.flush()
        self.assertIn("?", buffer.getvalue().decode("cp1252"))


if __name__ == "__main__":
    unittest.main()

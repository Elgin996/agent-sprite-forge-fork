from __future__ import annotations

import argparse
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "skills"
    / "generate2dmap"
    / "scripts"
    / "extract_prop_pack.py"
)
SPEC = importlib.util.spec_from_file_location("extract_prop_pack", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class PropLabelValidationTests(unittest.TestCase):
    @staticmethod
    def labels_args(
        labels: str | None = None, labels_file: Path | None = None
    ) -> argparse.Namespace:
        return argparse.Namespace(labels=labels, labels_file=labels_file)

    def test_rejects_exact_duplicate_labels(self) -> None:
        with self.assertRaisesRegex(ValueError, r"Duplicate prop labels.*rock.*cells \[0, 1\]"):
            MODULE.parse_labels(self.labels_args("rock,rock"), expected_count=2)

    def test_rejects_labels_that_collide_after_sanitization(self) -> None:
        with self.assertRaisesRegex(ValueError, r"Duplicate prop labels.*rock.*cells \[0, 1\]"):
            MODULE.parse_labels(self.labels_args("rock!,rock?"), expected_count=2)

    def test_repeated_skip_markers_are_allowed(self) -> None:
        labels = MODULE.parse_labels(self.labels_args("skip,empty,-"), expected_count=3)
        self.assertEqual(labels, ["", "", ""])

    def test_rejects_labels_and_labels_file_together(self) -> None:
        with self.assertRaisesRegex(ValueError, "either --labels or --labels-file"):
            MODULE.parse_labels(
                self.labels_args("rock", Path("labels.txt")), expected_count=1
            )

    def test_collision_fails_before_output_directory_or_manifest_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_dir = root / "props"
            original_argv = sys.argv
            sys.argv = [
                "extract_prop_pack.py",
                "--input", str(root / "missing.png"),
                "--rows", "1",
                "--cols", "2",
                "--output-dir", str(output_dir),
                "--labels", "rock!,rock?",
            ]
            try:
                with self.assertRaisesRegex(ValueError, "Duplicate prop labels"):
                    MODULE.main()
            finally:
                sys.argv = original_argv

            self.assertFalse(output_dir.exists())
            self.assertFalse((output_dir / "prop-pack.json").exists())

    def test_unique_labels_keep_the_existing_slug_layout(self) -> None:
        labels = MODULE.parse_labels(self.labels_args("Rock!,tree"), expected_count=2)
        self.assertEqual(labels, ["rock", "tree"])


if __name__ == "__main__":
    unittest.main()

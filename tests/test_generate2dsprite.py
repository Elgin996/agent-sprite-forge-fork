from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "skills"
    / "generate2dsprite"
    / "scripts"
    / "generate2dsprite.py"
)
SPEC = importlib.util.spec_from_file_location("generate2dsprite", SCRIPT_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

ANCHOR_SCRIPT_PATH = SCRIPT_PATH.with_name("make_anchor_layout.py")
ANCHOR_SPEC = importlib.util.spec_from_file_location("make_anchor_layout", ANCHOR_SCRIPT_PATH)
assert ANCHOR_SPEC and ANCHOR_SPEC.loader
ANCHOR_MODULE = importlib.util.module_from_spec(ANCHOR_SPEC)
sys.modules[ANCHOR_SPEC.name] = ANCHOR_MODULE
ANCHOR_SPEC.loader.exec_module(ANCHOR_MODULE)


MAGENTA = (255, 0, 255, 255)
SUBJECT = (20, 40, 60, 255)


class SplitGridTests(unittest.TestCase):
    def test_rgba_alpha_is_preserved_in_fit_preserve_and_sheet(self) -> None:
        source = Image.new("RGBA", (4, 4), (20, 40, 60, 255))
        source.putpixel((1, 1), (20, 40, 60, 128))
        source.putpixel((2, 2), (0, 0, 0, 0))

        for strategy in ("fit", "preserve"):
            frames, _info, _report = MODULE.split_grid(
                source,
                rows=1,
                cols=1,
                cell_size=4,
                threshold=100,
                edge_threshold=150,
                fit_scale=1.0,
                trim_border_px=0,
                edge_clean_depth=0,
                scale_strategy=strategy,
            )
            self.assertEqual(frames[0].tobytes(), source.tobytes())

            sheet = MODULE.compose_sheet(frames, rows=1, cols=1, cell_size=4)
            self.assertEqual(sheet.tobytes(), source.tobytes())

    def test_edge_touch_uses_trimmed_frame_dimensions(self) -> None:
        image = Image.new("RGBA", (20, 20), MAGENTA)
        image.paste(SUBJECT, (2, 5, 18, 15))

        _frames, info, _stabilize = MODULE.split_grid(
            image,
            rows=1,
            cols=1,
            cell_size=20,
            threshold=100,
            edge_threshold=150,
            trim_border_px=2,
            edge_clean_depth=0,
            component_mode="largest",
        )

        self.assertEqual(info[0]["source_frame_size"], [16, 16])
        self.assertTrue(info[0]["source_edge_touch"])
        self.assertTrue(info[0]["edge_touch"])

    def test_empty_frame_is_reported_as_empty(self) -> None:
        image = Image.new("RGBA", (20, 20), MAGENTA)

        frames, info, _stabilize = MODULE.split_grid(
            image,
            rows=1,
            cols=1,
            cell_size=20,
            threshold=100,
            edge_threshold=150,
            trim_border_px=0,
            edge_clean_depth=0,
            scale_strategy="preserve",
        )

        self.assertTrue(info[0]["is_empty"])
        self.assertEqual(info[0]["output_size"], [0, 0])
        self.assertIsNone(frames[0].getbbox())

    def test_preserve_aligns_equal_size_subjects_to_same_anchor(self) -> None:
        image = Image.new("RGBA", (40, 20), MAGENTA)
        image.paste(SUBJECT, (3, 3, 9, 15))
        image.paste(SUBJECT, (28, 6, 34, 18))

        frames, info, _stabilize = MODULE.split_grid(
            image,
            rows=1,
            cols=2,
            cell_size=40,
            threshold=100,
            edge_threshold=150,
            fit_scale=0.7,
            trim_border_px=0,
            edge_clean_depth=0,
            align="feet",
            component_mode="largest",
            scale_strategy="preserve",
        )

        bboxes = [frame.getbbox() for frame in frames]
        self.assertEqual(bboxes[0], bboxes[1])
        self.assertEqual(info[0]["anchor_target"], info[1]["anchor_target"])
        self.assertFalse(info[0]["paste_clamped"])
        self.assertFalse(info[1]["paste_clamped"])

    def test_qc_summary_reports_model_scale_and_anchor_drift(self) -> None:
        image = Image.new("RGBA", (48, 24), MAGENTA)
        image.paste(SUBJECT, (8, 8, 12, 16))
        image.paste(SUBJECT, (32, 4, 40, 20))

        _frames, info, _stabilize = MODULE.split_grid(
            image,
            rows=1,
            cols=2,
            cell_size=48,
            threshold=100,
            edge_threshold=150,
            fit_scale=0.7,
            trim_border_px=0,
            edge_clean_depth=0,
            align="feet",
            component_mode="largest",
            scale_strategy="preserve",
        )

        summary = MODULE.summarize_frame_qc(info)
        self.assertEqual(summary["frame_count"], 2)
        self.assertEqual(summary["valid_frame_count"], 2)
        self.assertGreater(summary["body_scale_mean"], 0)
        self.assertGreater(summary["output_subject_height_mean"], 0)
        self.assertGreater(summary["body_scale_cv"], 0.2)
        self.assertGreater(summary["anchor_y_std"], 0.05)


class AxisStabilizationTests(unittest.TestCase):
    def make_frame(self, size: int, body_x: int, arm_length: int) -> Image.Image:
        """A dense body block plus a thin horizontal arm/weapon extension."""
        frame = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        body = Image.new("RGBA", (10, 30), SUBJECT)
        frame.paste(body, (body_x, size - 34))
        if arm_length:
            arm = Image.new("RGBA", (arm_length, 3), SUBJECT)
            frame.paste(arm, (body_x + 10, size - 30))
        return frame

    def test_body_axis_ignores_thin_extension(self) -> None:
        short = MODULE.body_axis_x(self.make_frame(64, 20, 2))
        long = MODULE.body_axis_x(self.make_frame(64, 20, 24))

        self.assertIsNotNone(short)
        self.assertIsNotNone(long)
        # A growing weapon must not drag the body axis the way a bbox center would.
        self.assertLess(abs(long - short), 1.5)

    def test_stabilize_removes_horizontal_slide(self) -> None:
        frames = [
            self.make_frame(64, 20, 2),
            self.make_frame(64, 28, 20),
            self.make_frame(64, 14, 12),
        ]
        info: list[dict[str, object]] = [{} for _ in frames]

        report = MODULE.stabilize_axis(
            frames, info, "core-register", (0.0, 1.0), 0.30, 4, 1.0, 32
        )

        self.assertTrue(report["applied"])
        self.assertLess(report["post_axis_x_std_px"], 1.0)
        self.assertLess(report["post_axis_x_std_px"], report["pre_axis_x_std_px"])
        self.assertEqual(report["shift_clamped_count"], 0)
        self.assertEqual(report["edge_clamped_count"], 0)
        self.assertEqual([i["stabilize_shift_x"] for i in info], report["shifts"])

    def test_stabilize_lands_axis_on_cell_center_by_default(self) -> None:
        # Every frame sits left of centre, so a mean-based target would leave the shared
        # axis off-centre and break output_origin.
        frames = [self.make_frame(64, 8, 2), self.make_frame(64, 12, 6), self.make_frame(64, 10, 3)]
        info: list[dict[str, object]] = [{} for _ in frames]

        report = MODULE.stabilize_axis(
            frames, info, "core-register", (0.0, 1.0), 0.30, 4, 1.0, 64
        )

        self.assertEqual(report["target_axis_x"], 32.0)
        landed = [MODULE.body_axis_x(frame) for frame in frames]
        for axis in landed:
            self.assertLess(abs(axis - 32.0), 2.0)

    def test_stabilize_mean_target_keeps_sheet_average_axis(self) -> None:
        frames = [self.make_frame(64, 8, 2), self.make_frame(64, 12, 6), self.make_frame(64, 10, 3)]
        info: list[dict[str, object]] = [{} for _ in frames]

        report = MODULE.stabilize_axis(
            frames, info, "core", (0.0, 1.0), 0.30, 0, 1.0, 64, 0, "mean"
        )

        self.assertLess(report["target_axis_x"], 32.0)

    def test_stabilize_respects_max_shift(self) -> None:
        frames = [self.make_frame(64, 6, 2), self.make_frame(64, 40, 2)]
        info: list[dict[str, object]] = [{} for _ in frames]

        report = MODULE.stabilize_axis(frames, info, "core", (0.0, 1.0), 0.30, 0, 1.0, 3)

        self.assertTrue(all(abs(dx) <= 3 for dx in report["shifts"]))
        self.assertGreater(report["shift_clamped_count"], 0)

    def test_stabilize_never_pushes_subject_into_cell_edge(self) -> None:
        frames = [self.make_frame(64, 2, 2), self.make_frame(64, 40, 2)]
        info: list[dict[str, object]] = [{} for _ in frames]

        MODULE.stabilize_axis(frames, info, "core", (0.0, 1.0), 0.30, 0, 1.0, 64, 0)

        for frame in frames:
            bbox = frame.getbbox()
            self.assertIsNotNone(bbox)
            self.assertFalse(MODULE.bbox_touches_edge(bbox, frame.width, frame.height, 0))

    def test_split_grid_reports_axis_spread_without_stabilizing(self) -> None:
        image = Image.new("RGBA", (80, 40), MAGENTA)
        image.paste(SUBJECT, (5, 10, 15, 34))
        image.paste(SUBJECT, (58, 10, 68, 34))

        frames, info, report = MODULE.split_grid(
            image,
            rows=1,
            cols=2,
            cell_size=40,
            threshold=100,
            edge_threshold=150,
            trim_border_px=0,
            edge_clean_depth=0,
            align="feet",
            component_mode="largest",
            scale_strategy="preserve",
        )

        self.assertEqual(report["mode"], "none")
        self.assertFalse(report["applied"])
        self.assertIn("axis_x_std_px", report)
        self.assertEqual([i["stabilize_shift_x"] for i in info], [0, 0])
        self.assertEqual(len(frames), 2)

    def test_qc_summary_normalizes_axis_spread_by_cell_size(self) -> None:
        report = {
            "mode": "core-register",
            "pre_axis_x_std_px": 8.0,
            "post_axis_x_std_px": 2.0,
            "post_axis_x_range_px": 5.0,
            "max_applied_shift": 13,
            "shift_clamped_count": 0,
            "edge_clamped_count": 1,
        }

        summary = MODULE.summarize_frame_qc([], report, 200)

        self.assertAlmostEqual(summary["axis_x_std"], 0.01)
        self.assertAlmostEqual(summary["pre_stabilize_axis_x_std"], 0.04)
        self.assertEqual(summary["stabilize_max_shift_px"], 13)
        self.assertEqual(summary["stabilize_clamped_count"], 1)


class ScaleProfileTests(unittest.TestCase):
    def make_metadata(self) -> dict[str, object]:
        return {
            "target": "player",
            "mode": "run",
            "rows": 2,
            "cols": 3,
            "cell_size": 128,
            "fit_scale": 0.8,
            "trim_border": 4,
            "edge_clean_depth": 3,
            "align": "feet",
            "shared_scale": False,
            "scale_strategy": "preserve",
            "component_mode": "largest",
            "component_padding": 8,
            "min_component_area": 1,
            "edge_touch_margin": 0,
            "qc_summary": {
                "body_scale_mean": 0.2,
                "body_scale_cv": 0.03,
                "anchor_y_mean": 0.82,
            },
        }

    def test_profile_round_trip_and_processing_contract(self) -> None:
        profile = MODULE.build_scale_profile(self.make_metadata(), "ronin", 0.08)
        self.assertEqual(profile["output_origin"], [64.0, 116])
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "profile.json"
            path.write_text(json.dumps(profile), encoding="utf-8")
            loaded = MODULE.load_scale_profile(path)

        args = argparse.Namespace(
            cell_size=96,
            fit_scale=0.95,
            trim_border=0,
            edge_clean_depth=0,
            align="center",
            shared_scale=True,
            scale_strategy="fit",
            component_mode="all",
            component_padding=0,
            min_component_area=10,
            edge_touch_margin=2,
        )
        MODULE.apply_scale_profile(args, loaded)

        self.assertEqual(args.cell_size, 128)
        self.assertEqual(args.fit_scale, 0.8)
        self.assertEqual(args.align, "feet")
        self.assertEqual(args.scale_strategy, "preserve")
        self.assertEqual(args.component_mode, "largest")
        self.assertEqual(args.component_padding, 8)

    def test_profile_scale_drift_compares_generation_scale(self) -> None:
        profile = MODULE.build_scale_profile(self.make_metadata(), "ronin", 0.08)
        drift = MODULE.profile_scale_drift({"body_scale_mean": 0.22}, profile)
        self.assertAlmostEqual(drift, 0.10)

    def test_godot_sprite3d_contract_uses_feet_origin_and_subject_height(self) -> None:
        metadata = self.make_metadata()
        metadata["duration"] = 125
        metadata["frame_labels"] = ["idle-1", "idle-2"]
        metadata["output_origin"] = [64.0, 116.0]
        metadata["qc_summary"]["output_subject_height_mean"] = 100.0

        contract = MODULE.build_godot_sprite3d_metadata(metadata, world_height=0.7)

        self.assertEqual(contract["schema"], "generate2dsprite.godot_sprite3d.v1")
        self.assertEqual(contract["sprite3d_offset"], [0.0, 52.0])
        self.assertAlmostEqual(contract["recommended_pixel_size"], 0.007)
        self.assertEqual(contract["scale_source"], "measured_subject_height")
        self.assertEqual(contract["fps"], 8.0)
        self.assertEqual(contract["frames"], ["idle-1.png", "idle-2.png"])


class AnchorLayoutTests(unittest.TestCase):
    def test_anchor_layout_repeats_identical_scale_and_feet_line(self) -> None:
        source = Image.new("RGBA", (40, 40), MAGENTA)
        source.paste(SUBJECT, (10, 5, 30, 35))

        layout = ANCHOR_MODULE.build_anchor_layout(
            source,
            rows=2,
            cols=3,
            cell_width=40,
            cell_height=40,
            subject_height_ratio=0.5,
            subject_width_ratio=0.8,
            feet_ratio=0.8,
            threshold=100,
            edge_threshold=150,
        )

        self.assertEqual(layout.size, (120, 80))
        bboxes = []
        for row in range(2):
            for col in range(3):
                cell = layout.crop((col * 40, row * 40, (col + 1) * 40, (row + 1) * 40))
                cleaned = MODULE.remove_bg_magenta(cell, 100, 150)
                bboxes.append(cleaned.getbbox())
        self.assertTrue(all(bbox == bboxes[0] for bbox in bboxes))
        self.assertEqual(bboxes[0][3], 32)


class GodotSprite3DBundleTests(unittest.TestCase):
    def make_contract(self, world_height: float, pixel_size: float = 0.0042) -> dict[str, object]:
        return {
            "schema": "generate2dsprite.godot_sprite3d.v1",
            "world_height": world_height,
            "frames": ["frame-1.png", "frame-2.png"],
            "recommended_pixel_size": pixel_size,
        }

    def test_bundle_marks_one_shots_and_preserves_default(self) -> None:
        bundle = MODULE.build_godot_sprite3d_bundle(
            {
                "idle": ("idle/godot-sprite3d.json", self.make_contract(0.7)),
                "hurt": ("hurt/godot-sprite3d.json", self.make_contract(0.705)),
            },
            default_action="idle",
            one_shot_actions={"hurt"},
        )

        self.assertEqual(bundle["default_action"], "idle")
        self.assertTrue(bundle["actions"]["idle"]["loop"])
        self.assertFalse(bundle["actions"]["hurt"]["loop"])
        self.assertLess(bundle["world_height_max_drift"], 0.02)

    def test_bundle_rejects_cross_action_world_height_drift(self) -> None:
        with self.assertRaisesRegex(ValueError, "world-height drift"):
            MODULE.build_godot_sprite3d_bundle(
                {
                    "idle": ("idle.json", self.make_contract(0.7)),
                    "attack": ("attack.json", self.make_contract(0.9)),
                },
                default_action="idle",
                one_shot_actions={"attack"},
            )

    def test_bundle_rejects_per_action_runtime_rescaling(self) -> None:
        with self.assertRaisesRegex(ValueError, "pixel-size drift"):
            MODULE.build_godot_sprite3d_bundle(
                {
                    "idle": ("idle.json", self.make_contract(0.7, 0.0042)),
                    "hurt": ("hurt.json", self.make_contract(0.7, 0.0050)),
                },
                default_action="idle",
                one_shot_actions={"hurt"},
            )

    def test_bundle_scale_is_invariant_to_action_order(self) -> None:
        contracts = {
            "idle": ("idle.json", self.make_contract(0.7, 0.0042)),
            "hurt": ("hurt.json", self.make_contract(0.705, 0.00424)),
            "attack": ("attack.json", self.make_contract(0.704, 0.00423)),
        }
        bundles = []
        for order in itertools.permutations(contracts):
            ordered = {action: contracts[action] for action in order}
            bundles.append(
                MODULE.build_godot_sprite3d_bundle(ordered, default_action="idle")
            )

        for bundle in bundles:
            self.assertEqual(bundle["world_height"], 0.7)
            self.assertEqual(bundle["pixel_size"], 0.0042)
        self.assertEqual(
            {(bundle["world_height_max_drift"], bundle["pixel_size_max_drift"]) for bundle in bundles},
            {(bundles[0]["world_height_max_drift"], bundles[0]["pixel_size_max_drift"])},
        )

    def test_default_action_reference_is_used_when_default_is_last(self) -> None:
        contracts = {
            "attack": ("attack.json", self.make_contract(0.728, 0.00436)),
            "hurt": ("hurt.json", self.make_contract(0.714, 0.00428)),
            "idle": ("idle.json", self.make_contract(0.7, 0.0042)),
        }
        with self.assertRaisesRegex(ValueError, "world-height drift"):
            MODULE.build_godot_sprite3d_bundle(contracts, default_action="idle")

    def test_default_action_last_sets_canonical_scale(self) -> None:
        bundle = MODULE.build_godot_sprite3d_bundle(
            {
                "hurt": ("hurt.json", self.make_contract(0.705, 0.00424)),
                "idle": ("idle.json", self.make_contract(0.7, 0.0042)),
            },
            default_action="idle",
        )
        self.assertEqual(bundle["world_height"], 0.7)
        self.assertEqual(bundle["pixel_size"], 0.0042)

    def test_scale_profile_locks_runtime_pixel_size(self) -> None:
        metadata = ScaleProfileTests().make_metadata()
        metadata["godot_sprite3d"] = {
            "world_height": 0.7,
            "recommended_pixel_size": 0.0042,
        }
        profile = MODULE.build_scale_profile(metadata, "creature", 0.15)
        self.assertEqual(profile["godot_sprite3d"]["pixel_size"], 0.0042)


class ParserTests(unittest.TestCase):
    def test_source_edge_override_is_explicit_and_opt_in(self) -> None:
        parser = MODULE.build_parser()
        args = parser.parse_args(
            [
                "process",
                "--input", "raw.png",
                "--target", "creature",
                "--mode", "idle",
                "--output-dir", "out",
                "--allow-source-edge-touch",
            ]
        )
        self.assertTrue(args.allow_source_edge_touch)

    def test_custom_grid_prefixes_are_safe_and_slugged(self) -> None:
        self.assertEqual(MODULE.validate_filename_prefix("Heavy Attack"), "heavy-attack")
        for value in (
            "../outside",
            r"..\outside",
            "/absolute",
            r"C:\absolute",
            "CON",
            "NUL",
            "bad\u0085prefix",
            "bad\u200bprefix",
        ):
            with self.assertRaises(ValueError):
                MODULE.validate_filename_prefix(value)

    def test_invalid_custom_prefix_writes_no_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "raw.png"
            Image.new("RGBA", (4, 4), (20, 40, 60, 255)).save(input_path)
            output_dir = root / "out"
            args = MODULE.build_parser().parse_args(
                [
                    "process",
                    "--input", str(input_path),
                    "--target", "asset",
                    "--mode", "sheet",
                    "--output-dir", str(output_dir),
                    "--rows", "1",
                    "--cols", "1",
                    "--label-prefix", "../outside",
                    "--trim-border", "0",
                    "--edge-clean-depth", "0",
                ]
            )
            with self.assertRaises(ValueError):
                MODULE.cmd_process(args)
            self.assertFalse(output_dir.exists())

    def test_valid_custom_prefix_uses_contained_slugged_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "raw.png"
            Image.new("RGBA", (4, 4), (20, 40, 60, 255)).save(input_path)
            output_dir = root / "out"
            args = MODULE.build_parser().parse_args(
                [
                    "process",
                    "--input", str(input_path),
                    "--target", "asset",
                    "--mode", "sheet",
                    "--output-dir", str(output_dir),
                    "--rows", "1",
                    "--cols", "1",
                    "--label-prefix", "Heavy Attack",
                    "--fit-scale", "1",
                    "--trim-border", "0",
                    "--edge-clean-depth", "0",
                ]
            )
            MODULE.cmd_process(args)
            self.assertTrue((output_dir / "heavy-attack-1.png").is_file())
            self.assertFalse((root / "outside-1.png").exists())

    def test_source_edge_override_still_rejects_output_edge_touch(self) -> None:
        parser = MODULE.build_parser()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = Image.new("RGBA", (40, 40), MAGENTA)
            for row in range(2):
                for col in range(2):
                    left = col * 20
                    top = row * 20
                    image.paste(SUBJECT, (left, top + 5, left + 6, top + 15))
            source = root / "raw.png"
            image.save(source)

            common = [
                "process",
                "--input", str(source),
                "--target", "asset",
                "--mode", "idle",
                "--trim-border", "0",
                "--edge-clean-depth", "0",
                "--cell-size", "32",
                "--component-mode", "largest",
                "--strict-qc",
                "--reject-edge-touch",
                "--allow-source-edge-touch",
            ]
            MODULE.cmd_process(
                parser.parse_args(
                    [*common, "--output-dir", str(root / "safe"), "--fit-scale", "0.5"]
                )
            )

            with self.assertRaisesRegex(ValueError, "frames touch a cell edge"):
                MODULE.cmd_process(
                    parser.parse_args(
                        [*common, "--output-dir", str(root / "unsafe"), "--fit-scale", "1.0"]
                    )
                )

    def test_process_with_palette_despill_mixel_and_exports(self) -> None:
        parser = MODULE.build_parser()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "raw.png"
            out_dir = root / "out"

            # Create a 2x2 grid image with magenta background
            img = Image.new("RGBA", (64, 64), MAGENTA)
            for r in range(2):
                for c in range(2):
                    left = c * 32 + 8
                    top = r * 32 + 8
                    img.paste(SUBJECT, (left, top, left + 16, top + 16))
            img.save(source)

            args = parser.parse_args([
                "process",
                "--input", str(source),
                "--target", "asset",
                "--mode", "idle",
                "--output-dir", str(out_dir),
                "--cell-size", "32",
                "--rows", "2",
                "--cols", "2",
                "--despill",
                "--palette", "pico-8",
                "--pixel-grid", "16",
                "--export-aseprite",
                "--export-texturepacker",
            ])
            MODULE.cmd_process(args)

            self.assertTrue((out_dir / "sheet-transparent.png").is_file())
            self.assertTrue((out_dir / "idle.aseprite").is_file())
            self.assertTrue((out_dir / "sheet.json").is_file())

            meta = json.loads((out_dir / "pipeline-meta.json").read_text(encoding="utf-8"))
            self.assertIn("aseprite_output", meta)
            self.assertIn("texturepacker_output", meta)


if __name__ == "__main__":
    unittest.main()

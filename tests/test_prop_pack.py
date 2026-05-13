"""Regression tests for extract_prop_pack.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from conftest import (
    ROOT,
    PROP_PACK_SCRIPT,
    assert_golden,
    make_image,
)


def test_remove_bg_magenta_pure_magenta(prop_pack_mod, solid_magenta):
    result = prop_pack_mod.remove_bg_magenta(solid_magenta.copy(), 100, 150)
    arr = np.array(result)
    assert (arr[..., 3] == 0).all()


def test_remove_bg_magenta_keeps_subject(prop_pack_mod, magenta_with_green_subject):
    result = prop_pack_mod.remove_bg_magenta(magenta_with_green_subject.copy(), 100, 150)
    arr = np.array(result)
    assert (arr[5:11, 5:11, 3] == 255).all()
    assert arr[0, 0, 3] == 0


def test_remove_bg_magenta_internal_near_preserved(prop_pack_mod, subject_with_internal_near_magenta):
    result = prop_pack_mod.remove_bg_magenta(subject_with_internal_near_magenta.copy(), 100, 150)
    arr = np.array(result)
    assert arr[10, 10, 3] == 255
    assert arr[0, 0, 3] == 0


def test_remove_bg_magenta_golden(prop_pack_mod, magenta_with_green_subject):
    result = prop_pack_mod.remove_bg_magenta(magenta_with_green_subject.copy(), 100, 150)
    assert_golden("prop_pack_remove_bg_subject", np.array(result))


def test_clean_edges_depth_2(prop_pack_mod, dark_canvas_with_bright_center):
    result = prop_pack_mod.clean_edges(dark_canvas_with_bright_center.copy(), depth=2)
    arr = np.array(result)
    assert arr[0, 0, 3] == 0
    assert arr[1, 1, 3] == 0
    assert arr[5, 5, 3] == 255
    assert arr[7, 7, 3] == 255


def test_clean_edges_golden(prop_pack_mod, dark_canvas_with_bright_center):
    result = prop_pack_mod.clean_edges(dark_canvas_with_bright_center.copy(), depth=2)
    assert_golden("prop_pack_clean_edges_depth2", np.array(result))


def test_connected_components_largest_first(prop_pack_mod, two_component_canvas):
    comps = prop_pack_mod.connected_components(two_component_canvas, min_area=1)
    assert len(comps) == 2
    assert int(comps[0]["area"]) == 9
    assert tuple(comps[0]["bbox"]) == (2, 2, 5, 5)
    assert int(comps[1]["area"]) == 4
    assert tuple(comps[1]["bbox"]) == (10, 10, 12, 12)


def test_extract_prop_pack_end_to_end(grid_3x3_prop_pack, tmp_path):
    """Run the script via subprocess to exercise CLI + manifest output."""
    out_dir = tmp_path / "props"
    manifest = tmp_path / "manifest.json"
    result = subprocess.run(
        [
            sys.executable,
            str(PROP_PACK_SCRIPT),
            "--input", str(grid_3x3_prop_pack),
            "--rows", "3",
            "--cols", "3",
            "--output-dir", str(out_dir),
            "--manifest", str(manifest),
            "--min-component-area", "1",
            "--component-padding", "0",
            "--trim-border", "0",
            "--edge-clean-depth", "0",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"prop pack run failed: {result.stderr}"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert len(data["accepted"]) == 9, "Each of the 9 cells must yield one prop"
    for entry in data["accepted"]:
        assert entry["status"] == "accepted"
        assert entry["selected_component_area"] == 9
        assert entry["output_size"] != [0, 0]
        prop_path = Path(entry["image"])
        assert prop_path.exists()


def test_extract_prop_pack_edge_touch(prop_pack_mod, tmp_path):
    """Subject spilling to a cell edge must be flagged via the edge_touch metadata."""
    # Build a 20x20 image (2x2 grid of 10x10 cells), put a block that hits cell
    # 0,0's right edge.
    arr = np.full((20, 20, 4), [255, 0, 255, 255], dtype=np.uint8)
    arr[2:8, 5:10] = [0, 100, 200, 255]
    img = make_image(arr)
    img_path = tmp_path / "pack.png"
    img.save(img_path)

    out_dir = tmp_path / "out"
    manifest = tmp_path / "m.json"
    result = subprocess.run(
        [
            sys.executable,
            str(PROP_PACK_SCRIPT),
            "--input", str(img_path),
            "--rows", "2",
            "--cols", "2",
            "--output-dir", str(out_dir),
            "--manifest", str(manifest),
            "--min-component-area", "1",
            "--trim-border", "0",
            "--edge-clean-depth", "0",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(manifest.read_text(encoding="utf-8"))
    cell_00 = next(item for item in data["accepted"] if item["grid"] == [0, 0])
    assert cell_00["edge_touch"] is True

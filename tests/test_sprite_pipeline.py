"""Regression tests for generate2dsprite.py post-processing pipeline."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from conftest import assert_golden, make_image


def test_remove_bg_magenta_pure_magenta(sprite_mod, solid_magenta):
    result = sprite_mod.remove_bg_magenta(solid_magenta.copy())
    arr = np.array(result)
    assert (arr[..., 3] == 0).all(), "Solid magenta canvas must be fully transparent"


def test_remove_bg_magenta_keeps_subject(sprite_mod, magenta_with_green_subject):
    result = sprite_mod.remove_bg_magenta(magenta_with_green_subject.copy())
    arr = np.array(result)
    assert (arr[5:11, 5:11, 3] == 255).all(), "Subject interior must stay opaque"
    assert arr[0, 0, 3] == 0, "Background corner must be transparent"


def test_remove_bg_magenta_internal_pure_magenta_cleared(sprite_mod, subject_with_internal_pure_magenta):
    """Pure magenta inside subject is cleared by the unconditional first pass."""
    result = sprite_mod.remove_bg_magenta(subject_with_internal_pure_magenta.copy())
    arr = np.array(result)
    assert arr[8, 8, 3] == 0, "Pure magenta inside subject must be cleared (first pass)"


def test_remove_bg_magenta_internal_near_magenta_preserved(sprite_mod, subject_with_internal_near_magenta):
    """Near-magenta surrounded by opaque subject must NOT be cleared (BFS can't reach)."""
    result = sprite_mod.remove_bg_magenta(subject_with_internal_near_magenta.copy())
    arr = np.array(result)
    assert arr[10, 10, 3] == 255, "Near-magenta enclosed in opaque subject must remain"
    assert arr[0, 0, 3] == 0, "Background still cleared"


def test_remove_bg_magenta_edge_near_magenta_cleared(sprite_mod, near_magenta_touching_edge):
    result = sprite_mod.remove_bg_magenta(near_magenta_touching_edge.copy())
    arr = np.array(result)
    assert (arr[:, 0, 3] == 0).all(), "Near-magenta column at edge must be cleared by BFS"


def test_remove_bg_magenta_golden(sprite_mod, magenta_with_green_subject):
    result = sprite_mod.remove_bg_magenta(magenta_with_green_subject.copy())
    assert_golden("sprite_remove_bg_subject", np.array(result))


def test_remove_bg_magenta_internal_near_golden(sprite_mod, subject_with_internal_near_magenta):
    result = sprite_mod.remove_bg_magenta(subject_with_internal_near_magenta.copy())
    assert_golden("sprite_remove_bg_internal_near", np.array(result))


def test_clean_edges_depth_3_clears_outer_rings(sprite_mod, dark_canvas_with_bright_center):
    result = sprite_mod.clean_edges(dark_canvas_with_bright_center.copy(), depth=3)
    arr = np.array(result)
    assert arr[0, 0, 3] == 0, "Outer ring must be cleared"
    assert arr[2, 2, 3] == 0, "Ring at depth=2 must be cleared (dark pixel)"
    assert arr[5, 5, 3] == 255, "Beyond depth=3, interior dark pixels untouched"
    assert arr[7, 7, 3] == 255, "Bright center pixel preserved"


def test_clean_edges_golden(sprite_mod, dark_canvas_with_bright_center):
    result = sprite_mod.clean_edges(dark_canvas_with_bright_center.copy(), depth=3)
    assert_golden("sprite_clean_edges_depth3", np.array(result))


def test_connected_components_count_and_order(sprite_mod, two_component_canvas):
    comps = sprite_mod.connected_components(two_component_canvas, min_area=1)
    assert len(comps) == 2
    assert comps[0]["area"] == 9, "Largest (3x3) component listed first"
    assert comps[1]["area"] == 4, "Second (2x2) component listed second"
    assert comps[0]["bbox"] == (2, 2, 5, 5)
    assert comps[1]["bbox"] == (10, 10, 12, 12)
    assert comps[0]["touches_edge"] is False
    assert comps[1]["touches_edge"] is False


def test_connected_components_min_area_filter(sprite_mod, two_component_canvas):
    comps = sprite_mod.connected_components(two_component_canvas, min_area=5)
    assert len(comps) == 1, "min_area=5 filters out the 4-pixel component"
    assert comps[0]["area"] == 9


def test_split_grid_2x2_basic(sprite_mod):
    """End-to-end: 2x2 grid splitting must produce 4 frames with consistent layout."""
    arr = np.full((64, 64, 4), [255, 0, 255, 255], dtype=np.uint8)
    for ry in range(2):
        for cx in range(2):
            y0, x0 = ry * 32 + 8, cx * 32 + 8
            arr[y0 : y0 + 16, x0 : x0 + 16] = [50, 150, 250, 255]
    img = make_image(arr)
    frames, qc = sprite_mod.split_grid(
        img,
        rows=2,
        cols=2,
        cell_size=64,
        threshold=100,
        edge_threshold=150,
    )
    assert len(frames) == 4
    assert len(qc) == 4
    for frame in frames:
        assert frame.size == (64, 64)
    for info in qc:
        assert info["crop_bbox"] is not None
        assert info["output_size"] != [0, 0]


def test_save_transparent_gif_palette_index_zero(sprite_mod, tmp_path):
    frames = []
    for _ in range(3):
        a = np.zeros((16, 16, 4), dtype=np.uint8)
        a[4:12, 4:12] = [255, 100, 50, 255]
        frames.append(make_image(a))
    out = tmp_path / "test.gif"
    sprite_mod.save_transparent_gif(frames, out, duration=100)
    gif = Image.open(out)
    assert gif.info.get("transparency") == 0, "GIF transparency index must be 0"


def test_build_prompt_deterministic(sprite_mod):
    """Same inputs must produce byte-identical prompt strings."""
    a, seed_a = sprite_mod.build_prompt("creature", "evolution", "fire wolf")
    b, seed_b = sprite_mod.build_prompt("creature", "evolution", "fire wolf")
    assert a == b
    assert seed_a == seed_b

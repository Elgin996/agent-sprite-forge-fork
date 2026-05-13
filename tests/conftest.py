"""Shared fixtures for sprite/map pipeline regression tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SPRITE_SCRIPT = ROOT / "skills" / "generate2dsprite" / "scripts" / "generate2dsprite.py"
PROP_PACK_SCRIPT = ROOT / "skills" / "generate2dmap" / "scripts" / "extract_prop_pack.py"
GOLDEN_DIR = Path(__file__).parent / "fixtures"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def sprite_mod():
    return _load_module("sprite_mod", SPRITE_SCRIPT)


@pytest.fixture(scope="session")
def prop_pack_mod():
    return _load_module("prop_pack_mod", PROP_PACK_SCRIPT)


def make_image(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr.astype(np.uint8), mode="RGBA")


@pytest.fixture
def solid_magenta():
    arr = np.full((8, 8, 4), [255, 0, 255, 255], dtype=np.uint8)
    return make_image(arr)


@pytest.fixture
def magenta_with_green_subject():
    """16x16 magenta canvas with an 8x8 green block centered."""
    arr = np.full((16, 16, 4), [255, 0, 255, 255], dtype=np.uint8)
    arr[4:12, 4:12] = [0, 200, 0, 255]
    return make_image(arr)


@pytest.fixture
def subject_with_internal_pure_magenta():
    """Green subject containing a single pure-magenta pixel (cleared by first pass, not BFS)."""
    arr = np.full((16, 16, 4), [255, 0, 255, 255], dtype=np.uint8)
    arr[4:12, 4:12] = [0, 200, 0, 255]
    arr[8, 8] = [255, 0, 255, 255]
    return make_image(arr)


@pytest.fixture
def subject_with_internal_near_magenta():
    """Green subject containing a near-magenta pixel surrounded by opaque green.

    Chosen so dist-to-magenta is in (threshold=100, edge_threshold=150): pixel
    (200, 80, 220) has distance ~103. First pass leaves it (not within 100).
    BFS cannot reach it through opaque green pixels, so it must remain opaque.
    This is the critical regression case for the BFS edge-bleed implementation.
    """
    arr = np.full((20, 20, 4), [255, 0, 255, 255], dtype=np.uint8)
    arr[4:16, 4:16] = [0, 200, 0, 255]
    arr[10, 10] = [200, 80, 220, 255]
    return make_image(arr)


@pytest.fixture
def near_magenta_touching_edge():
    """Near-magenta column at the left edge — BFS must clear it."""
    arr = np.full((16, 16, 4), [0, 200, 0, 255], dtype=np.uint8)
    arr[:, 0] = [250, 5, 245, 255]
    return make_image(arr)


@pytest.fixture
def dark_canvas_with_bright_center():
    """16x16 dark gray (below threshold) with a bright center pixel."""
    arr = np.full((16, 16, 4), [30, 30, 30, 255], dtype=np.uint8)
    arr[7, 7] = [200, 200, 200, 255]
    return make_image(arr)


@pytest.fixture
def two_component_canvas():
    """Two disjoint opaque blocks for connected_components tests."""
    arr = np.zeros((16, 16, 4), dtype=np.uint8)
    arr[2:5, 2:5] = [100, 100, 100, 255]
    arr[10:12, 10:12] = [100, 100, 100, 255]
    return make_image(arr)


@pytest.fixture
def grid_3x3_prop_pack(tmp_path):
    """30x30 magenta canvas, 3x3 grid, 3x3 colored block in the center of each cell."""
    arr = np.full((30, 30, 4), [255, 0, 255, 255], dtype=np.uint8)
    for r in range(3):
        for c in range(3):
            cy, cx = r * 10 + 5, c * 10 + 5
            arr[cy - 1 : cy + 2, cx - 1 : cx + 2] = [0, 100, 200, 255]
    img = make_image(arr)
    path = tmp_path / "pack.png"
    img.save(path)
    return path


def assert_golden(name: str, array: np.ndarray) -> None:
    """Compare array to a saved golden .npy; create on first run."""
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    path = GOLDEN_DIR / f"{name}.npy"
    if not path.exists():
        np.save(path, array)
        pytest.skip(f"Created golden fixture {name}")
    expected = np.load(path)
    np.testing.assert_array_equal(array, expected, err_msg=f"Mismatch vs golden {name}")

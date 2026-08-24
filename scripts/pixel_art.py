#!/usr/bin/env python3
"""Pixel art quantization, standard color palette clamping, and mixel reduction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

BUILTIN_PALETTES: dict[str, list[tuple[int, int, int]]] = {
    "pico-8": [
        (0, 0, 0),        # 0: Black
        (29, 43, 83),     # 1: Dark Blue
        (126, 37, 83),    # 2: Dark Purple
        (0, 135, 81),     # 3: Dark Green
        (171, 82, 54),    # 4: Brown
        (95, 87, 79),     # 5: Dark Gray
        (194, 195, 199),  # 6: Light Gray
        (255, 241, 232),  # 7: White
        (255, 0, 77),     # 8: Red
        (255, 163, 0),    # 9: Orange
        (255, 236, 39),   # 10: Yellow
        (0, 228, 54),     # 11: Green
        (41, 173, 255),   # 12: Blue
        (131, 118, 156),  # 13: Indigo
        (255, 119, 168),  # 14: Pink
        (255, 204, 170),  # 15: Peach
    ],
    "endesga-32": [
        (190, 74, 47), (215, 118, 67), (234, 212, 170), (228, 166, 114),
        (184, 111, 80), (115, 62, 57), (62, 39, 49), (162, 38, 51),
        (228, 59, 68), (247, 118, 34), (254, 174, 52), (254, 231, 97),
        (99, 199, 77), (62, 137, 72), (38, 92, 66), (25, 60, 62),
        (18, 78, 137), (0, 153, 219), (44, 232, 244), (255, 255, 255),
        (192, 203, 220), (139, 155, 180), (90, 105, 136), (58, 68, 102),
        (38, 43, 68), (24, 20, 37), (255, 0, 68), (255, 106, 60),
        (175, 42, 107), (118, 35, 78), (73, 34, 60), (38, 20, 43)
    ],
    "gameboy": [
        (15, 56, 15),     # Darkest green
        (48, 98, 48),     # Dark green
        (139, 172, 15),   # Light green
        (155, 188, 15),   # Lightest green
    ],
    "db16": [
        (20, 12, 28), (68, 36, 52), (48, 52, 109), (78, 74, 78),
        (133, 76, 48), (52, 101, 36), (208, 70, 72), (117, 113, 102),
        (89, 125, 206), (210, 125, 44), (133, 149, 161), (109, 170, 44),
        (210, 170, 153), (109, 194, 202), (218, 212, 94), (222, 238, 214)
    ],
    "db32": [
        (0, 0, 0), (34, 32, 52), (69, 40, 60), (102, 57, 49),
        (143, 86, 59), (223, 113, 38), (217, 160, 102), (238, 195, 154),
        (251, 242, 54), (153, 229, 80), (106, 190, 48), (55, 148, 110),
        (75, 105, 47), (82, 75, 36), (50, 60, 57), (63, 63, 116),
        (48, 96, 130), (91, 110, 225), (99, 155, 255), (95, 205, 228),
        (203, 219, 252), (255, 255, 255), (155, 173, 183), (132, 126, 135),
        (105, 106, 106), (89, 86, 82), (118, 66, 138), (172, 50, 50),
        (217, 87, 99), (215, 123, 186), (143, 151, 74), (138, 111, 48)
    ],
    "sweetie-16": [
        (26, 28, 44), (93, 39, 93), (177, 62, 83), (239, 125, 87),
        (255, 205, 117), (167, 240, 112), (56, 183, 100), (37, 113, 121),
        (41, 54, 111), (59, 93, 201), (65, 166, 246), (115, 239, 247),
        (244, 244, 244), (148, 176, 194), (86, 108, 134), (51, 60, 87)
    ],
}


def parse_hex_color(hex_str: str) -> tuple[int, int, int]:
    clean = hex_str.strip().lstrip("#")
    if len(clean) == 3:
        clean = "".join(c * 2 for c in clean)
    if len(clean) != 6:
        raise ValueError(f"Invalid hex color: {hex_str}")
    return int(clean[0:2], 16), int(clean[2:4], 16), int(clean[4:6], 16)


def resolve_palette(palette_spec: str | list[tuple[int, int, int]]) -> list[tuple[int, int, int]]:
    """Resolve a palette name, comma-separated hex string, or list of RGB tuples."""
    if isinstance(palette_spec, list):
        return palette_spec
    
    key = palette_spec.strip().lower()
    if key in BUILTIN_PALETTES:
        return BUILTIN_PALETTES[key]
    
    # Check aliases
    alias_map = {
        "pico8": "pico-8",
        "endesga32": "endesga-32",
        "sweetie16": "sweetie-16",
        "dawnbringer16": "db16",
        "dawnbringer32": "db32",
        "gb": "gameboy",
    }
    if key in alias_map:
        return BUILTIN_PALETTES[alias_map[key]]

    # Parse comma/space separated hex codes
    tokens = [t.strip() for t in palette_spec.replace(",", " ").split() if t.strip()]
    if tokens:
        return [parse_hex_color(t) for t in tokens]
    
    raise ValueError(f"Unknown or invalid palette specification: {palette_spec}")


def quantize_to_palette(
    image: Image.Image,
    palette_spec: str | list[tuple[int, int, int]],
    alpha_threshold: int = 10,
) -> Image.Image:
    """Clamp non-transparent pixels in an RGBA image to the nearest color in the target palette."""
    palette_rgb = resolve_palette(palette_spec)
    palette_arr = np.array(palette_rgb, dtype=np.float32)  # Shape: (N, 3)

    img_rgba = image.convert("RGBA")
    arr = np.array(img_rgba)
    h, w, _ = arr.shape

    rgb = arr[:, :, :3].astype(np.float32)  # (H, W, 3)
    alpha = arr[:, :, 3]                    # (H, W)

    # Mask for non-transparent pixels
    valid_mask = alpha >= alpha_threshold

    if np.any(valid_mask):
        valid_rgb = rgb[valid_mask]  # (M, 3)
        # Compute squared distance to each palette color: (M, 1, 3) - (1, N, 3) -> (M, N)
        diff = valid_rgb[:, np.newaxis, :] - palette_arr[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=2)
        nearest_indices = np.argmin(dist_sq, axis=1)

        # Replace valid pixels with exact palette RGB
        mapped_rgb = palette_arr[nearest_indices].astype(np.uint8)
        arr[valid_mask, :3] = mapped_rgb

    return Image.fromarray(arr, mode="RGBA")


def reduce_mixels(
    image: Image.Image,
    target_grid: int | tuple[int, int],
    upscale_back: bool = True,
) -> Image.Image:
    """Downsample an image to a strict pixel grid to remove mixels, optionally upscaling back."""
    img_rgba = image.convert("RGBA")
    orig_w, orig_h = img_rgba.size

    if isinstance(target_grid, int):
        target_w = target_grid
        target_h = int(round(orig_h * (target_grid / orig_w)))
    else:
        target_w, target_h = target_grid

    target_w = max(1, target_w)
    target_h = max(1, target_h)

    # Downscale using Box / Area resampling to average subpixel mixels
    downscaled = img_rgba.resize((target_w, target_h), resample=Image.Resampling.BOX)

    if not upscale_back:
        return downscaled

    # Upscale back to original size with Nearest Neighbor for crisp pixel art
    return downscaled.resize((orig_w, orig_h), resample=Image.Resampling.NEAREST)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input image file path.")
    parser.add_argument("--out", required=True, type=Path, help="Output image file path.")
    parser.add_argument(
        "--palette",
        help=f"Target palette name ({', '.join(BUILTIN_PALETTES.keys())}) or custom hex list.",
    )
    parser.add_argument(
        "--pixel-grid",
        type=int,
        help="Target grid resolution (e.g. 32, 64) for mixel reduction.",
    )
    parser.add_argument(
        "--downscale-only",
        action="store_true",
        help="Keep the image downscaled at the pixel grid resolution without upscaling back.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite output file if it exists.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.input.is_file():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        return 1

    if args.out.exists() and not args.force:
        print(f"Error: output file already exists: {args.out} (use --force to overwrite)", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(args.input) as im:
        result = im.convert("RGBA")
        if args.pixel_grid:
            result = reduce_mixels(result, target_grid=args.pixel_grid, upscale_back=not args.downscale_only)
        if args.palette:
            result = quantize_to_palette(result, palette_spec=args.palette)
        result.save(args.out, "PNG")

    print(f"Wrote processed pixel-art image to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

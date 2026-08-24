#!/usr/bin/env python3
"""Remove chroma key background from images with optional soft matting, despill, and edge contraction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


def parse_hex_color(hex_str: str) -> tuple[int, int, int]:
    clean = hex_str.strip().lstrip("#")
    if len(clean) == 3:
        clean = "".join(c * 2 for c in clean)
    if len(clean) != 6:
        raise ValueError(f"Invalid hex color: {hex_str}. Expected 3 or 6 hex digits (e.g. #ff00ff).")
    return int(clean[0:2], 16), int(clean[2:4], 16), int(clean[4:6], 16)


def color_distance(arr: np.ndarray, key_rgb: tuple[int, int, int]) -> np.ndarray:
    """Calculate Euclidean distance in RGB color space."""
    rgb = arr[:, :, :3].astype(np.float32)
    key = np.array(key_rgb, dtype=np.float32)
    return np.sqrt(np.sum((rgb - key) ** 2, axis=2))


def apply_despill(
    rgb: np.ndarray,
    alpha: np.ndarray,
    key_rgb: tuple[int, int, int],
) -> np.ndarray:
    """Neutralize color spill around edges based on the key color."""
    kr, kg, kb = key_rgb
    out_rgb = rgb.astype(np.float32).copy()
    r, g, b = out_rgb[:, :, 0], out_rgb[:, :, 1], out_rgb[:, :, 2]

    # Magenta key despill (kr > 200, kb > 200, kg < 100)
    if kr > kg and kb > kg:
        excess = np.maximum(0.0, np.minimum(r, b) - g)
        fade = 1.0 - (alpha.astype(np.float32) / 255.0)
        weight = np.clip(fade * 1.5, 0.0, 1.0)
        out_rgb[:, :, 0] = np.clip(r - excess * weight, 0.0, 255.0)
        out_rgb[:, :, 2] = np.clip(b - excess * weight, 0.0, 255.0)
    # Green key despill (kg > kr, kg > kb)
    elif kg > kr and kg > kb:
        excess = np.maximum(0.0, g - np.maximum(r, b))
        fade = 1.0 - (alpha.astype(np.float32) / 255.0)
        weight = np.clip(fade * 1.5, 0.0, 1.0)
        out_rgb[:, :, 1] = np.clip(g - excess * weight, 0.0, 255.0)
    # Blue key despill (kb > kr, kb > kg)
    elif kb > kr and kb > kg:
        excess = np.maximum(0.0, b - np.maximum(r, g))
        fade = 1.0 - (alpha.astype(np.float32) / 255.0)
        weight = np.clip(fade * 1.5, 0.0, 1.0)
        out_rgb[:, :, 2] = np.clip(b - excess * weight, 0.0, 255.0)

    return out_rgb.astype(np.uint8)


def remove_chroma_key(
    image: Image.Image,
    key_color: tuple[int, int, int] = (255, 0, 255),
    soft_matte: bool = False,
    transparent_threshold: float = 35.0,
    opaque_threshold: float = 160.0,
    despill: bool = False,
    edge_contract: int = 0,
) -> Image.Image:
    """Process an image to remove the specified chroma key color and return RGBA image."""
    img_rgba = image.convert("RGBA")
    arr = np.array(img_rgba)

    dist = color_distance(arr, key_color)
    orig_alpha = arr[:, :, 3].astype(np.float32)

    if soft_matte:
        t_low = float(transparent_threshold)
        t_high = max(float(opaque_threshold), t_low + 1e-5)
        factor = np.clip((dist - t_low) / (t_high - t_low), 0.0, 1.0)
        new_alpha = (orig_alpha * factor).astype(np.uint8)
    else:
        mask = dist > float(transparent_threshold)
        new_alpha = np.where(mask, orig_alpha, 0.0).astype(np.uint8)

    if edge_contract > 0:
        alpha_img = Image.fromarray(new_alpha, mode="L")
        size = edge_contract * 2 + 1
        alpha_img = alpha_img.filter(ImageFilter.MinFilter(size))
        new_alpha = np.array(alpha_img)

    rgb = arr[:, :, :3]
    if despill:
        rgb = apply_despill(rgb, new_alpha, key_color)

    result_arr = np.dstack([rgb, new_alpha])
    return Image.fromarray(result_arr, mode="RGBA")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Path to input image file.")
    parser.add_argument("--out", "--output", dest="out", required=True, type=Path, help="Path to output PNG image file.")
    parser.add_argument("--key-color", default="#ff00ff", help="Chroma key hex color, e.g. '#ff00ff' (default).")
    parser.add_argument("--soft-matte", action="store_true", help="Enable soft alpha transition ramp.")
    parser.add_argument(
        "--transparent-threshold",
        type=float,
        default=35.0,
        help="Color distance below which pixels become completely transparent (default 35.0).",
    )
    parser.add_argument(
        "--opaque-threshold",
        type=float,
        default=160.0,
        help="Color distance above which pixels remain fully opaque (default 160.0).",
    )
    parser.add_argument("--despill", action="store_true", help="Neutralize key color spill along edges.")
    parser.add_argument(
        "--edge-contract",
        type=int,
        default=0,
        help="Erode alpha mask by N pixels to remove border halos (default 0).",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite output file if it already exists.")
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

    try:
        key_rgb = parse_hex_color(args.key_color)
    except ValueError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(args.input) as im:
        result = remove_chroma_key(
            im,
            key_color=key_rgb,
            soft_matte=args.soft_matte,
            transparent_threshold=args.transparent_threshold,
            opaque_threshold=args.opaque_threshold,
            despill=args.despill,
            edge_contract=args.edge_contract,
        )
        result.save(args.out, "PNG")

    print(f"Wrote transparent image to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Export sprite frames to native Aseprite binary files (.aseprite / .ase) and TexturePacker JSON atlases."""

from __future__ import annotations

import argparse
import io
import json
import struct
import sys
import zlib
from pathlib import Path

from PIL import Image


def create_aseprite_string(text: str) -> bytes:
    encoded = text.encode("utf-8")
    return struct.pack("<H", len(encoded)) + encoded


def build_layer_chunk(layer_name: str = "Layer 1") -> bytes:
    """Build an Aseprite Layer Chunk (0x2004)."""
    flags = 1 | 2  # Visible | Editable
    layer_type = 0  # Normal image layer
    child_level = 0
    default_w = 0
    default_h = 0
    blend_mode = 0  # Normal
    opacity = 255
    reserved = b"\x00\x00\x00"
    name_bytes = create_aseprite_string(layer_name)

    body = struct.pack(
        "<HHHHHBB3s",
        flags,
        layer_type,
        child_level,
        default_w,
        default_h,
        blend_mode,
        opacity,
        reserved,
    ) + name_bytes

    chunk_size = 4 + 2 + len(body)
    return struct.pack("<IH", chunk_size, 0x2004) + body


def build_cel_chunk(image: Image.Image, layer_index: int = 0) -> bytes:
    """Build an Aseprite Compressed Image Cel Chunk (0x2005)."""
    img_rgba = image.convert("RGBA")
    w, h = img_rgba.size
    raw_rgba = img_rgba.tobytes()
    compressed = zlib.compress(raw_rgba, level=6)

    x, y = 0, 0
    opacity = 255
    cel_type = 2  # Compressed image
    z_index = 0
    reserved = b"\x00" * 5

    header = struct.pack(
        "<HhhBHh5sHH",
        layer_index,
        x,
        y,
        opacity,
        cel_type,
        z_index,
        reserved,
        w,
        h,
    )
    body = header + compressed
    chunk_size = 4 + 2 + len(body)
    return struct.pack("<IH", chunk_size, 0x2005) + body


def build_tags_chunk(tags: list[tuple[str, int, int]]) -> bytes:
    """Build an Aseprite Animation Tags Chunk (0x2018).

    `tags` is a list of (tag_name, from_frame, to_frame) with 0-indexed frame indices.
    """
    body = bytearray(struct.pack("<H8s", len(tags), b"\x00" * 8))
    for name, from_idx, to_idx in tags:
        loop_direction = 0  # Forward
        repeat = 0          # Loop infinitely
        reserved = b"\x00" * 6
        tag_color = b"\x00\x00\x00\x00"
        name_bytes = create_aseprite_string(name)
        tag_data = struct.pack(
            "<HHBH6s4s",
            from_idx,
            to_idx,
            loop_direction,
            repeat,
            reserved,
            tag_color,
        ) + name_bytes
        body.extend(tag_data)

    chunk_size = 4 + 2 + len(body)
    return struct.pack("<IH", chunk_size, 0x2018) + bytes(body)


def write_aseprite_file(
    frames: list[Image.Image],
    out_path: Path,
    frame_duration_ms: int = 100,
    tags: list[tuple[str, int, int]] | None = None,
    layer_name: str = "Layer 1",
) -> None:
    """Serialize a list of PIL Images into a valid native .aseprite binary file."""
    if not frames:
        raise ValueError("Cannot write empty frames list to Aseprite file.")

    w, h = frames[0].size
    for f in frames:
        if f.size != (w, h):
            raise ValueError(f"All frames must share size ({w}, {h}), got {f.size}")

    frame_bytes_list: list[bytes] = []

    for i, frame in enumerate(frames):
        chunks = []
        if i == 0:
            chunks.append(build_layer_chunk(layer_name))
            if tags:
                chunks.append(build_tags_chunk(tags))
        chunks.append(build_cel_chunk(frame, layer_index=0))

        chunks_data = b"".join(chunks)
        frame_size = 16 + len(chunks_data)
        magic = 0xF1FA
        old_num_chunks = len(chunks) if len(chunks) < 0xFFFF else 0xFFFF
        reserved = 0
        new_num_chunks = len(chunks)

        frame_header = struct.pack(
            "<IHHHHI",
            frame_size,
            magic,
            old_num_chunks,
            frame_duration_ms,
            reserved,
            new_num_chunks,
        )
        frame_bytes_list.append(frame_header + chunks_data)

    all_frames_data = b"".join(frame_bytes_list)
    total_filesize = 128 + len(all_frames_data)

    header = struct.pack(
        "<IHHHHHIH4s4sB3sHBBhhHH84s",
        total_filesize,
        0xA5E0,                 # Magic number
        len(frames),             # Frame count
        w,                       # Width
        h,                       # Height
        32,                      # Color depth (32bpp = RGBA)
        1,                       # Flags (has valid layer opacity)
        frame_duration_ms,       # Deprecated speed
        b"\x00" * 4,             # Reserved
        b"\x00" * 4,             # Reserved
        0,                       # Transparent palette index
        b"\x00" * 3,             # Reserved
        32,                      # Num colors
        1,                       # Pixel width ratio
        1,                       # Pixel height ratio
        0, 0,                    # Grid X, Y
        16, 16,                  # Grid Width, Height
        b"\x00" * 84,            # Reserved
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(header + all_frames_data)


def export_texturepacker_json(
    frames: list[Image.Image],
    frame_names: list[str],
    sheet_image_name: str,
    sheet_width: int,
    sheet_height: int,
    frame_coords: list[tuple[int, int, int, int]],  # (x, y, w, h) on sheet
    out_json_path: Path,
    duration_ms: int = 100,
) -> None:
    """Generate TexturePacker Hash JSON format."""
    frames_dict: dict[str, dict[str, object]] = {}
    for name, (x, y, w, h) in zip(frame_names, frame_coords):
        frames_dict[name] = {
            "frame": {"x": x, "y": y, "w": w, "h": h},
            "rotated": False,
            "trimmed": False,
            "spriteSourceSize": {"x": 0, "y": 0, "w": w, "h": h},
            "sourceSize": {"w": w, "h": h},
            "duration": duration_ms,
        }

    atlas = {
        "frames": frames_dict,
        "meta": {
            "app": "agent-sprite-forge",
            "version": "1.0",
            "image": sheet_image_name,
            "format": "RGBA8888",
            "size": {"w": sheet_width, "h": sheet_height},
            "scale": "1",
        },
    }

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(atlas, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frames-dir",
        required=True,
        type=Path,
        help="Directory containing frame PNG images.",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Prefix filter for frame files (e.g. 'walk', 'idle').",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output .aseprite file path.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=100,
        help="Frame duration in milliseconds (default 100ms).",
    )
    parser.add_argument(
        "--tag",
        help="Optional animation tag name (e.g. 'walk').",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite output file.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    pattern = f"{args.prefix}*.png" if args.prefix else "*.png"
    frame_files = sorted(args.frames_dir.glob(pattern))
    if not frame_files:
        print(f"Error: no frame files found matching '{pattern}' in {args.frames_dir}", file=sys.stderr)
        return 1

    if args.out.exists() and not args.force:
        print(f"Error: output file already exists: {args.out} (use --force to overwrite)", file=sys.stderr)
        return 1

    frames = [Image.open(p).convert("RGBA") for p in frame_files]
    tags = [(args.tag, 0, len(frames) - 1)] if args.tag else None

    write_aseprite_file(
        frames=frames,
        out_path=args.out,
        frame_duration_ms=args.duration,
        tags=tags,
    )
    print(f"Exported {len(frames)} frames to Aseprite project: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

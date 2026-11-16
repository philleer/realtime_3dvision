#!/usr/bin/env python3
"""Visualize paired RGB-D frames and export a chronological MP4 video."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="RGB-D visualization video exporter")
    p.add_argument(
        "--data-root",
        type=Path,
        default=root / "data" / "jj_salon_room",
        help="Dataset root containing rgb/ and depth/",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=root / "tmp" / "rgbd_video",
        help="Output directory for preview frames and mp4",
    )
    p.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Output mp4 path (default: <outdir>/rgbd.mp4)",
    )
    p.add_argument("--fps", type=float, default=15.0, help="Output video FPS")
    p.add_argument("--max-depth-mm", type=float, default=5000.0, help="Depth colormap clip")
    p.add_argument("--stride", type=int, default=1, help="Frame stride")
    p.add_argument("--num-frames", type=int, default=0, help="<=0 means all frames")
    p.add_argument("--save-preview", type=int, default=0, help="Also save first N preview jpgs")
    return p.parse_args()


def collect_pairs(data_root: Path) -> list[tuple[Path, Path]]:
    rgb_dir = data_root / "rgb"
    depth_dir = data_root / "depth"
    if not rgb_dir.is_dir() or not depth_dir.is_dir():
        raise FileNotFoundError(f"missing rgb/ or depth/ under {data_root}")

    rgb_files = sorted(rgb_dir.glob("*.jpg")) + sorted(rgb_dir.glob("*.png"))
    pairs: list[tuple[Path, Path]] = []
    for rgb_path in rgb_files:
        depth_path = depth_dir / f"{rgb_path.stem}.png"
        if not depth_path.is_file():
            depth_path = depth_dir / f"{rgb_path.stem}.jpg"
        if depth_path.is_file():
            pairs.append((rgb_path, depth_path))
    if not pairs:
        raise FileNotFoundError(f"no rgb-depth pairs under {data_root}")
    return pairs


def colorize_depth(depth_mm: np.ndarray, max_depth_mm: float) -> np.ndarray:
    valid = depth_mm > 0
    depth_f = depth_mm.astype(np.float32)
    norm = np.zeros_like(depth_f, dtype=np.float32)
    if np.any(valid):
        clipped = np.clip(depth_f[valid], 0, max_depth_mm)
        norm[valid] = clipped / max_depth_mm
    u8 = (norm * 255.0).astype(np.uint8)
    colored = cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def make_panel(rgb_bgr: np.ndarray, depth_color: np.ndarray) -> np.ndarray:
    h, w = rgb_bgr.shape[:2]
    depth_resized = cv2.resize(depth_color, (w, h), interpolation=cv2.INTER_NEAREST)
    # Side-by-side RGB | depth
    panel = np.concatenate([rgb_bgr, depth_resized], axis=1)
    # Labels
    cv2.putText(panel, "RGB", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(panel, "Depth", (w + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    return panel


def main() -> None:
    args = parse_args()
    pairs = collect_pairs(args.data_root)
    pairs = pairs[:: max(args.stride, 1)]
    if args.num_frames > 0:
        pairs = pairs[: args.num_frames]

    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    video_path = (args.video or (outdir / "rgbd.mp4")).resolve()
    video_path.parent.mkdir(parents=True, exist_ok=True)

    first_rgb = cv2.imread(str(pairs[0][0]), cv2.IMREAD_COLOR)
    if first_rgb is None:
        raise RuntimeError(f"failed to read {pairs[0][0]}")
    h, w = first_rgb.shape[:2]
    out_w, out_h = w * 2, h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, args.fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open VideoWriter for {video_path}")

    print(f"[rgbd-video] pairs={len(pairs)} fps={args.fps} size={out_w}x{out_h}")
    print(f"[rgbd-video] out={video_path}")

    for i, (rgb_path, depth_path) in enumerate(pairs, start=1):
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if rgb is None or depth is None:
            print(f"  skip unreadable: {rgb_path.name}")
            continue
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        depth_color = colorize_depth(depth, args.max_depth_mm)
        panel = make_panel(rgb, depth_color)
        if panel.shape[1] != out_w or panel.shape[0] != out_h:
            panel = cv2.resize(panel, (out_w, out_h), interpolation=cv2.INTER_AREA)
        writer.write(panel)
        if args.save_preview > 0 and i <= args.save_preview:
            cv2.imwrite(str(outdir / f"preview_{rgb_path.stem}.jpg"), panel)
        if i == 1 or i % 100 == 0 or i == len(pairs):
            print(f"  [{i}/{len(pairs)}] {rgb_path.name}")

    writer.release()
    size_mb = video_path.stat().st_size / (1024 * 1024)
    print(f"[rgbd-video] done: {video_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()

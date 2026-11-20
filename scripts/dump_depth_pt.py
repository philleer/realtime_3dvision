#!/usr/bin/env python3
"""Dump Depth-Anything V2/V3 relative depth .npy for a RGB folder."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from depth_backend import add_depth_model_args, load_depth_backend  # noqa: E402


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Dump DA V2/V3 depth npy (GPU)")
    add_depth_model_args(p, default_version="v2")
    p.add_argument("--source", type=Path, default=root / "data" / "jj_salon_room" / "rgb")
    p.add_argument("--outdir", type=Path, default=root / "tmp" / "yolo_depth" / "depth_raw")
    p.add_argument("--num-frames", type=int, default=0)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--device", default="0")
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


def collect_images(source: Path, num_frames: int, stride: int) -> list[Path]:
    files = sorted(p for p in source.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    files = files[:: max(stride, 1)]
    if num_frames > 0:
        files = files[:num_frames]
    return files


def main() -> None:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    args = parse_args()
    images = collect_images(args.source, args.num_frames, args.stride)
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    backend = load_depth_backend(args.depth_version, args.depth_weights, args.device)
    print(
        f"[dump-depth] version={backend.version} weights={backend.weights} "
        f"device={backend.device} frames={len(images)} -> {outdir}"
    )

    for i, path in enumerate(images, 1):
        out_path = outdir / f"{path.stem}.npy"
        if args.skip_existing and out_path.is_file():
            if i == 1 or i % 100 == 0:
                print(f"  [{i}/{len(images)}] skip {path.name}", flush=True)
            continue
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"failed to read {path}")
        d = backend.predict(bgr)
        np.save(out_path, d.astype(np.float16))
        if i == 1 or i == len(images) or i % 50 == 0:
            print(f"  [{i}/{len(images)}] {path.name} depth[{d.min():.2f},{d.max():.2f}]", flush=True)

    print(f"[dump-depth] done -> {outdir}")


if __name__ == "__main__":
    main()

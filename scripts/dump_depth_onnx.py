#!/usr/bin/env python3
"""Dump Depth-Anything ONNX relative depth maps (.npy) for a RGB folder."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Dump DA ONNX depth npy for all RGB frames")
    p.add_argument("--source", type=Path, default=root / "data" / "jj_salon_room" / "rgb")
    p.add_argument(
        "--onnx",
        type=Path,
        default=root / "pretrained" / "depth_anything_v2_base_518.onnx",
    )
    p.add_argument("--outdir", type=Path, default=root / "tmp" / "yolo_depth" / "depth_raw")
    p.add_argument("--size", type=int, default=518)
    p.add_argument("--num-frames", type=int, default=0, help="<=0 all")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--skip-existing", action="store_true")
    return p.parse_args()


def collect_images(source: Path, num_frames: int, stride: int) -> list[Path]:
    files = sorted(p for p in source.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    files = files[:: max(stride, 1)]
    if num_frames > 0:
        files = files[:num_frames]
    return files


def preprocess(bgr: np.ndarray, size: int) -> np.ndarray:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_CUBIC)
    x = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    return np.ascontiguousarray(np.transpose(x, (2, 0, 1))[None, ...].astype(np.float32))


def main() -> None:
    args = parse_args()
    images = collect_images(args.source, args.num_frames, args.stride)
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = [
        ("CUDAExecutionProvider", {"device_id": args.device_id}),
        "CPUExecutionProvider",
    ]
    sess = ort.InferenceSession(str(args.onnx), sess_options=so, providers=providers)
    in_name = sess.get_inputs()[0].name
    print(f"[dump-depth] frames={len(images)} onnx={args.onnx} providers={sess.get_providers()} -> {outdir}")

    for i, path in enumerate(images, 1):
        out_path = outdir / f"{path.stem}.npy"
        if args.skip_existing and out_path.is_file():
            if i % 100 == 0 or i == 1:
                print(f"  [{i}/{len(images)}] skip {path.name}", flush=True)
            continue
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"failed to read {path}")
        h, w = bgr.shape[:2]
        x = preprocess(bgr, args.size)
        pred = sess.run(None, {in_name: x})[0]
        # squeeze to HxW model res then upsample to original
        d = np.squeeze(pred).astype(np.float32)
        if d.ndim != 2:
            raise RuntimeError(f"unexpected depth rank: {d.shape}")
        d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
        np.save(out_path, d.astype(np.float16))
        if i == 1 or i == len(images) or i % 50 == 0:
            print(f"  [{i}/{len(images)}] {path.name} depth[{d.min():.2f},{d.max():.2f}]", flush=True)

    print(f"[dump-depth] done -> {outdir}")


if __name__ == "__main__":
    main()

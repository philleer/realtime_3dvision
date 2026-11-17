#!/usr/bin/env python3
"""Back-project depth maps to colored point clouds (camera frame = world frame)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Depth -> PLY point cloud (cam=world origin)")
    p.add_argument("--rgb", type=Path, default=None, help="RGB image (for colors); or use --source dir")
    p.add_argument("--depth", type=Path, default=None, help="Depth: uint16 png (mm) or float .npy")
    p.add_argument(
        "--source",
        type=Path,
        default=root / "data" / "jj_salon_room" / "rgb",
        help="RGB directory when batching",
    )
    p.add_argument(
        "--depth-dir",
        type=Path,
        default=None,
        help="Depth dir (.png/.npy). Default: dataset depth/ for png, else tmp depth npy",
    )
    p.add_argument(
        "--calib",
        type=Path,
        default=root / "data" / "jj_salon_room" / "scan-003" / "calib.json",
    )
    p.add_argument("--outdir", type=Path, default=root / "tmp" / "depth_pointcloud")
    p.add_argument("--num-frames", type=int, default=1)
    p.add_argument("--stride", type=int, default=50)
    p.add_argument(
        "--camera",
        choices=["color", "depth"],
        default="color",
        help="Which intrinsics to use (color for predicted RGB-aligned depth)",
    )
    p.add_argument(
        "--depth-unit",
        choices=["auto", "m", "mm", "relative"],
        default="auto",
        help="Depth value unit. auto: u16->mm, float npy->relative (apply --depth-scale)",
    )
    p.add_argument(
        "--depth-scale",
        type=float,
        default=1.0,
        help="Multiply depth after unit conversion (useful for relative Depth Anything)",
    )
    p.add_argument(
        "--metricize-gt",
        action="store_true",
        help="Scale relative depth so median matches GT depth (mm/1000) on valid pixels",
    )
    p.add_argument("--gt-depth-dir", type=Path, default=root / "data" / "jj_salon_room" / "depth")
    p.add_argument("--max-depth-m", type=float, default=8.0, help="Drop points beyond this (metres)")
    p.add_argument("--min-depth-m", type=float, default=0.05)
    p.add_argument("--stride-px", type=int, default=2, help="Pixel stride when sampling points")
    p.add_argument("--voxel", type=float, default=0.0, help="Optional voxel downsample size (m), 0=off")
    return p.parse_args()


def load_calib(path: Path, camera: str) -> tuple[float, float, float, float, int, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cam = data[camera]
    K = np.asarray(cam["K"], dtype=np.float64)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    return fx, fy, cx, cy, int(cam["width"]), int(cam["height"])


def scale_intrinsics(
    fx: float, fy: float, cx: float, cy: float, w0: int, h0: int, w: int, h: int
) -> tuple[float, float, float, float]:
    sx, sy = w / w0, h / h0
    return fx * sx, fy * sy, cx * sx, cy * sy


def load_depth(path: Path, unit: str) -> tuple[np.ndarray, str]:
    """Return depth in metres (or relative*scale later) and resolved unit tag."""
    if path.suffix.lower() == ".npy":
        d = np.load(path).astype(np.float32)
        resolved = "relative" if unit == "auto" else unit
        if resolved == "mm":
            d = d / 1000.0
            resolved = "m"
        elif resolved == "relative":
            pass
        elif resolved == "m":
            pass
        return d, resolved

    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise RuntimeError(f"failed to read depth: {path}")
    d = raw.astype(np.float32)
    resolved = "mm" if unit == "auto" else unit
    if resolved == "mm":
        d = d / 1000.0
        resolved = "m"
    elif resolved == "relative":
        pass
    return d, resolved


def resize_depth_to_rgb(depth: np.ndarray, wh: tuple[int, int]) -> np.ndarray:
    w, h = wh
    if depth.shape[0] == h and depth.shape[1] == w:
        return depth
    return cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)


def metricize_with_gt(pred_m: np.ndarray, gt_path: Path, rgb_wh: tuple[int, int]) -> tuple[np.ndarray, str]:
    """Align relative Depth-Anything output to GT metres.

    HF Depth Anything ``predicted_depth`` is disparity-like (larger = closer),
    so a naive ``pred * scale`` inverts geometry and bends planar floors.
    We detect the relationship via correlation and use either:
      - disparity:  metric = median(gt * pred) / pred
      - depth-like: metric = median(gt / pred) * pred
    """
    gt, _ = load_depth(gt_path, "auto")
    gt = resize_depth_to_rgb(gt, rgb_wh)
    pred = resize_depth_to_rgb(pred_m, rgb_wh)
    valid = (gt > 0.2) & (gt < 5.0) & np.isfinite(pred) & (pred > 1e-3)
    if valid.sum() < 100:
        return pred, "relative(unscaled)"
    corr = float(np.corrcoef(pred[valid], gt[valid])[0, 1])
    if corr < 0.0:
        # disparity / inverse-depth style
        scale = float(np.median(gt[valid] * pred[valid]))
        return scale / np.clip(pred, 1e-6, None), f"m(disp/gt,corr={corr:.2f})"
    scale = float(np.median(gt[valid] / pred[valid]))
    return pred * scale, f"m(depth*s,corr={corr:.2f})"


def backproject(
    depth_m: np.ndarray,
    rgb_bgr: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    min_z: float,
    max_z: float,
    stride_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = depth_m.shape
    assert rgb_bgr.shape[0] == h and rgb_bgr.shape[1] == w
    us = np.arange(0, w, stride_px)
    vs = np.arange(0, h, stride_px)
    uu, vv = np.meshgrid(us, vs)
    z = depth_m[vv, uu]
    mask = np.isfinite(z) & (z >= min_z) & (z <= max_z) & (z > 0)
    uu, vv, z = uu[mask].astype(np.float32), vv[mask].astype(np.float32), z[mask].astype(np.float32)
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy
    # Camera frame: X right, Y down, Z forward (OpenCV). World = camera (identity).
    pts = np.stack([x, y, z], axis=1)
    cols = rgb_bgr[vv.astype(np.int32), uu.astype(np.int32)][:, ::-1]  # BGR->RGB
    return pts, cols


def voxel_downsample(pts: np.ndarray, cols: np.ndarray, voxel: float) -> tuple[np.ndarray, np.ndarray]:
    if voxel <= 0 or len(pts) == 0:
        return pts, cols
    keys = np.floor(pts / voxel).astype(np.int32)
    # unique rows
    _, idx = np.unique(keys, axis=0, return_index=True)
    return pts[idx], cols[idx]


def write_ply(path: Path, pts: np.ndarray, cols: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(pts)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    # structured array
    vert = np.empty(
        n,
        dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")],
    )
    vert["x"], vert["y"], vert["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    vert["r"], vert["g"], vert["b"] = cols[:, 0], cols[:, 1], cols[:, 2]
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(vert.tobytes())


def collect_jobs(args: argparse.Namespace) -> list[tuple[Path, Path]]:
    if args.rgb is not None and args.depth is not None:
        return [(args.rgb, args.depth)]

    rgb_dir = args.source
    files = sorted(p for p in rgb_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    files = files[:: max(args.stride, 1)]
    if args.num_frames > 0:
        files = files[: args.num_frames]

    depth_dir = args.depth_dir
    if depth_dir is None:
        # Prefer predicted npy if present, else GT depth png
        pred = args.outdir.parent / "yolo_depth" / "depth_raw"
        pred2 = args.outdir.parent / "depth_anything" / "raw"
        if pred.is_dir() and any(pred.glob("*.npy")):
            depth_dir = pred
        elif pred2.is_dir() and any(pred2.glob("*.npy")):
            depth_dir = pred2
        else:
            depth_dir = args.gt_depth_dir

    jobs: list[tuple[Path, Path]] = []
    for rgb in files:
        for ext in (".npy", ".png"):
            cand = depth_dir / f"{rgb.stem}{ext}"
            if cand.is_file():
                jobs.append((rgb, cand))
                break
        else:
            raise FileNotFoundError(f"no depth for {rgb.name} under {depth_dir}")
    return jobs


def main() -> None:
    args = parse_args()
    jobs = collect_jobs(args)
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    fx0, fy0, cx0, cy0, w0, h0 = load_calib(args.calib, args.camera)
    print(f"[pc] calib={args.calib} camera={args.camera} K0=({fx0:.1f},{fy0:.1f},{cx0:.1f},{cy0:.1f}) {w0}x{h0}")
    print(f"[pc] jobs={len(jobs)} cam=world(I) out={outdir}")

    for i, (rgb_path, depth_path) in enumerate(jobs, 1):
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"failed to read rgb: {rgb_path}")
        h, w = bgr.shape[:2]
        depth, unit = load_depth(depth_path, args.depth_unit)
        depth = resize_depth_to_rgb(depth, (w, h))
        if unit == "relative":
            depth = depth * float(args.depth_scale)
            if args.metricize_gt:
                gt = args.gt_depth_dir / f"{rgb_path.stem}.png"
                if gt.is_file():
                    depth, unit = metricize_with_gt(depth, gt, (w, h))
                else:
                    unit = "relative*scale"
            else:
                unit = "relative*scale"

        fx, fy, cx, cy = scale_intrinsics(fx0, fy0, cx0, cy0, w0, h0, w, h)
        # If using depth camera intrinsics but depth resized to RGB, K already scaled above.

        pts, cols = backproject(
            depth, bgr, fx, fy, cx, cy, args.min_depth_m, args.max_depth_m, args.stride_px
        )
        pts, cols = voxel_downsample(pts, cols, args.voxel)
        out_ply = outdir / f"{rgb_path.stem}.ply"
        write_ply(out_ply, pts, cols)
        print(
            f"  [{i}/{len(jobs)}] {rgb_path.name}: depth={depth_path.name} unit={unit} "
            f"points={len(pts)} z[{pts[:,2].min():.3f},{pts[:,2].max():.3f}] -> {out_ply.name}"
        )

    # Sidecar: note that world = camera
    (outdir / "README.txt").write_text(
        "Point clouds are in the camera coordinate frame.\n"
        "World origin = camera origin (T_world_cam = I).\n"
        "Axes: OpenCV camera (X right, Y down, Z forward).\n",
        encoding="utf-8",
    )
    print(f"[pc] done -> {outdir}")


if __name__ == "__main__":
    main()

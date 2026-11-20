#!/usr/bin/env python3
"""Benchmark YOLO26-SEG and Depth Anything: latency + CPU/GPU utilization."""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import sys

import cv2
import numpy as np
import psutil
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from depth_backend import add_depth_model_args, load_depth_backend  # noqa: E402


@dataclass
class SampleStats:
    name: str
    n_frames: int
    warmup: int
    latency_ms: list[float] = field(default_factory=list)
    cpu_proc_pct: list[float] = field(default_factory=list)
    cpu_sys_pct: list[float] = field(default_factory=list)
    gpu_util_pct: list[float] = field(default_factory=list)
    gpu_mem_used_mb: list[float] = field(default_factory=list)

    def summary(self) -> dict:
        def agg(xs: list[float]) -> dict:
            if not xs:
                return {"mean": None, "p50": None, "p90": None, "min": None, "max": None}
            xs_sorted = sorted(xs)
            p90_i = min(len(xs_sorted) - 1, int(round(0.9 * (len(xs_sorted) - 1))))
            return {
                "mean": statistics.fmean(xs),
                "p50": statistics.median(xs),
                "p90": xs_sorted[p90_i],
                "min": min(xs),
                "max": max(xs),
            }

        return {
            "name": self.name,
            "n_frames": self.n_frames,
            "warmup": self.warmup,
            "latency_ms": agg(self.latency_ms),
            "fps_mean": (1000.0 / statistics.fmean(self.latency_ms)) if self.latency_ms else None,
            "cpu_proc_pct": agg(self.cpu_proc_pct),
            "cpu_sys_pct": agg(self.cpu_sys_pct),
            "gpu_util_pct": agg(self.gpu_util_pct),
            "gpu_mem_used_mb": agg(self.gpu_mem_used_mb),
        }


class ResourceMonitor:
    """Background sampler for process CPU%, system CPU%, GPU util/mem."""

    def __init__(self, gpu_index: int, interval_s: float = 0.05) -> None:
        self.gpu_index = gpu_index
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.cpu_proc: list[float] = []
        self.cpu_sys: list[float] = []
        self.gpu_util: list[float] = []
        self.gpu_mem: list[float] = []
        self._proc = psutil.Process(os.getpid())
        self._handle = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        except Exception:
            self._pynvml = None

    def start(self) -> None:
        self.cpu_proc.clear()
        self.cpu_sys.clear()
        self.gpu_util.clear()
        self.gpu_mem.clear()
        self._proc.cpu_percent(None)  # prime
        psutil.cpu_percent(None)
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return {
            "cpu_proc_pct": list(self.cpu_proc),
            "cpu_sys_pct": list(self.cpu_sys),
            "gpu_util_pct": list(self.gpu_util),
            "gpu_mem_used_mb": list(self.gpu_mem),
        }

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.cpu_proc.append(self._proc.cpu_percent(None))
            self.cpu_sys.append(psutil.cpu_percent(None))
            if self._handle is not None:
                util = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                mem = self._pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                self.gpu_util.append(float(util.gpu))
                self.gpu_mem.append(mem.used / (1024**2))
            time.sleep(self.interval_s)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Seg / depth / joint latency benchmark")
    p.add_argument("--source", type=Path, default=root / "data" / "jj_salon_room" / "rgb")
    p.add_argument("--yolo-weights", type=Path, default=root / "pretrained" / "yolo26m-seg.pt")
    add_depth_model_args(p, default_version="v2")
    p.add_argument("--outdir", type=Path, default=root / "tmp" / "bench_detseg_depth")
    p.add_argument("--num-frames", type=int, default=50)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--device", default="0")
    return p.parse_args()


def collect_images(source: Path, num_frames: int, stride: int) -> list[Path]:
    files = sorted(p for p in source.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    sampled = files[:: max(stride, 1)]
    if num_frames > 0:
        sampled = sampled[:num_frames]
    if not sampled:
        raise FileNotFoundError(f"no images under {source}")
    return sampled


def sync_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def collect_system_info(gpu_index: int) -> dict:
    """CPU / RAM / GPU inventory for bench_report.json."""
    uname = os.uname()
    vm = psutil.virtual_memory()
    cpu_freq = psutil.cpu_freq()
    info: dict = {
        "host": uname.nodename,
        "kernel": uname.release,
        "cpu": {
            "model": _cpu_model_name(),
            "physical_cores": psutil.cpu_count(logical=False),
            "logical_cores": psutil.cpu_count(logical=True),
            "freq_mhz": {
                "current": cpu_freq.current if cpu_freq else None,
                "min": cpu_freq.min if cpu_freq else None,
                "max": cpu_freq.max if cpu_freq else None,
            },
        },
        "memory": {
            "total_gb": round(vm.total / (1024**3), 2),
            "available_gb": round(vm.available / (1024**3), 2),
            "used_gb": round(vm.used / (1024**3), 2),
        },
        "gpus": [],
        "selected_gpu_index": gpu_index,
    }
    try:
        import pynvml

        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        driver = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(driver, bytes):
            driver = driver.decode()
        info["nvidia_driver"] = driver
        for i in range(n):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            try:
                cc = pynvml.nvmlDeviceGetCudaComputeCapability(h)
                compute_cap = f"{cc[0]}.{cc[1]}"
            except Exception:
                compute_cap = None
            info["gpus"].append(
                {
                    "index": i,
                    "name": name,
                    "memory_total_mb": round(mem.total / (1024**2), 1),
                    "memory_free_mb": round(mem.free / (1024**2), 1),
                    "memory_used_mb": round(mem.used / (1024**2), 1),
                    "compute_capability": compute_cap,
                }
            )
    except Exception as e:
        info["gpu_query_error"] = str(e)
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info["gpus"].append(
                    {
                        "index": i,
                        "name": props.name,
                        "memory_total_mb": round(props.total_memory / (1024**2), 1),
                        "compute_capability": f"{props.major}.{props.minor}",
                    }
                )
    return info


def _cpu_model_name() -> str | None:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or None


def load_bgr_batch(paths: list[Path]) -> list[np.ndarray]:
    imgs = []
    for p in paths:
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None:
            raise RuntimeError(f"failed to read {p}")
        imgs.append(im)
    return imgs


def run_seg_only(model, images: list[np.ndarray], args, gpu_index: int) -> SampleStats:
    stats = SampleStats("seg_only_yolo26m_seg", n_frames=len(images), warmup=args.warmup)
    # warmup
    for im in images[: args.warmup]:
        model.predict(im, imgsz=args.imgsz, conf=args.conf, iou=args.iou, device=args.device, verbose=False)
    sync_cuda()

    mon = ResourceMonitor(gpu_index)
    mon.start()
    for im in images:
        sync_cuda()
        t0 = time.perf_counter()
        model.predict(im, imgsz=args.imgsz, conf=args.conf, iou=args.iou, device=args.device, verbose=False)
        sync_cuda()
        stats.latency_ms.append((time.perf_counter() - t0) * 1000.0)
    samples = mon.stop()
    stats.cpu_proc_pct = samples["cpu_proc_pct"]
    stats.cpu_sys_pct = samples["cpu_sys_pct"]
    stats.gpu_util_pct = samples["gpu_util_pct"]
    stats.gpu_mem_used_mb = samples["gpu_mem_used_mb"]
    return stats


def run_depth_only(backend, images: list[np.ndarray], warmup: int, gpu_index: int) -> SampleStats:
    stats = SampleStats(
        f"depth_only_depth_anything_{backend.version}",
        n_frames=len(images),
        warmup=warmup,
    )

    for im in images[:warmup]:
        _ = backend.predict(im)
    sync_cuda()

    mon = ResourceMonitor(gpu_index)
    mon.start()
    for im in images:
        sync_cuda()
        t0 = time.perf_counter()
        _ = backend.predict(im)
        sync_cuda()
        stats.latency_ms.append((time.perf_counter() - t0) * 1000.0)
    samples = mon.stop()
    stats.cpu_proc_pct = samples["cpu_proc_pct"]
    stats.cpu_sys_pct = samples["cpu_sys_pct"]
    stats.gpu_util_pct = samples["gpu_util_pct"]
    stats.gpu_mem_used_mb = samples["gpu_mem_used_mb"]
    return stats


def run_joint(yolo, backend, images: list[np.ndarray], args, gpu_index: int) -> SampleStats:
    stats = SampleStats("joint_seg_then_depth", n_frames=len(images), warmup=args.warmup)

    def infer_one(bgr: np.ndarray) -> None:
        yolo.predict(bgr, imgsz=args.imgsz, conf=args.conf, iou=args.iou, device=args.device, verbose=False)
        _ = backend.predict(bgr)

    for im in images[: args.warmup]:
        infer_one(im)
    sync_cuda()

    mon = ResourceMonitor(gpu_index)
    mon.start()
    for im in images:
        sync_cuda()
        t0 = time.perf_counter()
        infer_one(im)
        sync_cuda()
        stats.latency_ms.append((time.perf_counter() - t0) * 1000.0)
    samples = mon.stop()
    stats.cpu_proc_pct = samples["cpu_proc_pct"]
    stats.cpu_sys_pct = samples["cpu_sys_pct"]
    stats.gpu_util_pct = samples["gpu_util_pct"]
    stats.gpu_mem_used_mb = samples["gpu_mem_used_mb"]
    return stats


def main() -> None:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    args = parse_args()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    gpu_index = int(str(args.device).split(":")[-1]) if str(args.device) not in {"cpu", "CPU"} else 0
    device = torch.device(f"cuda:{gpu_index}" if torch.cuda.is_available() else "cpu")

    paths = collect_images(args.source, args.num_frames, args.stride)
    # include warmup frames from the same pool (reuse first images if short)
    need = max(args.num_frames, args.warmup)
    while len(paths) < need:
        paths = paths + paths
    paths = paths[: args.num_frames]
    images = load_bgr_batch(paths)
    h, w = images[0].shape[:2]

    env = {
        **collect_system_info(gpu_index),
        "software": {
            "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn_enabled": bool(torch.backends.cudnn.enabled),
            "device": str(device),
        },
        "input": {
            "dataset": str(args.source),
            "n_frames": len(images),
            "stride": args.stride,
            "resolution_hw": [h, w],
            "channels": 3,
            "dtype": "uint8",
            "format": paths[0].suffix.lower(),
            "colorspace": "BGR(OpenCV)/RGB(Depth)",
        },
        "yolo": {
            "weights": str(args.yolo_weights),
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
        },
        "depth": {
            "version": args.depth_version,
            "weights": args.depth_weights,
        },
    }

    from ultralytics import YOLO

    print("[bench] loading YOLO...", flush=True)
    yolo = YOLO(str(args.yolo_weights))
    print(f"[bench] loading Depth Anything {args.depth_version}...", flush=True)
    depth_backend = load_depth_backend(args.depth_version, args.depth_weights, device)
    env["depth"]["weights"] = depth_backend.weights
    env["depth"]["version"] = depth_backend.version

    print("[bench] (1) seg only", flush=True)
    seg = run_seg_only(yolo, images, args, gpu_index)
    torch.cuda.empty_cache()

    print("[bench] (2) depth only", flush=True)
    depth = run_depth_only(depth_backend, images, args.warmup, gpu_index)
    torch.cuda.empty_cache()

    print("[bench] (3) joint seg+depth", flush=True)
    joint = run_joint(yolo, depth_backend, images, args, gpu_index)

    report = {
        "env": env,
        "results": {
            "seg_only": seg.summary(),
            "depth_only": depth.summary(),
            "joint": joint.summary(),
        },
    }
    out_json = outdir / "bench_report.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # human-readable table
    lines = ["# bench_detseg_depth", ""]
    for key in ("seg_only", "depth_only", "joint"):
        s = report["results"][key]
        lat = s["latency_ms"]
        lines.append(f"## {s['name']}")
        lines.append(
            f"- latency ms: mean={lat['mean']:.2f}  p50={lat['p50']:.2f}  "
            f"p90={lat['p90']:.2f}  min={lat['min']:.2f}  max={lat['max']:.2f}"
        )
        lines.append(f"- fps_mean: {s['fps_mean']:.2f}" if s["fps_mean"] else "- fps_mean: n/a")
        lines.append(
            f"- CPU proc%: mean={s['cpu_proc_pct']['mean']:.1f}  "
            f"sys%: mean={s['cpu_sys_pct']['mean']:.1f}"
        )
        lines.append(
            f"- GPU util%: mean={s['gpu_util_pct']['mean']:.1f}  "
            f"mem_MB mean={s['gpu_mem_used_mb']['mean']:.0f}"
        )
        lines.append("")
    text = "\n".join(lines)
    (outdir / "bench_report.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"[bench] wrote {out_json}")


if __name__ == "__main__":
    main()

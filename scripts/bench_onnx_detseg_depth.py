#!/usr/bin/env python3
"""Benchmark ONNX YOLO26-SEG + Depth Anything: latency + CPU/GPU utilization."""

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

import cv2
import numpy as np
import onnxruntime as ort
import psutil
import torch
from ultralytics import YOLO


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
        self._proc.cpu_percent(None)
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
    p = argparse.ArgumentParser(description="ONNX seg/depth/joint benchmark")
    p.add_argument("--source", type=Path, default=root / "data" / "jj_salon_room" / "rgb")
    p.add_argument(
        "--yolo-onnx",
        type=Path,
        default=root / "pretrained" / "yolo26m-seg_736x1280.onnx",
    )
    p.add_argument(
        "--depth-onnx",
        type=Path,
        default=root / "pretrained" / "depth_anything_v2_base_518.onnx",
    )
    p.add_argument("--outdir", type=Path, default=root / "tmp" / "bench_onnx_detseg_depth")
    p.add_argument("--num-frames", type=int, default=50)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument(
        "--imgsz",
        type=int,
        nargs="+",
        default=[736, 1280],
        help="YOLO imgsz: one int or H W (default 736 1280 to match 1080p letterbox)",
    )
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--depth-size", type=int, default=518)
    p.add_argument("--device", default="0", help="GPU index for CUDA EP / YOLO")
    p.add_argument(
        "--providers",
        default="cuda",
        choices=["cuda", "tensorrt", "cpu"],
        help="ORT execution provider preference",
    )
    return p.parse_args()


def _cpu_model_name() -> str | None:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or None


def collect_system_info(gpu_index: int) -> dict:
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
    return info


def collect_images(source: Path, num_frames: int, stride: int) -> list[Path]:
    files = sorted(p for p in source.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    sampled = files[:: max(stride, 1)][:num_frames] if num_frames > 0 else files[:: max(stride, 1)]
    if not sampled:
        raise FileNotFoundError(f"no images under {source}")
    return sampled


def load_bgr_batch(paths: list[Path]) -> list[np.ndarray]:
    imgs = []
    for p in paths:
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None:
            raise RuntimeError(f"failed to read {p}")
        imgs.append(im)
    return imgs


def make_ort_session(onnx_path: Path, providers_pref: str, gpu_index: int) -> ort.InferenceSession:
    available = ort.get_available_providers()
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers: list = []
    if providers_pref == "tensorrt" and "TensorrtExecutionProvider" in available:
        providers.append(
            (
                "TensorrtExecutionProvider",
                {
                    "device_id": gpu_index,
                    "trt_fp16_enable": True,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": str(onnx_path.parent / "trt_cache"),
                },
            )
        )
    if providers_pref in {"cuda", "tensorrt"} and "CUDAExecutionProvider" in available:
        providers.append(("CUDAExecutionProvider", {"device_id": gpu_index}))
    providers.append("CPUExecutionProvider")
    print(f"[ort] {onnx_path.name} providers={providers}")
    return ort.InferenceSession(str(onnx_path), sess_options=so, providers=providers)


def preprocess_depth(bgr: np.ndarray, size: int) -> np.ndarray:
    """ImageNet-normalize RGB -> NCHW float32 for Depth Anything."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_CUBIC)
    x = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))[None, ...].astype(np.float32)
    return np.ascontiguousarray(x)


def _yolo_imgsz(args) -> int | list[int]:
    imgsz = args.imgsz
    if isinstance(imgsz, list):
        return imgsz[0] if len(imgsz) == 1 else imgsz
    return imgsz


def run_seg_only(yolo: YOLO, images: list[np.ndarray], args, gpu_index: int) -> SampleStats:
    stats = SampleStats("onnx_seg_only_yolo26m_seg", n_frames=len(images), warmup=args.warmup)
    imgsz = _yolo_imgsz(args)
    for im in images[: args.warmup]:
        yolo.predict(im, imgsz=imgsz, conf=args.conf, iou=args.iou, device=args.device, verbose=False)
    mon = ResourceMonitor(gpu_index)
    mon.start()
    for im in images:
        t0 = time.perf_counter()
        yolo.predict(im, imgsz=imgsz, conf=args.conf, iou=args.iou, device=args.device, verbose=False)
        stats.latency_ms.append((time.perf_counter() - t0) * 1000.0)
    samples = mon.stop()
    stats.cpu_proc_pct = samples["cpu_proc_pct"]
    stats.cpu_sys_pct = samples["cpu_sys_pct"]
    stats.gpu_util_pct = samples["gpu_util_pct"]
    stats.gpu_mem_used_mb = samples["gpu_mem_used_mb"]
    return stats


def run_depth_only(session: ort.InferenceSession, images: list[np.ndarray], size: int, warmup: int, gpu_index: int) -> SampleStats:
    stats = SampleStats("onnx_depth_only_depth_anything_base", n_frames=len(images), warmup=warmup)
    in_name = session.get_inputs()[0].name

    def infer_one(bgr: np.ndarray) -> None:
        x = preprocess_depth(bgr, size)
        session.run(None, {in_name: x})

    for im in images[:warmup]:
        infer_one(im)
    mon = ResourceMonitor(gpu_index)
    mon.start()
    for im in images:
        t0 = time.perf_counter()
        infer_one(im)
        stats.latency_ms.append((time.perf_counter() - t0) * 1000.0)
    samples = mon.stop()
    stats.cpu_proc_pct = samples["cpu_proc_pct"]
    stats.cpu_sys_pct = samples["cpu_sys_pct"]
    stats.gpu_util_pct = samples["gpu_util_pct"]
    stats.gpu_mem_used_mb = samples["gpu_mem_used_mb"]
    return stats


def run_joint(
    yolo: YOLO,
    session: ort.InferenceSession,
    images: list[np.ndarray],
    args,
    gpu_index: int,
) -> SampleStats:
    stats = SampleStats("onnx_joint_seg_then_depth", n_frames=len(images), warmup=args.warmup)
    in_name = session.get_inputs()[0].name

    imgsz = _yolo_imgsz(args)

    def infer_one(bgr: np.ndarray) -> None:
        yolo.predict(bgr, imgsz=imgsz, conf=args.conf, iou=args.iou, device=args.device, verbose=False)
        x = preprocess_depth(bgr, args.depth_size)
        session.run(None, {in_name: x})

    for im in images[: args.warmup]:
        infer_one(im)
    mon = ResourceMonitor(gpu_index)
    mon.start()
    for im in images:
        t0 = time.perf_counter()
        infer_one(im)
        stats.latency_ms.append((time.perf_counter() - t0) * 1000.0)
    samples = mon.stop()
    stats.cpu_proc_pct = samples["cpu_proc_pct"]
    stats.cpu_sys_pct = samples["cpu_sys_pct"]
    stats.gpu_util_pct = samples["gpu_util_pct"]
    stats.gpu_mem_used_mb = samples["gpu_mem_used_mb"]
    return stats


def main() -> None:
    args = parse_args()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    gpu_index = int(str(args.device).split(":")[-1])

    for path in (args.yolo_onnx, args.depth_onnx):
        if not path.is_file():
            raise FileNotFoundError(f"missing ONNX: {path}")

    paths = collect_images(args.source, args.num_frames, args.stride)
    images = load_bgr_batch(paths)
    h, w = images[0].shape[:2]

    print("[bench-onnx] load YOLO ONNX via ultralytics...", flush=True)
    yolo = YOLO(str(args.yolo_onnx))
    print("[bench-onnx] load Depth ORT session...", flush=True)
    depth_sess = make_ort_session(args.depth_onnx, args.providers, gpu_index)

    print("[bench-onnx] (1) seg only", flush=True)
    seg = run_seg_only(yolo, images, args, gpu_index)
    print("[bench-onnx] (2) depth only", flush=True)
    depth = run_depth_only(depth_sess, images, args.depth_size, args.warmup, gpu_index)
    print("[bench-onnx] (3) joint", flush=True)
    joint = run_joint(yolo, depth_sess, images, args, gpu_index)

    report = {
        "backend": "onnxruntime",
        "ort_version": ort.__version__,
        "available_providers": ort.get_available_providers(),
        "requested_providers": args.providers,
        "depth_session_providers": depth_sess.get_providers(),
        "env": {
            **collect_system_info(gpu_index),
            "software": {
                "python": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
                "platform": platform.platform(),
                "torch": torch.__version__,
                "onnxruntime": ort.__version__,
            },
            "input": {
                "n_frames": len(images),
                "stride": args.stride,
                "resolution_hw": [h, w],
                "channels": 3,
                "dtype": "uint8",
                "format": paths[0].suffix.lower(),
            },
            "yolo_onnx": str(args.yolo_onnx),
            "depth_onnx": str(args.depth_onnx),
            "imgsz": args.imgsz,
            "depth_size": args.depth_size,
        },
        "results": {
            "seg_only": seg.summary(),
            "depth_only": depth.summary(),
            "joint": joint.summary(),
        },
    }
    (outdir / "bench_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# bench_onnx_detseg_depth", ""]
    for key in ("seg_only", "depth_only", "joint"):
        s = report["results"][key]
        lat = s["latency_ms"]
        lines += [
            f"## {s['name']}",
            (
                f"- latency ms: mean={lat['mean']:.2f}  p50={lat['p50']:.2f}  "
                f"p90={lat['p90']:.2f}  min={lat['min']:.2f}  max={lat['max']:.2f}"
            ),
            f"- fps_mean: {s['fps_mean']:.2f}",
            (
                f"- CPU proc%: mean={s['cpu_proc_pct']['mean']:.1f}  "
                f"sys%: mean={s['cpu_sys_pct']['mean']:.1f}"
            ),
            (
                f"- GPU util%: mean={s['gpu_util_pct']['mean']:.1f}  "
                f"mem_MB mean={s['gpu_mem_used_mb']['mean']:.0f}"
            ),
            "",
        ]
    text = "\n".join(lines)
    (outdir / "bench_report.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

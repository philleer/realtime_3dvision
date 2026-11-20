# realtime_3dvision

室内实时三维视觉原型：RGB 实例分割 + 单目深度 → 点云 / 俯视 BEV。

- **算力**：远端 GPU 跑推理；本机影子仓编辑，经 `scripts/host/sync_*.sh` 同步（主机写在本地 `.remote_host`，不入库）
- **环境**：`conda activate yolov11`
- **数据**：`data/jj_salon_room`（1080p RGB + GT depth mm + 标定）
- **产物**：一律写 `tmp/`（不入库）；权重在 `pretrained/`

选型细节与问题归档见 [`docs/technical_report.md`](docs/technical_report.md)、[`docs/PASs.md`](docs/PASs.md)。

---

## Pipeline

```
RGB (1920×1080)
        │
        ├──────────────────────┐
        ▼                      ▼
 ┌──────────────┐      ┌───────────────────┐
 │ YOLO26-SEG   │      │ Depth Anything V2 │
 │ imgsz≈736×1280│      │ 518×518 (HF)      │
 └──────┬───────┘      └─────────┬─────────┘
        │ masks / cls            │ relative / disparity
        │                        │ metricize vs GT
        └──────────┬─────────────┘
                   ▼
         图像空间对齐融合
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
  depth → 点云 PLY     seg+depth → BEV
  (camera = world)     (majority / OBB)
```

| 阶段 | 做什么 | 主脚本 | 默认产物 |
|------|--------|--------|----------|
| 1. DetSeg | 实例分割 | `scripts/test_yolo_detseg.py` | `tmp/yolo_detseg/` |
| 2. Depth | 单目深度（**默认 V2 Base**） | `scripts/test_depth_anything.py` | `tmp/depth_anything/` |
| 3. Joint viz | RGB \| SEG \| Depth | `scripts/viz_yolo_depth.py` | `tmp/yolo_depth/` |
| 4. Depth dump | 全序列相对深度 `.npy` | `dump_depth_pt.py` / `dump_depth_onnx.py` | `tmp/yolo_depth/depth_raw/` |
| 5. Point cloud | 反投影相机系 PLY | `scripts/depth_to_pointcloud.py` | `tmp/depth_pointcloud/` |
| 6. BEV | 颜色点投影俯视图 | `scripts/seg_to_bev.py` | `tmp/seg_bev/` |
| 7. BEV-occ | 语义 majority + 凸包/OBB（仍在 `test-depth`） | `seg_to_bev_occ.py`（未合入 develop） | `tmp/seg_bev_occ/` |

度量深度：若 `corr(pred, GT)<0`（disparity 风格），用 `metric = median(gt·pred) / pred`，避免地面挠曲。

---

## Modules

### 感知

| Module | 说明 | 入口 |
|--------|------|------|
| **YOLO26-SEG** | 主分割路径，`retina_masks` 对齐原图 | `test_yolo_detseg.py`，权重 `pretrained/yolo26m-seg.pt` |
| **Depth backend** | V2（默认）/ V3（可选）/ V1（回滚）统一加载 | `depth_backend.py` |
| Depth Anything **V2 Base** | 主深度模型，HF Transformers | `depth-anything/Depth-Anything-V2-Base-hf` |
| Depth Anything V3 | 可选更高精度；需 `pip install depth-anything-3` | `--depth-version v3` |
| DETR / RF-DETR | 历史对比，非主路径 | `test_detr_seg.py` |

```bash
# 深度版本（所有 PT 深度脚本通用）
--depth-version v2|v3|v1
--depth-weights <HF_id_or_local_dir>   # 默认随 version
```

### 几何与可视化

| Module | 说明 | 入口 |
|--------|------|------|
| **Point cloud** | 针孔反投影；可选 GT metricize、stride/voxel | `depth_to_pointcloud.py` |
| **BEV（颜色点）** | mask 颜色随点投到俯视栅格 | `seg_to_bev.py` |
| **BEV-occ** | 格点 class majority + 凸包/OBB（`test-depth`） | 见分支 `test-depth`：`seg_to_bev_occ.py` |
| Joint / RGB-D viz | 三栏面板或 RGB-D 视频 | `viz_yolo_depth.py`、`viz_rgbd_video.py` |

坐标系：单帧下 **相机 = 世界**（OpenCV：X 右、Y 下、Z 前）。多帧融合再用数据集 `T_world_cam`。

### 导出与评测

| Module | 说明 | 入口 |
|--------|------|------|
| **ONNX export** | YOLO + Depth V2 → ONNX | `export_onnx.py` |
| PT bench | 分割 / 深度 / 联合延迟与利用率 | `bench_detseg_depth.py` |
| ONNX bench | 同上（ORT CUDA EP） | `bench_onnx_detseg_depth.py` |
| Version compare | V1/V2/V3 可视化 + vs-GT 指标 | `compare_depth_versions.py` |

默认 ONNX 权重：

- `pretrained/yolo26m-seg_736x1280.onnx`
- `pretrained/depth_anything_v2_base_518.onnx`（旧 `depth_anything_base_518.onnx` 为 V1）

远端联合 ONNX 约 **12 FPS** 量级（详见 `tmp/bench_*/`）。

### 工程

| Module | 说明 |
|--------|------|
| `scripts/host/sync_to_remote.sh` | 本机 → 远端（`code` \| `all`） |
| `scripts/host/sync_from_remote.sh` | 远端 → 本机 |
| `docs/` | 技术报告与 PAS 问题日志 |

---

## Quick start

```bash
conda activate yolov11
cd /path/to/realtime_3dvision

# 1) 分割冒烟
python scripts/test_yolo_detseg.py --num-frames 4 --stride 50

# 2) 深度冒烟（默认 V2）
python scripts/test_depth_anything.py --num-frames 4 --stride 50

# 3) 联合可视化
python scripts/viz_yolo_depth.py --num-frames 8 --stride 50 --write-video

# 4) 导出深度 npy → 点云 / BEV
python scripts/dump_depth_pt.py --stride 50 --num-frames 8
python scripts/depth_to_pointcloud.py --metricize-gt
# BEV-occ 仍在 test-depth：python scripts/seg_to_bev_occ.py

# 5) ONNX 导出与评测（远端 GPU 推荐）
python scripts/export_onnx.py --skip-yolo   # 或全量导出
python scripts/bench_onnx_detseg_depth.py
```

同步到算力机：

```bash
bash scripts/host/sync_to_remote.sh code
# 需要数据/权重时：
bash scripts/host/sync_to_remote.sh all
```

HF 国内镜像（可选）：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
```

---

## Layout

```
realtime_3dvision/
├── README.md                 # 本文档
├── docs/                     # 技术报告 / PAS
├── scripts/                  # 模块与流水线入口
│   ├── depth_backend.py      # 深度统一后端
│   ├── host/                 # rsync 本机↔远端
│   └── ...
├── data/                     # 演示数据（gitignore）
├── pretrained/               # 权重 / ONNX（gitignore）
└── tmp/                      # 实验产物（gitignore）
```

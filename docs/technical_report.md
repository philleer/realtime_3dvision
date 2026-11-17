# realtime_3dvision 技术选型与演进报告

> 分支：`test-depth`（自 `master` / `test-detseg` 演进）  
> 覆盖范围：仓库建立 → 检测分割 → 深度 → ONNX 评测 → 点云 / BEV → 后续可视化路线  
> 文档日期：2026-11-17

---

## 1. 项目目标

构建室内实时三维视觉原型管线，面向：

- **感知**：RGB 实例分割 + 单目 / RGB-D 深度
- **几何**：深度反投影点云、俯视 BEV
- **后续**：3DGS / 空间理解（规划中）

部署形态：**远端主算力机**（`your-gpu-host`）+ **本机影子仓**（`your-local-host`），代码双向 rsync，大权重 / 数据按需同步。

---

## 2. 仓库与工程基建

| 决策 | 选型 | 理由 |
|------|------|------|
| 代码布局 | 影子 Git 仓 + `scripts/host/sync_*.sh` | 本机编辑、远端 GPU 跑实验 |
| 产物目录 | 统一写入 `tmp/`（弃用 `outputs/`） | 与 gitignore / rsync 一致，避免污染仓 |
| 忽略规则 | `data/`、`pretrained/`、`tmp/`、`*.ply`、`third_party/` | 大数据与权重不入库 |
| 演示数据 | `data/jj_salon_room`（HF: SLAM_JJ_Salon_with_GT） | 1080p 连续 RGB-D + 标定 + 位姿 GT |

### 数据集要点

- RGB：1585 帧，**1920×1080**
- Depth GT：uint16 mm，**640×576**（与 RGB 异分辨率）
- 标定：`scan-003/calib.json`（color / depth 内参 + `T_color_depth`）
- 位姿：**已包含** keyframe GT（218 / 1585），TUM `groundtruth.txt` + `scan_poses.json` 的 `T_world_cam`
- 结论：位姿不必从零 SfM；中间帧可用插值 / RGB-D 里程计补密，SfM 仅作备选校验

---

## 3. 检测与分割选型

### 3.1 候选与取舍

| 方案 | 状态 | 结论 |
|------|------|------|
| YOLO26-SEG（`yolo26m-seg`） | **主路径** | Ultralytics 生态成熟，`retina_masks` 对齐原图，延迟可接受 |
| 经典 DETR / DETR-Seg | 曾试后弱化 | 权重与后处理重，非优先 |
| RF-DETR-Seg（s/m） | 保留对比 | 需独立 `rfdetr` 环境（PyTorch 2.6 + cu124），不升驱动前提下可用 |

### 3.2 运行约定

- 环境：`conda yolov11`（YOLO / Depth HF / 点云 BEV）
- 推理尺寸：`imgsz=1280` → letterbox 后约 **736×1280**（stride 32）
- 脚本：`scripts/test_yolo_detseg.py`，产物 `tmp/yolo_detseg/`

---

## 4. 深度估计选型

| 方案 | 选型细节 | 结论 |
|------|----------|------|
| Depth Anything Base（HF） | `LiheYoung/depth-anything-base-hf` | **主路径**；官方 `.pth` 与 HF 键不兼容，统一走 HF |
| 输入分辨率 | 模型侧约 **518×518**（/14 patch） | 与 YOLO 输入解耦，各自预处理 |
| 输出语义 | HF `predicted_depth` 为 **disparity 风格**（越大越近） | 度量标定必须用 `scale/pred`，禁止 `pred*scale` |
| GT 深度 | 数据集 depth png（mm） | 用于 metricize、评测与点云对照 |

### 关键修复（点云地面挠曲）

- **现象**：相对深度经 `median(gt/pred)*pred` 反投影后，地面随深度非线性弯曲。
- **根因**：DA 输出与度量深度 **负相关**（corr ≈ −0.6～−0.9），当作深度做正比例缩放会拧曲几何。
- **修复**：`corr(pred,gt)<0` 时采用 `metric = median(gt*pred) / pred`；地面平面残差约 **11 cm → 0.8 cm**。
- **代码**：`scripts/depth_to_pointcloud.py`、`scripts/seg_to_bev.py`（commit `4a68f20`）。

脚本：`test_depth_anything.py`、`viz_yolo_depth.py`（RGB | SEG | Depth 三栏 + 可选 MP4 / raw npy）。

---

## 5. 性能评测与 ONNX

### 5.1 导出

- `scripts/export_onnx.py`
- YOLO：`yolo26m-seg_736x1280.onnx`（与 PT letterbox 对齐）
- Depth：`depth_anything_base_518.onnx`

### 5.2 对齐要点

早期 ONNX YOLO 用正方形 1280²，比 PT letterbox **更慢且不公平**。对齐到 **736×1280** 后，seg ONNX ≈ PT。

### 5.3 延迟摘要（远端 GPU，见 `tmp/bench_*/bench_report.md`）

**PyTorch**

| 模式 | mean ms | 约 FPS |
|------|---------|--------|
| Seg only | 33.3 | 30 |
| Depth only | 126.5 | 7.9 |
| Joint | 161.7 | 6.2 |

**ONNX Runtime（CUDA EP）**

| 模式 | mean ms | 约 FPS |
|------|---------|--------|
| Seg only | 34.3 | 29 |
| Depth only | 51.4 | 19 |
| Joint | 86.0 | 12 |

**结论**

- Seg：ONNX 与 PT 持平（尺寸对齐后）。
- Depth：ONNX 明显更快（约 2.5×）。
- 联合：ONNX 更适合实时原型；部署时各模型独立预处理，在图像空间融合。

---

## 6. 深度 → 点云

### 6.1 约定

- **世界原点 = 相机原点**（`T_world_cam = I`，单帧相机系）
- 轴：OpenCV（X 右，Y 下，Z 前）
- RGB 对齐深度 → 用 **color** 内参；原生 depth 相机图 → 用 **depth** 内参
- 脚本：`scripts/depth_to_pointcloud.py` → `tmp/depth_pointcloud/*.ply`

### 6.2 实现要点

- 针孔反投影：`X=(u-cx)Z/fx` 等
- 相对深度：`--metricize-gt` + disparity 判定（见 §4）
- 可选像素 stride / voxel 下采样
- 彩色来自同步 RGB

### 6.3 已知限制

- 单帧不做多视角融合；多帧需乘数据集 `T_world_cam`
- color 畸变对地面弯曲影响次要（本数据上弱于错误 metricize）
- GT depth 与 RGB 异分辨率时，应用外参正确对齐，避免简单 resize 误用

---

## 7. RGB 分割 → BEV（当前实现）

### 7.1 管线

1. YOLO26-SEG → 2D mask / `plot()` 叠图  
2. Depth Anything → 度量深度（disparity metricize）  
3. 反投影到相机系 `(X,Y,Z)`  
4. 俯视栅格化：水平 **+X**，图像上方 **+Z**，黄叉为相机原点  
5. 输出：`tmp/seg_bev/`（面板、独立 BEV、MP4）

脚本：`scripts/seg_to_bev.py`

### 7.2 类别如何区分（现状）

- **无独立 class 栅格**：把 2D 分割叠图的 **BGR 颜色**随点投影到 BEV。
- 同类靠 Ultralytics 着色间接区分；无图例、无 class_id 投票、无 BEV 框。
- 未检出区域保留 RGB 纹理（如地板）。

### 7.3 观感问题

- 径向采样条纹、遮挡空洞、叠图混色 → 演示观感偏“调试图”，非产品级 BEV。

---

## 8. BEV / 占用可视化技术选型（归档）

面向「学术界 + 工业界最新方案里，**可视化效果**哪种最好」的结论（2023–2025 主潮流）。

### 8.1 按观感排序

| 优先级 | 方案 | 代表 | 观感 |
|--------|------|------|------|
| ★★★★★ | **语义 3D Occupancy**（体素按类上色，可俯视/透视） | Tesla Occupancy、OccNet / SurroundOcc / OccFormer | 最好：有高度、块面干净 |
| ★★★★ | **稠密 BEV 语义栅格** + **朝向框 / 矢量地图** | BEVFormer、MapTR、nuScenes 论文图 | 俯视清晰、产品感强 |
| ★★★ | **实例 footprint 多边形 / 轮廓** + 标签 | 部分 AV 工具链 | 比 AABB 贴合 |
| ★★ | **BEV OBB / AABB only** | Waymo / nuScenes 检测可视化 | 干净但缺形状 |
| ★ | **反投影点云按类着色**（当前） | 调试常用 | 条纹/孔洞，演示最弱 |

### 8.2 对本项目的路线建议

| 阶段 | 做法 | 目标 |
|------|------|------|
| **P0（现有 YOLO+Depth 上）** | mask 反投影 → **BEV 格点 class_id 多数投票填色** → 叠加 **footprint 多边形或 OBB + 类别标签** | 性价比最高，接近论文/工业俯视图 |
| **P1** | 稠密语义 BEV 头或占用头（可用现有点云伪标签 / GT depth 监督） | 观感与可融合性上台阶 |
| **P2** | 多帧 `T_world_cam` 融合占用 + 3DGS | 房间级一致地图 |

**不推荐**作为终态演示：仅投影 `plot()` 颜色点，或仅 2D 框四角抬到 BEV（角点深度不稳）。

### 8.3 若只加「类别框」的最小正确做法

1. 每实例取 mask + `cls` + `conf`  
2. mask 像素 + 深度 → `(X,Z)`（可选近地 Y 带做 footprint）  
3. AABB 或 PCA/最小面积矩形 → OBB  
4. 与 BEV 同一 `mpp / x_range / z_range` 画框与标签  
5. 过滤：点数过少、无效深度占比过高、面积过小

---

## 9. 坐标系与位姿策略

| 场景 | 策略 |
|------|------|
| 单帧点云 / BEV（当前） | 相机 = 世界，`T = I` |
| 多帧融合 / 建图 | 使用数据集 keyframe `T_world_cam`；非 keyframe 插值或 RGB-D 里程计 |
| SfM | **非必需**；纹理尚可时可作校验，弱纹理墙面有风险，尺度需 depth/GT 对齐 |

---

## 10. 目录与脚本索引

```
scripts/
  test_yolo_detseg.py          # YOLO26-SEG 冒烟
  test_detr_seg.py             # DETR-Seg 对比（历史）
  test_depth_anything.py       # Depth Anything 冒烟
  viz_yolo_depth.py            # RGB|SEG|Depth 联合可视化 / 视频
  viz_rgbd_video.py            # RGB-D 视频
  export_onnx.py               # YOLO / Depth → ONNX
  bench_detseg_depth.py        # PT 延迟 + 利用率
  bench_onnx_detseg_depth.py   # ONNX 延迟 + 利用率
  depth_to_pointcloud.py       # 深度 → PLY（cam=world）
  seg_to_bev.py                # 分割+深度 → 俯视 BEV
  host/sync_*.sh               # 本机 ↔ 远端同步

tmp/                           # 全部实验产物（不入库）
docs/technical_report.md       # 本文档
```

### 关键 commit 轨迹（`test-depth`）

| Commit | 主题 |
|--------|------|
| `0365190` | 影子仓与远端同步基建 |
| `0833997` | det/seg 与 RGB-D 可视化 |
| `9e6d949` | 产物统一到 `tmp/` |
| `2c7ddf9` | Depth Anything + YOLO–Depth 联合可视化 |
| `a1787ab` | ONNX 导出与 PT/ONNX 评测 |
| `3513666` | 深度 → 相机系点云 |
| `e96830a` | 分割 → BEV |
| `4a68f20` | DA disparity 度量标定修复 |

---

## 11. 当前结论与下一步

**已落地**

- YOLO26-SEG + Depth Anything 主链路，ONNX 联合约 **12 FPS** 量级（评测机）
- 点云 / BEV 单帧相机系可用；DA 度量标定按 disparity 处理
- 数据自带 keyframe 位姿与房间点云 `room.ply`

**建议下一步（按可视化收益）**

1. BEV：**class 栅格投票 + 实例 footprint/OBB 标签**（替换纯颜色点投影）  
2. 可选：语义 occupancy 体素可视化（演示上限）  
3. 多帧：GT 位姿融合 → 世界系 BEV / 占用  
4. 3DGS：在稳定位姿与深度上接入

---

## 12. 参考

- 数据集：[pollen-robotics/SLAM_JJ_Salon_with_GT](https://huggingface.co/datasets/pollen-robotics/SLAM_JJ_Salon_with_GT)
- Depth Anything / HF Depth Estimation
- BEV / Occupancy 综述方向：BEVFormer、OccNet、SurroundOcc、OccFormer；工业演示常见语义占用 + BEV 框叠加
- 本地评测报告：`tmp/bench_detseg_depth/bench_report.md`、`tmp/bench_onnx_detseg_depth/bench_report.md`

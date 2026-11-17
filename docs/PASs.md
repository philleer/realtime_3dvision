# Problems And Solutions (PASs)

对话与实验过程中遇到的关键问题及对应方案。  
相关细节亦见 `docs/technical_report.md`。

- **Part 1**：技术问题（算法、模型、可视化、坐标系）
- **Part 2**：环境问题（依赖、网络、磁盘、路径与同步）

---

# Part 1 — 技术问题

## T1. 官方 Depth Anything `.pth` 与 HF 权重键不兼容

**Problem**  
本地 `depth_anything_vitb14.pth`（原仓库格式）无法直接灌入 Transformers HF 模型，state dict 键不匹配。

**Solution**  
统一使用 HF：`LiheYoung/depth-anything-base-hf`（或对应本地 HF 目录），不混用官方 `.pth`。

---

## T2. 经典 DETR / DETR-Seg 不适合作为主路径

**Problem**  
DETR 权重重、后处理复杂，与实时分割目标不匹配；用户更倾向 RF-DETR-Seg 而非经典 DETR。

**Solution**  
- **主路径**：YOLO26-SEG（`yolo26m-seg`）  
- RF-DETR-Seg 仅作对比  
- 清理未使用 / 不完整 DETR 权重

---

## T3. ONNX YOLO 分割不升反降（比 PT 慢）

**Problem**  
导出 ONNX 后 seg 延迟高于 PyTorch。

**Solution**  
根因是 **输入尺寸不对齐**：PT letterbox ≈ **736×1280**，ONNX 曾用 **1280×1280**。  
对齐到 736×1280 后 ONNX seg ≈ PT；Depth ONNX 仍明显快于 PT。

---

## T4. 部署分辨率：模型输入 vs 采集分辨率混淆

**Problem**  
采集 1080p，YOLO / Depth 输入尺寸不同，不清楚线上该统一到哪一档。

**Solution**  
采集保持 1080p；各模型 **独立 preprocess**；在 **原图像素空间** 对齐融合（mask / depth resize 回 HxW）。

---

## T5. Depth Anything 点云地面「挠曲」

**Problem**  
相对深度 `--metricize-gt` 后反投影，地面随距离非线性弯曲（平面残差约十余 cm）。

**Solution**  
HF `predicted_depth` 为 **disparity 风格**（越大越近）。禁止 `pred * scale`；  
`corr(pred,gt)<0` 时用 `metric = median(gt*pred) / pred`。  
残差约 **11 cm → 0.8 cm**（`4a68f20`）。

---

## T6. 深度 / RGB 相机内参误配

**Problem**  
GT depth 640×576、RGB 1920×1080。简单 resize + 错误 K，或混用 color/depth 内参，几何与着色偏移。

**Solution**  
- 预测 RGB 对齐深度 → **color** 内参  
- 原生 depth 图 → **depth** 内参（尽量原生分辨率）  
- 跨相机用 `T_color_depth` 正确投影，不只靠 resize

---

## T7. BEV 拼图中俯视图过小、难读

**Problem**  
1080p RGB|SEG 与小尺寸 BEV 直接拼接，俯视图缩在一角。

**Solution**  
**2×2** 等高等宽布局；BEV 用 NEAREST 上采样填半宽单元格；另存独立 `*_bev_*.jpg`。

---

## T8. BEV「类别」不清晰

**Problem**  
俯视只有彩色点，难以对应语义类别。

**Solution**  
- **现状**：投影 YOLO `plot()` 颜色，无 class_id 栅格 / 图例  
- **改进**：mask → BEV **class 投票填色** + **footprint/OBB 类别框**（见 technical_report §8）

---

## T9. 联合可视化含冗余 Seg×Depth 面板

**Problem**  
分割×深度交叉面板信息冗余。

**Solution**  
固定三栏：**RGB | SEG | Depth**（`viz_yolo_depth.py`）。

---

## T10. 是否缺相机位姿、要不要上 SfM

**Problem**  
多帧建图前不确定数据是否含位姿；若无则是否用 SfM。

**Solution**  
已有 keyframe GT（约 218/1585）：`groundtruth.txt`、`scan_poses.json`。  
SfM **非必需**；中间帧用插值或 RGB-D 里程计。

---

## T11. 单帧点云 / BEV 世界系如何定义

**Problem**  
需明确世界原点，避免随意外参。

**Solution**  
单帧 **`T_world_cam = I`**（相机 = 世界）；多帧再乘数据集位姿。轴为 OpenCV（X 右，Y 下，Z 前）。

---

### Part 1 速查

| ID | Problem（短） | Solution（短） |
|----|---------------|----------------|
| T1 | DA pth ≠ HF | 只用 HF 权重 |
| T2 | DETR 不适配 | YOLO26 主路径 |
| T3 | ONNX seg 更慢 | 对齐 736×1280 |
| T4 | 部署分辨率 | 分模型预处理再融合 |
| T5 | 地面挠曲 | disparity：`scale/pred` |
| T6 | 深/彩内参混用 | 按相机选 K / 外参对齐 |
| T7 | BEV 太小 | 2×2 上采样拼图 |
| T8 | BEV 无类别 | class 栅格 + 框 |
| T9 | Seg×Depth 冗余 | 只保留三栏 |
| T10 | 位姿 / SfM | 已有 GT；SfM 非必需 |
| T11 | 世界系原点 | 单帧 cam=world |

---

# Part 2 — 环境问题

## E1. 远端 PyTorch 2.6 与 CUDA 驱动不匹配

**Problem**  
RF-DETR 等需要 PyTorch 2.6，无可用 `cu121` 轮子；不愿升级系统驱动。

**Solution**  
独立 conda 环境 **`rfdetr`：PyTorch 2.6 + cu124**；主路径继续用 `yolov11`。

---

## E2. Hugging Face / Roboflow 权重下载超时

**Problem**  
拉取 HF / 部分检测权重时网络超时或不稳定。

**Solution**  
- `HF_ENDPOINT=https://hf-mirror.com`，必要时 `HF_HUB_DISABLE_XET=1`  
- 大文件本机 ↔ 远端 **rsync**，少从公网重复拉

---

## E3. 产物目录混乱（`outputs/` vs `tmp/`）

**Problem**  
脚本与 sync 指向 `outputs/`，实际结果在 `tmp/`，gitignore / rsync 不一致。

**Solution**  
统一 **`tmp/`**；更新 `.gitignore` 与 `scripts/host/sync_*.sh`（`9e6d949`）。

---

## E4. 本机 / 远端磁盘与路径不一致

**Problem**  
- `$HOME` 满导致 heredoc / 临时文件失败（`No space left on device`）  
- `depth_raw/*.npy` 有时只在远端  
- `/home/...` 与 `/mnt/DATA/...` 不同步

**Solution**  
- `TMPDIR=/mnt/DATA/tmp`（或有空间的盘）  
- 缺文件时 rsync `tmp/` 产物  
- 大文件写在 **`/mnt/sdc`** 工作树；脚本改完 rsync 到远端再跑 GPU

---

### Part 2 速查

| ID | Problem（短） | Solution（短） |
|----|---------------|----------------|
| E1 | torch 2.6 / CUDA | cu124 独立 env |
| E2 | HF 下载超时 | 镜像 + rsync |
| E3 | outputs vs tmp | 统一 tmp/ |
| E4 | 磁盘 / 路径不同步 | TMPDIR + rsync sdc |

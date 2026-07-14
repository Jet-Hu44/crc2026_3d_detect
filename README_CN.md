# CRC 2026 — 3D 识别项目

> **2026 中国机器人大赛 & RoboCup 中国公开赛**
> 机器人先进视觉赛项 — 3D 识别
> 队伍: **NEEPU-VF** 观薪 VisionFire (东北电力大学)

基于 **OrangePi AI Pro** (8T Ascend NPU, 16 GB RAM) 与 **ORBBEC Astra Pro Plus** RGBD 深度相机，通过 YOLO 实时检测并分类家居物品，通过 TCP 二进制协议将结果发送至裁判盒评分。

[English README](README.md)

---

## 比赛运行全流程

```
                    ┌─────────────────────┐
                    │   Windows PC        │
                    │   192.168.1.66      │
                    │   judgeGui.exe      │
                    │   (裁判盒评分软件)    │
                    │   端口 6666          │
                    └─────────┬───────────┘
                              │ TCP 二进制协议
                              │ (DataType 0/1/2/3)
                              │
┌─────────────────────────────┴─────────────────────────────┐
│                       OrangePi AI Pro                     │
│                       192.168.1.67                        │
│                                                           │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────┐ │
│  │ ORBBEC   │   │ YOLO     │   │ 多帧投票  │   │ 结果文  │ │
│  │ 相机     │──▶│ 检测     │──▶│ Multi-   │──▶│ 件.txt │ │
│  │ RGBD     │   │ (NPU/CPU)│   │ frame    │   │        │ │
│  └──────────┘   └──────────┘   │ Voting   │   └────┬───┘ │
│                                └──────────┘        │     │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐       │     │
│  │ OCR 识别 │   │ 深度测距  │   │ PyQt5    │       │     │
│  │ (W 类)   │   │ Filter   │   │ GUI      │       │     │
│  └──────────┘   └──────────┘   └──────────┘       │     │
│                                                    │     │
│  输出格式:                                           │     │
│  Goal_ID=CA001;Num=2;Table=1 ──────────────────────┘     │
│  Goal_ID=W001;Num=1;Table=2                               │
└───────────────────────────────────────────────────────────┘
```

### Round 1 (第一轮 — 单桌台, 固定相机)

```
识别.sh 1 → 连接裁判盒 → DataType 0 (开始计时)
         → 检测 1 张桌台 (20s)
         → DataType 1 (结果文件 + 停止计时)
         → 断开连接
```

### Round 2 (第二轮 — 三桌台, 云台旋转相机)

```
识别.sh 2 → 连接裁判盒 → DataType 0 (开始计时)
         → 检测桌台 1 → DataType 3 (旋转信号)
         → 检测桌台 2 → DataType 3 (旋转信号)
         → 检测桌台 3
         → DataType 1 (合并结果 + 停止计时)
         → 断开连接
```

---

## 代码模块架构

```
huanshibest.py          # 入口 — 解析命令行参数, 创建 detector + GUI
    │
    ├── config.py       # RoundConfig (轮次参数), 网络 IP, 各类阈值
    │
    ├── detector.py     # YoloOrbbecDetector — 相机采集 + YOLO 推理
    │   ├── NPU 路径:  npu_detector.py → Ascend 310B4 ACL API
    │   └── CPU 路径:  ultralytics.YOLO → PyTorch ARM CPU
    │
    ├── gui.py          # MainWindow — PyQt5 界面 + DetectWorker (后台线程)
    │   ├── Round 1: _run_round1() → 单桌检测
    │   └── Round 2: _run_round2() → 逐桌旋转→检测→投票→合并
    │
    ├── network.py      # JudgeBoxClient — TCP 二进制协议 (DataType 0/1/2/3)
    │
    └── ocr_module.py   # LightweightOCR — Tesseract 引擎, 识别 W 类物品表面文字
```

### 内部数据流 (逐帧)

```
相机帧 (640×480 BGR)
    │
    ▼
[detector.py] frame_to_bgr_image()  ← ORBBEC SDK → OpenCV
    │
    ▼
[detector.py] inference_image()
    │
    ├── NPU (USE_NPU=True):
    │   │  _preprocess(): BGR → letterbox(640×640) → RGB → HWC→CHW → /255
    │   │  NPUDetector.infer(): H2D → acl.mdl.execute → D2H (FP16→FP32)
    │   │  decode_npu_output(): (1,12,8400) → NMS → (N,6) boxes
    │   │  速度: ~20ms
    │   │
    │   └── CPU (USE_NPU=False):
    │       YOLO(model).predict() → (N,6) boxes
    │       速度: ~568ms
    │
    ▼
[gui.py] DetectWorker (后台线程)
    │  if conf ≥ threshold → 写入帧结果到 runss/labels/
    │  emit result_ready(image, boxes) → GUI 显示
    │
    ▼
[gui.py] _process_folder()  ← 多帧投票
    │  frame_presence[obj] ≥ min_occurrences(5)
    │  Counter.most_common(1) → 取众数
    │
    ▼
[gui.py] _finish()          ← 生成结果文件
    │  写入: START / Goal_ID=xxx;Num=n;Table=t / END
    │  拷贝到 ~/Desktop/result_r/
    │
    ▼
[network.py] JudgeBoxClient.send_result_and_stop()
    │  TCP 二进制: [DataType=1][Length][TXT 内容]
    │  裁判盒停止计时 → 计算得分
    └── close()
```

---

## 启动脚本原理 (`识别.sh`)

### 脚本逐行解析

```bash
#!/bin/bash
ROUND=${1:-2}           # 第 1 个参数: 轮次 (默认 2)
WEIGHTS=${2:-best4.pt}  # 第 2 个参数: 模型权重 (默认 best4.pt)

① sudo ifconfig eth0 192.168.1.67 netmask 255.255.255.0
   # 将香橙派有线网口设置为静态 IP 192.168.1.67

② ping -c 2 -W 2 192.168.1.66
   # 检查裁判盒 (Windows PC) 是否在线
   # 注意: Windows 防火墙默认禁 Ping, 不通可跳过

③ export PYTHONPATH=...
   source /usr/local/miniconda3/etc/profile.d/conda.sh
   conda deactivate
   # 配置 Python 环境路径, 激活 conda

④ cd /home/HwHiAiUser/Desktop/Round/ultralytics-main
   # 切换到项目工作目录

⑤ DISPLAY=:0 python3 huanshibest.py --round ${ROUND} --weights ${WEIGHTS}
   # DISPLAY=:0: 通过 SSH 启动时, 将 GUI 重定向到 HDMI 显示器
   # --round: 比赛轮次
   # --weights: 模型权重文件
```

### 不同启动条件

| 命令 | 轮次 | 桌台数 | 相机 | 光源 | 输出文件 |
|---------|:--:|:--:|------|------|-------------|
| `./识别.sh` | 2 (默认) | 3 | 旋转云台 | 特定光源 | NEEPU-VF-R2.txt |
| `./识别.sh 1` | 1 | 1 | 固定 | 环境光 | NEEPU-VF-R1.txt |
| `./识别.sh 1 best6.pt` | 1 | 1 | 固定 | 环境光 | NEEPU-VF-R1.txt |
| `./识别.sh 2 best5.pt` | 2 | 3 | 旋转云台 | 特定光源 | NEEPU-VF-R2.txt |

### 为什么需要 `DISPLAY=:0`?

通过 SSH 远程登录时, 终端没有图形显示目标。`DISPLAY=:0` 指定 PyQt5 GUI 窗口输出到香橙派 HDMI 接口连接的物理显示器上。不加此环境变量, 程序会报 `could not connect to display` 并崩溃。

### NPU / CPU 切换

```python
# config.py
USE_NPU = True   # 使用 Ascend 310B4 NPU, 推理 ~20ms/帧
USE_NPU = False  # 使用 PyTorch ARM CPU, 推理 ~568ms/帧
```

程序启动时自动检测 `best4.om` 文件是否存在, 若不存在则自动回退到 CPU 模式。

---

## 比赛轮次

| | 第一轮 (Round 1) | 第二轮 (Round 2) |
|---|---|---|
| 桌台数 | 1 | 3 (三角布局) |
| 物品数 | 7–15 | 21–45 (每桌 7–15) |
| 相机 | 固定 | 云台旋转 |
| 光源 | 环境光 | 2 张桌有特定夹持光源 (黄+白) |
| 时间限制 | 20–50s | 70–150s |
| 权重 | 40% | 60% |

**输出格式**:
```
START
Goal_ID=CA001;Num=2;Table=1
Goal_ID=CB003;Num=1;Table=1
Goal_ID=W01;Num=1;Table=2
END
```

**计分**: 完全正确 (ID + 数量 + 桌号) = +3 分。错误 ID = −3 分。超过分数阈值可获得时间奖励分。

---

## 快速开始

### 香橙派环境准备 (一次性)

```bash
# 安装 opencv 完整版 (不能装 headless)
pip install opencv-python

# 修复 opencv Qt 插件与系统 PyQt5 的冲突
rm -rf ~/.local/lib/python3.8/site-packages/cv2/qt/

# NPU 模型转换 (一次性)
python3 export_onnx.py                    # best4.pt → best4.onnx
source /usr/local/Ascend/ascend-toolkit/set_env.sh
atc --model=best4.onnx --framework=5 --output=best4 \
    --soc_version=Ascend310B4 --input_shape="images:1,3,640,640" \
    --input_format=NCHW --output_type=FP16
# → 生成 best4.om (6.5MB)
```

### 运行

```bash
cd ~/Desktop
./识别.sh 1   # Round 1
./识别.sh 2   # Round 2
```

### 裁判盒操作 (Windows PC)

```
① 双击 judgeGui_240328/judgeGui.exe 启动裁判盒
② 选择轮次 → 点击锁定
③ 输入比赛真值 (物品 ID + 数量)
④ 等待香橙派 TCP 连接
⑤ 得分表自动保存到 Result/ 目录
```

> 注意: Windows 必须安装 Microsoft Excel, 裁判盒通过 COM 接口调用 Excel 模板 (`Template_excel/`) 计算得分。

---

## 裁判盒 TCP 协议

数据包结构 (大端, Big-Endian):

```
| DataType (int32, 4byte) | DataLength (int32, 4byte) | Data (bytes) |
```

| DataType | 含义 | 发送时机 | Data 内容 |
|----------|------|----------|-----------|
| **0** | 队伍 ID + **开始计时** | Round 开始 | 队伍 ID 字符串 |
| **1** | 结果文件 + **停止计时** | 检测完成 | TXT 文件全部内容 |
| **2** | 结果文件 (工业测量) | — | TXT 内容, 得分=0 |
| **3** | 转台旋转信号 | 每桌检测完毕 | `"0000"` |

时间窗口:
- Round 1: MinTime=20s, MaxTime=50s
- Round 2: MinTime=70s, MaxTime=150s

---

## 工程目录结构

```
├── ultralytics-main/           # ★ 主工程 (部署于香橙派)
│   ├── huanshibest.py          # 入口 (argparse CLI)
│   ├── config.py               # RoundConfig, 网络配置, 阈值, USE_NPU 开关
│   ├── detector.py             # YoloOrbbecDetector — 相机+YOLO (NPU/CPU 双路径)
│   ├── npu_detector.py         # Ascend 310B4 NPU 推理封装
│   ├── gui.py                  # PyQt5 GUI + DetectWorker + 检测流程控制
│   ├── network.py              # JudgeBoxClient — TCP 二进制协议
│   ├── ocr_module.py           # LightweightOCR — Tesseract 引擎
│   ├── train.py                # YOLO11 训练脚本
│   ├── export_onnx.py          # .pt → .onnx 导出 (NPU 用)
│   ├── best4.pt                # 比赛模型 (当前 8 类, 待采集训练数据)
│   ├── dataset/                # 数据集目录 (images/labels — 待采集)
│   ├── archive/                # 历史版本/变体/开发测试快照
│   └── ultralytics/            # YOLO 库 (修改版 fork)
├── pyorbbecsdk-main/           # ORBBEC Astra Pro Plus Python SDK
├── 2_相机旋转云台/              # 云台硬件 — Arduino 固件 + 组装图纸 PDF
├── judgeGui_240328/            # 裁判盒软件 (Windows .exe + Excel 评分模板)
├── docs/                       # 比赛规则, 分析报告, 行动计划
├── 识别.sh                      # 一键启动脚本 (chmod +x)
├── README.md                   # 英文版 README
└── 比赛规则.pdf                 # 原始比赛规则 PDF
```

---

## 核心功能

| 功能 | 实现方式 |
|------|---------|
| **YOLO 检测** | YOLOv8/YOLO11, 8–18 类, NPU/CPU 双路径可切换 |
| **NPU 加速** | Ascend 310B4 via CANN 7.0 ACL API, 相比 CPU 加速 28× |
| **OCR 文字识别** | Tesseract — 识别 W 类未知物品表面文字以区分小类 |
| **深度测距** | 100–1600mm 范围过滤, 时域平滑 (α=0.7) |
| **多帧投票** | min_occurrences=5, 取每类物品的众数数量 |
| **裁判盒通信** | TCP 二进制: DataType 0 开始 / 1 结果+停止 / 3 旋转 |
| **轮次自适应** | `RoundConfig` 类 + `--round` CLI + `识别.sh` 传参 |
| **PyQt5 界面** | 实时画面, 检测标签叠加, 结果列表, 一键关闭 |
| **一键启动** | `识别.sh` — 设 IP, 检查连通, 配环境, 启动识别 |
| **资格认证视频** | `record_video_opi.py` — cv2.VideoCapture + YOLO 叠加 |

---

## 当前状态 (2026-07-15)

| 系统 | 状态 |
|------|:--:|
| ORBBEC 相机 + 深度 | ✅ |
| YOLO 检测 (8 类) | ✅ |
| NPU 推理 (20ms, 28×) | ✅ |
| TCP 裁判盒通信 | ✅ |
| Round 1 端到端 | ✅ |
| PyQt5 GUI | ✅ |
| 18 类训练数据采集 | 🔜 计划 7/18–7/20 |
| Round 2 云台 + 测试 | 🔜 计划 7/21–7/23 |
| OCR (Tesseract) | 🔜 待 apt install |
| NPU GUI 渲染 Bug | 🔧 已知, 下次修复 |

---

## 参考与致谢

- 基于 [crc2025_3d_detect_dyl](https://github.com/xensedyl/crc2025_3d_detect_dyl) by @xensedyl
- 使用 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) 与 [pyorbbecsdk](https://github.com/orbbec/pyorbbecsdk)
- OCR 引擎: [Tesseract](https://github.com/tesseract-ocr/tesseract)
- NPU 加速: [华为昇腾 CANN](https://www.hiascend.com/)

## 许可证

本项目仅供教育及比赛使用。Ultralytics 许可证条款见 `ultralytics-main/LICENSE`。

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Auto-sync**: A `Stop` hook in `.claude/settings.json` automatically runs `git add -A && git commit && git push origin master` every time the Claude Code session ends. No manual git commands needed — your work is always saved and pushed to GitHub.

## Project Overview

中国机器人大赛 RoboCup 机器人先进视觉赛项 — **3D识别项目** (2026赛季)。
Hardware: OrangePi AI Pro (8T Ascend NPU, 16GB RAM) + ORBBEC Astra Pro Plus RGBD camera.
The system detects household objects (16 known classes + unknown items) on 1–3 tabletop platforms and communicates results to a judge box via TCP.

## Repository Structure

```
3D_Recongise/
├── ultralytics-main/          # ★ 主工程 — 部署于香橙派 /home/HwHiAiUser/ultralytics-main/
│   ├── huanshibest.py         # 主识别入口 (识别.sh 启动)
│   ├── huanshi.py/huanshi1.py # 幻视变体 (Round 1/2 variants)
│   ├── huanjuebest.py         # 幻决变体 (alternate detection strategy)
│   ├── train.py               # 训练脚本 (YOLO11s, 需修改data/epochs)
│   ├── best.pt~best6.pt       # 比赛模型权重 (~6MB, YOLOv8n-scale)
│   ├── runss/                 # 运行时输出: labels/, d/, result/, output_video.avi
│   └── ultralytics/           # YOLO library (modified fork)
├── pyorbbecsdk-main/          # ORBBEC Astra Pro Plus Python SDK wrapper
│   └── sdk/                   # Compiled .so/.pyd native libraries
├── crc2025_3d_detect_dyl-main/# 开源参考项目 (GitHub: xensedyl/crc2025_3d_detect_dyl)
│   ├── detect_qt.py           # 参考GUI识别程序 (当前代码的90%来源)
│   └── crc_2025.yaml          # 参考训练配置 (10类, 需扩充至16类+)
├── result_r/                  # 桌面结果文件夹 (裁判盒读取此路径)
├── 识别.sh                    # 一键启动脚本 (设置IP→conda→python3 huanshibest.py)
├── docs/
│   ├── pdfscribe/比赛规则/     # 比赛规则 Markdown (已从PDF转换)
│   ├── 项目分析报告.md         # 完成度分析 + 改进方案
│   └── P0_四天行动计划.md      # 四天压缩行动计划
├── OrangePi_AI_Pro_昇腾_用户手册_v1.2.pdf  # 香橙派手册 (207页)
└── 比赛规则.pdf                # 原始比赛规则PDF
```

## Architecture: Detection Pipeline

```
Camera (ORBBEC Astra Pro Plus)               TCP Socket
   │ RGBD frames                               │
   ▼                                           │
TemporalFilter (depth stabilization α=0.7)     │
   │                                           │
   ▼                                           │
YOLO Detection (best4.pt, conf_thres=0.50)    │
   │                                           │
   ├──► Known items (CAxxx–CDxxx) ─────────┐  │
   │                                        │  │
   └──► Unknown items (Wxxx) ──► OCR ──────┤  │
                                            │  │
   ▼                                        │  │
Depth Filter (100–1600mm range) ◄───────────┘  │
   │                                           │
   ▼                                           │
Multi-frame Voting (min_occurrences=5, mode)    │
   │                                           │
   ▼                                           │
Result Format: START                            │
  Goal_ID=CA001;Num=2;Table=1                   │
  Goal_ID=W001;Num=1;Table=2                    │
END                                            │
   │                                           │
   └───────────────────────────────────────────┘
        send_file(1, path) → 192.168.1.88:6666
```

## Key Hardware & Network

- **Camera**: ORBBEC Astra Pro Plus (0.6-8m range), USB3.0, accessed via `pyorbbecsdk.Pipeline`
- **Compute**: OrangePi AI Pro — 8T INT8 Ascend NPU, 16GB RAM, Ubuntu/openEuler
- **Network**: Static IP `192.168.1.67` (eth0), judge box at `192.168.1.88:6666`
- **NPU Path**: PyTorch → ONNX export → ATC tool → Ascend .om model → AscendCL Python API
  - **Current state: NPU NOT used** — all inference runs on ARM CPU via PyTorch

## Competition Rules (Critical Constraints)

From `docs/pdfscribe/比赛规则/比赛规则.md`:

| Requirement | Status |
|-------------|--------|
| 16 known classes (CA001–CD004, 4 per category) | ⚠️ Currently 10 classes |
| Unknown items with OCR text recognition (2025 new) | ❌ Not implemented |
| Table ID in output (`;Table=n`) (2025 new) | ❌ Not implemented |
| Camera rotation control for 3-table Round 2 | ❌ Not implemented |
| Model load time counts toward score (2025 new) | ❌ Not optimized |
| Send start/end signals to judge box | ❌ Not implemented |
| `start.sh` one-click launch, no parameter changes after start | ✅ Implemented |
| Result file: `{unit}-{team}-R{x}.txt` with START/END markers | ✅ Format correct, missing Table field |

**Output format required**: Each line: `Goal_ID={code};Num={count};Table={table_num}` (分号分隔)
**Round 2**: 3 tables in triangle, camera rotates via motor, 2 tables have specific light sources

## Common Commands

### On Orange Pi (deployment target)
```bash
# Launch recognition (from Desktop)
./识别.sh              # Round 2 (default)
./识别.sh 1            # Round 1 (if arg support added)

# Manual launch
cd /home/HwHiAiUser/ultralytics-main
python3 huanshibest.py --round 1 --weights best4.pt

# Check camera
python3 -c "from pyorbbecsdk import Pipeline, Config; print('OK')"

# Check model classes
python3 -c "from ultralytics import YOLO; m=YOLO('best4.pt'); print(m.names)"

# Install Tesseract OCR
sudo apt-get install -y tesseract-ocr tesseract-ocr-chi-sim
pip install pytesseract
```

### On dev machine (Windows, this repo)
```bash
# Train model
cd ultralytics-main
python3 train.py   # Edit data/epochs/weights first

# Test detection on image
python3 -c "
from ultralytics import YOLO
m = YOLO('best4.pt')
r = m('test.jpg', conf=0.5)
r[0].show()
"

# Export to ONNX (for NPU conversion)
python3 -c "from ultralytics import YOLO; YOLO('best4.pt').export(format='onnx')"

# Convert competition PDF rules to markdown
/pdf-to-md 比赛规则.pdf --single

# Analyze code structure
codegraph explore "huanshibest YoloOrbbecDetector process_detection_cycle"
```

## Key Files to Modify for Competition Readiness

| Priority | File | What to change |
|----------|------|----------------|
| **P0** | `huanshibest.py:503` | Add `;Table={current_table}` to output format |
| **P0** | `huanshibest.py:process_detection_cycle()` | Add per-table detection loop for Round 2 |
| **P0** | `huanshibest.py:MainWindow` | Add `send_rotate_command()` for camera rotation |
| **P0** | New: `ocr_module.py` | LightweightOCR class using Tesseract for W-class items |
| **P1** | `train.py` | Expand from 10→18 classes, use yolo11s.pt pretrained |
| **P1** | `huanshibest.py:YoloOrbbecDetector` | Add AscendCL NPU inference path |
| **P1** | `huanshibest.py:send_string()` | Add START/END signal message types |

## Code Relationships

- `huanshibest.py` ≈ `detect_qt.py` (from crc2025) — ~90% code overlap, same architecture
- `huanjuebest.py` ≈ alternate strategy variant (幻决 vs 幻视), same camera/YOLO base
- `test.py` through `test9.py` — incremental development snapshots of the main script
- Model weights: `best.pt` series are competition-trained; `yolo11n.pt`/`yolov8n.pt` are base models
- The `ultralytics/` folder is a modified fork — do NOT replace with pip version

## Project Documentation

- `docs/项目分析报告.md` — Full gap analysis vs competition rules, improvement recommendations
- `docs/P0_四天行动计划.md` — 4-day compressed P0 action plan with daily milestones
- `docs/pdfscribe/比赛规则/比赛规则.md` — Full competition rules in Markdown (32 pages)
- `docs/pdfscribe/OrangePi_AI_Pro_昇腾_用户手册_v1.2/` — Orange Pi manual extracts (207 pages, partially converted)

## Reference Projects

- **crc2025_3d_detect_dyl**: https://github.com/xensedyl/crc2025_3d_detect_dyl — original open-source base (10 classes, Qt GUI, same pipeline)
- **QQ Group**: 1027375571 (先进视觉赛技术交流群) — judge box protocol, rule supplements, community support
- **Ultralytics**: https://github.com/ultralytics/ultralytics — upstream YOLO framework (this repo uses a modified fork)

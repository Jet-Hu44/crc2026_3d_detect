# CRC 2026 — 3D Object Recognition

> **2026 China Robot Competition & RoboCup China Open**
> Robot Advanced Vision — 3D Recognition Event
> Team: **NEEPU-VF** 观薪 VisionFire (Northeast Electric Power University)

Built on **OrangePi AI Pro** (8T Ascend NPU, 16 GB RAM) with an **ORBBEC Astra Pro Plus** RGBD camera, this system performs real-time YOLO object detection and classification across two competition rounds, communicating results to a judge box via TCP.

📖 [中文版 README](README_CN.md)

---

## Competition Workflow

```
                    ┌─────────────────────┐
                    │   Windows PC        │
                    │   192.168.1.66      │
                    │   judgeGui.exe      │
                    │   (Judge Box)       │
                    │   Port 6666         │
                    └─────────┬───────────┘
                              │ TCP Binary Protocol
                              │ (DataType 0/1/2/3)
                              │
┌─────────────────────────────┴─────────────────────────────┐
│                       OrangePi AI Pro                     │
│                       192.168.1.67                        │
│                                                           │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────┐ │
│  │ ORBBEC   │   │ YOLO     │   │ Multi-   │   │ Result │ │
│  │ Camera   │──▶│ Detection│──▶│ frame    │──▶│ File   │ │
│  │ RGBD     │   │ (NPU/CPU)│   │ Voting   │   │ .txt   │ │
│  └──────────┘   └──────────┘   └──────────┘   └────┬───┘ │
│                                                    │     │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐       │     │
│  │ OCR      │   │ Depth    │   │ PyQt5    │       │     │
│  │ (W class)│   │ Filter   │   │ GUI      │       │     │
│  └──────────┘   └──────────┘   └──────────┘       │     │
│                                                    │     │
│  Output:                                            │     │
│  Goal_ID=CA001;Num=2;Table=1 ──────────────────────┘     │
│  Goal_ID=W001;Num=1;Table=2                               │
└───────────────────────────────────────────────────────────┘
```

### Round 1 (single table, fixed camera)

```
识别.sh 1 → connect judge box → DataType 0 (start timer)
         → detect 1 table (20s)
         → DataType 1 (result file + stop timer)
         → disconnect
```

### Round 2 (three tables, rotating camera)

```
识别.sh 2 → connect judge box → DataType 0 (start timer)
         → detect table 1 → DataType 3 (rotate signal)
         → detect table 2 → DataType 3 (rotate signal)
         → detect table 3
         → DataType 1 (merged result + stop timer)
         → disconnect
```

---

## Code Architecture

```
huanshibest.py          # Entry point — parses CLI args, creates detector + GUI
    │
    ├── config.py       # RoundConfig (Round 1/2 params), network IPs, thresholds
    │
    ├── detector.py     # YoloOrbbecDetector — camera capture + YOLO inference
    │   ├── NPU path:  npu_detector.py → Ascend 310B4 ACL API
    │   └── CPU path:  ultralytics.YOLO → PyTorch ARM CPU
    │
    ├── gui.py          # MainWindow — PyQt5 GUI + DetectWorker (background thread)
    │   ├── Round 1: _run_round1() → single-table detection
    │   └── Round 2: _run_round2() → per-table rotate → detect → vote → merge
    │
    ├── network.py      # JudgeBoxClient — TCP binary protocol (DataType 0/1/2/3)
    │
    └── ocr_module.py   # LightweightOCR — Tesseract engine for W-class items
```

### Internal Data Flow

```
Camera Frame (640×480 BGR)
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
    │   │  Speed: ~20ms
    │   │
    │   └── CPU (USE_NPU=False):
    │       YOLO(model).predict() → (N,6) boxes
    │       Speed: ~568ms
    │
    ▼
[gui.py] DetectWorker
    │  if conf ≥ threshold → write frame results to runss/labels/
    │  emit result_ready(image, boxes) → GUI display
    │
    ▼
[gui.py] _process_folder()  ← Multi-frame Voting
    │  frame_presence[obj] ≥ min_occurrences(5)
    │  Counter.most_common(1) → majority count
    │
    ▼
[gui.py] _finish()          ← Result Generation
    │  write: START / Goal_ID=xxx;Num=n;Table=t / END
    │  copy to ~/Desktop/result_r/
    │
    ▼
[network.py] JudgeBoxClient.send_result_and_stop()
    │  TCP binary: [DataType=1][Length][TXT content]
    │  Judge box stops timer → computes score
    └── close()
```

---

## Launch Script (`识别.sh`)

### Script Flow

```bash
#!/bin/bash
ROUND=${1:-2}        # Argument 1: round number (default 2)
WEIGHTS=${2:-best4.pt}  # Argument 2: model weights (default best4.pt)

① sudo ifconfig eth0 192.168.1.67    # Set static IP
② ping 192.168.1.66                   # Check judge box reachability
③ conda activate / deactivate         # Python environment
④ cd ultralytics-main                 # Working directory
⑤ DISPLAY=:0 python3 huanshibest.py --round $ROUND --weights $WEIGHTS
```

### Launch Conditions

| Command | Round | Tables | Camera | Light | Output File |
|---------|:-----:|:------:|--------|-------|-------------|
| `./识别.sh` | 2 (default) | 3 | Rotating | Specific lamps | NEEPU-VF-R2.txt |
| `./识别.sh 1` | 1 | 1 | Fixed | Ambient | NEEPU-VF-R1.txt |
| `./识别.sh 1 best6.pt` | 1 | 1 | Fixed | Ambient | NEEPU-VF-R1.txt |
| `./识别.sh 2 best5.pt` | 2 | 3 | Rotating | Specific lamps | NEEPU-VF-R2.txt |

### Why `DISPLAY=:0`?

When running via SSH, Qt has no display target. `DISPLAY=:0` routes the PyQt5 GUI window to the HDMI-connected monitor. Without it, the program crashes with "could not connect to display."

### NPU / CPU Switch

```python
# config.py
USE_NPU = True   # Ascend 310B4 NPU ~20ms/frame
USE_NPU = False  # PyTorch ARM CPU ~568ms/frame
```

The program auto-detects `best4.om` at startup and falls back to CPU if absent.

---

## Competition Rounds

| | Round 1 | Round 2 |
|---|---|---|
| Tables | 1 | 3 (triangle layout) |
| Items | 7–15 | 21–45 (7–15 per table) |
| Camera | Fixed | Motorized rotation (pan-tilt) |
| Lighting | Ambient | 2 tables with clip-lamps (yellow + white) |
| Time limit | 20–50s | 70–150s |
| Weight | 40% | 60% |

**Output format**:
```
START
Goal_ID=CA001;Num=2;Table=1
Goal_ID=CB003;Num=1;Table=1
Goal_ID=W01;Num=1;Table=2
END
```

**Scoring**: +3 per correct match (ID + count + table). −3 per false positive. Time bonus available above score threshold.

---

## Quick Start

### Prerequisites (OrangePi)

```bash
# One-time setup
pip install opencv-python  # NOT headless version
rm -rf ~/.local/lib/python3.8/site-packages/cv2/qt/  # Fix Qt conflict

# NPU model (one-time conversion)
python3 export_onnx.py                    # best4.pt → best4.onnx
source /usr/local/Ascend/ascend-toolkit/set_env.sh
atc --model=best4.onnx --framework=5 --output=best4 \
    --soc_version=Ascend310B4 --input_shape="images:1,3,640,640" \
    --input_format=NCHW --output_type=FP16
```

### Run

```bash
cd ~/Desktop
./识别.sh 1   # Round 1
./识别.sh 2   # Round 2
```

### Judge Box (Windows PC)

```
① Double-click judgeGui_240328/judgeGui.exe
② Select round → Lock
③ Input ground truth items (ID + count)
④ Wait for OrangePi connection
⑤ Score auto-saved to Result/
```

### Data Collection (Training Dataset)

```bash
# Orange Pi — ORBBEC camera (RGB + Depth .npy)
DISPLAY=:0 python3 capture_data_opi.py --class CB001 --count 60

# Windows — any USB camera (RGB only)
python capture_data_win.py --class CB001 --count 60 --camera 0
```

| Key | Action |
|-----|--------|
| **Enter** | Capture one photo |
| **A** | Switch angle (0°/30°/60°/90°/top) |
| **L** | Switch lighting (daylight/whiteLED/yellowLED/mix) |
| **D** | Switch distance (0.6m/1.2m/1.8m) |
| **Q** | Quit |

Output: `dataset/images/{class}/{class}_{angle}_{light}_{dist}_{seq:04d}.jpg`

---

## Repository Structure

```
├── ultralytics-main/           # ★ Main application (deployed on OrangePi)
│   ├── huanshibest.py          # Entry point (argparse CLI)
│   ├── config.py               # RoundConfig, network, thresholds, USE_NPU switch
│   ├── detector.py             # YoloOrbbecDetector — camera + YOLO (NPU/CPU)
│   ├── npu_detector.py         # Ascend 310B4 NPU inference wrapper
│   ├── gui.py                  # PyQt5 GUI + DetectWorker + detection flow
│   ├── network.py              # JudgeBoxClient — TCP binary protocol
│   ├── ocr_module.py           # LightweightOCR — Tesseract engine
│   ├── train.py                # YOLO11 training script
│   ├── capture_data_opi.py     # Training data capture (Orange Pi + ORBBEC)
│   ├── capture_data_win.py     # Training data capture (Windows + USB cam)
│   ├── export_onnx.py          # .pt → .onnx export for NPU
│   ├── best4.pt                # Competition model (8-class, training data pending)
│   ├── dataset/                # Training dataset (images/labels — pending collection)
│   ├── archive/                # Historical snapshots, variants & dev experiments
│   └── ultralytics/            # Modified YOLO library (fork)
├── pyorbbecsdk-main/           # ORBBEC Astra Pro Plus Python SDK
├── 2_相机旋转云台/              # Pan-tilt hardware — Arduino firmware + assembly PDF
├── judgeGui_240328/            # Judge box software (Windows .exe + Excel templates)
├── docs/                       # Competition rules, analysis reports, action plans
├── 识别.sh                      # One-click launch script (chmod +x)
├── README_CN.md                # 中文版 README
└── 比赛规则.pdf                 # Original competition rulebook
```

---

## Key Features

| Feature | Implementation |
|---------|---------------|
| **YOLO Detection** | YOLOv8/YOLO11, 8–18 classes, dual NPU/CPU inference |
| **NPU Acceleration** | Ascend 310B4 via CANN 7.0 ACL API, 28× speedup vs CPU |
| **OCR** | Tesseract — unknown-item text recognition (W-class) |
| **Depth Filtering** | 100–1600 mm range, temporal smoothing (α=0.7) |
| **Multi-frame Voting** | min_occurrences=5, majority-count per class |
| **Judge Box Protocol** | TCP binary: DataType 0 (start) / 1 (result+stop) / 3 (rotate) |
| **Round Auto-adaptation** | `RoundConfig` + `--round` CLI + `识别.sh` parameter passing |
| **PyQt5 GUI** | Live feed, detection overlay, result list, one-click close |
| **One-click Launch** | `识别.sh` — sets IP, checks connectivity, activates env, starts detection |
| **Qualification Video** | `record_video_opi.py` — cv2.VideoCapture + YOLO overlay |

---

## Current Status (2026-07-15)

| System | Status |
|--------|:--:|
| ORBBEC Camera + Depth | ✅ |
| YOLO Detection (8-class) | ✅ |
| NPU Inference (20ms, 28×) | ✅ |
| TCP Judge Box Communication | ✅ |
| Round 1 End-to-End | ✅ |
| PyQt5 GUI | ✅ |
| Training Data (18-class) | 🔜 Collection planned 7/18–7/20 |
| Round 2 Pan-Tilt + Testing | 🔜 Assembly + code planned 7/21–7/23 |
| OCR (Tesseract) | 🔜 Pending apt install |
| NPU GUI Rendering Bug | 🔧 Known issue, fix next session |

---

## Acknowledgments

- Based on [crc2025_3d_detect_dyl](https://github.com/xensedyl/crc2025_3d_detect_dyl) by @xensedyl
- Uses [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) and [pyorbbecsdk](https://github.com/orbbec/pyorbbecsdk)
- OCR powered by [Tesseract](https://github.com/tesseract-ocr/tesseract)
- NPU acceleration via [Huawei Ascend CANN](https://www.hiascend.com/)

## License

This project is for educational and competition use. See `ultralytics-main/LICENSE` for the Ultralytics license terms.

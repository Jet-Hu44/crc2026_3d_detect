# CRC 2026 — 3D Object Recognition

> **2026 China Robot Competition & RoboCup China Open**
> Robot Advanced Vision — 3D Recognition Event
> Team: **NEEPU-HS** (Northeast Electric Power University)

Built on **OrangePi AI Pro** (8T Ascend NPU, 16 GB RAM) with an **ORBBEC Astra Pro Plus** RGBD camera, this system performs real-time object detection and classification across two competition rounds.

---

## Competition Rounds

| | Round 1 | Round 2 |
|---|---|---|
| Tables | 1 square | 3 (2 square + 1 rotating circular) |
| Items | 7–15 | 21–45 (7–15 per table) |
| Camera | Fixed | Motorized rotation |
| Lighting | Ambient | 2 tables with specific clip-lamps (yellow + white, 7 W) |
| Weight | 40% of final score | 60% of final score |

**Output format** (per competition rules):
```
START
Goal_ID=CA001;Num=2;Table=1
Goal_ID=W01;Num=1;Table=2
END
```

**Scoring**: 3 points per correct (ID + count + table) match. Wrong ID = −3 pts. Time bonus available above a score threshold. Final rank = Round 1 × 40% + Round 2 × 60%.

---

## Pipeline

```
ORBBEC Astra Pro Plus (RGBD)
       │
       ▼
Depth Alignment + Temporal Filtering (α=0.7)
       │
       ▼
YOLO Detection (conf ≥ 0.50, IoU ≥ 0.45)
       │
       ├── Known items (CA001–CD004) → direct classification
       │
       └── Unknown items (Wxxx) → ROI crop → CLAHE + OTSU → Tesseract OCR
                                       │
                                  keyword match → W01/W02/…
       │
       ▼
Depth Filter (100–1600 mm)
       │
       ▼
Multi-frame Voting (min_occurrences=5, mode)
       │
       ▼
Result file → TCP binary protocol → Judge Box (192.168.1.88:6666)
```

---

## Repository Structure

```
├── ultralytics-main/           # Main application (deployed on OrangePi)
│   ├── huanshibest.py          # Entry point
│   ├── config.py               # Round config + all constants
│   ├── detector.py             # YOLO + camera + OCR integration
│   ├── network.py              # TCP protocol (judge box communication)
│   ├── gui.py                  # PyQt5 GUI + detection flow control
│   ├── ocr_module.py           # Lightweight OCR (Tesseract)
│   ├── train.py                # Training script
│   ├── best4.pt ~ best6.pt     # Competition model weights
│   ├── archive/                # Historical snapshots, variants & old weights
│   └── ultralytics/            # Modified YOLO library
├── pyorbbecsdk-main/           # ORBBEC camera Python SDK
├── crc2025_3d_detect_dyl-main/ # Upstream open-source reference
├── docs/                       # Competition rules, analysis reports, action plans
├── 识别.sh                     # One-click launch script
└── 比赛规则.pdf                 # Original competition rulebook (Chinese)
```

---

## Hardware

- **Compute**: OrangePi AI Pro — 8T INT8 NPU (Ascend), 16 GB LPDDR4X, Ubuntu/openEuler
- **Camera**: ORBBEC Astra Pro Plus (0.6–8 m range), USB 3.0, RGBD (color+depth)
- **Network**: Static IP `192.168.1.67` (eth0) → judge box at `192.168.1.88:6666`

> **Note**: NPU acceleration is not yet utilized — all inference currently runs on ARM CPU via PyTorch. See `docs/项目分析报告.md` for the NPU deployment roadmap.

---

## Quick Start (on OrangePi)

```bash
# One-click launch
./识别.sh        # Round 2 (default)
./识别.sh 1      # Round 1

# Manual launch
cd ultralytics-main
python3 huanshibest.py --round 1 --weights best4.pt

# Install OCR dependencies (one-time)
sudo apt-get install -y tesseract-ocr tesseract-ocr-chi-sim
pip install pytesseract Pillow
```

---

## Key Features

- Real-time YOLO object detection (16 known item classes + unknown items)
- Lightweight OCR via Tesseract for unknown-item subclassification by surface text
- Depth-based filtering with temporal smoothing
- Multi-frame voting for robust detection under occlusion and distractors
- TCP binary protocol for judge box communication (team ID → START signal → detection → END signal → result file)
- PyQt5 GUI with live camera feed, detection overlay, and one-button force-close
- One-click `start.sh` launch per competition rules (no parameter changes allowed after start)
- Round 1 / Round 2 auto-adaptation via `RoundConfig` and CLI `--round` flag

---

## Acknowledgments

- Based on [crc2025_3d_detect_dyl](https://github.com/xensedyl/crc2025_3d_detect_dyl) by @xensedyl
- Uses [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) and [pyorbbecsdk](https://github.com/orbbec/pyorbbecsdk)
- OCR powered by [Tesseract](https://github.com/tesseract-ocr/tesseract)

## License

This project is for educational and competition use. See `ultralytics-main/LICENSE` for the Ultralytics license terms.

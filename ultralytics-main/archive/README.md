# Archive — 历史开发文件

本目录存放开发过程中的快照和变体，仅供参考，不参与当前比赛流程。

## dev_snapshots/
`test.py` ~ `test9.py` — 功能增量开发快照，记录了从原始detect_qt.py到huanshibest.py的演进过程。当前版本的功能已抽取为config.py/detector.py/gui.py/network.py/ocr_module.py。

## variants/
历史版本变体，与当前huanshibest.py共享相同基础架构：

| 文件 | 说明 |
|------|------|
| `detect.py` | 原始检测脚本（参考项目来源） |
| `huanshi.py` | 幻视 Round 1 变体 |
| `huanshi1.py` | 幻视 Round 1 变体(v2) |
| `huanshibest1.py` | 幻视最佳版本(旧版) |
| `huanjue.py` | 幻决 变体 |
| `huanjue1.py` | 幻决 Round 1 变体 |
| `huanjuebest.py` | 幻决最佳版本 |

## old_weights/
历史模型权重文件：

| 文件 | 大小 | 说明 |
|------|------|------|
| `best.pt` | 6.0MB | 早期最佳模型 |
| `best3.pt` | 6.0MB | v3 模型 |
| `zgjqr2024best.pt` | 6.0MB | 2024年中国机器人大赛最佳模型 |
| `zgjqr2024last.pt` | 6.0MB | 2024年中国机器人大赛最终模型 |
| `yolo11n.pt` | 5.4MB | YOLO11n 基础模型（可重新下载） |
| `yolo11s.pt` | 18.4MB | YOLO11s 基础模型（可重新下载） |
| `yolov8n.pt` | 6.2MB | YOLOv8n 基础模型（可重新下载） |

当前比赛使用的权重文件在上级目录：`best4.pt` ~ `best6.pt`, `bbest.pt`, `bestt.pt`

"""
资质认证 — 功能展示视频录制脚本 (Windows 版)
==============================================
使用普通 USB 摄像头 + YOLO 模型完成四场景录制。
无需 ORBBEC 相机，无需香橙派，在 Windows 上直接运行。

用法:
  python record_video_win.py --scene 1   # 不同背景
  python record_video_win.py --scene 2   # 不同距离
  python record_video_win.py --scene 3   # 实物vs贴纸
  python record_video_win.py --scene 4   # 运动物品
  python record_video_win.py --scene all # 全部录制

输出: ./videos/scene{1-4}_{timestamp}.mp4
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np
from datetime import datetime

from ultralytics import YOLO

# ── 配置 ──
DEFAULT_WEIGHTS = 'best4.pt'
CONF_THRES = 0.50
IOU_THRES = 0.45
OUTPUT_DIR = './videos'

# 场景定义
SCENE_INFO = {
    1: {
        'name': '不同随机背景下的识别',
        'duration': 90,
        'fps': 15,
        'desc': '纯色→条纹→图案，三换背景',
    },
    2: {
        'name': '目标台与相机不同距离下的识别',
        'duration': 90,
        'fps': 15,
        'desc': '1.0m→1.5m→1.8m，三换距离',
    },
    3: {
        'name': '实物与贴纸的辨识',
        'duration': 60,
        'fps': 15,
        'desc': '真实物品 vs 彩色打印贴纸',
    },
    4: {
        'name': '运动中物品的识别',
        'duration': 60,
        'fps': 15,
        'desc': '旋转平台，慢速→快速',
    },
}

# 类别颜色
CLASS_COLORS = {}
def get_color(cls_name):
    if cls_name not in CLASS_COLORS:
        np.random.seed(hash(cls_name) % 2**32)
        CLASS_COLORS[cls_name] = tuple(int(c) for c in np.random.randint(80, 255, 3))
    return CLASS_COLORS[cls_name]


def draw_overlay(image, boxes, names, show_depth=False):
    """在图像上叠加检测框"""
    if boxes is None or len(boxes) == 0:
        return image

    for box in boxes:
        x1, y1, x2, y2, conf, cls_id = box
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cls_name = names[int(cls_id)]
        color = get_color(cls_name)

        # 检测框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # 标签
        label = f'{cls_name} {conf:.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(image, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(image, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    return image


def draw_info_bar(image, scene_num, elapsed, duration, fps_real, obj_count):
    """底部信息栏"""
    h, w = image.shape[:2]
    bar_h = 46

    overlay = image.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)

    info = SCENE_INFO[scene_num]
    cv2.putText(image, f'Scene {scene_num}: {info["name"]}',
                (10, h - bar_h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(image, f'{elapsed:.0f}s / {duration}s  |  Objects: {obj_count}  |  FPS: {fps_real:.1f}',
                (10, h - bar_h + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # 进度条
    p = min(elapsed / duration, 1.0) if duration > 0 else 0
    px, py, pw, ph = 10, h - 6, w - 20, 3
    cv2.rectangle(image, (px, py), (px + pw, py + ph), (60, 60, 60), -1)
    cv2.rectangle(image, (px, py), (px + int(pw * p), py + ph), (0, 220, 0), -1)

    return image


def add_title(image, text, duration_sec=3.0, elapsed=0):
    """场景标题（前N秒显示）"""
    if elapsed > duration_sec:
        return image
    alpha = 1.0 - (elapsed / duration_sec) * 0.5
    h, w = image.shape[:2]
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)
    cv2.putText(image, text, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    return image


def record_scene(scene_num, model, names, output_dir, camera_id=0, custom_duration=None):
    """录制单个场景"""
    info = SCENE_INFO[scene_num]
    duration = custom_duration or info['duration']
    fps = info['fps']
    interval = 1.0 / fps

    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f'scene{scene_num}_{ts}.mp4')

    # 打开摄像头
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f'[ERROR] 无法打开摄像头 (ID={camera_id})')
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # VideoWriter
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_path = output_path.replace('.mp4', '.avi')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

    print(f'\n{"="*60}')
    print(f'  Scene {scene_num}: {info["name"]}')
    print(f'  Duration: {duration}s  |  FPS: {fps}  |  Camera: {camera_id}')
    print(f'  Output: {output_path}')
    print(f'{"="*60}\n')
    print('[REC] Press Ctrl+C to stop early\n')

    start_time = time.time()
    frame_count = 0
    last_cap_time = 0

    try:
        while True:
            elapsed = time.time() - start_time
            if elapsed >= duration:
                break

            if elapsed - last_cap_time < interval:
                time.sleep(0.003)
                continue

            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                continue

            # YOLO 推理
            results = model(frame, conf=CONF_THRES, iou=IOU_THRES, verbose=False)
            boxes = results[0].boxes.data.cpu().numpy() if len(results) > 0 and results[0].boxes is not None else []

            # 可视化
            frame = draw_overlay(frame, boxes, names)
            frame = draw_info_bar(frame, scene_num, elapsed, duration, 1.0/(time.time()-t0) if t0 else 0, len(boxes))

            if elapsed < 3:
                frame = add_title(frame, f'Scene {scene_num}: {info["name"]}', 3, elapsed)

            writer.write(frame)
            frame_count += 1
            last_cap_time = elapsed

            if frame_count % 30 == 0:
                print(f'  Frame {frame_count}  |  {elapsed:.0f}s  |  {len(boxes)} objects')

    except KeyboardInterrupt:
        print('\n[REC] Stopped by user')

    cap.release()
    writer.release()

    actual = time.time() - start_time
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f'\n[DONE] {output_path}')
    print(f'       Frames: {frame_count}  |  Duration: {actual:.1f}s  |  Size: {size_mb:.1f} MB')
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Qualification Video Recorder (Windows)')
    parser.add_argument('--scene', type=str, default='1',
                        help='Scene 1-4, or "all" for all four')
    parser.add_argument('--weights', type=str, default=DEFAULT_WEIGHTS,
                        help='Model weights path')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device ID (0=内置, 1=外接)')
    parser.add_argument('--duration', type=int, default=None,
                        help='Custom duration per scene (seconds)')
    parser.add_argument('--output', type=str, default=OUTPUT_DIR,
                        help='Output directory')
    args = parser.parse_args()

    # 加载模型
    print(f'Loading model: {args.weights} ...')
    model = YOLO(args.weights)
    names = model.names
    print(f'Classes ({len(names)}): {list(names.values())}')

    # 测试摄像头
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f'[ERROR] Cannot open camera {args.camera}')
        print('Try: --camera 0 (built-in) or --camera 1 (USB)')
        sys.exit(1)
    cap.release()
    print(f'Camera {args.camera} OK')

    # 确定要录制的场景
    if args.scene == 'all':
        scenes = [1, 2, 3, 4]
    else:
        scenes = [int(args.scene)]

    # 录制
    for s in scenes:
        print(f'\n{"#"*60}')
        print(f'# 准备录制 Scene {s}: {SCENE_INFO[s]["name"]}')
        print(f'# {SCENE_INFO[s]["desc"]}')
        print(f'# 按 Enter 开始录制...')
        print(f'{"#"*60}')
        input()

        record_scene(s, model, names, args.output, args.camera, args.duration)

        if s != scenes[-1]:
            print(f'\n准备下一个场景... (按 Enter 继续)')
            input()

    # 汇总
    print(f'\n{"="*60}')
    print('All scenes recorded:')
    for f in sorted(os.listdir(args.output)):
        fpath = os.path.join(args.output, f)
        print(f'  {f}  ({os.path.getsize(fpath)/1024/1024:.1f} MB)')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()

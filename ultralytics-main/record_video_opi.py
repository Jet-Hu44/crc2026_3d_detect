"""
资质认证 — 功能展示视频录制脚本 (OrangePi 版)
==============================================
使用普通 USB 摄像头 + YOLO 模型完成四场景录制。
不依赖 ORBBEC 深度相机，在香橙派上直接运行。

用法:
  python3 record_video_opi.py --scene 4        # 仅录制第四场景
  python3 record_video_opi.py --scene all      # 全部四场景
  python3 record_video_opi.py --scene 3 --camera 1  # 指定摄像头

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
        'fps': 12,  # 香橙派 ARM CPU 降低帧率
        'desc': '纯色->条纹->图案，三换背景',
    },
    2: {
        'name': '目标台与相机不同距离下的识别',
        'duration': 90,
        'fps': 12,
        'desc': '1.0m->1.5m->1.8m，三换距离',
    },
    3: {
        'name': '实物与贴纸的辨识',
        'duration': 60,
        'fps': 12,
        'desc': '真实物品 vs 彩色打印贴纸',
    },
    4: {
        'name': '运动中物品的识别',
        'duration': 60,
        'fps': 12,
        'desc': '旋转平台，慢速->快速移动',
    },
}

# 类别颜色缓存
CLASS_COLORS = {}


def get_color(cls_name):
    """为每个类别分配固定颜色"""
    if cls_name not in CLASS_COLORS:
        np.random.seed(hash(cls_name) % 2**32)
        CLASS_COLORS[cls_name] = tuple(int(c) for c in np.random.randint(80, 255, 3))
    return CLASS_COLORS[cls_name]


def draw_overlay(image, boxes, names):
    """在图像上叠加 YOLO 检测框和标签"""
    if boxes is None or len(boxes) == 0:
        return image, 0

    for box in boxes:
        x1, y1, x2, y2, conf, cls_id = box
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cls_name = names.get(int(cls_id), f'cls{int(cls_id)}')
        color = get_color(cls_name)

        # 检测框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # 标签 + 置信度
        label = f'{cls_name} {conf:.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(image, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(image, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    return image, len(boxes)


def draw_info_bar(image, scene_num, elapsed, duration, fps_real, obj_count):
    """底部信息栏 + 进度条"""
    h, w = image.shape[:2]
    bar_h = 46

    overlay = image.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, image, 0.25, 0, image)

    info = SCENE_INFO[scene_num]
    cv2.putText(image, f'Scene {scene_num}: {info["name"]}',
                (10, h - bar_h + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1)
    cv2.putText(image, f'{elapsed:.0f}s / {duration}s  |  Objects: {obj_count}  |  FPS: {fps_real:.1f}',
                (10, h - bar_h + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (200, 200, 200), 1)

    # 进度条
    p = min(elapsed / duration, 1.0) if duration > 0 else 0
    px, py, pw, ph = 10, h - 6, w - 20, 3
    cv2.rectangle(image, (px, py), (px + pw, py + ph), (60, 60, 60), -1)
    cv2.rectangle(image, (px, py), (px + int(pw * p), py + ph),
                  (0, 220, 0), -1)

    return image


def add_title(image, text, elapsed=0):
    """场景标题（前3秒显示，渐变消失）"""
    if elapsed > 3.0:
        return image
    h, w = image.shape[:2]
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (w, 56), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)
    cv2.putText(image, text, (16, 38), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 255), 2)
    return image


def find_camera():
    """自动查找可用的摄像头"""
    for cam_id in range(4):
        cap = cv2.VideoCapture(cam_id)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                return cam_id
    return None


def record_scene(scene_num, model, names, output_dir, camera_id=0,
                 custom_duration=None, resolution=(640, 480)):
    """录制单个场景视频"""
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

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓冲延迟

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
    print(f'  Desc: {info["desc"]}')
    print(f'{"="*60}\n')
    print('[REC] 按 Ctrl+C 可提前停止\n')

    start_time = time.time()
    frame_count = 0
    last_cap_time = 0

    try:
        while True:
            elapsed = time.time() - start_time
            if elapsed >= duration:
                break

            # 控制帧率
            if elapsed - last_cap_time < interval:
                time.sleep(0.01)
                continue

            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                continue

            # YOLO 推理（香橙派上用小图加速）
            h, w = frame.shape[:2]
            infer_frame = cv2.resize(frame, (480, 360)) if w > 640 else frame
            results = model(infer_frame, conf=CONF_THRES, iou=IOU_THRES,
                          verbose=False, imgsz=320)

            # 提取检测框，缩放回原始尺寸
            boxes_raw = results[0].boxes
            if boxes_raw is not None and len(boxes_raw) > 0:
                boxes_data = boxes_raw.data.cpu().numpy()
                scale_x = w / infer_frame.shape[1]
                scale_y = h / infer_frame.shape[0]
                boxes = []
                for b in boxes_data:
                    bx1, by1, bx2, by2, bconf, bcls = b
                    boxes.append([
                        bx1 * scale_x, by1 * scale_y,
                        bx2 * scale_x, by2 * scale_y,
                        bconf, bcls
                    ])
                boxes = np.array(boxes)
            else:
                boxes = np.array([])

            # 叠加检测结果
            frame, obj_count = draw_overlay(frame, boxes, names)

            # 信息栏 + 标题
            real_fps = 1.0 / (time.time() - t0) if (time.time() - t0) > 0 else 0
            frame = draw_info_bar(frame, scene_num, elapsed, duration,
                                 real_fps, obj_count)
            frame = add_title(frame, f'Scene {scene_num}: {info["name"]}',
                            elapsed)

            writer.write(frame)
            frame_count += 1
            last_cap_time = elapsed

            if frame_count % 30 == 0:
                print(f'  Frame {frame_count:4d}  |  '
                      f'{elapsed:5.1f}s  |  '
                      f'{obj_count} objects  |  '
                      f'FPS ~{real_fps:.1f}')

    except KeyboardInterrupt:
        print('\n[REC] 用户中断')

    cap.release()
    writer.release()

    actual = time.time() - start_time
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f'\n[DONE] {output_path}')
    print(f'       Frames: {frame_count}  |  '
          f'Duration: {actual:.1f}s  |  Size: {size_mb:.1f} MB')
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='资质认证视频录制 (OrangePi 版)')
    parser.add_argument('--scene', type=str, default='4',
                        help='场景编号: 1/2/3/4 或 "all"')
    parser.add_argument('--weights', type=str, default=DEFAULT_WEIGHTS,
                        help='模型权重路径 (默认 best4.pt)')
    parser.add_argument('--camera', type=int, default=None,
                        help='摄像头 ID (不指定则自动检测)')
    parser.add_argument('--duration', type=int, default=None,
                        help='自定义录制时长(秒)')
    parser.add_argument('--output', type=str, default=OUTPUT_DIR,
                        help='输出目录')
    parser.add_argument('--resolution', type=str, default='640x480',
                        help='录制分辨率 (宽x高)')
    args = parser.parse_args()

    # 解析分辨率
    res_parts = args.resolution.split('x')
    resolution = (int(res_parts[0]), int(res_parts[1]))

    # 自动检测摄像头
    if args.camera is None:
        print('正在检测摄像头...')
        args.camera = find_camera()
        if args.camera is None:
            print('[ERROR] 未找到可用摄像头!')
            print('请确认 USB 摄像头已连接，或用 --camera 指定 ID')
            sys.exit(1)
        print(f'自动选择摄像头: ID={args.camera}')

    # 加载 YOLO 模型
    print(f'加载模型: {args.weights} ...')
    model = YOLO(args.weights)
    names = model.names
    print(f'类别数: {len(names)}')
    for k, v in names.items():
        print(f'  [{k}] {v}')
    print()

    # 测试摄像头
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f'[ERROR] 无法打开摄像头 ID={args.camera}')
        sys.exit(1)
    ret, test_frame = cap.read()
    cap.release()
    if not ret:
        print(f'[ERROR] 摄像头 {args.camera} 可打开但无法读取帧')
        sys.exit(1)
    print(f'摄像头 OK ({test_frame.shape[1]}x{test_frame.shape[0]})')

    # 确定场景
    if args.scene == 'all':
        scenes = [1, 2, 3, 4]
    else:
        scenes = [int(s) for s in args.scene.split(',')]

    # 逐个录制
    recorded = []
    for i, s in enumerate(scenes):
        print(f'\n{"#"*60}')
        print(f'# 准备录制 Scene {s}: {SCENE_INFO[s]["name"]}')
        print(f'# 提示: {SCENE_INFO[s]["desc"]}')
        if i == 0:
            print(f'# 按 Enter 开始录制...')
        else:
            print(f'# 上一个场景已完成，按 Enter 继续录制本场景...')
        print(f'{"#"*60}')
        input()

        path = record_scene(s, model, names, args.output, args.camera,
                           args.duration, resolution)
        if path:
            recorded.append(path)

        if s != scenes[-1]:
            print(f'\n准备下一个场景... (按 Enter 继续)')
            input()

    # 汇总
    print(f'\n{"="*60}')
    print(f'全部录制完成! 共 {len(recorded)} 个视频:')
    total_size = 0
    for f in sorted(os.listdir(args.output)):
        fpath = os.path.join(args.output, f)
        size = os.path.getsize(fpath) / 1024 / 1024
        total_size += size
        print(f'  {f}  ({size:.1f} MB)')
    print(f'  总大小: {total_size:.1f} MB')
    print(f'{"="*60}')
    print(f'\n提示: 如需重新录制某个场景，运行:')
    print(f'  python3 record_video_opi.py --scene <编号>')


if __name__ == '__main__':
    main()

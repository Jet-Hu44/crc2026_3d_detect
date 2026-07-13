"""
资质认证 — 功能展示视频录制脚本
=================================
四个场景的独立录制模式，每个场景录制一段带检测叠加的 MP4 视频。

用法（在香橙派上）:
  python3 record_video.py --scene 1   # 不同背景
  python3 record_video.py --scene 2   # 不同距离
  python3 record_video.py --scene 3   # 实物vs贴纸
  python3 record_video.py --scene 4   # 运动物品

输出: runss/scene{1-4}_{timestamp}.mp4
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np
from datetime import datetime

from detector import YoloOrbbecDetector
from config import (
    CONF_THRES, IOU_THRES, DEPTH_MIN_MM, DEPTH_MAX_MM,
    DEFAULT_WEIGHTS, RUNSS_DIR,
)


# ── 场景元信息 ──
SCENE_INFO = {
    1: {
        'name': '不同随机背景下的识别',
        'duration': 90,   # 90秒: 3种背景 × 各30秒
        'fps': 15,        # 录制帧率（降低以减少文件大小）
        'description': '展示纯色、条纹、图案三种背景下的检测效果',
    },
    2: {
        'name': '目标台与相机不同距离下的识别',
        'duration': 90,
        'fps': 15,
        'description': '展示1.0m、1.5m、1.8m三个距离下的检测效果',
    },
    3: {
        'name': '实物与贴纸的辨识',
        'duration': 60,
        'fps': 15,
        'description': '展示真实3D物品与2D打印贴纸的深度区分',
    },
    4: {
        'name': '运动中物品的识别',
        'duration': 60,
        'fps': 15,
        'description': '展示旋转平台上运动物品的实时检测',
    },
}


def draw_overlay(image, result_list, depth_data=None, show_depth=True):
    """在图像上叠加检测框和标签"""
    for r in result_list:
        name, conf, x1, y1, x2, y2 = r[:6]
        distance = r[6] if len(r) > 6 else None

        # 颜色：已知物品绿色框，W类/未知黄色框
        if name.startswith('W'):
            color = (0, 255, 255)  # 黄色
        else:
            color = (0, 255, 0)    # 绿色

        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        # 标签
        label = f'{name} {conf:.2f}'
        if show_depth and distance is not None:
            label += f' {distance:.0f}mm'

        # 标签背景
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (int(x1), int(y1) - th - 8),
                      (int(x1) + tw, int(y1)), color, -1)
        cv2.putText(image, label, (int(x1), int(y1) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return image


def draw_info_bar(image, scene_num, elapsed, duration, fps_real):
    """绘制底部信息栏"""
    h, w = image.shape[:2]
    bar_h = 40

    # 半透明背景条
    overlay = image.copy()
    cv2.rectangle(overlay, (0, h - bar_h), (w, h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

    # 信息文字
    info = SCENE_INFO[scene_num]
    lines = [
        f'Scene {scene_num}: {info["name"]}',
        f'Time: {elapsed:.0f}s / {duration}s  |  FPS: {fps_real:.1f}',
    ]
    for i, line in enumerate(lines):
        cv2.putText(image, line, (10, h - bar_h + 18 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 进度条
    progress = elapsed / duration if duration > 0 else 0
    bar_x, bar_y, bar_w, bar_h2 = 10, h - 8, w - 20, 4
    cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h2),
                  (80, 80, 80), -1)
    cv2.rectangle(image, (bar_x, bar_y),
                  (bar_x + int(bar_w * progress), bar_y + bar_h2),
                  (0, 200, 0), -1)

    return image


def add_scene_title(image, text, alpha=0.85):
    """在画面中央叠加场景标题（淡入效果）"""
    h, w = image.shape[:2]
    overlay = image.copy()

    # 标题背景
    cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

    cv2.putText(image, text, (20, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return image


def record_scene(scene_num, detector, output_dir=None):
    """录制单个场景视频"""
    info = SCENE_INFO[scene_num]
    duration = info['duration']
    fps = info['fps']
    frame_interval = 1.0 / fps

    # 输出路径
    if output_dir is None:
        output_dir = RUNSS_DIR
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(output_dir, f'scene{scene_num}_{timestamp}.mp4')

    # VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_w, frame_h = 640, 480
    writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))

    if not writer.isOpened():
        # 备用编码器
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_path = output_path.replace('.mp4', '.avi')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_w, frame_h))
        print(f'[录制] 使用 XVID/AVI 编码')

    print(f'\n{"="*60}')
    print(f'  场景 {scene_num}: {info["name"]}')
    print(f'  时长: {duration}秒  |  FPS: {fps}  |  总帧数: ~{duration * fps}')
    print(f'  输出: {output_path}')
    print(f'  说明: {info["description"]}')
    print(f'{"="*60}\n')

    start_time = time.time()
    frame_count = 0
    last_frame_time = 0

    print('[录制] 按 Ctrl+C 提前停止录制')

    try:
        while True:
            elapsed = time.time() - start_time
            if elapsed >= duration:
                break

            # 控制帧率
            if elapsed - last_frame_time < frame_interval:
                time.sleep(0.005)
                continue

            loop_start = time.time()

            # 检测一帧
            result_list, image = detector.inference_image()
            if image is None or image.size == 0:
                continue

            # 深度过滤: 只保留有效距离内的检测
            filtered_results = []
            for r in result_list:
                dist = r[6] if len(r) > 6 else None
                if dist is None or (DEPTH_MIN_MM < dist < DEPTH_MAX_MM):
                    filtered_results.append(r)
                else:
                    # 超出深度范围的标记为灰色
                    name, conf, x1, y1, x2, y2 = r[:6]
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)),
                                  (128, 128, 128), 1)
                    cv2.putText(image, f'{name} (out of range)',
                                (int(x1), int(y1) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)

            # 叠加可视化
            image = draw_overlay(image, filtered_results, show_depth=(scene_num == 3))
            image = draw_info_bar(image, scene_num, elapsed, duration, fps)

            # 前3秒显示场景标题
            if elapsed < 3:
                image = add_scene_title(image, f'Scene {scene_num}: {info["name"]}')

            # 确保尺寸一致
            if image.shape[0] != frame_h or image.shape[1] != frame_w:
                image = cv2.resize(image, (frame_w, frame_h))

            writer.write(image)
            frame_count += 1
            last_frame_time = elapsed

            # 实时FPS
            real_fps = 1.0 / (time.time() - loop_start) if loop_start > 0 else 0

            if frame_count % 30 == 0:
                print(f'  [录制中] 帧={frame_count}  '
                      f'耗时={elapsed:.1f}s  '
                      f'检测={len(filtered_results)}个物体  '
                      f'FPS={real_fps:.1f}')

    except KeyboardInterrupt:
        print('\n[录制] 用户中断')
    finally:
        writer.release()
        actual_duration = time.time() - start_time
        print(f'\n[录制] 完成!')
        print(f'  文件: {output_path}')
        print(f'  帧数: {frame_count}')
        print(f'  实际时长: {actual_duration:.1f}秒')
        print(f'  文件大小: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB')

    return output_path


def main():
    parser = argparse.ArgumentParser(description='资质认证视频录制')
    parser.add_argument('--scene', type=int, choices=[1, 2, 3, 4],
                        required=True, help='场景编号 1-4')
    parser.add_argument('--weights', type=str, default=DEFAULT_WEIGHTS,
                        help='模型权重路径')
    parser.add_argument('--duration', type=int, default=None,
                        help='自定义录制时长(秒)')
    parser.add_argument('--output', type=str, default=None,
                        help='自定义输出目录')
    args = parser.parse_args()

    # 自定义时长
    if args.duration:
        SCENE_INFO[args.scene]['duration'] = args.duration

    # 初始化检测器
    print('[初始化] 加载模型和相机...')
    detector = YoloOrbbecDetector(weights=args.weights, device='0')
    print(f'[初始化] 模型类别: {list(detector.names.values())}')
    print(f'[初始化] 深度可用: {detector.depth_available}')
    print(f'[初始化] OCR可用: {detector.ocr.available}')

    # 预热几帧
    print('[初始化] 预热相机...')
    for _ in range(10):
        detector.inference_image()

    # 录制
    output_path = record_scene(args.scene, detector, args.output)

    # 清理
    detector.close()

    print(f'\n录制完成 → {output_path}')
    print('如需重新录制，再次运行以上命令即可。')


if __name__ == '__main__':
    main()

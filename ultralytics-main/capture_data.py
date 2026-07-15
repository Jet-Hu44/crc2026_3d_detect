"""
训练数据拍照采集工具

用法:
  python3 capture_data.py --class CB001 --count 60

操作:
  - 按 Enter/Space: 拍一张照片
  - 按 A: 切换到下一个角度 (0°/30°/60°/90°/顶视)
  - 按 L: 切换光照条件 (日光/白光/黄光/混合)
  - 按 D: 切换距离 (近0.6m/中1.2m/远1.8m)
  - 按 Q 或 Esc: 退出

输出: dataset/images/CB001/CB001_00_日光_中_0001.jpg
"""
import os
import sys
import time
import cv2
import numpy as np
import argparse

# 拍摄参数
ANGLES = ['0deg', '30deg', '60deg', '90deg', 'top']
LIGHTS = ['daylight', 'whiteLED', 'yellowLED', 'mix']
DISTANCES = ['near_06m', 'mid_12m', 'far_18m']

def main():
    parser = argparse.ArgumentParser(description='训练数据采集')
    parser.add_argument('--class', dest='cls_id', type=str, required=True,
                        help='物品类别, 如 CB001')
    parser.add_argument('--count', type=int, default=60,
                        help='目标拍摄张数')
    parser.add_argument('--camera', type=int, default=0,
                        help='摄像头ID (默认0)')
    args = parser.parse_args()

    cls_id = args.cls_id
    target = args.count

    # 创建输出目录
    out_dir = f'dataset/images/{cls_id}'
    os.makedirs(out_dir, exist_ok=True)

    # 统计已有照片
    existing = len([f for f in os.listdir(out_dir) if f.endswith('.jpg')])
    print(f'类别: {cls_id}  |  目标: {target} 张  |  已有: {existing} 张')
    print(f'输出: {out_dir}/')
    print()

    # 打开摄像头
    print('正在打开摄像头...')
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f'[ERROR] 无法打开摄像头 ID={args.camera}')
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    angle_idx = 0
    light_idx = 0
    dist_idx = 0
    count = existing

    print('=' * 55)
    print('  操作说明')
    print('  Enter/Space = 拍照    A = 换角度    L = 换光照')
    print('  D = 换距离           Q/Esc = 退出')
    print('=' * 55)

    while count < target:
        angle = ANGLES[angle_idx]
        light = LIGHTS[light_idx]
        dist = DISTANCES[dist_idx]

        # 读取帧
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        # 预览画面
        display = frame.copy()
        # 信息栏
        info = f'{cls_id} | {angle} | {light} | {dist} | {count}/{target}'
        cv2.rectangle(display, (0, 0), (640, 40), (0, 0, 0), -1)
        cv2.putText(display, info, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # 十字准心
        h, w = display.shape[:2]
        cv2.line(display, (w // 2, 0), (w // 2, h), (0, 255, 0), 1)
        cv2.line(display, (0, h // 2), (w, h // 2), (0, 255, 0), 1)

        cv2.imshow('Data Capture — Press Enter to shoot, Q to quit', display)
        key = cv2.waitKey(30) & 0xFF

        if key == 13 or key == 32:  # Enter or Space
            count += 1
            fname = f'{cls_id}_{angle}_{light}_{dist}_{count:04d}.jpg'
            fpath = os.path.join(out_dir, fname)
            cv2.imwrite(fpath, frame)
            print(f'  [{count}/{target}] {fname} saved')
            # 自动切换角度 (每个角度拍够了换下一个)
            per_angle = target // len(ANGLES)
            if count % per_angle == 0:
                angle_idx = (angle_idx + 1) % len(ANGLES)

        elif key == ord('a') or key == ord('A'):
            angle_idx = (angle_idx + 1) % len(ANGLES)
            print(f'  角度 → {ANGLES[angle_idx]}')

        elif key == ord('l') or key == ord('L'):
            light_idx = (light_idx + 1) % len(LIGHTS)
            print(f'  光照 → {LIGHTS[light_idx]}')

        elif key == ord('d') or key == ord('D'):
            dist_idx = (dist_idx + 1) % len(DISTANCES)
            print(f'  距离 → {DISTANCES[dist_idx]}')

        elif key == ord('q') or key == ord('Q') or key == 27:
            print(f'\n退出, 已拍 {count} 张')
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f'\n完成! 共 {count} 张照片保存在 {out_dir}/')
    print(f'下一步: 用 LabelImg 标注, 或用 python3 -m ultralytics.data.annotator 辅助标注')


if __name__ == '__main__':
    main()

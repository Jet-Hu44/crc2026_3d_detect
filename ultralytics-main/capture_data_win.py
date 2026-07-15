"""
Windows 版 — 任意 USB 摄像头 (cv2.VideoCapture)
用法: python capture_data_win.py --class CB001 --count 60 --camera 0
"""
import os, time, cv2, argparse

ANGLES   = ['0deg','30deg','60deg','90deg','top']
LIGHTS   = ['daylight','whiteLED','yellowLED','mix']
DISTANCES = ['near_06m','mid_12m','far_18m']

def main():
    p = argparse.ArgumentParser(description='训练数据采集 (Windows)')
    p.add_argument('--class', dest='c', type=str, required=True, help='物品类别, 如 CB001')
    p.add_argument('--count', type=int, default=60, help='目标张数')
    p.add_argument('--camera', type=int, default=0, help='摄像头 ID, 默认 0')
    a = p.parse_args()

    out = f'dataset/images/{a.c}'
    os.makedirs(out, exist_ok=True)
    n = len([f for f in os.listdir(out) if f.endswith('.jpg')])
    print(f'{a.c} | target:{a.count} | exist:{n} | out:{out}/')

    cap = cv2.VideoCapture(a.camera)
    if not cap.isOpened():
        print(f'ERROR: camera {a.camera} not found')
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    time.sleep(0.5)
    print('Camera ready')

    ai = li = di = 0
    while n < a.count:
        angle, light, dist = ANGLES[ai], LIGHTS[li], DISTANCES[di]
        ret, frame = cap.read()
        if not ret: continue

        disp = frame.copy()
        cv2.rectangle(disp, (0, 0), (640, 48), (0, 0, 0), -1)
        cv2.putText(disp, f'{a.c} | {angle} | {light} | {dist} | {n}/{a.count}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.imshow('Win Capture — Enter:shoot Q:quit', disp)
        k = cv2.waitKey(30) & 0xFF

        if k in (13, 32):  # Enter / Space
            n += 1
            stem = f'{a.c}_{angle}_{light}_{dist}_{n:04d}'
            cv2.imwrite(f'{out}/{stem}.jpg', frame)
            print(f'[{n}/{a.count}] {stem}')
        elif k in (ord('a'), ord('A')): ai = (ai+1) % 5; print(f'angle→{angle}')
        elif k in (ord('l'), ord('L')): li = (li+1) % 4; print(f'light→{light}')
        elif k in (ord('d'), ord('D')): di = (di+1) % 3; print(f'dist→{dist}')
        elif k in (ord('q'), ord('Q'), 27): break

    cap.release(); cv2.destroyAllWindows()
    print(f'Done: {n} images → {out}/')


if __name__ == '__main__':
    main()

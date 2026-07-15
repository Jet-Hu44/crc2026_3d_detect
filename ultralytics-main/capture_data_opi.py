"""
香橙派版 — ORBBEC Astra Pro Plus (RGB + Depth)
用法: DISPLAY=:0 python3 capture_data_opi.py --class CB001 --count 60
"""
import os, sys, time, cv2, numpy as np, argparse
from pyorbbecsdk import Config, OBSensorType, Pipeline

ANGLES   = ['0deg','30deg','60deg','90deg','top']
LIGHTS   = ['daylight','whiteLED','yellowLED','mix']
DISTANCES = ['near_06m','mid_12m','far_18m']

def frame_to_bgr(f):
    if f is None: return None
    try:
        h,w = f.get_height(),f.get_width(); d = f.get_data()
        if len(d)!=w*h*3:
            return cv2.imdecode(np.frombuffer(d,np.uint8),cv2.IMREAD_COLOR)
        return cv2.cvtColor(np.frombuffer(d,np.uint8).reshape(h,w,3),cv2.COLOR_RGB2BGR)
    except: return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--class',dest='c',type=str,required=True)
    p.add_argument('--count',type=int,default=60)
    a = p.parse_args()

    img_dir = f'dataset/images/{a.c}'; dep_dir = f'dataset/depth/{a.c}'
    os.makedirs(img_dir,exist_ok=True); os.makedirs(dep_dir,exist_ok=True)
    n = len([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    print(f'{a.c} | target:{a.count} | exist:{n}')

    cfg = Config(); pipe = Pipeline()
    cfg.enable_stream(pipe.get_stream_profile_list(OBSensorType.COLOR_SENSOR).get_default_video_stream_profile())
    try:
        cfg.enable_stream(pipe.get_stream_profile_list(OBSensorType.DEPTH_SENSOR).get_default_video_stream_profile())
        has_d = True
    except: has_d = False
    pipe.start(cfg); time.sleep(1); print('ORBBEC ready')

    ai=li=di=0
    while n < a.count:
        angle,light,dist = ANGLES[ai],LIGHTS[li],DISTANCES[di]
        try:
            fs = pipe.wait_for_frames(200)
            if fs is None: continue
            bgr = frame_to_bgr(fs.get_color_frame())
            if bgr is None: continue
            dm = None
            if has_d:
                df = fs.get_depth_frame()
                if df is not None:
                    dm = np.frombuffer(df.get_data(),np.uint16).reshape(df.get_height(),df.get_width()).astype(np.float32)*df.get_depth_scale()
            disp = bgr.copy()
            cv2.rectangle(disp,(0,0),(640,48),(0,0,0),-1)
            cv2.putText(disp,f'{a.c} | {angle} | {light} | {dist} | {n}/{a.count}',(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,255,0),2)
            cv2.imshow('OPi Capture',disp)
            k = cv2.waitKey(30)&0xFF
            if k in (13,32):
                n+=1; stem=f'{a.c}_{angle}_{light}_{dist}_{n:04d}'
                cv2.imwrite(f'{img_dir}/{stem}.jpg',bgr)
                if dm is not None: np.save(f'{dep_dir}/{stem}.npy',dm)
                print(f'[{n}/{a.count}] {stem}')
            elif k in (ord('a'),ord('A')): ai=(ai+1)%5; print(f'angle→{angle}')
            elif k in (ord('l'),ord('L')): li=(li+1)%4; print(f'light→{light}')
            elif k in (ord('d'),ord('D')): di=(di+1)%3; print(f'dist→{dist}')
            elif k in (ord('q'),ord('Q'),27): break
        except KeyboardInterrupt: break
        except: time.sleep(0.1)
    pipe.stop(); cv2.destroyAllWindows()
    print(f'Done: {n} pairs → {img_dir}/ + {dep_dir}/')

if __name__=='__main__': main()

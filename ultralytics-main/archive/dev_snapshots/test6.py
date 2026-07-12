import socket
import struct
import time
import os
import shutil
from collections import defaultdict, Counter
import glob
import csv
import matplotlib
matplotlib.use('Agg')
import argparse
from pathlib import Path
import cv2
import torch
import numpy as np
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTextEdit
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QTimer, QThread, QObject, pyqtSignal
import threading
from ultralytics import YOLO
from pyorbbecsdk import Config, OBSensorType, OBFormat, Pipeline, OBError, AlignFilter, OBStreamType

# ======= FrameSmoother =======
class FrameSmoother:
    def __init__(self, blend_ratio=0.5):
        self.blend_ratio = blend_ratio
        self.last_frame = None
        
    def smooth_frame(self, current):
        if self.last_frame is None:
            result = current
        else:
            result = cv2.addWeighted(current, self.blend_ratio, 
                                    self.last_frame, 1 - self.blend_ratio, 0)
        self.last_frame = result
        return result

# ======= VisionDetector =======
class VisionDetector:
    def __init__(self, model_file='yolov8n.pt', compute_device='0', use_fp16=False):
        self.min_confidence = 0.50 
        self.compute_device = compute_device
        self.use_fp16 = use_fp16
        try:
            self.detection_engine = YOLO(model_file)
            print(f"已加载YOLOv8模型,类别数:{len(self.detection_engine.names)}")
            self.category_names = self.detection_engine.names
        except Exception as model_err:
            print(f"YOLOv8模型加载失败:{model_err}")
        self.camera_stream = None
        self.depth_aligner = None
        self.has_depth = False
        self.frame_smoother = FrameSmoother(blend_ratio=0.7)
        print('打开软件界面中---30%')
        self.setup_camera()
        print('打开软件界面中---50%')

    def setup_camera(self):
        try:
            os.system('sudo sh -c "echo 2048 > /sys/module/usbcore/parameters/usbfs_memory_mb"')
            print("已尝试增加USB缓冲区大小")
        except:
            print("注意:无法增加USB缓冲区,可能需要管理员权限")
        stream_config = Config()
        self.camera_stream = Pipeline()
        try:
            color_profiles = self.camera_stream.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            color_profile = color_profiles.get_default_video_stream_profile()
            print(f"使用彩色流:{color_profile}")
            stream_config.enable_stream(color_profile)
            self.camera_stream.start(stream_config)
            print("彩色流启动成功")
            print("预热相机中...")
            for _ in range(30):
                try:
                    frames = self.camera_stream.wait_for_frames(200)
                    if frames is None:
                        continue
                    color_frame = frames.get_color_frame()
                    if color_frame is None:
                        continue
                except:
                    pass
                time.sleep(0.3)
            self.camera_stream.stop()
            print("彩色流预热完成")
            stream_config = Config()
            color_profiles = self.camera_stream.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            color_profile = color_profiles.get_default_video_stream_profile()
            stream_config.enable_stream(color_profile)
            try:
                depth_profiles = self.camera_stream.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
                depth_profile = depth_profiles.get_default_video_stream_profile()
                print(f"使用深度流:{depth_profile}")
                stream_config.enable_stream(depth_profile)
                self.depth_aligner = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
                print("已创建深度对齐滤镜,深度图将对齐到彩色图")
                try:
                    self.camera_stream.enable_frame_sync()
                    print("已启用帧同步")
                except Exception as sync_err:
                    print(f"帧同步失败:{sync_err}")
                self.has_depth = True
            except Exception as depth_err:
                print(f"深度流配置失败:{depth_err},将使用仅彩色流模式")
                self.has_depth = False
            self.camera_stream.start(stream_config)
            print("相机启动成功,模式:", "彩色+深度" if self.has_depth else "仅彩色")
        except Exception as cam_err:
            print(f"相机启动失败:{cam_err}")
            self.camera_stream = None

    def convert_to_bgr(self, frame):
        if frame is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        try:
            img_h, img_w = frame.get_height(), frame.get_width()
            raw_bytes = frame.get_data()
            if len(raw_bytes) != img_w * img_h * 3:
                try:
                    img_data = np.frombuffer(raw_bytes, dtype=np.uint8)
                    decoded_img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                    if decoded_img is not None:
                        return decoded_img
                except Exception as decode_err:
                    print(f"解码MJPG失败:{decode_err}")
                return np.zeros((img_h, img_w, 3), dtype=np.uint8)
            else:
                img_array = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((img_h, img_w, 3))
                bgr_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                return bgr_img
        except Exception as convert_err:
            print(f"帧转换失败:{convert_err}")
            return np.zeros((480, 640, 3), dtype=np.uint8)

    def detect(self, input_img=None):
        detections = []
        depth_values = None
        if self.camera_stream is None:
            if input_img is None:
                return [], input_img
            else:
                return [], input_img
        else:
            try:
                while self.camera_stream.poll_for_frames():
                    _ = self.camera_stream.wait_for_frames(1)
            except:
                pass
            try:
                frames = self.camera_stream.wait_for_frames(200)
                if frames is None:
                    print("未获取到帧")
                    return [], input_img if input_img is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                if self.has_depth and self.depth_aligner is not None:
                    try:
                        frames = self.depth_aligner.process(frames)
                    except Exception as align_err:
                        print(f"深度对齐失败:{align_err}")
                color_frame = frames.get_color_frame()
                if color_frame is None:
                    print("未获取到彩色帧")
                    return [], input_img if input_img is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                depth_frame = None if not self.has_depth else frames.get_depth_frame()
                input_img = self.convert_to_bgr(color_frame)
                if self.has_depth and depth_frame is not None:
                    try:
                        raw_depth = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                        d_h, d_w = depth_frame.get_height(), depth_frame.get_width()
                        depth_values = raw_depth.reshape((d_h, d_w)).astype(np.float32) * depth_frame.get_depth_scale()
                        depth_values = self.frame_smoother.smooth_frame(depth_values)
                    except Exception as depth_err:
                        print(f"深度数据处理失败:{depth_err}")
            except Exception as frame_err:
                print(f"获取相机图像失败:{frame_err}")
                if input_img is None:
                    return [], np.zeros((480, 640, 3), dtype=np.uint8)
                else:
                    return [], input_img

        # 执行目标检测
        try:
            outputs = self.detection_engine(input_img, conf=self.min_confidence, iou=0.45)
            if len(outputs) > 0:
                result = outputs[0]
                if hasattr(result, 'boxes') and len(result.boxes) > 0:
                    boxes = result.boxes.data.cpu().numpy()
                    for box in boxes:
                        left, top, right, bottom, conf_score, class_idx = box
                        left, top, right, bottom = map(int, [left, top, right, bottom])
                        class_idx = int(class_idx)
                        class_name = self.category_names[class_idx]
                        center_x = int((left + right) / 2)
                        center_y = int((top + bottom) / 2)
                        dist_value = None
                        if depth_values is not None:
                            if 0 <= center_y < depth_values.shape[0] and 0 <= center_x < depth_values.shape[1]:
                                depth_val = depth_values[center_y, center_x]
                                if 0 < depth_val < 10000:  # 0到10米
                                    dist_value = depth_val
                        detections.append([class_name, float(conf_score), left, top, right, bottom, dist_value])
                        cv2.rectangle(input_img, (left, top), (right, bottom), (0, 0, 255), 2)
                        label = f"{class_name},{conf_score:.2f}"
                        if dist_value is not None:
                            label += f"{dist_value:.0f}mm"
                        cv2.putText(input_img, label, (left-5, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        except Exception as detect_err:
            print(f"目标检测失败:{detect_err}")
        return detections, input_img

    def detect_from_file(self, image_path):
        img = cv2.imread(image_path)
        detections, img = self.detect(img)
        return detections, img

    def save_to_file(self, detections, frame_num, save_dir):
        valid_detections = [d for d in detections if d[1] >= 0.5]
        if valid_detections:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"frame_{frame_num}.txt")
            with open(save_path, 'w', newline='') as f:
                for det in valid_detections:
                    name = det[0]
                    conf = round(det[1], 2)
                    x1 = det[2]
                    y1 = det[3]
                    x2 = det[4]
                    y2 = det[5]
                    cx = int(((x2 - x1) / 2) + x1)
                    cy = int(((y2 - y1) / 2) + y1)
                    depth = "0"
                    if len(det) > 6 and det[6] is not None:
                        depth = f"{int(det[6])}"
                    f.write(f"{name} {cx} {cy} {x1} {x2} {y1} {y2} {conf} {depth}\n")

    def shutdown(self):
        if self.camera_stream is not None:
            self.camera_stream.stop()
            print("相机已关闭")

class DetectionProcessor(QObject):
    processed = pyqtSignal(object, object)  # (processed_img, detections)

    def __init__(self, detector, txt_save_dir, record_video=False, video_dir=None):
        super().__init__()
        self.detector = detector
        self.txt_save_dir = txt_save_dir
        self.frame_count = 0
        self.active = False
        self.record_video = record_video
        self.video_dir = video_dir
        self.video_writer = None

    def run(self):
        self.active = True
        if self.record_video:
            if not os.path.exists(self.video_dir):
                os.makedirs(self.video_dir)
            _, first_img = self.detector.detect()
            if first_img is not None:
                h, w = first_img.shape[:2]
                codec = cv2.VideoWriter_fourcc(*'XVID')
                self.video_writer = cv2.VideoWriter(os.path.join(self.video_dir, 'output_video.avi'),
                                          codec, 15, (w, h))
            else:
                print("无法获取第一帧,视频保存可能失败")
                self.record_video = False
        self.frame_count = 0
        while self.active:
            detections, processed_img = self.detector.detect()
            if processed_img is None or np.all(processed_img == 0):
                print("采集到黑图，跳过本帧")
                time.sleep(0.05)
                continue

            if any(d[1] >= 0.5 for d in detections):
                self.detector.save_to_file(detections, self.frame_count, self.txt_save_dir)
            if self.record_video and self.video_writer is not None:
                self.video_writer.write(processed_img)
            self.processed.emit(processed_img, detections)
            self.frame_count += 1
            time.sleep(0.03)

    def stop(self):
        self.active = False
        if self.video_writer is not None:
            self.video_writer.release()
        self.detector.shutdown()

class AppWindow(QMainWindow):
    def __init__(self, detector):
        super().__init__()
        self.detector = detector
        self.setup_ui()
        self.setup_connection()
        self.processing_thread = None
        self.processor = None
        self.frame_counter = 0
        self.save_video = True
        self.video_path = "/home/HwHiAiUser/ultralytics-main/runss" 
        self.label_dir = "/home/HwHiAiUser/ultralytics-main/runss/labels" 
        self.file_timer = QTimer(self)
        self.file_timer.timeout.connect(self.check_files)
        self.file_timer.start(5000)
        self.start_timer = QTimer(self)
        self.start_timer.timeout.connect(self.auto_start)
        self.start_timer.setSingleShot(True)  
        self.start_timer.start(1000) 

    def setup_ui(self):
        self.setWindowTitle('3D识别')
        self.setGeometry(100, 100, 800, 500)
        self.setStyleSheet("background-color:white;")
        main_layout = QHBoxLayout()
        self.img_label = QLabel(self)
        self.img_label.setFixedSize(500, 400)
        self.img_label.setStyleSheet("border:1px solid black;")
        main_layout.addWidget(self.img_label)
        side_layout = QVBoxLayout()
        side_layout.addSpacing(20)
        result_title = QLabel('识别结果输出区', self)
        result_title.setFont(QFont('Arial', 14))
        result_title.setAlignment(Qt.AlignLeft)
        side_layout.addWidget(result_title)
        self.result_box = QTextEdit(self)
        self.result_box.setReadOnly(True)
        self.result_box.setFixedSize(300, 300)
        self.result_box.setStyleSheet("border:1px solid black;")
        font = QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.result_box.setFont(font)
        side_layout.addWidget(self.result_box)
        self.status_label = QLabel("准备中...", self)  
        self.status_label.setFont(QFont('Arial', 16))
        self.status_label.setFixedSize(300, 60)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("background-color:lightgray;color:black;")
        side_layout.addWidget(self.status_label)
        self.start_btn = QPushButton('自动启动中...', self)
        self.start_btn.setFont(QFont('Arial', 15))
        self.start_btn.setStyleSheet(
            "QPushButton{background-color:#F9C49A;color:black;border-radius:50px;}"
            "QPushButton:pressed{background-color:#FF8C00;}"
        )
        self.start_btn.setFixedSize(100, 100)
        self.start_btn.setEnabled(False)  
        self.start_btn.clicked.connect(self.start_processing)
        side_layout.addWidget(self.start_btn)
        side_layout.addStretch()
        main_layout.addLayout(side_layout)
        central = QWidget(self)
        central.setLayout(main_layout)
        self.setCentralWidget(central)

    def check_files(self):
        target_dir = '/home/HwHiAiUser/ultralytics-main/runss/result' 
        if any(f.endswith('.txt') for f in os.listdir(target_dir)):
            print("已发送xuexiao-tuanduiid-R1.txt到裁判盒,识别结束,准备关闭软件界面。")
            self.close()
        else:
            print("结果文件未生成")

    def setup_connection(self):
        self.server_ip = '192.168.1.66'
        self.server_port = 6666
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.sock.connect((self.server_ip, self.server_port))
            print("打开软件界面中---80%")
            print("Connected to server successfully.")
        except Exception as conn_err:
            print(f"Failed to connect to server: {conn_err}")

    def send_text(self, data_type, text):
        encoded = text.encode()
        length = len(encoded)
        msg = struct.pack('>II', data_type, length) + encoded
        self.sock.sendall(msg)
        
    def send_id(self, data_type, team_id):
        self.send_text(data_type, team_id)
        
    def send_file_data(self, data_type, file_path):
        while not os.path.exists(file_path):
            print(f"文件{file_path}未找到,等待中...")
            time.sleep(0.1)
        with open(file_path, 'rb') as f:
            data = f.read()
        size = len(data)
        header = struct.pack('>II', data_type, size)
        self.sock.sendall(header)
        self.sock.sendall(data)
        
    def send_results(self, result_file):
        result_file = '/home/HwHiAiUser/Desktop/result_r/xuexiao-tuanduiid-R1.txt' 
        self.send_file_data(1, result_file)
        
    def clean_dir(self, dir_path):
        for f in os.listdir(dir_path):
            file_path = os.path.join(dir_path, f)
            os.remove(file_path)
            print(f"文件夹{dir_path}/{f}已被删除")
            
    def move_files(self, src, dest):
        if not os.path.exists(dest):
            os.makedirs(dest)
        for f in os.listdir(src):
            if f.endswith('.txt'):
                src_file = os.path.join(src, f)
                dest_file = os.path.join(dest, f)
                shutil.move(src_file, dest_file)
                print(f"文件{f}从{src}移动到{dest}")
                
    def calc_mode(self, values):
        if not values:
            return None
        freq = Counter(values)
        mode = freq.most_common(1)[0][0]
        return mode
        
    def process_dir(self, dir_path, min_count=5):
        txt_files = glob.glob(os.path.join(dir_path, '*.txt'))
        obj_counts = defaultdict(list)
        frame_count = defaultdict(int)
        for file in txt_files:
            with open(file, 'r') as f:
                lines = f.readlines()
            cur_counts = defaultdict(int)
            for line in lines:
                parts = line.split()
                if not parts or parts[0] in ['Table', 'R_Table']:
                    continue
                if len(parts) >= 8:
                    try:
                        depth = float(parts[-1])
                        if 1000 <= depth <= 1800:
                            obj = parts[0]
                            cur_counts[obj] += 1
                    except ValueError:
                        pass
            for obj, cnt in cur_counts.items():
                obj_counts[obj].append(cnt)
                frame_count[obj] += 1
        result = {}
        for obj in obj_counts:
            if frame_count[obj] >= min_count:
                counter = Counter(obj_counts[obj])
                mode = counter.most_common(1)[0][0]
                result[obj] = mode
        return result
        
    def choose_table(self, tables, lines):
        if not tables:
            return None
        max_cnt = 0
        selected = None
        for table in tables:
            cnt = self.count_in_table(lines, table)
            if cnt > 6 and cnt > max_cnt:
                max_cnt = cnt
                selected = table
        return selected
        
    def count_in_table(self, lines, table):
        x1, x2, y1, y2 = table
        count = 0
        for line in lines:
            parts = line.split()
            if parts and parts[0] not in ['Table', 'R_Table']:
                obj, obj_x1, obj_x2, obj_y1, obj_y2 = parts[0], float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])
                if obj_x1 > x1 and obj_x2 < x2 and obj_y2 < y2 and obj_y2 > y1:
                    count += 1
        return count
        
    def get_table_objects(self, lines, table):
        x1, x2, y1, y2 = table
        objs = []
        exclude = {'Table', 'R_Table'}
        for line in lines:
            parts = line.split()
            if parts and parts[0] not in ['Table', 'R_Table']:
                obj, obj_x1, obj_x2, obj_y1, obj_y2 = parts[0], float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])
                if obj_x1 > x1 and obj_x2 < x2 and obj_y2 < y2 and obj_y2 > y1 and (obj not in exclude):
                    objs.append(obj)
        return objs
        
    def update_counts(self, count_dict, objs):
        local_count = defaultdict(int)
        for obj in objs:
            local_count[obj] += 1
        for obj, cnt in local_count.items():
            count_dict[obj].append(cnt)
            
    def copy_txts(self, src, dest):
        if not os.path.exists(dest):
            os.makedirs(dest)
        txts = glob.glob(os.path.join(src, '*.txt'))
        for txt in txts:
            name = os.path.basename(txt)
            dest_file = os.path.join(dest, name)
            shutil.copy(txt, dest_file)
            print(f"Copied '{txt}' to '{dest_file}'")
            
    def run_detection_cycle(self):
        dirs = ['/home/HwHiAiUser/ultralytics-main/runss/d'] 
        output_file = '/home/HwHiAiUser/ultralytics-main/runss/result/xuexiao-tuanduiid-R1.txt' 
        time.sleep(2)
        self.clean_dir('/home/HwHiAiUser/ultralytics-main/runss/labels') 
        self.clean_dir('/home/HwHiAiUser/ultralytics-main/runss/d') 
        time.sleep(10)
        self.move_files('/home/HwHiAiUser/ultralytics-main/runss/labels', '/home/HwHiAiUser/ultralytics-main/runss/d') 
        time.sleep(0.5)
        total = defaultdict(int)
        for d in dirs:
            stable = self.process_dir(d, min_count=5)
            for obj, cnt in stable.items():
                total[obj] = cnt
        with open(output_file, 'w') as f:
            f.write("START\n")
            for obj, cnt in total.items():
                f.write(f"Goal_ID={obj};Num={cnt}\n")
            f.write("END\n")
        print(f"xuexiao-tuanduiid-R1.txt已生成,路径为：{output_file}")
        self.clean_dir('/home/HwHiAiUser/Desktop/result_r') 
        time.sleep(0.5)
        self.copy_txts('/home/HwHiAiUser/ultralytics-main/runss/result', '/home/HwHiAiUser/Desktop/result_r')
        result_path = '/home/HwHiAiUser/Desktop/result_r/xuexiao-tuanduiid-R1.txt' 
        self.send_file_data(1, result_path)
        self.sock.close()
        print("关闭socket")
        time.sleep(5)
        self.clean_dir('/home/HwHiAiUser/ultralytics-main/runss/result') 
   
    def start_processing(self):
        self.status_label.setText("识别中") 
        self.send_id(0, "幻视")
        os.makedirs(self.label_dir, exist_ok=True)
        self.frame_counter = 0
        self.processing_thread = QThread()
        self.processor = DetectionProcessor(self.detector, txt_save_dir=self.label_dir,
                                  record_video=self.save_video, video_dir=self.video_path)
        self.processor.moveToThread(self.processing_thread)
        self.processor.processed.connect(self.handle_result)
        self.processing_thread.started.connect(self.processor.run)
        self.processing_thread.start()
        threading.Thread(target=self.run_detection_cycle).start()

    def stop_processing(self):
        if self.processor:
            self.processor.stop()
        if self.processing_thread:
            self.processing_thread.quit()
            self.processing_thread.wait()
        self.detector.shutdown()
        cv2.destroyAllWindows()

    def handle_result(self, img, detections):
        if img is not None:
            self.show_image(img)
        if detections is not None:
            self.show_results(detections)
        self.frame_counter += 1

    def show_image(self, img):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_line = ch * w
        qt_img = QImage(rgb.data, w, h, bytes_line, QImage.Format_RGB888)
        scaled = qt_img.scaled(self.img_label.width(), self.img_label.height())
        self.img_label.setPixmap(QPixmap.fromImage(scaled))

    def show_results(self, detections):
        self.result_box.clear()
        counts = {}
        for d in detections:
            obj = d[0]
            if obj in counts:
                counts[obj] += 1
            else:
                counts[obj] = 1
        for d in detections:
            obj = d[0]
            cnt = counts.get(obj, 1)
            dist_txt = ""
            formatted_dist = ""
            if len(d) > 6 and d[6] is not None:
                dist_m = d[6] / 1000
                dist_txt = f"{dist_m:.1f}m"
                if 1.0 <= dist_m <= 1.8: 
                    formatted_dist = f'<span style="color:red;">{dist_txt}</span>'
                else:
                    formatted_dist = dist_txt
            self.result_box.append(f'目标ID:{obj} 数量:{cnt},{formatted_dist}')
            
    def auto_start(self):
        print("界面已显示，自动启动检测...")
        self.start_processing()

if __name__ == '__main__':
    detector = VisionDetector(model_file='best3.pt', compute_device='0') 
    app = QApplication(sys.argv)
    window = AppWindow(detector)
    window.show()
    sys.exit(app.exec_())
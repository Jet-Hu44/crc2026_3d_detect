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

class TemporalFilter:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.previous = None
    def process(self, frame):
        if self.previous is None:
            out = frame
        else:
            out = cv2.addWeighted(frame, self.alpha, self.previous, 1 - self.alpha, 0)
        self.previous = out
        return out

class YoloOrbbecDetector:
    def __init__(self, weights='yolov8n.pt', device='0', half=False):
        self.conf_thres = 0.50                                                         #注意调整
        self.device = device
        self.half = half
        try:
            self.model = YOLO(weights)
            print(f"加载YOLO模型,类别数:{len(self.model.names)}")
            self.names = self.model.names
        except Exception as e:
            print(f"YOLO模型加载失败:{e}")
        self.pipeline = None
        self.align = None
        self.depth_available = False
        self.temporal_filter = TemporalFilter(alpha=0.7)
        #print('打开软件界面中---30%')
        self.init_camera()
        #print('打开软件界面中---50%')

    def init_camera(self):
        try:
            os.system('sudo sh -c "echo 2048 > /sys/module/usbcore/parameters/usbfs_memory_mb"')
            print("已尝试增加USB缓冲区大小")
        except:
            print("无法增加USB缓冲区")
        config = Config()
        self.pipeline = Pipeline()
        try:
            color_profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            color_profile = color_profiles.get_default_video_stream_profile()
            print(f"使用彩色流:{color_profile}")
            config.enable_stream(color_profile)
            self.pipeline.start(config)
            print("彩色流启动成功")
            print("预热相机中...")
            for _ in range(30):
                try:
                    frames = self.pipeline.wait_for_frames(200)
                    if frames is None:
                        continue
                    color_frame = frames.get_color_frame()
                    if color_frame is None:
                        continue
                except:
                    pass
                time.sleep(0.3)
            self.pipeline.stop()
            print("彩色流预热完成")
            config = Config()
            color_profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            color_profile = color_profiles.get_default_video_stream_profile()
            config.enable_stream(color_profile)
            try:
                depth_profiles = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
                depth_profile = depth_profiles.get_default_video_stream_profile()
                print(f"使用深度:{depth_profile}")
                config.enable_stream(depth_profile)
                self.align = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
                print("深度图将对齐到彩色图")
                try:
                    self.pipeline.enable_frame_sync()
                    print("已启用帧同步")
                except Exception as e:
                    print(f"帧同步失败:{e}")
                self.depth_available = True
            except Exception as e:
                print(f"深度流配置失败:{e},使用彩色流模式")
                self.depth_available = False
            self.pipeline.start(config)
            print("相机启动成功,模式:", "彩色+深度" if self.depth_available else "彩色")
        except Exception as e:
            print(f"相机启动失败:{e}")
            self.pipeline = None

    def frame_to_bgr_image(self, frame):
        if frame is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        try:
            h, w = frame.get_height(), frame.get_width()
            data = frame.get_data()
            if len(data) != w * h * 3:
                try:
                    img_array = np.frombuffer(data, dtype=np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if img is not None:
                        return img
                except Exception as e:
                    print(f"解码MJPG失败:{e}")
                return np.zeros((h, w, 3), dtype=np.uint8)
            else:
                img = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                return img
        except Exception as e:
            print(f"帧转换失败:{e}")
            return np.zeros((480, 640, 3), dtype=np.uint8)

    def inference_image(self, opencv_image=None):
        result_list = []
        depth_data = None
        if self.pipeline is None:
            if opencv_image is None:
                return [], opencv_image
            else:
                return [], opencv_image
        else:
            try:
                while self.pipeline.poll_for_frames():
                    _ = self.pipeline.wait_for_frames(1)
            except:
                pass
            try:
                frames = self.pipeline.wait_for_frames(200)
                if frames is None:
                    print("未获取到帧")
                    return [], opencv_image if opencv_image is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                if self.depth_available and self.align is not None:
                    try:
                        frames = self.align.process(frames)
                    except Exception as e:
                        print(f"深度对齐失败:{e}")
                color_frame = frames.get_color_frame()
                if color_frame is None:
                    print("未获取到彩色帧")
                    return [], opencv_image if opencv_image is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                depth_frame = None if not self.depth_available else frames.get_depth_frame()
                opencv_image = self.frame_to_bgr_image(color_frame)
                if self.depth_available and depth_frame is not None:
                    try:
                        raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                        h, w = depth_frame.get_height(), depth_frame.get_width()
                        depth_data = raw.reshape((h, w)).astype(np.float32) * depth_frame.get_depth_scale()
                        depth_data = self.temporal_filter.process(depth_data)
                    except Exception as e:
                        print(f"深度数据处理失败:{e}")
            except Exception as e:
                print(f"获取相机图像失败:{e}")
                if opencv_image is None:
                    return [], np.zeros((480, 640, 3), dtype=np.uint8)
                else:
                    return [], opencv_image

        try:
            results = self.model(opencv_image, conf=self.conf_thres, iou=0.45)
            if len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and len(result.boxes) > 0:
                    boxes_data = result.boxes.data.cpu().numpy()
                    for box in boxes_data:
                        x1, y1, x2, y2, conf, cls_id = box
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                        cls_id = int(cls_id)
                        name = self.names[cls_id]
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        distance = None
                        if depth_data is not None:
                            if 0 <= cy < depth_data.shape[0] and 0 <= cx < depth_data.shape[1]:
                                depth_mm = depth_data[cy, cx]
                                if 0 < depth_mm < 10000:  
                                    distance = depth_mm
                        result_list.append([name, float(conf), x1, y1, x2, y2, distance])
                        cv2.rectangle(opencv_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        label_text = f"{name},{conf:.2f}"
                        if distance is not None:
                            label_text += f"{distance:.0f}mm"
                        cv2.putText(opencv_image, label_text, (x1-5, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        except Exception as e:
            print(f"检测失败:{e}")
        return result_list, opencv_image

    def inference_image_from_file(self, image_file):
        img = cv2.imread(image_file)
        result_list, img = self.inference_image(img)
        return result_list, img

    def write_results_to_txt(self, result_list, frame_idx, output_folder):
        filtered_results = [result for result in result_list if result[1] >= 0.5]
        if filtered_results:
            os.makedirs(output_folder, exist_ok=True)
            output_file = os.path.join(output_folder, f"frame_{frame_idx}.txt")
            with open(output_file, 'w', newline='') as file:
                for result in filtered_results:
                    name = result[0]
                    conf = round(result[1], 2)
                    xmin = result[2]
                    ymin = result[3]
                    xmax = result[4]
                    ymax = result[5]
                    x = int(((xmax - xmin) / 2) + xmin)
                    y = int(((ymax - ymin) / 2) + ymin)
                    depth_info = "0"
                    if len(result) > 6 and result[6] is not None:
                        depth_info = f"{int(result[6])}"
                    file.write(f"{name} {x} {y} {xmin} {xmax} {ymin} {ymax} {conf} {depth_info}\n")

    def close(self):
        if self.pipeline is not None:
            self.pipeline.stop()
            print("相机已关闭")

class DetectWorker(QObject):
    result_ready = pyqtSignal(object, object)  

    def __init__(self, detector, txt_output_folder, save_video=False, video_folder=None):
        super().__init__()
        self.detector = detector
        self.txt_output_folder = txt_output_folder
        self.frame_idx = 0
        self.running = False
        self.save_video = save_video
        self.video_folder = video_folder
        self.out = None

    def start(self):
        self.running = True
        if self.save_video:
            if not os.path.exists(self.video_folder):
                os.makedirs(self.video_folder)
            _, first_frame = self.detector.inference_image()
            if first_frame is not None:
                h, w = first_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.out = cv2.VideoWriter(os.path.join(self.video_folder, 'output_video.avi'),
                                          fourcc, 15, (w, h))
            else:
                print("无法获取第一帧")
                self.save_video = False
        self.frame_idx = 0
        while self.running:
            result_list, result_image = self.detector.inference_image()
            if result_image is None or np.all(result_image == 0):
                print("跳过本帧")
                time.sleep(0.05)
                continue

            if any(result[1] >= 0.5 for result in result_list):
                self.detector.write_results_to_txt(result_list, self.frame_idx, self.txt_output_folder)
            if self.save_video and self.out is not None:
                self.out.write(result_image)
            self.result_ready.emit(result_image, result_list)
            self.frame_idx += 1
            time.sleep(0.03)


    def stop(self):
        self.running = False
        if self.out is not None:
            self.out.release()
        self.detector.close()

class MainWindow(QMainWindow):
    def __init__(self, detector):
        super().__init__()
        self.detector = detector
        self.initUI()
        self.initSocket()
        self.worker_thread = None
        self.worker = None
        self.frame_idx = 0
        self.save_video = True
        self.video_folder = "/home/HwHiAiUser/ultralytics-main/runss" 
        self.txt_output_folder = "/home/HwHiAiUser/ultralytics-main/runss/labels" 
        self.file_timer = QTimer(self)
        self.file_timer.timeout.connect(self.check_txt_files)
        self.file_timer.start(5000)
        self.auto_start_timer = QTimer(self)
        self.auto_start_timer.timeout.connect(self.auto_start_detection)
        self.auto_start_timer.setSingleShot(True)  
        self.auto_start_timer.start(1000)  

def initUI(self):
        self.setWindowTitle('3D视觉识别系统')
        self.setGeometry(100, 100, 1000, 600)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2c3e50;
            }
            QLabel, QTextEdit {
                background-color: #34495e;
                color: #ecf0f1;
                border-radius: 5px;
                padding: 5px;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border-radius: 25px;
                font-weight: bold;
                padding: 10px;
                min-width: 100px;
                min-height: 40px;
            }
            QPushButton:pressed {
                background-color: #2980b9;
            }
            QPushButton:disabled {
                background-color: #7f8c8d;
            }
        """)
        
        # 创建主布局
        main_layout = QHBoxLayout()
        
        # 左侧区域 - 图像显示
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # 图像显示区域
        image_container = QWidget()
        image_container.setStyleSheet("background-color: #1a1a2e; border-radius: 10px;")
        image_layout = QVBoxLayout(image_container)
        
        image_title = QLabel("实时图像")
        image_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #3498db;")
        image_title.setAlignment(Qt.AlignCenter)
        image_layout.addWidget(image_title)
        
        self.image_label = QLabel()
        self.image_label.setFixedSize(640, 480)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: black;")
        image_layout.addWidget(self.image_label, alignment=Qt.AlignCenter)
        
        # 帧率显示
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setStyleSheet("font-size: 14px; color: #2ecc71;")
        self.fps_label.setAlignment(Qt.AlignRight)
        image_layout.addWidget(self.fps_label)
        
        left_layout.addWidget(image_container)
        
        # 右侧区域 - 控制和信息显示
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(15)
        
        # 系统状态面板
        status_panel = QWidget()
        status_panel.setStyleSheet("background-color: #34495e; border-radius: 10px;")
        status_layout = QVBoxLayout(status_panel)
        
        status_title = QLabel("系统状态")
        status_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #3498db;")
        status_title.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(status_title)
        
        # 状态指示器
        self.status_indicator = QLabel()
        self.status_indicator.setFixedSize(30, 30)
        self.status_indicator.setStyleSheet("background-color: #e74c3c; border-radius: 15px;")
        
        status_container = QWidget()
        status_container_layout = QHBoxLayout(status_container)
        status_container_layout.addWidget(QLabel("当前状态:"))
        status_container_layout.addWidget(self.status_indicator)
        status_container_layout.addStretch()
        
        status_layout.addWidget(status_container)
        
        self.status_label = QLabel("系统初始化中...")
        self.status_label.setStyleSheet("font-size: 14px; color: #f1c40f;")
        self.status_label.setAlignment(Qt.AlignCenter)
        status_layout.addWidget(self.status_label)
        
        # 连接状态
        self.connection_status = QLabel("裁判盒: 未连接")
        self.connection_status.setStyleSheet("font-size: 14px; color: #e74c3c;")
        status_layout.addWidget(self.connection_status)
        
        right_layout.addWidget(status_panel)
        
        # 识别结果面板
        results_panel = QWidget()
        results_panel.setStyleSheet("background-color: #34495e; border-radius: 10px;")
        results_layout = QVBoxLayout(results_panel)
        
        results_title = QLabel("识别结果")
        results_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #3498db;")
        results_title.setAlignment(Qt.AlignCenter)
        results_layout.addWidget(results_title)
        
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setFixedHeight(200)
        self.result_text.setStyleSheet("""
            QTextEdit {
                font-family: Consolas, monospace;
                font-size: 14px;
                background-color: #2c3e50;
                color: #ecf0f1;
                border: 1px solid #3498db;
            }
        """)
        results_layout.addWidget(self.result_text)
        
        right_layout.addWidget(results_panel)
        
        # 控制面板
        control_panel = QWidget()
        control_layout = QVBoxLayout(control_panel)
        control_layout.setSpacing(10)
        
        self.start_button = QPushButton('启动识别')
        self.start_button.setFixedHeight(60)
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #27ae60;
                font-size: 18px;
                font-weight: bold;
            }
            QPushButton:pressed {
                background-color: #219653;
            }
        """)
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_detection)
        control_layout.addWidget(self.start_button)
        
        # 统计信息
        stats_container = QWidget()
        stats_layout = QHBoxLayout(stats_container)
        
        self.objects_label = QLabel("目标: 0")
        self.objects_label.setStyleSheet("font-size: 14px; color: #3498db;")
        stats_layout.addWidget(self.objects_label)
        
        self.frames_label = QLabel("帧数: 0")
        self.frames_label.setStyleSheet("font-size: 14px; color: #3498db;")
        stats_layout.addWidget(self.frames_label)
        
        control_layout.addWidget(stats_container)
        
        right_layout.addWidget(control_panel)
        
        # 添加左右区域到主布局
        main_layout.addWidget(left_widget, 60)
        main_layout.addWidget(right_widget, 40)
        
        # 设置中央部件
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        # 初始化计时器用于FPS计算
        self.frame_count = 0
        self.fps_timer = QTimer(self)
        self.fps_timer.timeout.connect(self.update_fps)
        self.last_time = time.time()

    def update_fps(self):
        current_time = time.time()
        elapsed = current_time - self.last_time
        if elapsed > 0:
            fps = self.frame_count / elapsed
            self.fps_label.setText(f"FPS: {fps:.1f}")
        self.last_time = current_time
        self.frame_count = 0
    
    def check_txt_files(self):
        directory = '/home/HwHiAiUser/ultralytics-main/runss/result' 
        if any(file.endswith('.txt') for file in os.listdir(directory)):
            print("已发送NEEPU-HS-R1.txt到裁判盒,识别结束")
            self.close()
        else:
            print("结果文件未生成")

    def initSocket(self):
        self.host = '192.168.1.66'
        self.port = 6666
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.s.connect((self.host, self.port))
            self.connection_status.setText("裁判盒: 已连接")
            self.connection_status.setStyleSheet("font-size: 14px; color: #2ecc71;")
        except Exception as e:
            self.connection_status.setText(f"裁判盒: 连接失败 ({e})")
            self.connection_status.setStyleSheet("font-size: 14px; color: #e74c3c;")

    def send_string(self, datatype, data):
        encoded_data = data.encode()
        data_length = len(encoded_data)
        message = struct.pack('>II', datatype, data_length) + encoded_data
        self.s.sendall(message)
    def send_team_id(self, data_type, team_id):
        self.send_string(data_type, team_id)
    def send_file(self, datatype, file_path):
        while not os.path.exists(file_path):
            print(f"文件{file_path}未找到,等待中")
            time.sleep(0.1)
        with open(file_path, 'rb') as file:
            file_data = file.read()
        data_length = len(file_data)
        header = struct.pack('>II', datatype, data_length)
        self.s.sendall(header)
        self.s.sendall(file_data)
    def send_result(self, file_path):
        file_path = '/home/HwHiAiUser/Desktop/result_r/NEEPU-HS-R1.txt'
        self.send_file(1, file_path)
    def delete_all_files_in_folder(self, folder_path):
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            os.remove(file_path)
            print(f"文件夹{folder_path}/{file_name}已被删除")
    def move_txt_files(self, src_folder, dest_folder):
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
        for file_name in os.listdir(src_folder):
            if file_name.endswith('.txt'):
                src_path = os.path.join(src_folder, file_name)
                dest_path = os.path.join(dest_folder, file_name)
                shutil.move(src_path, dest_path)
                print(f"文件{file_name}从{src_folder}移动到{dest_folder}")
    def get_mode(self, counts):
        if not counts:
            return None
        count_freq = Counter(counts)
        mode = count_freq.most_common(1)[0][0]
        return mode
    def process_folder(self, folder, min_occurrences=5):
        txt_files = glob.glob(os.path.join(folder, '*.txt'))
        object_counts = defaultdict(list)
        frame_presence = defaultdict(int)
        for file in txt_files:
            with open(file, 'r') as f:
                lines = f.readlines()
            current_frame_counts = defaultdict(int)
            for line in lines:
                words = line.split()
                if not words or words[0] in ['Table', 'R_Table']:
                    continue
                if len(words) >= 8:
                    try:
                        depth = float(words[-1])
                        if 800 <= depth <= 1800:                                                  #注意调整
                            obj_name = words[0]
                            current_frame_counts[obj_name] += 1
                    except ValueError:
                        pass
            for obj, count in current_frame_counts.items():
                object_counts[obj].append(count)
                frame_presence[obj] += 1
        result = {}
        for obj in object_counts:
            if frame_presence[obj] >= min_occurrences:
                counter = Counter(object_counts[obj])
                mode = counter.most_common(1)[0][0]
                result[obj] = mode
        return result
    def select_table(self, tables, lines):
        if not tables:
            return None
        max_count = 0
        selected_table = None
        for table in tables:
            count = self.count_words_in_table(lines, table)
            if count > 6 and count > max_count:
                max_count = count
                selected_table = table
        return selected_table
    def count_words_in_table(self, lines, table):
        x_min, x_max, y_min, y_max = table
        count = 0
        for line in lines:
            words = line.split()
            if words and words[0] not in ['Table', 'R_Table']:
                word, word_x_min, word_x_max, word_y_min, word_y_max = words[0], float(words[3]), float(words[4]), float(words[5]), float(words[6])
                if word_x_min > x_min and word_x_max < x_max and word_y_max < y_max and word_y_max > y_min:
                    count += 1
        return count
    def extract_words_in_table(self, lines, table):
        x_min, x_max, y_min, y_max = table
        words_in_table = []
        excluded_words = {'Table', 'R_Table'}
        for line in lines:
            words = line.split()
            if words and words[0] not in ['Table', 'R_Table']:
                word, word_x_min, word_x_max, word_y_min, word_y_max = words[0], float(words[3]), float(words[4]), float(words[5]), float(words[6])
                if word_x_min > x_min and word_x_max < x_max and word_y_max < y_max and word_y_max > y_min and (word not in excluded_words):
                    words_in_table.append(word)
        return words_in_table
    def update_word_counts(self, word_counts, words):
        local_max_count = defaultdict(int)
        for word in words:
            local_max_count[word] += 1
        for word, count in local_max_count.items():
            word_counts[word].append(count)
    def copy_txt_files(self, source_folder, destination_folder):
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
        txt_files = glob.glob(os.path.join(source_folder, '*.txt'))
        for file in txt_files:
            base_name = os.path.basename(file)
            destination_file = os.path.join(destination_folder, base_name)
            shutil.copy(file, destination_file)
            print(f"Copied '{file}' to '{destination_file}'")
    def process_detection_cycle(self):
        folders = ['/home/HwHiAiUser/ultralytics-main/runss/d'] 
        output_path = '/home/HwHiAiUser/ultralytics-main/runss/result/NEEPU-HS-R1.txt' 
        time.sleep(2)
        self.delete_all_files_in_folder('/home/HwHiAiUser/ultralytics-main/runss/labels') 
        self.delete_all_files_in_folder('/home/HwHiAiUser/ultralytics-main/runss/d') 
        time.sleep(17)
        self.move_txt_files('/home/HwHiAiUser/ultralytics-main/runss/labels', '/home/HwHiAiUser/ultralytics-main/runss/d') 
        time.sleep(0.5)
        total_counts = defaultdict(int)
        for folder in folders:
            stable_targets = self.process_folder(folder, min_occurrences=7)                                           #注意调整
            for word, count in stable_targets.items():
                total_counts[word] = count
        with open(output_path, 'w') as f:
            f.write("START\n")
            for word, count in total_counts.items():
                f.write(f"Goal_ID={word};Num={count}\n")
            f.write("END\n")
        print(f"NEEPU-HS-R1.txt已生成,路径为：{output_path}")
        self.delete_all_files_in_folder('/home/HwHiAiUser/Desktop/result_r') 
        time.sleep(0.5)
        self.copy_txt_files('/home/HwHiAiUser/ultralytics-main/runss/result', '/home/HwHiAiUser/Desktop/result_r') 
        file_path = '/home/HwHiAiUser/Desktop/result_r/NEEPU-HS-R1.txt' 
        self.send_file(1, file_path)
        self.s.close()
        print("关闭socket")
        time.sleep(5)
        self.delete_all_files_in_folder('/home/HwHiAiUser/ultralytics-main/runss/result') 

    def start_detection(self):
        self.status_label.setText("识别中") 
        self.send_team_id(0, "幻觉")  
        os.makedirs(self.txt_output_folder, exist_ok=True)
        self.frame_idx = 0
        self.worker_thread = QThread()
        self.worker = DetectWorker(self.detector, txt_output_folder=self.txt_output_folder,
                                  save_video=self.save_video, video_folder=self.video_folder)
        self.worker.moveToThread(self.worker_thread)
        self.worker.result_ready.connect(self.on_result_ready)
        self.worker_thread.started.connect(self.worker.start)
        self.worker_thread.start()
        threading.Thread(target=self.process_detection_cycle).start()

    def stop_detection(self):
        if self.worker:
            self.worker.stop()
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
        self.detector.close()
        cv2.destroyAllWindows()

    def on_result_ready(self, result_image, result_list):
        if result_image is not None:
            self.display_image(result_image)
            self.frame_count += 1  # 更新帧计数
        
        if result_list is not None:
            self.display_results(result_list)
            self.objects_label.setText(f"目标: {len(result_list)}")
        
        self.frames_label.setText(f"帧数: {self.frame_idx}")
        self.frame_idx += 1

    def display_image(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.image_label.width(), self.image_label.height())
        self.image_label.setPixmap(QPixmap.fromImage(p))

    def display_results(self, result_list):
        self.result_text.clear()
        if not result_list:
            self.result_text.append("未检测到目标")
            return
        
        # 按类别分组
        categories = defaultdict(list)
        for result in result_list:
            categories[result[0]].append(result)
        
        # 显示每个类别的信息
        for category, items in categories.items():
            self.result_text.append(f"<b>{category} ({len(items)}个):</b>")
            for item in items:
                distance_text = ""
                if len(item) > 6 and item[6] is not None:
                    distance_m = item[6] / 1000
                    distance_text = f"距离: {distance_m:.2f}m"
                
                conf_text = f"置信度: {item[1]*100:.1f}%"
                position_text = f"位置: ({item[2]}, {item[3]})"
                
                # 高亮近距离目标
                if distance_text and 0.8 <= distance_m <= 1.8:
                    self.result_text.append(f"  <span style='color:#e74c3c'>{conf_text}, {distance_text}, {position_text}</span>")
                else:
                    self.result_text.append(f"  {conf_text}, {distance_text}, {position_text}")
        
        self.result_text.append("")

    def start_detection(self):
        self.status_indicator.setStyleSheet("background-color: #2ecc71; border-radius: 15px;")
        self.status_label.setText("识别中")
        self.status_label.setStyleSheet("font-size: 14px; color: #2ecc71;")

    def auto_start_detection(self):
        print("自动启动中")
        self.start_detection()
        self.fps_timer.start(1000)

if __name__ == '__main__':
    detector = YoloOrbbecDetector(weights='best4.pt', device='0') 
    app = QApplication(sys.argv)
    main_window = MainWindow(detector)
    main_window.show()
    sys.exit(app.exec_())
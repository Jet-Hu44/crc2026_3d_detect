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

# ======= TimeSmoother =======
class TimeSmoother:
    """时间滤波器, 用于平滑深度数据"""
    def __init__(self, smoothing_factor=0.5):
        self.smoothing_factor = smoothing_factor
        self.prev_frame = None
        
    def apply_filter(self, current_frame):
        if self.prev_frame is None:
            output = current_frame
        else:
            output = cv2.addWeighted(current_frame, self.smoothing_factor, self.prev_frame, 1 - self.smoothing_factor, 0)
        self.prev_frame = output
        return output

# ======= ObjectDetectorWithCamera =======
class ObjectDetectorWithCamera:
    def __init__(self, model_weights='yolov8n.pt', device_id='0', use_half=False):
        self.confidence_threshold = 0.50  # 置信度 #--------------------------------------------------------------------------------------------------比赛注意
        self.device_id = device_id
        self.use_half = use_half
        try:
            self.detection_model = YOLO(model_weights)
            print(f"已加载YOLOv8模型,类别数:{len(self.detection_model.names)}")
            self.class_names = self.detection_model.names
        except Exception as e:
            print(f"YOLOv8模型加载失败:{e}")
        self.camera_pipeline = None
        self.alignment_filter = None
        self.depth_supported = False
        self.time_filter = TimeSmoother(smoothing_factor=0.7)
        print('打开软件界面中---30%')
        self.initialize_camera()
        print('打开软件界面中---50%')

    def initialize_camera(self):
        try:
            os.system('sudo sh -c "echo 2048 > /sys/module/usbcore/parameters/usbfs_memory_mb"')
            print("已尝试增加USB缓冲区大小")
        except:
            print("注意:无法增加USB缓冲区,可能需要管理员权限")
        camera_config = Config()
        self.camera_pipeline = Pipeline()
        try:
            color_profiles = self.camera_pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            color_profile = color_profiles.get_default_video_stream_profile()
            print(f"使用彩色流:{color_profile}")
            camera_config.enable_stream(color_profile)
            self.camera_pipeline.start(camera_config)
            print("彩色流启动成功")
            print("预热相机中...")
            for _ in range(30):
                try:
                    frame_set = self.camera_pipeline.wait_for_frames(200)
                    if frame_set is None:
                        continue
                    color_frame = frame_set.get_color_frame()
                    if color_frame is None:
                        continue
                except:
                    pass
                time.sleep(0.3)
            self.camera_pipeline.stop()
            print("彩色流预热完成")
            camera_config = Config()
            color_profiles = self.camera_pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            color_profile = color_profiles.get_default_video_stream_profile()
            camera_config.enable_stream(color_profile)
            try:
                depth_profiles = self.camera_pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
                depth_profile = depth_profiles.get_default_video_stream_profile()
                print(f"使用深度流:{depth_profile}")
                camera_config.enable_stream(depth_profile)
                self.alignment_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
                print("已创建深度对齐滤镜,深度图将对齐到彩色图")
                try:
                    self.camera_pipeline.enable_frame_sync()
                    print("已启用帧同步")
                except Exception as e:
                    print(f"帧同步失败:{e}")
                self.depth_supported = True
            except Exception as e:
                print(f"深度流配置失败:{e},将使用仅彩色流模式")
                self.depth_supported = False
            self.camera_pipeline.start(camera_config)
            print("相机启动成功,模式:", "彩色+深度" if self.depth_supported else "仅彩色")
        except Exception as e:
            print(f"相机启动失败:{e}")
            self.camera_pipeline = None

    def convert_frame_to_image(self, frame_data):
        if frame_data is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        try:
            height, width = frame_data.get_height(), frame_data.get_width()
            raw_data = frame_data.get_data()
            if len(raw_data) != width * height * 3:
                try:
                    image_array = np.frombuffer(raw_data, dtype=np.uint8)
                    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                    if image is not None:
                        return image
                except Exception as e:
                    print(f"解码MJPG失败:{e}")
                return np.zeros((height, width, 3), dtype=np.uint8)
            else:
                image = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width, 3))
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                return image
        except Exception as e:
            print(f"帧转换失败:{e}")
            return np.zeros((480, 640, 3), dtype=np.uint8)

    def detect_objects(self, input_image=None):
        detection_results = []
        depth_map = None
        if self.camera_pipeline is None:
            if input_image is None:
                return [], input_image
            else:
                return [], input_image
        else:
            try:
                while self.camera_pipeline.poll_for_frames():
                    _ = self.camera_pipeline.wait_for_frames(1)
            except:
                pass
            try:
                frame_set = self.camera_pipeline.wait_for_frames(200)
                if frame_set is None:
                    print("未获取到帧")
                    return [], input_image if input_image is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                if self.depth_supported and self.alignment_filter is not None:
                    try:
                        frame_set = self.alignment_filter.process(frame_set)
                    except Exception as e:
                        print(f"深度对齐失败:{e}")
                color_frame = frame_set.get_color_frame()
                if color_frame is None:
                    print("未获取到彩色帧")
                    return [], input_image if input_image is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                depth_frame = None if not self.depth_supported else frame_set.get_depth_frame()
                input_image = self.convert_frame_to_image(color_frame)
                if self.depth_supported and depth_frame is not None:
                    try:
                        raw_depth = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                        d_height, d_width = depth_frame.get_height(), depth_frame.get_width()
                        depth_map = raw_depth.reshape((d_height, d_width)).astype(np.float32) * depth_frame.get_depth_scale()
                        depth_map = self.time_filter.apply_filter(depth_map)
                    except Exception as e:
                        print(f"深度数据处理失败:{e}")
            except Exception as e:
                print(f"获取相机图像失败:{e}")
                if input_image is None:
                    return [], np.zeros((480, 640, 3), dtype=np.uint8)
                else:
                    return [], input_image

        # 执行YOLO目标检测
        try:
            detection_output = self.detection_model(input_image, conf=self.confidence_threshold, iou=0.45)
            if len(detection_output) > 0:
                detection_result = detection_output[0]
                if hasattr(detection_result, 'boxes') and len(detection_result.boxes) > 0:
                    bounding_boxes = detection_result.boxes.data.cpu().numpy()
                    for box in bounding_boxes:
                        x1, y1, x2, y2, conf, class_id = box
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                        class_id = int(class_id)
                        class_name = self.class_names[class_id]
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        distance_value = None
                        if depth_map is not None:
                            if 0 <= center_y < depth_map.shape[0] and 0 <= center_x < depth_map.shape[1]:
                                depth_value = depth_map[center_y, center_x]
                                if 0 < depth_value < 10000:  # 0到10米
                                    distance_value = depth_value
                        detection_results.append([class_name, float(conf), x1, y1, x2, y2, distance_value])
                        cv2.rectangle(input_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        label_text = f"{class_name},{conf:.2f}"
                        if distance_value is not None:
                            label_text += f"{distance_value:.0f}mm"
                        cv2.putText(input_image, label_text, (x1-5, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        except Exception as e:
            print(f"目标检测失败:{e}")
        return detection_results, input_image

    def detect_from_image_file(self, image_path):
        img = cv2.imread(image_path)
        detection_results, img = self.detect_objects(img)
        return detection_results, img

    def save_detection_results(self, results, frame_index, output_directory):
        filtered_results = [res for res in results if res[1] >= 0.5]
        if filtered_results:
            os.makedirs(output_directory, exist_ok=True)
            output_file = os.path.join(output_directory, f"frame_{frame_index}.txt")
            with open(output_file, 'w', newline='') as file:
                for res in filtered_results:
                    class_name = res[0]
                    confidence = round(res[1], 2)
                    x_min = res[2]
                    y_min = res[3]
                    x_max = res[4]
                    y_max = res[5]
                    center_x = int(((x_max - x_min) / 2) + x_min)
                    center_y = int(((y_max - y_min) / 2) + y_min)
                    depth_value = "0"
                    if len(res) > 6 and res[6] is not None:
                        depth_value = f"{int(res[6])}"
                    file.write(f"{class_name} {center_x} {center_y} {x_min} {x_max} {y_min} {y_max} {confidence} {depth_value}\n")

    def shutdown(self):
        if self.camera_pipeline is not None:
            self.camera_pipeline.stop()
            print("相机已关闭")

# ======= DetectionWorkerThread 子线程 =======
class DetectionWorkerThread(QObject):
    detection_complete = pyqtSignal(object, object)  # (result_image, result_list)

    def __init__(self, detector_instance, output_directory, save_video_stream=False, video_directory=None):
        super().__init__()
        self.detector = detector_instance
        self.output_directory = output_directory
        self.current_frame_index = 0
        self.is_running = False
        self.save_video_stream = save_video_stream
        self.video_directory = video_directory
        self.video_writer = None

    def start_processing(self):
        self.is_running = True
        if self.save_video_stream:
            if not os.path.exists(self.video_directory):
                os.makedirs(self.video_directory)
            _, initial_frame = self.detector.detect_objects()
            if initial_frame is not None:
                height, width = initial_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.video_writer = cv2.VideoWriter(os.path.join(self.video_directory, 'output_video.avi'),
                                          fourcc, 15, (width, height))
            else:
                print("无法获取第一帧,视频保存可能失败")
                self.save_video_stream = False
        self.current_frame_index = 0
        while self.is_running:
            detection_results, processed_image = self.detector.detect_objects()
            # 判断是否黑图或采集失败，黑图不emit，不刷新UI
            if processed_image is None or np.all(processed_image == 0):
                print("采集到黑图，跳过本帧")
                time.sleep(0.05)
                continue

            if any(res[1] >= 0.5 for res in detection_results):
                self.detector.save_detection_results(detection_results, self.current_frame_index, self.output_directory)
            if self.save_video_stream and self.video_writer is not None:
                self.video_writer.write(processed_image)
            self.detection_complete.emit(processed_image, detection_results)
            self.current_frame_index += 1
            time.sleep(0.03)

    def stop_processing(self):
        self.is_running = False
        if self.video_writer is not None:
            self.video_writer.release()
        self.detector.shutdown()

# ======= PrimaryWindow ============
class PrimaryWindow(QMainWindow):
    def __init__(self, detector_instance):
        super().__init__()
        self.detector = detector_instance
        self.setup_ui()
        self.initialize_socket()
        self.worker_thread = None
        self.worker = None
        self.frame_counter = 0
        self.save_video_flag = True
        self.video_storage_path = "/home/HwHiAiUser/ultralytics-main/runss" # 目录名 #--------------------------------------------------------------------------------------------------比赛注意
        self.output_directory = "/home/HwHiAiUser/ultralytics-main/runss/labels" # 目录名 #--------------------------------------------------------------------------------------------------比赛注意
        self.file_check_timer = QTimer(self)
        self.file_check_timer.timeout.connect(self.check_for_result_files)
        self.file_check_timer.start(5000)
        self.auto_start_timer = QTimer(self)
        self.auto_start_timer.timeout.connect(self.auto_start_detection)
        self.auto_start_timer.setSingleShot(True)  # 只执行一次
        self.auto_start_timer.start(1000)  # 1秒后自动启动

    def setup_ui(self):
        self.setWindowTitle('3D识别')
        self.setGeometry(100, 100, 800, 500)
        self.setStyleSheet("background-color:blue;")
        main_layout = QHBoxLayout()
        self.image_display = QLabel(self)
        self.image_display.setFixedSize(500, 400)
        self.image_display.setStyleSheet("border:1px solid black;")
        main_layout.addWidget(self.image_display)
        side_layout = QVBoxLayout()
        side_layout.addSpacing(20)
        results_header = QLabel('识别结果输出区', self)
        results_header.setFont(QFont('Arial', 14))
        results_header.setAlignment(Qt.AlignLeft)
        side_layout.addWidget(results_header)
        self.result_display = QTextEdit(self)
        self.result_display.setReadOnly(True)
        self.result_display.setFixedSize(300, 300)
        self.result_display.setStyleSheet("border:1px solid black;")
        text_font = QFont()
        text_font.setFamily("Arial")
        text_font.setPointSize(14)
        self.result_display.setFont(text_font)
        side_layout.addWidget(self.result_display)
        self.status_indicator = QLabel("准备中...", self)  # 修改初始状态文本
        self.status_indicator.setFont(QFont('Arial', 16))
        self.status_indicator.setFixedSize(300, 60)
        self.status_indicator.setAlignment(Qt.AlignCenter)
        self.status_indicator.setStyleSheet("background-color:lightgray;color:black;")
        side_layout.addWidget(self.status_indicator)
        self.start_button = QPushButton('自动启动中...', self)
        self.start_button.setFont(QFont('Arial', 15))
        self.start_button.setStyleSheet(
            "QPushButton{background-color:#F9C49A;color:black;border-radius:50px;}"
            "QPushButton:pressed{background-color:#FF8C00;}"
        )
        self.start_button.setFixedSize(100, 100)
        self.start_button.setEnabled(False)  # 禁用按钮
        self.start_button.clicked.connect(self.begin_detection)
        side_layout.addWidget(self.start_button)
        side_layout.addStretch()
        main_layout.addLayout(side_layout)
        central_widget = QWidget(self)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def check_for_result_files(self):
        result_dir = '/home/HwHiAiUser/ultralytics-main/runss/result' # 目录名 #--------------------------------------------------------------------------------------------------比赛注意
        if any(filename.endswith('.txt') for filename in os.listdir(result_dir)):
            print("已发送xuexiao-tuanduiid-R1.txt到裁判盒,识别结束,准备关闭软件界面。")
            self.close()
        else:
            print("结果文件未生成")

    def initialize_socket(self):
        self.server_host = '192.168.1.66'
        self.server_port = 6666
        self.socket_connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.socket_connection.connect((self.server_host, self.server_port))
            print("打开软件界面中---80%")
            print("Connected to server successfully.")
        except Exception as e:
            print(f"Failed to connect to server: {e}")

    def send_data_string(self, data_type, data_content):
        encoded_data = data_content.encode()
        data_length = len(encoded_data)
        message = struct.pack('>II', data_type, data_length) + encoded_data
        self.socket_connection.sendall(message)
        
    def send_team_identifier(self, data_type, team_id):
        self.send_data_string(data_type, team_id)
        
    def send_file_data(self, data_type, file_path):
        while not os.path.exists(file_path):
            print(f"文件{file_path}未找到,等待中...")
            time.sleep(0.1)
        with open(file_path, 'rb') as file:
            file_content = file.read()
        data_length = len(file_content)
        header = struct.pack('>II', data_type, data_length)
        self.socket_connection.sendall(header)
        self.socket_connection.sendall(file_content)
        
    def transmit_results(self, result_file_path):
        result_file_path = '/home/HwHiAiUser/Desktop/result_r/xuexiao-tuanduiid-R1.txt' # 目录名 #--------------------------------------------------------------------------------------------------比赛注意
        self.send_file_data(1, result_file_path)
        
    def remove_directory_contents(self, target_directory):
        for filename in os.listdir(target_directory):
            file_path = os.path.join(target_directory, filename)
            os.remove(file_path)
            print(f"文件夹{target_directory}/{filename}已被删除")
            
    def transfer_text_files(self, source_dir, destination_dir):
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
        for filename in os.listdir(source_dir):
            if filename.endswith('.txt'):
                source_path = os.path.join(source_dir, filename)
                destination_path = os.path.join(destination_dir, filename)
                shutil.move(source_path, destination_path)
                print(f"文件{filename}从{source_dir}移动到{destination_dir}")
                
    def calculate_mode(self, values):
        if not values:
            return None
        frequency_count = Counter(values)
        mode_value = frequency_count.most_common(1)[0][0]
        return mode_value
        
    def analyze_folder(self, folder_path, min_detections=5):
        text_files = glob.glob(os.path.join(folder_path, '*.txt'))
        object_occurrences = defaultdict(list)
        detection_count = defaultdict(int)
        for file_path in text_files:
            with open(file_path, 'r') as file:
                lines = file.readlines()
            current_detections = defaultdict(int)
            for line in lines:
                elements = line.split()
                if not elements or elements[0] in ['Table', 'R_Table']:
                    continue
                if len(elements) >= 8:
                    try:
                        depth_value = float(elements[-1])
                        """
                        depth范围为比赛要求范围，比赛前会测量，根据测量结果修改
                        """
                        if 1000 <= depth_value <= 1800: # 桌面距离 #--------------------------------------------------------------------------------------------------比赛注意
                            object_name = elements[0]
                            current_detections[object_name] += 1
                    except ValueError:
                        pass
            for obj, count in current_detections.items():
                object_occurrences[obj].append(count)
                detection_count[obj] += 1
        final_result = {}
        for obj in object_occurrences:
            if detection_count[obj] >= min_detections:
                frequency = Counter(object_occurrences[obj])
                mode = frequency.most_common(1)[0][0]
                final_result[obj] = mode
        return final_result
        
    def select_table_area(self, table_areas, frame_lines):
        if not table_areas:
            return None
        max_object_count = 0
        selected_table = None
        for table in table_areas:
            object_count = self.count_objects_in_area(frame_lines, table)
            if object_count > 6 and object_count > max_object_count:
                max_object_count = object_count
                selected_table = table
        return selected_table
        
    def count_objects_in_area(self, lines, table_coords):
        x_min, x_max, y_min, y_max = table_coords
        object_count = 0
        for line in lines:
            elements = line.split()
            if elements and elements[0] not in ['Table', 'R_Table']:
                obj_name, obj_x_min, obj_x_max, obj_y_min, obj_y_max = elements[0], float(elements[3]), float(elements[4]), float(elements[5]), float(elements[6])
                if obj_x_min > x_min and obj_x_max < x_max and obj_y_max < y_max and obj_y_max > y_min:
                    object_count += 1
        return object_count
        
    def extract_objects_in_area(self, lines, table_coords):
        x_min, x_max, y_min, y_max = table_coords
        objects_in_area = []
        excluded_objects = {'Table', 'R_Table'}
        for line in lines:
            elements = line.split()
            if elements and elements[0] not in ['Table', 'R_Table']:
                obj_name, obj_x_min, obj_x_max, obj_y_min, obj_y_max = elements[0], float(elements[3]), float(elements[4]), float(elements[5]), float(elements[6])
                if obj_x_min > x_min and obj_x_max < x_max and obj_y_max < y_max and obj_y_max > y_min and (obj_name not in excluded_objects):
                    objects_in_area.append(obj_name)
        return objects_in_area
        
    def update_object_counts(self, count_dict, objects_list):
        local_counts = defaultdict(int)
        for obj in objects_list:
            local_counts[obj] += 1
        for obj, count in local_counts.items():
            count_dict[obj].append(count)
            
    def duplicate_text_files(self, source_folder, target_folder):
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        txt_files = glob.glob(os.path.join(source_folder, '*.txt'))
        for file_path in txt_files:
            base_name = os.path.bas(file_path)
            target_path = os.path.join(target_folder, base_name)
            shutil.copy(file_path, target_path)
            print(f"Copied '{file_path}' to '{target_path}'")
            
    def run_detection_cycle(self):
        data_folders = ['/home/HwHiAiUser/ultralytics-main/runss/d'] # 目录名 #--------------------------------------------------------------------------------------------------比赛注意
        result_file_path = '/home/HwHiAiUser/ultralytics-main/runss/result/xuexiao-tuanduiid-R1.txt' # 目录名 #--------------------------------------------------------------------------------------------------比赛注意
        # 开始后2秒删除所有文件
        time.sleep(2)
        self.remove_directory_contents('/home/HwHiAiUser/ultralytics-main/runss/labels') # 目录名 #--------------------------------------------------------------------------------------------------比赛注意
        self.remove_directory_contents('/home/HwHiAiUser/ultralytics-main/runss/d') # 目录名 #--------------------------------------------------------------------------------------------------比赛注意
        # 检测10秒
        time.sleep(10)
        self.transfer_text_files('/home/HwHiAiUser/ultralytics-main/runss/labels', '/home/HwHiAiUser/ultralytics-main/runss/d') # 目录名 #--------------------------------------------------------------------------------------------------比赛注意
        # 等待0.5秒后处理结果
        time.sleep(0.5)
        total_counts = defaultdict(int)
        for folder in data_folders:

            stable_objects = self.analyze_folder(folder, min_detections=5) # 调整稳定结果次数 #--------------------------------------------------------------------------------------------------比赛注意
            for obj, count in stable_objects.items():
                total_counts[obj] = count
        with open(result_file_path, 'w') as output_file:
            output_file.write("START\n")
            for obj, count in total_counts.items():
                output_file.write(f"Goal_ID={obj};Num={count}\n")
            output_file.write("END\n")
        print(f"xuexiao-tuanduiid-R1.txt已生成,路径为：{result_file_path}")
        self.remove_directory_contents('/home/HwHiAiUser/Desktop/result_r') # 目录名 #--------------------------------------------------------------------------------------------------比赛注意
        time.sleep(0.5)
        self.duplicate_text_files('/home/HwHiAiUser/ultralytics-main/runss/result', '/home/HwHiAiUser/Desktop/result_r')
        final_file_path = '/home/HwHiAiUser/Desktop/result_r/xuexiao-tuanduiid-R1.txt' # 目录名 #--------------------------------------------------------------------------------------------------比赛注意
        self.send_file_data(1, final_file_path)
        self.socket_connection.close()
        print("关闭socket")
        time.sleep(5)
        self.remove_directory_contents('/home/HwHiAiUser/ultralytics-main/runss/result') # 目录名 #--------------------------------------------------------------------------------------------------比赛注意

    # ============ 关键线程控制 ============
    def begin_detection(self):
        self.status_indicator.setText("识别中") # 更新状态为“识别中”
        self.send_team_identifier(0, "tuanduiid")  # 发送团队ID #--------------------------------------------------------------------------------------------------比赛调试：队伍ID
        os.makedirs(self.output_directory, exist_ok=True)
        self.frame_counter = 0
        self.worker_thread = QThread()
        self.worker = DetectionWorkerThread(self.detector, output_directory=self.output_directory,
                                  save_video_stream=self.save_video_flag, video_directory=self.video_storage_path)
        self.worker.moveToThread(self.worker_thread)
        self.worker.detection_complete.connect(self.handle_detection_results)
        self.worker_thread.started.connect(self.worker.start_processing)
        self.worker_thread.start()
        threading.Thread(target=self.run_detection_cycle).start()

    def stop_detection(self):
        if self.worker:
            self.worker.stop_processing()
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
        self.detector.shutdown()
        cv2.destroyAllWindows()

    def handle_detection_results(self, result_image, detection_results):
        if result_image is not None:
            self.show_image(result_image)
        if detection_results is not None:
            self.display_results(detection_results)
        self.frame_counter += 1

    def show_image(self, image_data):
        rgb_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_image.shape
        bytes_per_line = channels * width
        qt_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        scaled_image = qt_image.scaled(self.image_display.width(), self.image_display.height())
        self.image_display.setPixmap(QPixmap.fromImage(scaled_image))

    def display_results(self, detection_results):
        self.result_display.clear()
        object_counts = {}
        for result in detection_results:
            object_type = result[0]
            if object_type in object_counts:
                object_counts[object_type] += 1
            else:
                object_counts[object_type] = 1
        for result in detection_results:
            obj = result[0]
            count = object_counts.get(obj, 1)
            distance_text = ""
            formatted_distance = ""
            if len(result) > 6 and result[6] is not None:
                distance_meters = result[6] / 1000
                distance_text = f"{distance_meters:.1f}m"
                """
                distance_m范围为比赛要求范围，比赛前会测量，根据测量结果修改
                """
                if 1.0 <= distance_meters <= 1.8: # 桌面距离 #--------------------------------------------------------------------------------------------------比赛注意
                    # 红色字体
                    formatted_distance = f'<span style="color:red;">{distance_text}</span>'
                else:
                    formatted_distance = distance_text
            self.result_display.append(f'目标ID:{obj} 数量:{count},{formatted_distance}')

    # 添加自动启动方法
    def auto_start_detection(self):
        """界面显示后自动启动检测"""
        print("界面已显示，自动启动检测...")
        self.begin_detection()

if __name__ == '__main__':
    """更换自己的pt权重文件"""
    detector_instance = ObjectDetectorWithCamera(model_weights='bbest.pt', device_id='0') # 权重替换 #--------------------------------------------------------------------------------------------------比赛注意
    app = QApplication(sys.argv)
    main_window = PrimaryWindow(detector_instance)
    main_window.show()
    sys.exit(app.exec_())
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

# ======= TemporalSmoother =======
class TemporalSmoother:
    def __init__(self, smoothing_factor=0.5):
        self.smoothing_factor = smoothing_factor
        self.previous_frame = None
        
    def apply(self, current_frame):
        if self.previous_frame is None:
            output_frame = current_frame
        else:
            output_frame = cv2.addWeighted(current_frame, self.smoothing_factor, 
                                          self.previous_frame, 1 - self.smoothing_factor, 0)
        self.previous_frame = output_frame
        return output_frame

# ======= YOLOOrbbecDetector =======
class YOLOOrbbecDetector:
    def __init__(self, model_weights='yolov8n.pt', processing_device='0', use_half_precision=False):
        self.confidence_threshold = 0.50 
        self.processing_device = processing_device
        self.use_half_precision = use_half_precision
        try:
            self.detection_model = YOLO(model_weights)
            print(f"已加载YOLOv8模型,类别数:{len(self.detection_model.names)}")
            self.class_names = self.detection_model.names
        except Exception as model_error:
            print(f"YOLOv8模型加载失败:{model_error}")
        self.camera_pipeline = None
        self.align_filter = None
        self.depth_supported = False
        self.temporal_smoother = TemporalSmoother(smoothing_factor=0.7)
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
                self.align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
                print("已创建深度对齐滤镜,深度图将对齐到彩色图")
                try:
                    self.camera_pipeline.enable_frame_sync()
                    print("已启用帧同步")
                except Exception as sync_error:
                    print(f"帧同步失败:{sync_error}")
                self.depth_supported = True
            except Exception as depth_error:
                print(f"深度流配置失败:{depth_error},将使用仅彩色流模式")
                self.depth_supported = False
            self.camera_pipeline.start(camera_config)
            print("相机启动成功,模式:", "彩色+深度" if self.depth_supported else "仅彩色")
        except Exception as camera_error:
            print(f"相机启动失败:{camera_error}")
            self.camera_pipeline = None

    def convert_frame_to_bgr(self, frame_data):
        if frame_data is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        try:
            frame_height, frame_width = frame_data.get_height(), frame_data.get_width()
            raw_data = frame_data.get_data()
            if len(raw_data) != frame_width * frame_height * 3:
                try:
                    image_array = np.frombuffer(raw_data, dtype=np.uint8)
                    decoded_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                    if decoded_image is not None:
                        return decoded_image
                except Exception as decode_error:
                    print(f"解码MJPG失败:{decode_error}")
                return np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
            else:
                image_data = np.frombuffer(raw_data, dtype=np.uint8).reshape((frame_height, frame_width, 3))
                converted_image = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
                return converted_image
        except Exception as conversion_error:
            print(f"帧转换失败:{conversion_error}")
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
                if self.depth_supported and self.align_filter is not None:
                    try:
                        frame_set = self.align_filter.process(frame_set)
                    except Exception as align_error:
                        print(f"深度对齐失败:{align_error}")
                color_frame = frame_set.get_color_frame()
                if color_frame is None:
                    print("未获取到彩色帧")
                    return [], input_image if input_image is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                depth_frame = None if not self.depth_supported else frame_set.get_depth_frame()
                input_image = self.convert_frame_to_bgr(color_frame)
                if self.depth_supported and depth_frame is not None:
                    try:
                        depth_raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                        depth_height, depth_width = depth_frame.get_height(), depth_frame.get_width()
                        depth_map = depth_raw.reshape((depth_height, depth_width)).astype(np.float32) * depth_frame.get_depth_scale()
                        depth_map = self.temporal_smoother.apply(depth_map)
                    except Exception as depth_error:
                        print(f"深度数据处理失败:{depth_error}")
            except Exception as frame_error:
                print(f"获取相机图像失败:{frame_error}")
                if input_image is None:
                    return [], np.zeros((480, 640, 3), dtype=np.uint8)
                else:
                    return [], input_image

        # 执行YOLO目标检测
        try:
            model_outputs = self.detection_model(input_image, conf=self.confidence_threshold, iou=0.45)
            if len(model_outputs) > 0:
                detection_result = model_outputs[0]
                if hasattr(detection_result, 'boxes') and len(detection_result.boxes) > 0:
                    bounding_boxes = detection_result.boxes.data.cpu().numpy()
                    for bbox in bounding_boxes:
                        x1, y1, x2, y2, confidence_score, class_id = bbox
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
                        detection_results.append([class_name, float(confidence_score), x1, y1, x2, y2, distance_value])
                        cv2.rectangle(input_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        label_content = f"{class_name},{confidence_score:.2f}"
                        if distance_value is not None:
                            label_content += f"{distance_value:.0f}mm"
                        cv2.putText(input_image, label_content, (x1-5, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        except Exception as detection_error:
            print(f"目标检测失败:{detection_error}")
        return detection_results, input_image

    def detect_from_image_file(self, image_path):
        image_data = cv2.imread(image_path)
        detection_results, processed_image = self.detect_objects(image_data)
        return detection_results, processed_image

    def save_detection_results(self, detection_data, frame_number, output_directory):
        filtered_results = [result for result in detection_data if result[1] >= 0.5]
        if filtered_results:
            os.makedirs(output_directory, exist_ok=True)
            output_file_path = os.path.join(output_directory, f"frame_{frame_number}.txt")
            with open(output_file_path, 'w', newline='') as output_file:
                for result in filtered_results:
                    class_name = result[0]
                    confidence_score = round(result[1], 2)
                    x_min = result[2]
                    y_min = result[3]
                    x_max = result[4]
                    y_max = result[5]
                    center_x = int(((x_max - x_min) / 2) + x_min)
                    center_y = int(((y_max - y_min) / 2) + y_min)
                    depth_value = "0"
                    if len(result) > 6 and result[6] is not None:
                        depth_value = f"{int(result[6])}"
                    output_file.write(f"{class_name} {center_x} {center_y} {x_min} {x_max} {y_min} {y_max} {confidence_score} {depth_value}\n")

    def close_camera(self):
        if self.camera_pipeline is not None:
            self.camera_pipeline.stop()
            print("相机已关闭")

class DetectionWorker(QObject):
    detection_complete = pyqtSignal(object, object)  # (processed_image, detection_results)

    def __init__(self, detector_instance, txt_output_path, enable_video_save=False, video_output_path=None):
        super().__init__()
        self.detector = detector_instance
        self.txt_output_path = txt_output_path
        self.current_frame_index = 0
        self.is_running = False
        self.enable_video_save = enable_video_save
        self.video_output_path = video_output_path
        self.video_writer = None

    def start_detection(self):
        self.is_running = True
        if self.enable_video_save:
            if not os.path.exists(self.video_output_path):
                os.makedirs(self.video_output_path)
            _, initial_frame = self.detector.detect_objects()
            if initial_frame is not None:
                frame_height, frame_width = initial_frame.shape[:2]
                video_codec = cv2.VideoWriter_fourcc(*'XVID')
                self.video_writer = cv2.VideoWriter(os.path.join(self.video_output_path, 'output_video.avi'),
                                          video_codec, 15, (frame_width, frame_height))
            else:
                print("无法获取第一帧,视频保存可能失败")
                self.enable_video_save = False
        self.current_frame_index = 0
        while self.is_running:
            detection_results, processed_image = self.detector.detect_objects()
            if processed_image is None or np.all(processed_image == 0):
                print("采集到黑图，跳过本帧")
                time.sleep(0.05)
                continue

            if any(result[1] >= 0.5 for result in detection_results):
                self.detector.save_detection_results(detection_results, self.current_frame_index, self.txt_output_path)
            if self.enable_video_save and self.video_writer is not None:
                self.video_writer.write(processed_image)
            self.detection_complete.emit(processed_image, detection_results)
            self.current_frame_index += 1
            time.sleep(0.03)

    def stop_detection(self):
        self.is_running = False
        if self.video_writer is not None:
            self.video_writer.release()
        self.detector.close_camera()

class MainApplicationWindow(QMainWindow):
    def __init__(self, detector_instance):
        super().__init__()
        self.detector = detector_instance
        self.initialize_ui()
        self.initialize_network()
        self.detection_thread = None
        self.detection_worker = None
        self.current_frame_index = 0
        self.save_video_flag = True
        self.video_storage_path = "/home/HwHiAiUser/ultralytics-main/runss" 
        self.txt_output_directory = "/home/HwHiAiUser/ultralytics-main/runss/labels" 
        self.file_check_timer = QTimer(self)
        self.file_check_timer.timeout.connect(self.check_for_result_files)
        self.file_check_timer.start(5000)
        self.auto_start_timer = QTimer(self)
        self.auto_start_timer.timeout.connect(self.auto_launch_detection)
        self.auto_start_timer.setSingleShot(True)  
        self.auto_start_timer.start(1000) 

    def initialize_ui(self):
        self.setWindowTitle('3D识别')
        self.setGeometry(100, 100, 800, 500)
        self.setStyleSheet("background-color:white;")
        horizontal_layout = QHBoxLayout()
        self.display_label = QLabel(self)
        self.display_label.setFixedSize(500, 400)
        self.display_label.setStyleSheet("border:1px solid black;")
        horizontal_layout.addWidget(self.display_label)
        vertical_layout = QVBoxLayout()
        vertical_layout.addSpacing(20)
        results_title = QLabel('识别结果输出区', self)
        results_title.setFont(QFont('Arial', 14))
        results_title.setAlignment(Qt.AlignLeft)
        vertical_layout.addWidget(results_title)
        self.results_display = QTextEdit(self)
        self.results_display.setReadOnly(True)
        self.results_display.setFixedSize(300, 300)
        self.results_display.setStyleSheet("border:1px solid black;")
        text_font = QFont()
        text_font.setFamily("Arial")
        text_font.setPointSize(14)
        self.results_display.setFont(text_font)
        vertical_layout.addWidget(self.results_display)
        self.status_indicator = QLabel("准备中...", self)  
        self.status_indicator.setFont(QFont('Arial', 16))
        self.status_indicator.setFixedSize(300, 60)
        self.status_indicator.setAlignment(Qt.AlignCenter)
        self.status_indicator.setStyleSheet("background-color:lightgray;color:black;")
        vertical_layout.addWidget(self.status_indicator)
        self.start_detection_button = QPushButton('自动启动中...', self)
        self.start_detection_button.setFont(QFont('Arial', 15))
        self.start_detection_button.setStyleSheet(
            "QPushButton{background-color:#F9C49A;color:black;border-radius:50px;}"
            "QPushButton:pressed{background-color:#FF8C00;}"
        )
        self.start_detection_button.setFixedSize(100, 100)
        self.start_detection_button.setEnabled(False)  
        self.start_detection_button.clicked.connect(self.begin_detection)
        vertical_layout.addWidget(self.start_detection_button)
        vertical_layout.addStretch()
        horizontal_layout.addLayout(vertical_layout)
        main_widget = QWidget(self)
        main_widget.setLayout(horizontal_layout)
        self.setCentralWidget(main_widget)

    def check_for_result_files(self):
        target_directory = '/home/HwHiAiUser/ultralytics-main/runss/result' 
        if any(filename.endswith('.txt') for filename in os.listdir(target_directory)):
            print("已发送xuexiao-tuanduiid-R1.txt到裁判盒,识别结束,准备关闭软件界面。")
            self.close()
        else:
            print("结果文件未生成")

    def initialize_network(self):
        self.server_address = '192.168.1.66'
        self.server_port = 6666
        self.network_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.network_socket.connect((self.server_address, self.server_port))
            print("打开软件界面中---80%")
            print("Connected to server successfully.")
        except Exception as connection_error:
            print(f"Failed to connect to server: {connection_error}")

    def send_text_data(self, data_type, text_content):
        encoded_content = text_content.encode()
        content_length = len(encoded_content)
        message_packet = struct.pack('>II', data_type, content_length) + encoded_content
        self.network_socket.sendall(message_packet)
        
    def send_identifier(self, data_type, identifier):
        self.send_text_data(data_type, identifier)
        
    def transfer_file(self, data_type, file_location):
        while not os.path.exists(file_location):
            print(f"文件{file_location}未找到,等待中...")
            time.sleep(0.1)
        with open(file_location, 'rb') as file_handle:
            file_contents = file_handle.read()
        file_size = len(file_contents)
        header_info = struct.pack('>II', data_type, file_size)
        self.network_socket.sendall(header_info)
        self.network_socket.sendall(file_contents)
        
    def transfer_results(self, result_location):
        result_location = '/home/HwHiAiUser/Desktop/result_r/xuexiao-tuanduiid-R1.txt' 
        self.transfer_file(1, result_location)
        
    def clear_directory(self, directory_path):
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            os.remove(file_path)
            print(f"文件夹{directory_path}/{filename}已被删除")
            
    def move_text_files(self, source_path, destination_path):
        if not os.path.exists(destination_path):
            os.makedirs(destination_path)
        for filename in os.listdir(source_path):
            if filename.endswith('.txt'):
                source_file = os.path.join(source_path, filename)
                destination_file = os.path.join(destination_path, filename)
                shutil.move(source_file, destination_file)
                print(f"文件{filename}从{source_path}移动到{destination_path}")
                
    def calculate_mode(self, values):
        if not values:
            return None
        frequency_count = Counter(values)
        most_common = frequency_count.most_common(1)[0][0]
        return most_common
        
    def analyze_directory(self, directory_path, min_frames=5):
        text_files = glob.glob(os.path.join(directory_path, '*.txt'))
        object_counts = defaultdict(list)
        frame_counts = defaultdict(int)
        for file_path in text_files:
            with open(file_path, 'r') as file_reader:
                file_lines = file_reader.readlines()
            current_counts = defaultdict(int)
            for line in file_lines:
                words = line.split()
                if not words or words[0] in ['Table', 'R_Table']:
                    continue
                if len(words) >= 8:
                    try:
                        depth_value = float(words[-1])
                        if 1000 <= depth_value <= 1800:
                            object_name = words[0]
                            current_counts[object_name] += 1
                    except ValueError:
                        pass
            for object_name, count_value in current_counts.items():
                object_counts[object_name].append(count_value)
                frame_counts[object_name] += 1
        final_result = {}
        for object_name in object_counts:
            if frame_counts[object_name] >= min_frames:
                count_frequency = Counter(object_counts[object_name])
                mode_value = count_frequency.most_common(1)[0][0]
                final_result[object_name] = mode_value
        return final_result
        
    def select_table_area(self, tables, file_lines):
        if not tables:
            return None
        max_object_count = 0
        selected_table = None
        for table_area in tables:
            object_count = self.count_objects_in_table(file_lines, table_area)
            if object_count > 6 and object_count > max_object_count:
                max_object_count = object_count
                selected_table = table_area
        return selected_table
        
    def count_objects_in_table(self, lines, table_area):
        x_min, x_max, y_min, y_max = table_area
        count = 0
        for line in lines:
            words = line.split()
            if words and words[0] not in ['Table', 'R_Table']:
                word, word_x_min, word_x_max, word_y_min, word_y_max = words[0], float(words[3]), float(words[4]), float(words[5]), float(words[6])
                if word_x_min > x_min and word_x_max < x_max and word_y_max < y_max and word_y_max > y_min:
                    count += 1
        return count
        
    def extract_table_objects(self, lines, table_area):
        x_min, x_max, y_min, y_max = table_area
        table_objects = []
        excluded_objects = {'Table', 'R_Table'}
        for line in lines:
            words = line.split()
            if words and words[0] not in ['Table', 'R_Table']:
                word, word_x_min, word_x_max, word_y_min, word_y_max = words[0], float(words[3]), float(words[4]), float(words[5]), float(words[6])
                if word_x_min > x_min and word_x_max < x_max and word_y_max < y_max and word_y_max > y_min and (word not in excluded_objects):
                    table_objects.append(word)
        return table_objects
        
    def update_object_counts(self, object_counter, objects_list):
        local_counts = defaultdict(int)
        for object_name in objects_list:
            local_counts[object_name] += 1
        for object_name, count_value in local_counts.items():
            object_counter[object_name].append(count_value)
            
    def duplicate_text_files(self, source_folder, target_folder):
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        text_files = glob.glob(os.path.join(source_folder, '*.txt'))
        for file_path in text_files:
            base_name = os.path.basename(file_path)
            target_file = os.path.join(target_folder, base_name)
            shutil.copy(file_path, target_file)
            print(f"Copied '{file_path}' to '{target_file}'")
            
    def process_detection_cycle(self):
        source_directories = ['/home/HwHiAiUser/ultralytics-main/runss/d'] 
        output_location = '/home/HwHiAiUser/ultralytics-main/runss/result/xuexiao-tuanduiid-R1.txt' 
        time.sleep(2)
        self.clear_directory('/home/HwHiAiUser/ultralytics-main/runss/labels') 
        self.clear_directory('/home/HwHiAiUser/ultralytics-main/runss/d') 
        time.sleep(10)
        self.move_text_files('/home/HwHiAiUser/ultralytics-main/runss/labels', '/home/HwHiAiUser/ultralytics-main/runss/d') 
        time.sleep(0.5)
        total_objects = defaultdict(int)
        for directory in source_directories:
            stable_objects = self.analyze_directory(directory, min_frames=5)
            for object_name, count_value in stable_objects.items():
                total_objects[object_name] = count_value
        with open(output_location, 'w') as output_file:
            output_file.write("START\n")
            for object_name, count_value in total_objects.items():
                output_file.write(f"Goal_ID={object_name};Num={count_value}\n")
            output_file.write("END\n")
        print(f"xuexiao-tuanduiid-R1.txt已生成,路径为：{output_location}")
        self.clear_directory('/home/HwHiAiUser/Desktop/result_r') 
        time.sleep(0.5)
        self.duplicate_text_files('/home/HwHiAiUser/ultralytics-main/runss/result', '/home/HwHiAiUser/Desktop/result_r')
        result_file = '/home/HwHiAiUser/Desktop/result_r/xuexiao-tuanduiid-R1.txt' 
        self.transfer_file(1, result_file)
        self.network_socket.close()
        print("关闭socket")
        time.sleep(5)
        self.clear_directory('/home/HwHiAiUser/ultralytics-main/runss/result') 
   
    def begin_detection(self):
        self.status_indicator.setText("识别中") 
        self.send_identifier(0, "幻视")
        os.makedirs(self.txt_output_directory, exist_ok=True)
        self.current_frame_index = 0
        self.detection_thread = QThread()
        self.detection_worker = DetectionWorker(self.detector, txt_output_path=self.txt_output_directory,
                                  enable_video_save=self.save_video_flag, video_output_path=self.video_storage_path)
        self.detection_worker.moveToThread(self.detection_thread)
        self.detection_worker.detection_complete.connect(self.handle_detection_results)
        self.detection_thread.started.connect(self.detection_worker.start_detection)
        self.detection_thread.start()
        threading.Thread(target=self.process_detection_cycle).start()

    def stop_detection_process(self):
        if self.detection_worker:
            self.detection_worker.stop_detection()
        if self.detection_thread:
            self.detection_thread.quit()
            self.detection_thread.wait()
        self.detector.close_camera()
        cv2.destroyAllWindows()

    def handle_detection_results(self, processed_image, detection_results):
        if processed_image is not None:
            self.display_processed_image(processed_image)
        if detection_results is not None:
            self.display_detection_results(detection_results)
        self.current_frame_index += 1

    def display_processed_image(self, image_data):
        rgb_converted = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        image_height, image_width, channels = rgb_converted.shape
        bytes_per_line = channels * image_width
        qt_image = QImage(rgb_converted.data, image_width, image_height, bytes_per_line, QImage.Format_RGB888)
        scaled_image = qt_image.scaled(self.display_label.width(), self.display_label.height())
        self.display_label.setPixmap(QPixmap.fromImage(scaled_image))

    def display_detection_results(self, detection_results):
        self.results_display.clear()
        object_counters = {}
        for result in detection_results:
            object_type = result[0]
            if object_type in object_counters:
                object_counters[object_type] += 1
            else:
                object_counters[object_type] = 1
        for result in detection_results:
            object_name = result[0]
            count_value = object_counters.get(object_name, 1)
            distance_text = ""
            formatted_distance = ""
            if len(result) > 6 and result[6] is not None:
                distance_meters = result[6] / 1000
                distance_text = f"{distance_meters:.1f}m"
                if 1.0 <= distance_meters <= 1.8: 
                    formatted_distance = f'<span style="color:red;">{distance_text}</span>'
                else:
                    formatted_distance = distance_text
            self.results_display.append(f'目标ID:{object_name} 数量:{count_value},{formatted_distance}')
            
    def auto_launch_detection(self):
        print("界面已显示，自动启动检测...")
        self.begin_detection()

if __name__ == '__main__':
    detector_instance = YOLOOrbbecDetector(model_weights='best3.pt', processing_device='0') 
    app_instance = QApplication(sys.argv)
    application_window = MainApplicationWindow(detector_instance)
    application_window.show()
    sys.exit(app_instance.exec_())
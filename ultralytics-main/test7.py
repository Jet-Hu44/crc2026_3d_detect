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

# ======= ImageStabilizer =======
class ImageStabilizer:
    def __init__(self, smoothing=0.5):
        self.smoothing = smoothing
        self.previous_image = None
        
    def stabilize(self, current_image):
        if self.previous_image is None:
            output_image = current_image
        else:
            output_image = cv2.addWeighted(current_image, self.smoothing, 
                                          self.previous_image, 1 - self.smoothing, 0)
        self.previous_image = output_image
        return output_image

# ======= ObjectDetectionSystem =======
class ObjectDetectionSystem:
    def __init__(self, model_path='yolov8n.pt', device_id='0', half_precision=False):
        self.confidence_level = 0.50 
        self.device_id = device_id
        self.half_precision = half_precision
        try:
            self.detection_model = YOLO(model_path)
            print(f"已加载YOLOv8模型,类别数:{len(self.detection_model.names)}")
            self.class_labels = self.detection_model.names
        except Exception as model_error:
            print(f"YOLOv8模型加载失败:{model_error}")
        self.camera_pipeline = None
        self.depth_adapter = None
        self.depth_enabled = False
        self.image_stabilizer = ImageStabilizer(smoothing=0.7)
        print('打开软件界面中---30%')
        self.initialize_camera_hardware()
        print('打开软件界面中---50%')

    def initialize_camera_hardware(self):
        try:
            os.system('sudo sh -c "echo 2048 > /sys/module/usbcore/parameters/usbfs_memory_mb"')
            print("已尝试增加USB缓冲区大小")
        except:
            print("注意:无法增加USB缓冲区,可能需要管理员权限")
        configuration = Config()
        self.camera_pipeline = Pipeline()
        try:
            color_streams = self.camera_pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            color_config = color_streams.get_default_video_stream_profile()
            print(f"使用彩色流:{color_config}")
            configuration.enable_stream(color_config)
            self.camera_pipeline.start(configuration)
            print("彩色流启动成功")
            print("预热相机中...")
            for _ in range(30):
                try:
                    frame_data = self.camera_pipeline.wait_for_frames(200)
                    if frame_data is None:
                        continue
                    color_image = frame_data.get_color_frame()
                    if color_image is None:
                        continue
                except:
                    pass
                time.sleep(0.3)
            self.camera_pipeline.stop()
            print("彩色流预热完成")
            configuration = Config()
            color_streams = self.camera_pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            color_config = color_streams.get_default_video_stream_profile()
            configuration.enable_stream(color_config)
            try:
                depth_streams = self.camera_pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
                depth_config = depth_streams.get_default_video_stream_profile()
                print(f"使用深度流:{depth_config}")
                configuration.enable_stream(depth_config)
                self.depth_adapter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
                print("已创建深度对齐滤镜,深度图将对齐到彩色图")
                try:
                    self.camera_pipeline.enable_frame_sync()
                    print("已启用帧同步")
                except Exception as sync_error:
                    print(f"帧同步失败:{sync_error}")
                self.depth_enabled = True
            except Exception as depth_error:
                print(f"深度流配置失败:{depth_error},将使用仅彩色流模式")
                self.depth_enabled = False
            self.camera_pipeline.start(configuration)
            print("相机启动成功,模式:", "彩色+深度" if self.depth_enabled else "仅彩色")
        except Exception as camera_error:
            print(f"相机启动失败:{camera_error}")
            self.camera_pipeline = None

    def frame_to_bgr_conversion(self, frame_object):
        if frame_object is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        try:
            height, width = frame_object.get_height(), frame_object.get_width()
            raw_data = frame_object.get_data()
            if len(raw_data) != width * height * 3:
                try:
                    image_array = np.frombuffer(raw_data, dtype=np.uint8)
                    decoded_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                    if decoded_image is not None:
                        return decoded_image
                except Exception as decode_error:
                    print(f"解码MJPG失败:{decode_error}")
                return np.zeros((height, width, 3), dtype=np.uint8)
            else:
                image_data = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width, 3))
                bgr_image = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
                return bgr_image
        except Exception as conversion_error:
            print(f"帧转换失败:{conversion_error}")
            return np.zeros((480, 640, 3), dtype=np.uint8)

    def perform_detection(self, input_frame=None):
        detection_results = []
        depth_map = None
        if self.camera_pipeline is None:
            if input_frame is None:
                return [], input_frame
            else:
                return [], input_frame
        else:
            try:
                while self.camera_pipeline.poll_for_frames():
                    _ = self.camera_pipeline.wait_for_frames(1)
            except:
                pass
            try:
                frames = self.camera_pipeline.wait_for_frames(200)
                if frames is None:
                    print("未获取到帧")
                    return [], input_frame if input_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                if self.depth_enabled and self.depth_adapter is not None:
                    try:
                        frames = self.depth_adapter.process(frames)
                    except Exception as align_error:
                        print(f"深度对齐失败:{align_error}")
                color_frame = frames.get_color_frame()
                if color_frame is None:
                    print("未获取到彩色帧")
                    return [], input_frame if input_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                depth_frame = None if not self.depth_enabled else frames.get_depth_frame()
                input_frame = self.frame_to_bgr_conversion(color_frame)
                if self.depth_enabled and depth_frame is not None:
                    try:
                        raw_depth = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                        d_height, d_width = depth_frame.get_height(), depth_frame.get_width()
                        depth_map = raw_depth.reshape((d_height, d_width)).astype(np.float32) * depth_frame.get_depth_scale()
                        depth_map = self.image_stabilizer.stabilize(depth_map)
                    except Exception as depth_error:
                        print(f"深度数据处理失败:{depth_error}")
            except Exception as frame_error:
                print(f"获取相机图像失败:{frame_error}")
                if input_frame is None:
                    return [], np.zeros((480, 640, 3), dtype=np.uint8)
                else:
                    return [], input_frame

        # 执行目标检测
        try:
            model_output = self.detection_model(input_frame, conf=self.confidence_level, iou=0.45)
            if len(model_output) > 0:
                results = model_output[0]
                if hasattr(results, 'boxes') and len(results.boxes) > 0:
                    bounding_boxes = results.boxes.data.cpu().numpy()
                    for box in bounding_boxes:
                        left, top, right, bottom, score, class_index = box
                        left, top, right, bottom = map(int, [left, top, right, bottom])
                        class_index = int(class_index)
                        class_name = self.class_labels[class_index]
                        center_x = int((left + right) / 2)
                        center_y = int((top + bottom) / 2)
                        distance = None
                        if depth_map is not None:
                            if 0 <= center_y < depth_map.shape[0] and 0 <= center_x < depth_map.shape[1]:
                                depth_value = depth_map[center_y, center_x]
                                if 0 < depth_value < 10000:  # 0到10米
                                    distance = depth_value
                        detection_results.append([class_name, float(score), left, top, right, bottom, distance])
                        cv2.rectangle(input_frame, (left, top), (right, bottom), (0, 0, 255), 2)
                        label_text = f"{class_name},{score:.2f}"
                        if distance is not None:
                            label_text += f"{distance:.0f}mm"
                        cv2.putText(input_frame, label_text, (left-5, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        except Exception as detection_error:
            print(f"目标检测失败:{detection_error}")
        return detection_results, input_frame

    def detect_from_file(self, image_path):
        image = cv2.imread(image_path)
        results, image = self.perform_detection(image)
        return results, image

    def save_detection_data(self, detections, frame_index, output_path):
        valid_detections = [d for d in detections if d[1] >= 0.5]
        if valid_detections:
            os.makedirs(output_path, exist_ok=True)
            file_path = os.path.join(output_path, f"frame_{frame_index}.txt")
            with open(file_path, 'w', newline='') as f:
                for detection in valid_detections:
                    name = detection[0]
                    confidence = round(detection[1], 2)
                    xmin = detection[2]
                    ymin = detection[3]
                    xmax = detection[4]
                    ymax = detection[5]
                    center_x = int(((xmax - xmin) / 2) + xmin)
                    center_y = int(((ymax - ymin) / 2) + ymin)
                    depth_info = "0"
                    if len(detection) > 6 and detection[6] is not None:
                        depth_info = f"{int(detection[6])}"
                    f.write(f"{name} {center_x} {center_y} {xmin} {xmax} {ymin} {ymax} {confidence} {depth_info}\n")

    def terminate(self):
        if self.camera_pipeline is not None:
            self.camera_pipeline.stop()
            print("相机已关闭")

class DetectionHandler(QObject):
    detection_completed = pyqtSignal(object, object)  # (processed_image, detection_list)

    def __init__(self, detector, text_output_dir, record_video=False, video_output_dir=None):
        super().__init__()
        self.detector = detector
        self.text_output_dir = text_output_dir
        self.current_frame = 0
        self.active = True
        self.record_video = record_video
        self.video_output_dir = video_output_dir
        self.video_recorder = None

    def start_processing(self):
        self.active = True
        if self.record_video:
            if not os.path.exists(self.video_output_dir):
                os.makedirs(self.video_output_dir)
            _, first_image = self.detector.perform_detection()
            if first_image is not None:
                height, width = first_image.shape[:2]
                codec = cv2.VideoWriter_fourcc(*'XVID')
                self.video_recorder = cv2.VideoWriter(os.path.join(self.video_output_dir, 'output_video.avi'),
                                          codec, 15, (width, height))
            else:
                print("无法获取第一帧,视频保存可能失败")
                self.record_video = False
        self.current_frame = 0
        while self.active:
            detections, processed_image = self.detector.perform_detection()
            if processed_image is None or np.all(processed_image == 0):
                print("采集到黑图，跳过本帧")
                time.sleep(0.05)
                continue

            if any(d[1] >= 0.5 for d in detections):
                self.detector.save_detection_data(detections, self.current_frame, self.text_output_dir)
            if self.record_video and self.video_recorder is not None:
                self.video_recorder.write(processed_image)
            self.detection_completed.emit(processed_image, detections)
            self.current_frame += 1
            time.sleep(0.03)

    def stop_processing(self):
        self.active = False
        if self.video_recorder is not None:
            self.video_recorder.release()
        self.detector.terminate()

class MainInterface(QMainWindow):
    def __init__(self, detector):
        super().__init__()
        self.detector = detector
        self.setup_interface()
        self.initialize_network_connection()
        self.detection_thread = None
        self.detection_handler = None
        self.frame_count = 0
        self.enable_recording = True
        self.video_storage = "/home/HwHiAiUser/ultralytics-main/runss" 
        self.label_directory = "/home/HwHiAiUser/ultralytics-main/runss/labels" 
        self.file_checker = QTimer(self)
        self.file_checker.timeout.connect(self.check_result_files)
        self.file_checker.start(5000)
        self.auto_launch = QTimer(self)
        self.auto_launch.timeout.connect(self.automatic_start)
        self.auto_launch.setSingleShot(True)  
        self.auto_launch.start(1000) 

    def setup_interface(self):
        self.setWindowTitle('3D识别')
        self.setGeometry(100, 100, 800, 500)
        self.setStyleSheet("background-color:white;")
        main_layout = QHBoxLayout()
        self.display_panel = QLabel(self)
        self.display_panel.setFixedSize(500, 400)
        self.display_panel.setStyleSheet("border:1px solid black;")
        main_layout.addWidget(self.display_panel)
        sidebar = QVBoxLayout()
        sidebar.addSpacing(20)
        results_header = QLabel('识别结果输出区', self)
        results_header.setFont(QFont('Arial', 14))
        results_header.setAlignment(Qt.AlignLeft)
        sidebar.addWidget(results_header)
        self.result_display = QTextEdit(self)
        self.result_display.setReadOnly(True)
        self.result_display.setFixedSize(300, 300)
        self.result_display.setStyleSheet("border:1px solid black;")
        text_font = QFont()
        text_font.setFamily("Arial")
        text_font.setPointSize(14)
        self.result_display.setFont(text_font)
        sidebar.addWidget(self.result_display)
        self.status_panel = QLabel("准备中...", self)  
        self.status_panel.setFont(QFont('Arial', 16))
        self.status_panel.setFixedSize(300, 60)
        self.status_panel.setAlignment(Qt.AlignCenter)
        self.status_panel.setStyleSheet("background-color:lightgray;color:black;")
        sidebar.addWidget(self.status_panel)
        self.start_button = QPushButton('自动启动中...', self)
        self.start_button.setFont(QFont('Arial', 15))
        self.start_button.setStyleSheet(
            "QPushButton{background-color:#F9C49A;color:black;border-radius:50px;}"
            "QPushButton:pressed{background-color:#FF8C00;}"
        )
        self.start_button.setFixedSize(100, 100)
        self.start_button.setEnabled(False)  
        self.start_button.clicked.connect(self.initiate_detection)
        sidebar.addWidget(self.start_button)
        sidebar.addStretch()
        main_layout.addLayout(sidebar)
        central_content = QWidget(self)
        central_content.setLayout(main_layout)
        self.setCentralWidget(central_content)

    def check_result_files(self):
        target_directory = '/home/HwHiAiUser/ultralytics-main/runss/result' 
        if any(file.endswith('.txt') for file in os.listdir(target_directory)):
            print("已发送xuexiao-tuanduiid-R1.txt到裁判盒,识别结束,准备关闭软件界面。")
            self.close()
        else:
            print("结果文件未生成")

    def initialize_network_connection(self):
        self.server_address = '192.168.1.66'
        self.server_port = 6666
        self.network_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.network_socket.connect((self.server_address, self.server_port))
            print("打开软件界面中---80%")
            print("Connected to server successfully.")
        except Exception as connection_error:
            print(f"Failed to connect to server: {connection_error}")

    def transmit_text(self, data_type, content):
        encoded_content = content.encode()
        length = len(encoded_content)
        packet = struct.pack('>II', data_type, length) + encoded_content
        self.network_socket.sendall(packet)
        
    def send_identifier(self, data_type, identifier):
        self.transmit_text(data_type, identifier)
        
    def transmit_file(self, data_type, file_location):
        while not os.path.exists(file_location):
            print(f"文件{file_location}未找到,等待中...")
            time.sleep(0.1)
        with open(file_location, 'rb') as file:
            file_content = file.read()
        file_size = len(file_content)
        header = struct.pack('>II', data_type, file_size)
        self.network_socket.sendall(header)
        self.network_socket.sendall(file_content)
        
    def send_final_results(self, result_location):
        result_location = '/home/HwHiAiUser/Desktop/result_r/xuexiao-tuanduiid-R1.txt' 
        self.transmit_file(1, result_location)
        
    def clear_directory(self, directory):
        for item in os.listdir(directory):
            item_path = os.path.join(directory, item)
            os.remove(item_path)
            print(f"文件夹{directory}/{item}已被删除")
            
    def relocate_files(self, source, destination):
        if not os.path.exists(destination):
            os.makedirs(destination)
        for item in os.listdir(source):
            if item.endswith('.txt'):
                source_file = os.path.join(source, item)
                destination_file = os.path.join(destination, item)
                shutil.move(source_file, destination_file)
                print(f"文件{item}从{source}移动到{destination}")
                
    def determine_mode(self, values):
        if not values:
            return None
        frequency = Counter(values)
        mode_value = frequency.most_common(1)[0][0]
        return mode_value
        
    def analyze_directory(self, directory, min_occurrences=5):
        text_files = glob.glob(os.path.join(directory, '*.txt'))
        object_data = defaultdict(list)
        frame_occurrence = defaultdict(int)
        for file in text_files:
            with open(file, 'r') as f:
                lines = f.readlines()
            current_count = defaultdict(int)
            for line in lines:
                parts = line.split()
                if not parts or parts[0] in ['Table', 'R_Table']:
                    continue
                if len(parts) >= 8:
                    try:
                        depth = float(parts[-1])
                        if 1000 <= depth <= 1800:
                            object_name = parts[0]
                            current_count[object_name] += 1
                    except ValueError:
                        pass
            for object_name, count in current_count.items():
                object_data[object_name].append(count)
                frame_occurrence[object_name] += 1
        final_result = {}
        for object_name in object_data:
            if frame_occurrence[object_name] >= min_occurrences:
                counter = Counter(object_data[object_name])
                mode = counter.most_common(1)[0][0]
                final_result[object_name] = mode
        return final_result
        
    def select_main_table(self, tables, lines):
        if not tables:
            return None
        max_count = 0
        chosen_table = None
        for table in tables:
            count = self.count_objects_in_table(lines, table)
            if count > 6 and count > max_count:
                max_count = count
                chosen_table = table
        return chosen_table
        
    def count_objects_in_table(self, lines, table_area):
        x_min, x_max, y_min, y_max = table_area
        count = 0
        for line in lines:
            parts = line.split()
            if parts and parts[0] not in ['Table', 'R_Table']:
                obj, obj_xmin, obj_xmax, obj_ymin, obj_ymax = parts[0], float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])
                if obj_xmin > x_min and obj_xmax < x_max and obj_ymax < y_max and obj_ymax > y_min:
                    count += 1
        return count
        
    def extract_table_objects(self, lines, table_area):
        x_min, x_max, y_min, y_max = table_area
        table_objects = []
        excluded = {'Table', 'R_Table'}
        for line in lines:
            parts = line.split()
            if parts and parts[0] not in ['Table', 'R_Table']:
                obj, obj_xmin, obj_xmax, obj_ymin, obj_ymax = parts[0], float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])
                if obj_xmin > x_min and obj_xmax < x_max and obj_ymax < y_max and obj_ymax > y_min and (obj not in excluded):
                    table_objects.append(obj)
        return table_objects
        
    def update_object_counts(self, count_dict, objects_list):
        local_count = defaultdict(int)
        for obj in objects_list:
            local_count[obj] += 1
        for obj, count in local_count.items():
            count_dict[obj].append(count)
            
    def duplicate_files(self, source_folder, target_folder):
        if not os.path.exists(target_folder):
            os.makedirs(target_folder)
        text_files = glob.glob(os.path.join(source_folder, '*.txt'))
        for file in text_files:
            filename = os.path.basename(file)
            target_file = os.path.join(target_folder, filename)
            shutil.copy(file, target_file)
            print(f"Copied '{file}' to '{target_file}'")
            
    def execute_detection_cycle(self):
        directories = ['/home/HwHiAiUser/ultralytics-main/runss/d'] 
        output_file = '/home/HwHiAiUser/ultralytics-main/runss/result/xuexiao-tuanduiid-R1.txt' 
        time.sleep(2)
        self.clear_directory('/home/HwHiAiUser/ultralytics-main/runss/labels') 
        self.clear_directory('/home/HwHiAiUser/ultralytics-main/runss/d') 
        time.sleep(10)
        self.relocate_files('/home/HwHiAiUser/ultralytics-main/runss/labels', '/home/HwHiAiUser/ultralytics-main/runss/d') 
        time.sleep(0.5)
        total_counts = defaultdict(int)
        for directory in directories:
            stable_objects = self.analyze_directory(directory, min_occurrences=5)
            for object_name, count in stable_objects.items():
                total_counts[object_name] = count
        with open(output_file, 'w') as f:
            f.write("START\n")
            for object_name, count in total_counts.items():
                f.write(f"Goal_ID={object_name};Num={count}\n")
            f.write("END\n")
        print(f"xuexiao-tuanduiid-R1.txt已生成,路径为：{output_file}")
        self.clear_directory('/home/HwHiAiUser/Desktop/result_r') 
        time.sleep(0.5)
        self.duplicate_files('/home/HwHiAiUser/ultralytics-main/runss/result', '/home/HwHiAiUser/Desktop/result_r')
        result_file = '/home/HwHiAiUser/Desktop/result_r/xuexiao-tuanduiid-R1.txt' 
        self.transmit_file(1, result_file)
        self.network_socket.close()
        print("关闭socket")
        time.sleep(5)
        self.clear_directory('/home/HwHiAiUser/ultralytics-main/runss/result') 
   
    def initiate_detection(self):
        self.status_panel.setText("识别中") 
        self.send_identifier(0, "幻视")
        os.makedirs(self.label_directory, exist_ok=True)
        self.frame_count = 0
        self.detection_thread = QThread()
        self.detection_handler = DetectionHandler(self.detector, text_output_dir=self.label_directory,
                                  record_video=self.enable_recording, video_output_dir=self.video_storage)
        self.detection_handler.moveToThread(self.detection_thread)
        self.detection_handler.detection_completed.connect(self.display_results)
        self.detection_thread.started.connect(self.detection_handler.start_processing)
        self.detection_thread.start()
        threading.Thread(target=self.execute_detection_cycle).start()

    def stop_detection(self):
        if self.detection_handler:
            self.detection_handler.stop_processing()
        if self.detection_thread:
            self.detection_thread.quit()
            self.detection_thread.wait()
        self.detector.terminate()
        cv2.destroyAllWindows()

    def display_results(self, image, detections):
        if image is not None:
            self.show_image(image)
        if detections is not None:
            self.show_detection_results(detections)
        self.frame_count += 1

    def show_image(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_image.shape
        bytes_per_line = channels * width
        qt_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        scaled_image = qt_image.scaled(self.display_panel.width(), self.display_panel.height())
        self.display_panel.setPixmap(QPixmap.fromImage(scaled_image))

    def show_detection_results(self, detections):
        self.result_display.clear()
        object_counts = {}
        for detection in detections:
            object_type = detection[0]
            if object_type in object_counts:
                object_counts[object_type] += 1
            else:
                object_counts[object_type] = 1
        for detection in detections:
            object_name = detection[0]
            count = object_counts.get(object_name, 1)
            distance_text = ""
            formatted_distance = ""
            if len(detection) > 6 and detection[6] is not None:
                distance_m = detection[6] / 1000
                distance_text = f"{distance_m:.1f}m"
                if 1.0 <= distance_m <= 1.8: 
                    formatted_distance = f'<span style="color:red;">{distance_text}</span>'
                else:
                    formatted_distance = distance_text
            self.result_display.append(f'目标ID:{object_name} 数量:{count},{formatted_distance}')
            
    def automatic_start(self):
        print("界面已显示，自动启动检测...")
        self.initiate_detection()

if __name__ == '__main__':
    detector = ObjectDetectionSystem(model_path='yolo11s.pt', device_id='0') 
    application = QApplication(sys.argv)
    main_window = MainInterface(detector)
    main_window.show()
    sys.exit(application.exec_())
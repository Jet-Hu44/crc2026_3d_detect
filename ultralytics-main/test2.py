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

# ======= TemporalFilter =======
class TemporalFilter:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.previous_frame = None
        
    def process(self, frame):
        if self.previous_frame is None:
            output = frame
        else:
            output = cv2.addWeighted(frame, self.alpha, self.previous_frame, 1 - self.alpha, 0)
        self.previous_frame = output
        return output

# ======= YoloOrbbecDetector =======
class YoloOrbbecDetector:
    def __init__(self, weights='yolov8n.pt', device='0', half=False):
        self.conf_threshold = 0.50 
        self.device = device
        self.half_precision = half
        try:
            self.model = YOLO(weights)
            print(f"Loaded YOLOv8 model, classes count: {len(self.model.names)}")
            self.class_names = self.model.names
        except Exception as e:
            print(f"YOLOv8 model loading failed: {e}")
        self.pipeline = None
        self.align_filter = None
        self.depth_enabled = False
        self.temporal_filter = TemporalFilter(alpha=0.7)
        print('Initializing interface ---30%')
        self.initialize_camera()
        print('Initializing interface ---50%')

    def initialize_camera(self):
        try:
            os.system('sudo sh -c "echo 2048 > /sys/module/usbcore/parameters/usbfs_memory_mb"')
            print("Increased USB buffer size")
        except:
            print("Warning: Failed to increase USB buffer, may need admin rights")
        config = Config()
        self.pipeline = Pipeline()
        try:
            color_profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            color_profile = color_profiles.get_default_video_stream_profile()
            print(f"Using color stream: {color_profile}")
            config.enable_stream(color_profile)
            self.pipeline.start(config)
            print("Color stream started successfully")
            print("Warming up camera...")
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
            print("Color stream warmup complete")
            config = Config()
            color_profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            color_profile = color_profiles.get_default_video_stream_profile()
            config.enable_stream(color_profile)
            try:
                depth_profiles = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
                depth_profile = depth_profiles.get_default_video_stream_profile()
                print(f"Using depth stream: {depth_profile}")
                config.enable_stream(depth_profile)
                self.align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
                print("Created depth alignment filter, depth map aligned to color")
                try:
                    self.pipeline.enable_frame_sync()
                    print("Frame sync enabled")
                except Exception as e:
                    print(f"Frame sync failed: {e}")
                self.depth_enabled = True
            except Exception as e:
                print(f"Depth stream configuration failed: {e}, using color-only mode")
                self.depth_enabled = False
            self.pipeline.start(config)
            print("Camera started successfully, mode:", "Color+Depth" if self.depth_enabled else "Color-only")
        except Exception as e:
            print(f"Camera initialization failed: {e}")
            self.pipeline = None

    def convert_frame_to_bgr(self, frame):
        if frame is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        try:
            height, width = frame.get_height(), frame.get_width()
            frame_data = frame.get_data()
            if len(frame_data) != width * height * 3:
                try:
                    img_array = np.frombuffer(frame_data, dtype=np.uint8)
                    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if image is not None:
                        return image
                except Exception as e:
                    print(f"MJPG decoding failed: {e}")
                return np.zeros((height, width, 3), dtype=np.uint8)
            else:
                image = np.frombuffer(frame_data, dtype=np.uint8).reshape((height, width, 3))
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                return image
        except Exception as e:
            print(f"Frame conversion failed: {e}")
            return np.zeros((480, 640, 3), dtype=np.uint8)

    def detect_objects(self, input_image=None):
        detection_results = []
        depth_map = None
        if self.pipeline is None:
            if input_image is None:
                return [], input_image
            else:
                return [], input_image
        else:
            try:
                while self.pipeline.poll_for_frames():
                    _ = self.pipeline.wait_for_frames(1)
            except:
                pass
            try:
                frames = self.pipeline.wait_for_frames(200)
                if frames is None:
                    print("No frames received")
                    return [], input_image if input_image is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                if self.depth_enabled and self.align_filter is not None:
                    try:
                        frames = self.align_filter.process(frames)
                    except Exception as e:
                        print(f"Depth alignment failed: {e}")
                color_frame = frames.get_color_frame()
                if color_frame is None:
                    print("No color frame received")
                    return [], input_image if input_image is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                depth_frame = None if not self.depth_enabled else frames.get_depth_frame()
                input_image = self.convert_frame_to_bgr(color_frame)
                if self.depth_enabled and depth_frame is not None:
                    try:
                        raw_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                        height, width = depth_frame.get_height(), depth_frame.get_width()
                        depth_map = raw_data.reshape((height, width)).astype(np.float32) * depth_frame.get_depth_scale()
                        depth_map = self.temporal_filter.process(depth_map)
                    except Exception as e:
                        print(f"Depth data processing failed: {e}")
            except Exception as e:
                print(f"Camera frame capture failed: {e}")
                if input_image is None:
                    return [], np.zeros((480, 640, 3), dtype=np.uint8)
                else:
                    return [], input_image

        # Perform YOLO object detection
        try:
            results = self.model(input_image, conf=self.conf_threshold, iou=0.45)
            if len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and len(result.boxes) > 0:
                    boxes_data = result.boxes.data.cpu().numpy()
                    for box in boxes_data:
                        x1, y1, x2, y2, confidence, class_id = box
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                        class_id = int(class_id)
                        class_name = self.class_names[class_id]
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        distance_value = None
                        if depth_map is not None:
                            if 0 <= center_y < depth_map.shape[0] and 0 <= center_x < depth_map.shape[1]:
                                depth_value = depth_map[center_y, center_x]
                                if 0 < depth_value < 10000:  # 0-10 meters
                                    distance_value = depth_value
                        detection_results.append([class_name, float(confidence), x1, y1, x2, y2, distance_value])
                        cv2.rectangle(input_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        label_text = f"{class_name},{confidence:.2f}"
                        if distance_value is not None:
                            label_text += f"{distance_value:.0f}mm"
                        cv2.putText(input_image, label_text, (x1-5, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        except Exception as e:
            print(f"Object detection failed: {e}")
        return detection_results, input_image

    def detect_from_file(self, image_path):
        image = cv2.imread(image_path)
        detection_results, image = self.detect_objects(image)
        return detection_results, image

    def save_detection_results(self, detection_results, frame_index, output_directory):
        filtered_results = [result for result in detection_results if result[1] >= 0.5]
        if filtered_results:
            os.makedirs(output_directory, exist_ok=True)
            output_file = os.path.join(output_directory, f"frame_{frame_index}.txt")
            with open(output_file, 'w', newline='') as file:
                for result in filtered_results:
                    class_name = result[0]
                    confidence = round(result[1], 2)
                    xmin = result[2]
                    ymin = result[3]
                    xmax = result[4]
                    ymax = result[5]
                    center_x = int(((xmax - xmin) / 2) + xmin)
                    center_y = int(((ymax - ymin) / 2) + ymin)
                    depth_info = "0"
                    if len(result) > 6 and result[6] is not None:
                        depth_info = f"{int(result[6])}"
                    file.write(f"{class_name} {center_x} {center_y} {xmin} {xmax} {ymin} {ymax} {confidence} {depth_info}\n")

    def close(self):
        if self.pipeline is not None:
            self.pipeline.stop()
            print("Camera closed")

class DetectWorker(QObject):
    result_ready = pyqtSignal(object, object)  # (result_image, detection_results)

    def __init__(self, detector, text_output_directory, save_video=False, video_directory=None):
        super().__init__()
        self.detector = detector
        self.text_output_directory = text_output_directory
        self.frame_index = 0
        self.is_running = False
        self.save_video = save_video
        self.video_directory = video_directory
        self.video_writer = None

    def start_detection(self):
        self.is_running = True
        if self.save_video:
            if not os.path.exists(self.video_directory):
                os.makedirs(self.video_directory)
            _, first_frame = self.detector.detect_objects()
            if first_frame is not None:
                height, width = first_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.video_writer = cv2.VideoWriter(os.path.join(self.video_directory, 'output_video.avi'),
                                          fourcc, 15, (width, height))
            else:
                print("Failed to get first frame, video saving may fail")
                self.save_video = False
        self.frame_index = 0
        while self.is_running:
            detection_results, result_frame = self.detector.detect_objects()
            if result_frame is None or np.all(result_frame == 0):
                print("Received black frame, skipping")
                time.sleep(0.05)
                continue

            if any(result[1] >= 0.5 for result in detection_results):
                self.detector.save_detection_results(detection_results, self.frame_index, self.text_output_directory)
            if self.save_video and self.video_writer is not None:
                self.video_writer.write(result_frame)
            self.result_ready.emit(result_frame, detection_results)
            self.frame_index += 1
            time.sleep(0.03)


    def stop_detection(self):
        self.is_running = False
        if self.video_writer is not None:
            self.video_writer.release()
        self.detector.close()

class MainWindow(QMainWindow):
    def __init__(self, detector):
        super().__init__()
        self.detector = detector
        self.initialize_ui()
        self.initialize_socket()
        self.worker_thread = None
        self.worker = None
        self.frame_index = 0
        self.save_video = True
        self.video_directory = "/home/HwHiAiUser/ultralytics-main/runss" 
        self.text_output_directory = "/home/HwHiAiUser/ultralytics-main/runss/labels" 
        self.file_check_timer = QTimer(self)
        self.file_check_timer.timeout.connect(self.check_text_files)
        self.file_check_timer.start(5000)
        self.auto_start_timer = QTimer(self)
        self.auto_start_timer.timeout.connect(self.start_detection_automatically)
        self.auto_start_timer.setSingleShot(True)  
        self.auto_start_timer.start(1000) 

    def initialize_ui(self):
        self.setWindowTitle('3D Recognition')
        self.setGeometry(100, 100, 800, 500)
        self.setStyleSheet("background-color:white;")
        horizontal_layout = QHBoxLayout()
        self.image_display = QLabel(self)
        self.image_display.setFixedSize(500, 400)
        self.image_display.setStyleSheet("border:1px solid black;")
        horizontal_layout.addWidget(self.image_display)
        vertical_layout = QVBoxLayout()
        vertical_layout.addSpacing(20)
        results_label = QLabel('Recognition Results', self)
        results_label.setFont(QFont('Arial', 14))
        results_label.setAlignment(Qt.AlignLeft)
        vertical_layout.addWidget(results_label)
        self.result_display = QTextEdit(self)
        self.result_display.setReadOnly(True)
        self.result_display.setFixedSize(300, 300)
        self.result_display.setStyleSheet("border:1px solid black;")
        font = QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.result_display.setFont(font)
        vertical_layout.addWidget(self.result_display)
        self.status_display = QLabel("Initializing...", self)  
        self.status_display.setFont(QFont('Arial', 16))
        self.status_display.setFixedSize(300, 60)
        self.status_display.setAlignment(Qt.AlignCenter)
        self.status_display.setStyleSheet("background-color:lightgray;color:black;")
        vertical_layout.addWidget(self.status_display)
        self.start_button = QPushButton('Auto-starting...', self)
        self.start_button.setFont(QFont('Arial', 15))
        self.start_button.setStyleSheet(
            "QPushButton{background-color:#F9C49A;color:black;border-radius:50px;}"
            "QPushButton:pressed{background-color:#FF8C00;}"
        )
        self.start_button.setFixedSize(100, 100)
        self.start_button.setEnabled(False)  
        self.start_button.clicked.connect(self.start_detection_process)
        vertical_layout.addWidget(self.start_button)
        vertical_layout.addStretch()
        horizontal_layout.addLayout(vertical_layout)
        central_widget = QWidget(self)
        central_widget.setLayout(horizontal_layout)
        self.setCentralWidget(central_widget)

    def check_text_files(self):
        directory = '/home/HwHiAiUser/ultralytics-main/runss/result' 
        if any(file.endswith('.txt') for file in os.listdir(directory)):
            print("Sent xuexiao-tuanduiid-R1.txt to referee box, recognition completed, closing interface.")
            self.close()
        else:
            print("Result file not generated")

    def initialize_socket(self):
        self.host_address = '192.168.1.66'
        self.port_number = 6666
        self.socket_connection = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.socket_connection.connect((self.host_address, self.port_number))
            print("Initializing interface ---80%")
            print("Connected to server successfully.")
        except Exception as e:
            print(f"Failed to connect to server: {e}")

    def send_string_data(self, data_type, data_string):
        encoded_data = data_string.encode()
        data_length = len(encoded_data)
        message = struct.pack('>II', data_type, data_length) + encoded_data
        self.socket_connection.sendall(message)
        
    def send_team_identifier(self, data_type, team_id):
        self.send_string_data(data_type, team_id)
        
    def send_file_data(self, data_type, file_path):
        while not os.path.exists(file_path):
            print(f"File {file_path} not found, waiting...")
            time.sleep(0.1)
        with open(file_path, 'rb') as file:
            file_data = file.read()
        data_length = len(file_data)
        header = struct.pack('>II', data_type, data_length)
        self.socket_connection.sendall(header)
        self.socket_connection.sendall(file_data)
        
    def send_result_file(self, file_path):
        file_path = '/home/HwHiAiUser/Desktop/result_r/xuexiao-tuanduiid-R1.txt' 
        self.send_file_data(1, file_path)
        
    def clear_directory(self, directory_path):
        for file_name in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file_name)
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
            
    def move_text_files(self, source_directory, destination_directory):
        if not os.path.exists(destination_directory):
            os.makedirs(destination_directory)
        for file_name in os.listdir(source_directory):
            if file_name.endswith('.txt'):
                source_path = os.path.join(source_directory, file_name)
                destination_path = os.path.join(destination_directory, file_name)
                shutil.move(source_path, destination_path)
                print(f"Moved {file_name} from {source_directory} to {destination_directory}")
                
    def find_most_common(self, count_list):
        if not count_list:
            return None
        count_frequency = Counter(count_list)
        most_common = count_frequency.most_common(1)[0][0]
        return most_common
        
    def process_directory(self, directory, min_frames=5):
        text_files = glob.glob(os.path.join(directory, '*.txt'))
        object_counts = defaultdict(list)
        object_presence = defaultdict(int)
        for file_path in text_files:
            with open(file_path, 'r') as file:
                lines = file.readlines()
            frame_counts = defaultdict(int)
            for line in lines:
                parts = line.split()
                if not parts or parts[0] in ['Table', 'R_Table']:
                    continue
                if len(parts) >= 8:
                    try:
                        depth_value = float(parts[-1])
                        if 1000 <= depth_value <= 1800:
                            object_name = parts[0]
                            frame_counts[object_name] += 1
                    except ValueError:
                        pass
            for obj, count in frame_counts.items():
                object_counts[obj].append(count)
                object_presence[obj] += 1
        stable_objects = {}
        for obj in object_counts:
            if object_presence[obj] >= min_frames:
                counter = Counter(object_counts[obj])
                most_common_count = counter.most_common(1)[0][0]
                stable_objects[obj] = most_common_count
        return stable_objects
        
    def select_table_area(self, tables, lines):
        if not tables:
            return None
        max_count = 0
        selected_table = None
        for table in tables:
            count = self.count_objects_in_table(lines, table)
            if count > 6 and count > max_count:
                max_count = count
                selected_table = table
        return selected_table
        
    def count_objects_in_table(self, lines, table):
        x_min, x_max, y_min, y_max = table
        count = 0
        for line in lines:
            parts = line.split()
            if parts and parts[0] not in ['Table', 'R_Table']:
                obj_name, obj_x_min, obj_x_max, obj_y_min, obj_y_max = parts[0], float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])
                if obj_x_min > x_min and obj_x_max < x_max and obj_y_max < y_max and obj_y_max > y_min:
                    count += 1
        return count
        
    def extract_objects_in_table(self, lines, table):
        x_min, x_max, y_min, y_max = table
        objects_in_area = []
        excluded_objects = {'Table', 'R_Table'}
        for line in lines:
            parts = line.split()
            if parts and parts[0] not in ['Table', 'R_Table']:
                obj_name, obj_x_min, obj_x_max, obj_y_min, obj_y_max = parts[0], float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])
                if obj_x_min > x_min and obj_x_max < x_max and obj_y_max < y_max and obj_y_max > y_min and (obj_name not in excluded_objects):
                    objects_in_area.append(obj_name)
        return objects_in_area
        
    def update_object_counts(self, object_counts, objects):
        current_frame_counts = defaultdict(int)
        for obj in objects:
            current_frame_counts[obj] += 1
        for obj, count in current_frame_counts.items():
            object_counts[obj].append(count)
            
    def copy_text_files(self, source_directory, destination_directory):
        if not os.path.exists(destination_directory):
            os.makedirs(destination_directory)
        text_files = glob.glob(os.path.join(source_directory, '*.txt'))
        for file_path in text_files:
            base_name = os.path.basename(file_path)
            destination_path = os.path.join(destination_directory, base_name)
            shutil.copy(file_path, destination_path)
            print(f"Copied '{file_path}' to '{destination_path}'")
            
    def run_detection_cycle(self):
        directories = ['/home/HwHiAiUser/ultralytics-main/runss/d'] 
        output_path = '/home/HwHiAiUser/ultralytics-main/runss/result/xuexiao-tuanduiid-R1.txt' 
        time.sleep(2)
        self.clear_directory('/home/HwHiAiUser/ultralytics-main/runss/labels') 
        self.clear_directory('/home/HwHiAiUser/ultralytics-main/runss/d') 
        time.sleep(10)
        self.move_text_files('/home/HwHiAiUser/ultralytics-main/runss/labels', '/home/HwHiAiUser/ultralytics-main/runss/d') 
        time.sleep(0.5)
        total_counts = defaultdict(int)
        for directory in directories:
            stable_objects = self.process_directory(directory, min_frames=5)
            for obj, count in stable_objects.items():
                total_counts[obj] = count
        with open(output_path, 'w') as file:
            file.write("START\n")
            for obj, count in total_counts.items():
                file.write(f"Goal_ID={obj};Num={count}\n")
            file.write("END\n")
        print(f"xuexiao-tuanduiid-R1.txt generated at: {output_path}")
        self.clear_directory('/home/HwHiAiUser/Desktop/result_r') 
        time.sleep(0.5)
        self.copy_text_files('/home/HwHiAiUser/ultralytics-main/runss/result', '/home/HwHiAiUser/Desktop/result_r')
        file_path = '/home/HwHiAiUser/Desktop/result_r/xuexiao-tuanduiid-R1.txt' 
        self.send_file_data(1, file_path)
        self.socket_connection.close()
        print("Socket closed")
        time.sleep(5)
        self.clear_directory('/home/HwHiAiUser/ultralytics-main/runss/result') 
   
    def start_detection_process(self):
        self.status_display.setText("Detecting") 
        self.send_team_identifier(0, "幻视")
        os.makedirs(self.text_output_directory, exist_ok=True)
        self.frame_index = 0
        self.worker_thread = QThread()
        self.worker = DetectWorker(self.detector, text_output_directory=self.text_output_directory,
                                  save_video=self.save_video, video_directory=self.video_directory)
        self.worker.moveToThread(self.worker_thread)
        self.worker.result_ready.connect(self.handle_detection_result)
        self.worker_thread.started.connect(self.worker.start_detection)
        self.worker_thread.start()
        threading.Thread(target=self.run_detection_cycle).start()

    def stop_detection_process(self):
        if self.worker:
            self.worker.stop_detection()
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
        self.detector.close()
        cv2.destroyAllWindows()

    def handle_detection_result(self, result_frame, detection_results):
        if result_frame is not None:
            self.display_image(result_frame)
        if detection_results is not None:
            self.display_detection_results(detection_results)
        self.frame_index += 1

    def display_image(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb_image.shape
        bytes_per_line = channels * width
        qt_image = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        scaled_pixmap = qt_image.scaled(self.image_display.width(), self.image_display.height())
        self.image_display.setPixmap(QPixmap.fromImage(scaled_pixmap))

    def display_detection_results(self, detection_results):
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
                if 1.0 <= distance_meters <= 1.8: 
                    formatted_distance = f'<span style="color:red;">{distance_text}</span>'
                else:
                    formatted_distance = distance_text
            self.result_display.append(f'Object ID: {obj} Count: {count}, {formatted_distance}')
            
    def start_detection_automatically(self):
        print("Interface displayed, starting detection automatically...")
        self.start_detection_process()

if __name__ == '__main__':
    detector = YoloOrbbecDetector(weights='best3.pt', device='0') 
    app = QApplication(sys.argv)
    main_window = MainWindow(detector)
    main_window.show()
    sys.exit(app.exec_())
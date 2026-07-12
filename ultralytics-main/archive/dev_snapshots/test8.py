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

# ======= TemporalSmoothingFilter =======
class TemporalSmoothingFilter:
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.previous_frame = None
        
    def apply_smoothing(self, current_frame):
        if self.previous_frame is None:
            smoothed_frame = current_frame
        else:
            smoothed_frame = cv2.addWeighted(current_frame, self.alpha, 
                                            self.previous_frame, 1 - self.alpha, 0)
        self.previous_frame = smoothed_frame
        return smoothed_frame

# ======= ObjectDetectionModule =======
class ObjectDetectionModule:
    def __init__(self, model_weights='yolov8n.pt', device='0', use_half=False):
        self.confidence_threshold = 0.50 
        self.device = device
        self.use_half = use_half
        try:
            self.detection_model = YOLO(model_weights)
            print(f"YOLOv8 model loaded successfully, number of classes: {len(self.detection_model.names)}")
            self.class_names = self.detection_model.names
        except Exception as model_error:
            print(f"Failed to load YOLOv8 model: {model_error}")
        self.camera_pipeline = None
        self.depth_aligner = None
        self.depth_enabled = False
        self.temporal_filter = TemporalSmoothingFilter(alpha=0.7)
        print('Opening software interface---30%')
        self.initialize_camera()
        print('Opening software interface---50%')

    def initialize_camera(self):
        try:
            os.system('sudo sh -c "echo 2048 > /sys/module/usbcore/parameters/usbfs_memory_mb"')
            print("Attempted to increase USB buffer size")
        except:
            print("Warning: Unable to increase USB buffer, may require admin privileges")
        config = Config()
        self.camera_pipeline = Pipeline()
        try:
            color_profiles = self.camera_pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            color_profile = color_profiles.get_default_video_stream_profile()
            print(f"Using color stream: {color_profile}")
            config.enable_stream(color_profile)
            self.camera_pipeline.start(config)
            print("Color stream started successfully")
            print("Warming up camera...")
            for _ in range(30):
                try:
                    frames = self.camera_pipeline.wait_for_frames(200)
                    if frames is None:
                        continue
                    color_frame = frames.get_color_frame()
                    if color_frame is None:
                        continue
                except:
                    pass
                time.sleep(0.3)
            self.camera_pipeline.stop()
            print("Color stream warmup completed")
            config = Config()
            color_profiles = self.camera_pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            color_profile = color_profiles.get_default_video_stream_profile()
            config.enable_stream(color_profile)
            try:
                depth_profiles = self.camera_pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
                depth_profile = depth_profiles.get_default_video_stream_profile()
                print(f"Using depth stream: {depth_profile}")
                config.enable_stream(depth_profile)
                self.depth_aligner = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
                print("Created depth alignment filter, depth map will be aligned to color image")
                try:
                    self.camera_pipeline.enable_frame_sync()
                    print("Frame synchronization enabled")
                except Exception as sync_error:
                    print(f"Frame synchronization failed: {sync_error}")
                self.depth_enabled = True
            except Exception as depth_error:
                print(f"Depth stream configuration failed: {depth_error}, using color-only mode")
                self.depth_enabled = False
            self.camera_pipeline.start(config)
            print("Camera started successfully, mode:", "Color+Depth" if self.depth_enabled else "Color Only")
        except Exception as camera_error:
            print(f"Camera initialization failed: {camera_error}")
            self.camera_pipeline = None

    def convert_frame_to_bgr(self, frame):
        if frame is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        try:
            height, width = frame.get_height(), frame.get_width()
            data = frame.get_data()
            if len(data) != width * height * 3:
                try:
                    img_array = np.frombuffer(data, dtype=np.uint8)
                    decoded_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if decoded_img is not None:
                        return decoded_img
                except Exception as decode_error:
                    print(f"MJPG decoding failed: {decode_error}")
                return np.zeros((height, width, 3), dtype=np.uint8)
            else:
                img = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                return img
        except Exception as conversion_error:
            print(f"Frame conversion failed: {conversion_error}")
            return np.zeros((480, 640, 3), dtype=np.uint8)

    def detect_objects(self, input_image=None):
        detection_results = []
        depth_data = None
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
                frames = self.camera_pipeline.wait_for_frames(200)
                if frames is None:
                    print("No frames received")
                    return [], input_image if input_image is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                if self.depth_enabled and self.depth_aligner is not None:
                    try:
                        frames = self.depth_aligner.process(frames)
                    except Exception as align_error:
                        print(f"Depth alignment failed: {align_error}")
                color_frame = frames.get_color_frame()
                if color_frame is None:
                    print("No color frame received")
                    return [], input_image if input_image is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                depth_frame = None if not self.depth_enabled else frames.get_depth_frame()
                input_image = self.convert_frame_to_bgr(color_frame)
                if self.depth_enabled and depth_frame is not None:
                    try:
                        raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                        d_height, d_width = depth_frame.get_height(), depth_frame.get_width()
                        depth_data = raw.reshape((d_height, d_width)).astype(np.float32) * depth_frame.get_depth_scale()
                        depth_data = self.temporal_filter.apply_smoothing(depth_data)
                    except Exception as depth_error:
                        print(f"Depth data processing failed: {depth_error}")
            except Exception as frame_error:
                print(f"Failed to get camera image: {frame_error}")
                if input_image is None:
                    return [], np.zeros((480, 640, 3), dtype=np.uint8)
                else:
                    return [], input_image

        # Perform object detection
        try:
            results = self.detection_model(input_image, conf=self.confidence_threshold, iou=0.45)
            if len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and len(result.boxes) > 0:
                    boxes_data = result.boxes.data.cpu().numpy()
                    for box in boxes_data:
                        x1, y1, x2, y2, conf, cls_id = box
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                        cls_id = int(cls_id)
                        name = self.class_names[cls_id]
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        distance = None
                        if depth_data is not None:
                            if 0 <= cy < depth_data.shape[0] and 0 <= cx < depth_data.shape[1]:
                                depth_value = depth_data[cy, cx]
                                if 0 < depth_value < 10000:  # 0 to 10 meters
                                    distance = depth_value
                        detection_results.append([name, float(conf), x1, y1, x2, y2, distance])
                        cv2.rectangle(input_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        label_text = f"{name},{conf:.2f}"
                        if distance is not None:
                            label_text += f"{distance:.0f}mm"
                        cv2.putText(input_image, label_text, (x1-5, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        except Exception as detection_error:
            print(f"Object detection failed: {detection_error}")
        return detection_results, input_image

    def detect_from_image_file(self, image_file):
        img = cv2.imread(image_file)
        result_list, img = self.detect_objects(img)
        return result_list, img

    def save_results(self, result_list, frame_index, output_folder):
        filtered_results = [result for result in result_list if result[1] >= 0.5]
        if filtered_results:
            os.makedirs(output_folder, exist_ok=True)
            output_file = os.path.join(output_folder, f"frame_{frame_index}.txt")
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

    def shutdown(self):
        if self.camera_pipeline is not None:
            self.camera_pipeline.stop()
            print("Camera closed")

class DetectionWorker(QObject):
    result_ready = pyqtSignal(object, object)  # (result_image, result_list)

    def __init__(self, detector, txt_output_folder, save_video=False, video_folder=None):
        super().__init__()
        self.detector = detector
        self.txt_output_folder = txt_output_folder
        self.frame_index = 0
        self.running = False
        self.save_video = save_video
        self.video_folder = video_folder
        self.video_writer = None

    def start_detection(self):
        self.running = True
        if self.save_video:
            if not os.path.exists(self.video_folder):
                os.makedirs(self.video_folder)
            _, first_frame = self.detector.detect_objects()
            if first_frame is not None:
                h, w = first_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                self.video_writer = cv2.VideoWriter(os.path.join(self.video_folder, 'output_video.avi'),
                                          fourcc, 15, (w, h))
            else:
                print("Unable to get first frame, video saving may fail")
                self.save_video = False
        self.frame_index = 0
        while self.running:
            result_list, result_image = self.detector.detect_objects()
            if result_image is None or np.all(result_image == 0):
                print("Captured black frame, skipping")
                time.sleep(0.05)
                continue

            if any(result[1] >= 0.5 for result in result_list):
                self.detector.save_results(result_list, self.frame_index, self.txt_output_folder)
            if self.save_video and self.video_writer is not None:
                self.video_writer.write(result_image)
            self.result_ready.emit(result_image, result_list)
            self.frame_index += 1
            time.sleep(0.03)

    def stop(self):
        self.running = False
        if self.video_writer is not None:
            self.video_writer.release()
        self.detector.shutdown()

class MainWindow(QMainWindow):
    def __init__(self, detector):
        super().__init__()
        self.detector = detector
        self.init_ui()
        self.init_socket()
        self.worker_thread = None
        self.worker = None
        self.frame_index = 0
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

    def init_ui(self):
        self.setWindowTitle('3D Object Recognition')
        self.setGeometry(100, 100, 800, 500)
        self.setStyleSheet("background-color:white;")
        h_layout = QHBoxLayout()
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(500, 400)
        self.image_label.setStyleSheet("border:1px solid black;")
        h_layout.addWidget(self.image_label)
        v_layout = QVBoxLayout()
        v_layout.addSpacing(20)
        results_label = QLabel('Detection Results Output', self)
        results_label.setFont(QFont('Arial', 14))
        results_label.setAlignment(Qt.AlignLeft)
        v_layout.addWidget(results_label)
        self.result_text = QTextEdit(self)
        self.result_text.setReadOnly(True)
        self.result_text.setFixedSize(300, 300)
        self.result_text.setStyleSheet("border:1px solid black;")
        font = QFont()
        font.setFamily("Arial")
        font.setPointSize(14)
        self.result_text.setFont(font)
        v_layout.addWidget(self.result_text)
        self.status_label = QLabel("Initializing...", self)  
        self.status_label.setFont(QFont('Arial', 16))
        self.status_label.setFixedSize(300, 60)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("background-color:lightgray;color:black;")
        v_layout.addWidget(self.status_label)
        self.start_button = QPushButton('Auto Starting...', self)
        self.start_button.setFont(QFont('Arial', 15))
        self.start_button.setStyleSheet(
            "QPushButton{background-color:#F9C49A;color:black;border-radius:50px;}"
            "QPushButton:pressed{background-color:#FF8C00;}"
        )
        self.start_button.setFixedSize(100, 100)
        self.start_button.setEnabled(False)  
        self.start_button.clicked.connect(self.start_detection)
        v_layout.addWidget(self.start_button)
        v_layout.addStretch()
        h_layout.addLayout(v_layout)
        central_widget = QWidget(self)
        central_widget.setLayout(h_layout)
        self.setCentralWidget(central_widget)

    def check_txt_files(self):
        directory = '/home/HwHiAiUser/ultralytics-main/runss/result' 
        if any(file.endswith('.txt') for file in os.listdir(directory)):
            print("xuexiao-tuanduiid-R1.txt sent to judge box, recognition completed, closing interface.")
            self.close()
        else:
            print("Result files not generated")

    def init_socket(self):
        self.host = '192.168.1.66'
        self.port = 6666
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.socket.connect((self.host, self.port))
            print("Opening software interface---80%")
            print("Connected to server successfully.")
        except Exception as e:
            print(f"Failed to connect to server: {e}")

    def send_string(self, datatype, data):
        encoded_data = data.encode()
        data_length = len(encoded_data)
        message = struct.pack('>II', datatype, data_length) + encoded_data
        self.socket.sendall(message)
        
    def send_team_id(self, data_type, team_id):
        self.send_string(data_type, team_id)
        
    def send_file(self, datatype, file_path):
        while not os.path.exists(file_path):
            print(f"File {file_path} not found, waiting...")
            time.sleep(0.1)
        with open(file_path, 'rb') as file:
            file_data = file.read()
        data_length = len(file_data)
        header = struct.pack('>II', datatype, data_length)
        self.socket.sendall(header)
        self.socket.sendall(file_data)
        
    def send_result_file(self, file_path):
        file_path = '/home/HwHiAiUser/Desktop/result_r/xuexiao-tuanduiid-R1.txt' 
        self.send_file(1, file_path)
        
    def delete_all_files(self, folder_path):
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            os.remove(file_path)
            print(f"File {folder_path}/{file_name} deleted")
            
    def move_txt_files(self, src_folder, dest_folder):
        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)
        for file_name in os.listdir(src_folder):
            if file_name.endswith('.txt'):
                src_path = os.path.join(src_folder, file_name)
                dest_path = os.path.join(dest_folder, file_name)
                shutil.move(src_path, dest_path)
                print(f"File {file_name} moved from {src_folder} to {dest_folder}")
                
    def calculate_mode(self, values):
        if not values:
            return None
        count_freq = Counter(values)
        mode = count_freq.most_common(1)[0][0]
        return mode
        
    def process_directory(self, folder, min_occurrences=5):
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
                        if 1000 <= depth <= 1800:
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
        output_path = '/home/HwHiAiUser/ultralytics-main/runss/result/xuexiao-tuanduiid-R1.txt' 
        time.sleep(2)
        self.delete_all_files('/home/HwHiAiUser/ultralytics-main/runss/labels') 
        self.delete_all_files('/home/HwHiAiUser/ultralytics-main/runss/d') 
        time.sleep(10)
        self.move_txt_files('/home/HwHiAiUser/ultralytics-main/runss/labels', '/home/HwHiAiUser/ultralytics-main/runss/d') 
        time.sleep(0.5)
        total_counts = defaultdict(int)
        for folder in folders:
            stable_targets = self.process_directory(folder, min_occurrences=5)
            for word, count in stable_targets.items():
                total_counts[word] = count
        with open(output_path, 'w') as f:
            f.write("START\n")
            for word, count in total_counts.items():
                f.write(f"Goal_ID={word};Num={count}\n")
            f.write("END\n")
        print(f"xuexiao-tuanduiid-R1.txt generated at: {output_path}")
        self.delete_all_files('/home/HwHiAiUser/Desktop/result_r') 
        time.sleep(0.5)
        self.copy_txt_files('/home/HwHiAiUser/ultralytics-main/runss/result', '/home/HwHiAiUser/Desktop/result_r')
        file_path = '/home/HwHiAiUser/Desktop/result_r/xuexiao-tuanduiid-R1.txt' 
        self.send_file(1, file_path)
        self.socket.close()
        print("Socket closed")
        time.sleep(5)
        self.delete_all_files('/home/HwHiAiUser/ultralytics-main/runss/result') 
   
    def start_detection(self):
        self.status_label.setText("Detecting") 
        self.send_team_id(0, "幻视")
        os.makedirs(self.txt_output_folder, exist_ok=True)
        self.frame_index = 0
        self.worker_thread = QThread()
        self.worker = DetectionWorker(self.detector, txt_output_folder=self.txt_output_folder,
                                  save_video=self.save_video, video_folder=self.video_folder)
        self.worker.moveToThread(self.worker_thread)
        self.worker.result_ready.connect(self.on_result_ready)
        self.worker_thread.started.connect(self.worker.start_detection)
        self.worker_thread.start()
        threading.Thread(target=self.process_detection_cycle).start()

    def stop_detection(self):
        if self.worker:
            self.worker.stop()
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
        self.detector.shutdown()
        cv2.destroyAllWindows()

    def on_result_ready(self, result_image, result_list):
        if result_image is not None:
            self.display_image(result_image)
        if result_list is not None:
            self.display_results(result_list)
        self.frame_index += 1

    def display_image(self, image):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.image_label.width(), self.image_label.height())
        self.image_label.setPixmap(QPixmap.fromImage(p))

    def display_results(self, result_list):
        self.result_text.clear()
        object_counts = {}
        for result in result_list:
            object_type = result[0]
            if object_type in object_counts:
                object_counts[object_type] += 1
            else:
                object_counts[object_type] = 1
        for result in result_list:
            obj = result[0]
            count = object_counts.get(obj, 1)
            distance_text = ""
            colored_distance = ""
            if len(result) > 6 and result[6] is not None:
                distance_m = result[6] / 1000
                distance_text = f"{distance_m:.1f}m"
                if 1.0 <= distance_m <= 1.8: 
                    colored_distance = f'<span style="color:red;">{distance_text}</span>'
                else:
                    colored_distance = distance_text
            self.result_text.append(f'Object ID:{obj} Count:{count},{colored_distance}')
            
    def auto_start_detection(self):
        print("Interface displayed, auto-starting detection...")
        self.start_detection()

if __name__ == '__main__':
    detector = ObjectDetectionModule(model_weights='yolo11s.pt', device='0') 
    app = QApplication(sys.argv)
    main_window = MainWindow(detector)
    main_window.show()
    sys.exit(app.exec_())
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

# ======= FrameStabilizer =======
class FrameStabilizer:
    def __init__(self, blend_factor=0.5):
        self.blend_factor = blend_factor
        self.prev_frame = None
        
    def stabilize(self, current):
        if self.prev_frame is None:
            output = current
        else:
            output = cv2.addWeighted(current, self.blend_factor, 
                                    self.prev_frame, 1 - self.blend_factor, 0)
        self.prev_frame = output
        return output

# ======= VisionSystem =======
class VisionSystem:
    def __init__(self, model_file='yolov8n.pt', device_id='0', half_precision=False):
        self.min_confidence = 0.50 
        self.device_id = device_id
        self.half_precision = half_precision
        try:
            self.detection_engine = YOLO(model_file)
            print(f"YOLOv8 model loaded, classes: {len(self.detection_engine.names)}")
            self.class_labels = self.detection_engine.names
        except Exception as model_err:
            print(f"Model load error: {model_err}")
        self.camera_interface = None
        self.depth_processor = None
        self.depth_support = False
        self.frame_stabilizer = FrameStabilizer(blend_factor=0.7)
        print('Initializing interface---30%')
        self.setup_camera()
        print('Initializing interface---50%')

    def setup_camera(self):
        try:
            os.system('sudo sh -c "echo 2048 > /sys/module/usbcore/parameters/usbfs_memory_mb"')
            print("Increased USB buffer size")
        except:
            print("Warning: USB buffer increase failed")
        cam_config = Config()
        self.camera_interface = Pipeline()
        try:
            color_profiles = self.camera_interface.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            color_config = color_profiles.get_default_video_stream_profile()
            print(f"Color stream: {color_config}")
            cam_config.enable_stream(color_config)
            self.camera_interface.start(cam_config)
            print("Color stream active")
            print("Camera warmup...")
            for _ in range(30):
                try:
                    frames = self.camera_interface.wait_for_frames(200)
                    if frames is None:
                        continue
                    color_frame = frames.get_color_frame()
                    if color_frame is None:
                        continue
                except:
                    pass
                time.sleep(0.3)
            self.camera_interface.stop()
            print("Warmup complete")
            cam_config = Config()
            color_profiles = self.camera_interface.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            color_config = color_profiles.get_default_video_stream_profile()
            cam_config.enable_stream(color_config)
            try:
                depth_profiles = self.camera_interface.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
                depth_config = depth_profiles.get_default_video_stream_profile()
                print(f"Depth stream: {depth_config}")
                cam_config.enable_stream(depth_config)
                self.depth_processor = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
                print("Depth alignment enabled")
                try:
                    self.camera_interface.enable_frame_sync()
                    print("Frame sync active")
                except Exception as sync_err:
                    print(f"Sync error: {sync_err}")
                self.depth_support = True
            except Exception as depth_err:
                print(f"Depth setup failed: {depth_err}, using RGB only")
                self.depth_support = False
            self.camera_interface.start(cam_config)
            print("Camera ready:", "RGB+Depth" if self.depth_support else "RGB Only")
        except Exception as cam_err:
            print(f"Camera init error: {cam_err}")
            self.camera_interface = None

    def convert_to_bgr(self, frame_data):
        if frame_data is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        try:
            img_h, img_w = frame_data.get_height(), frame_data.get_width()
            raw_bytes = frame_data.get_data()
            if len(raw_bytes) != img_w * img_h * 3:
                try:
                    img_arr = np.frombuffer(raw_bytes, dtype=np.uint8)
                    decoded_img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                    if decoded_img is not None:
                        return decoded_img
                except Exception as decode_err:
                    print(f"Decode error: {decode_err}")
                return np.zeros((img_h, img_w, 3), dtype=np.uint8)
            else:
                img_data = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((img_h, img_w, 3))
                bgr_img = cv2.cvtColor(img_data, cv2.COLOR_RGB2BGR)
                return bgr_img
        except Exception as convert_err:
            print(f"Conversion error: {convert_err}")
            return np.zeros((480, 640, 3), dtype=np.uint8)

    def detect_objects(self, input_img=None):
        detections = []
        depth_values = None
        if self.camera_interface is None:
            if input_img is None:
                return [], input_img
            else:
                return [], input_img
        else:
            try:
                while self.camera_interface.poll_for_frames():
                    _ = self.camera_interface.wait_for_frames(1)
            except:
                pass
            try:
                frames = self.camera_interface.wait_for_frames(200)
                if frames is None:
                    print("No frames received")
                    return [], input_img if input_img is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                if self.depth_support and self.depth_processor is not None:
                    try:
                        frames = self.depth_processor.process(frames)
                    except Exception as align_err:
                        print(f"Alignment error: {align_err}")
                color_frame = frames.get_color_frame()
                if color_frame is None:
                    print("No color frame")
                    return [], input_img if input_img is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                depth_frame = None if not self.depth_support else frames.get_depth_frame()
                input_img = self.convert_to_bgr(color_frame)
                if self.depth_support and depth_frame is not None:
                    try:
                        raw_depth = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                        d_h, d_w = depth_frame.get_height(), depth_frame.get_width()
                        depth_values = raw_depth.reshape((d_h, d_w)).astype(np.float32) * depth_frame.get_depth_scale()
                        depth_values = self.frame_stabilizer.stabilize(depth_values)
                    except Exception as depth_err:
                        print(f"Depth processing error: {depth_err}")
            except Exception as frame_err:
                print(f"Camera error: {frame_err}")
                if input_img is None:
                    return [], np.zeros((480, 640, 3), dtype=np.uint8)
                else:
                    return [], input_img

        # Perform object detection
        try:
            model_output = self.detection_engine(input_img, conf=self.min_confidence, iou=0.45)
            if len(model_output) > 0:
                detection_result = model_output[0]
                if hasattr(detection_result, 'boxes') and len(detection_result.boxes) > 0:
                    boxes = detection_result.boxes.data.cpu().numpy()
                    for bbox in boxes:
                        left, top, right, bottom, conf_score, class_idx = bbox
                        left, top, right, bottom = map(int, [left, top, right, bottom])
                        class_idx = int(class_idx)
                        class_name = self.class_labels[class_idx]
                        center_x = int((left + right) / 2)
                        center_y = int((top + bottom) / 2)
                        distance = None
                        if depth_values is not None:
                            if 0 <= center_y < depth_values.shape[0] and 0 <= center_x < depth_values.shape[1]:
                                depth_val = depth_values[center_y, center_x]
                                if 0 < depth_val < 10000:  # 0-10 meters
                                    distance = depth_val
                        detections.append([class_name, float(conf_score), left, top, right, bottom, distance])
                        cv2.rectangle(input_img, (left, top), (right, bottom), (0, 0, 255), 2)
                        label = f"{class_name},{conf_score:.2f}"
                        if distance is not None:
                            label += f"{distance:.0f}mm"
                        cv2.putText(input_img, label, (left-5, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        except Exception as detect_err:
            print(f"Detection error: {detect_err}")
        return detections, input_img

    def detect_image_file(self, image_path):
        img_data = cv2.imread(image_path)
        results, processed_img = self.detect_objects(img_data)
        return results, processed_img

    def save_detections(self, detections, frame_num, output_dir):
        valid_detections = [d for d in detections if d[1] >= 0.5]
        if valid_detections:
            os.makedirs(output_dir, exist_ok=True)
            file_path = os.path.join(output_dir, f"frame_{frame_num}.txt")
            with open(file_path, 'w', newline='') as f:
                for det in valid_detections:
                    name = det[0]
                    conf = round(det[1], 2)
                    x1 = det[2]
                    y1 = det[3]
                    x2 = det[4]
                    y2 = det[5]
                    cx = int(((x2 - x1) / 2) + x1)
                    cy = int(((y2 - y1) / 2) + y1)
                    depth_val = "0"
                    if len(det) > 6 and det[6] is not None:
                        depth_val = f"{int(det[6])}"
                    f.write(f"{name} {cx} {cy} {x1} {x2} {y1} {y2} {conf} {depth_val}\n")

    def shutdown(self):
        if self.camera_interface is not None:
            self.camera_interface.stop()
            print("Camera shutdown")

class VisionWorker(QObject):
    processing_complete = pyqtSignal(object, object)  # (processed_image, detections)

    def __init__(self, vision_system, output_dir, record_video=False, video_dir=None):
        super().__init__()
        self.vision_system = vision_system
        self.output_dir = output_dir
        self.frame_counter = 0
        self.active = True
        self.record_video = record_video
        self.video_dir = video_dir
        self.video_recorder = None

    def start_processing(self):
        self.active = True
        if self.record_video:
            if not os.path.exists(self.video_dir):
                os.makedirs(self.video_dir)
            _, first_img = self.vision_system.detect_objects()
            if first_img is not None:
                h, w = first_img.shape[:2]
                codec = cv2.VideoWriter_fourcc(*'XVID')
                self.video_recorder = cv2.VideoWriter(os.path.join(self.video_dir, 'output_video.avi'),
                                          codec, 15, (w, h))
            else:
                print("First frame missing, video disabled")
                self.record_video = False
        self.frame_counter = 0
        while self.active:
            detections, processed_img = self.vision_system.detect_objects()
            if processed_img is None or np.all(processed_img == 0):
                print("Black frame skipped")
                time.sleep(0.05)
                continue

            if any(d[1] >= 0.5 for d in detections):
                self.vision_system.save_detections(detections, self.frame_counter, self.output_dir)
            if self.record_video and self.video_recorder is not None:
                self.video_recorder.write(processed_img)
            self.processing_complete.emit(processed_img, detections)
            self.frame_counter += 1
            time.sleep(0.03)

    def stop_processing(self):
        self.active = False
        if self.video_recorder is not None:
            self.video_recorder.release()
        self.vision_system.shutdown()

class VisionApp(QMainWindow):
    def __init__(self, vision_system):
        super().__init__()
        self.vision_system = vision_system
        self.setup_interface()
        self.setup_network()
        self.worker_thread = None
        self.vision_worker = None
        self.frame_index = 0
        self.enable_recording = True
        self.video_storage = "/home/HwHiAiUser/ultralytics-main/runss" 
        self.label_storage = "/home/HwHiAiUser/ultralytics-main/runss/labels" 
        self.file_checker = QTimer(self)
        self.file_checker.timeout.connect(self.check_files)
        self.file_checker.start(5000)
        self.auto_launch = QTimer(self)
        self.auto_launch.timeout.connect(self.automatic_start)
        self.auto_launch.setSingleShot(True)  
        self.auto_launch.start(1000) 

    def setup_interface(self):
        self.setWindowTitle('3D Vision System')
        self.setGeometry(100, 100, 800, 500)
        self.setStyleSheet("background-color:white;")
        main_layout = QHBoxLayout()
        self.display_area = QLabel(self)
        self.display_area.setFixedSize(500, 400)
        self.display_area.setStyleSheet("border:1px solid black;")
        main_layout.addWidget(self.display_area)
        control_panel = QVBoxLayout()
        control_panel.addSpacing(20)
        results_title = QLabel('Detection Results', self)
        results_title.setFont(QFont('Arial', 14))
        results_title.setAlignment(Qt.AlignLeft)
        control_panel.addWidget(results_title)
        self.results_display = QTextEdit(self)
        self.results_display.setReadOnly(True)
        self.results_display.setFixedSize(300, 300)
        self.results_display.setStyleSheet("border:1px solid black;")
        text_font = QFont()
        text_font.setFamily("Arial")
        text_font.setPointSize(14)
        self.results_display.setFont(text_font)
        control_panel.addWidget(self.results_display)
        self.status_panel = QLabel("Initializing...", self)  
        self.status_panel.setFont(QFont('Arial', 16))
        self.status_panel.setFixedSize(300, 60)
        self.status_panel.setAlignment(Qt.AlignCenter)
        self.status_panel.setStyleSheet("background-color:lightgray;color:black;")
        control_panel.addWidget(self.status_panel)
        self.start_button = QPushButton('Auto Start...', self)
        self.start_button.setFont(QFont('Arial', 15))
        self.start_button.setStyleSheet(
            "QPushButton{background-color:#F9C49A;color:black;border-radius:50px;}"
            "QPushButton:pressed{background-color:#FF8C00;}"
        )
        self.start_button.setFixedSize(100, 100)
        self.start_button.setEnabled(False)  
        self.start_button.clicked.connect(self.start_vision)
        control_panel.addWidget(self.start_button)
        control_panel.addStretch()
        main_layout.addLayout(control_panel)
        central_widget = QWidget(self)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

    def check_files(self):
        result_dir = '/home/HwHiAiUser/ultralytics-main/runss/result' 
        if any(file.endswith('.txt') for file in os.listdir(result_dir)):
            print("Result file sent, closing interface")
            self.close()
        else:
            print("Waiting for result files")

    def setup_network(self):
        self.server_ip = '192.168.1.66'
        self.server_port = 6666
        self.net_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.net_socket.connect((self.server_ip, self.server_port))
            print("Interface ready---80%")
            print("Server connected")
        except Exception as conn_err:
            print(f"Connection error: {conn_err}")

    def send_text(self, data_type, content):
        encoded = content.encode()
        length = len(encoded)
        packet = struct.pack('>II', data_type, length) + encoded
        self.net_socket.sendall(packet)
        
    def send_identifier(self, data_type, identifier):
        self.send_text(data_type, identifier)
        
    def send_file(self, data_type, file_path):
        while not os.path.exists(file_path):
            print(f"File missing: {file_path}")
            time.sleep(0.1)
        with open(file_path, 'rb') as file:
            file_data = file.read()
        size = len(file_data)
        header = struct.pack('>II', data_type, size)
        self.net_socket.sendall(header)
        self.net_socket.sendall(file_data)
        
    def send_result(self, result_file):
        result_file = '/home/HwHiAiUser/Desktop/result_r/xuexiao-tuanduiid-R1.txt' 
        self.send_file(1, result_file)
        
    def clean_directory(self, dir_path):
        for item in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item)
            os.remove(item_path)
            print(f"Cleaned: {dir_path}/{item}")
            
    def move_files(self, source, destination):
        if not os.path.exists(destination):
            os.makedirs(destination)
        for item in os.listdir(source):
            if item.endswith('.txt'):
                src_file = os.path.join(source, item)
                dest_file = os.path.join(destination, item)
                shutil.move(src_file, dest_file)
                print(f"Moved {item} to {destination}")
                
    def compute_mode(self, values):
        if not values:
            return None
        frequency = Counter(values)
        mode_value = frequency.most_common(1)[0][0]
        return mode_value
        
    def analyze_directory(self, directory, min_frames=5):
        txt_files = glob.glob(os.path.join(directory, '*.txt'))
        object_data = defaultdict(list)
        frame_count = defaultdict(int)
        for file_path in txt_files:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            current_counts = defaultdict(int)
            for line in lines:
                parts = line.split()
                if not parts or parts[0] in ['Table', 'R_Table']:
                    continue
                if len(parts) >= 8:
                    try:
                        depth = float(parts[-1])
                        if 1000 <= depth <= 1800:
                            obj_name = parts[0]
                            current_counts[obj_name] += 1
                    except ValueError:
                        pass
            for obj_name, count in current_counts.items():
                object_data[obj_name].append(count)
                frame_count[obj_name] += 1
        result = {}
        for obj_name in object_data:
            if frame_count[obj_name] >= min_frames:
                counter = Counter(object_data[obj_name])
                mode = counter.most_common(1)[0][0]
                result[obj_name] = mode
        return result
        
    def select_primary_table(self, tables, lines):
        if not tables:
            return None
        max_objects = 0
        selected_table = None
        for table in tables:
            count = self.count_in_table(lines, table)
            if count > 6 and count > max_objects:
                max_objects = count
                selected_table = table
        return selected_table
        
    def count_in_table(self, lines, table_area):
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
        
    def update_counts(self, count_dict, objects):
        local_count = defaultdict(int)
        for obj in objects:
            local_count[obj] += 1
        for obj, count in local_count.items():
            count_dict[obj].append(count)
            
    def copy_files(self, source, destination):
        if not os.path.exists(destination):
            os.makedirs(destination)
        txt_files = glob.glob(os.path.join(source, '*.txt'))
        for file in txt_files:
            filename = os.path.basename(file)
            dest_file = os.path.join(destination, filename)
            shutil.copy(file, dest_file)
            print(f"Copied {file} to {destination}")
            
    def process_cycle(self):
        dirs = ['/home/HwHiAiUser/ultralytics-main/runss/d'] 
        output_file = '/home/HwHiAiUser/ultralytics-main/runss/result/xuexiao-tuanduiid-R1.txt' 
        time.sleep(2)
        self.clean_directory('/home/HwHiAiUser/ultralytics-main/runss/labels') 
        self.clean_directory('/home/HwHiAiUser/ultralytics-main/runss/d') 
        time.sleep(10)
        self.move_files('/home/HwHiAiUser/ultralytics-main/runss/labels', '/home/HwHiAiUser/ultralytics-main/runss/d') 
        time.sleep(0.5)
        total_counts = defaultdict(int)
        for directory in dirs:
            stable_objects = self.analyze_directory(directory, min_frames=5)
            for obj_name, count in stable_objects.items():
                total_counts[obj_name] = count
        with open(output_file, 'w') as f:
            f.write("START\n")
            for obj_name, count in total_counts.items():
                f.write(f"Goal_ID={obj_name};Num={count}\n")
            f.write("END\n")
        print(f"Result file generated: {output_file}")
        self.clean_directory('/home/HwHiAiUser/Desktop/result_r') 
        time.sleep(0.5)
        self.copy_files('/home/HwHiAiUser/ultralytics-main/runss/result', '/home/HwHiAiUser/Desktop/result_r')
        result_path = '/home/HwHiAiUser/Desktop/result_r/xuexiao-tuanduiid-R1.txt' 
        self.send_file(1, result_path)
        self.net_socket.close()
        print("Network closed")
        time.sleep(5)
        self.clean_directory('/home/HwHiAiUser/ultralytics-main/runss/result') 
   
    def start_vision(self):
        self.status_panel.setText("Processing") 
        self.send_identifier(0, "幻视")
        os.makedirs(self.label_storage, exist_ok=True)
        self.frame_index = 0
        self.worker_thread = QThread()
        self.vision_worker = VisionWorker(self.vision_system, output_dir=self.label_storage,
                                  record_video=self.enable_recording, video_dir=self.video_storage)
        self.vision_worker.moveToThread(self.worker_thread)
        self.vision_worker.processing_complete.connect(self.display_results)
        self.worker_thread.started.connect(self.vision_worker.start_processing)
        self.worker_thread.start()
        threading.Thread(target=self.process_cycle).start()

    def stop_vision(self):
        if self.vision_worker:
            self.vision_worker.stop_processing()
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
        self.vision_system.shutdown()
        cv2.destroyAllWindows()

    def display_results(self, image, detections):
        if image is not None:
            self.show_image(image)
        if detections is not None:
            self.show_detections(detections)
        self.frame_index += 1

    def show_image(self, image):
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        scaled_img = qt_img.scaled(self.display_area.width(), self.display_area.height())
        self.display_area.setPixmap(QPixmap.fromImage(scaled_img))

    def show_detections(self, detections):
        self.results_display.clear()
        obj_counts = {}
        for detection in detections:
            obj_type = detection[0]
            if obj_type in obj_counts:
                obj_counts[obj_type] += 1
            else:
                obj_counts[obj_type] = 1
        for detection in detections:
            obj_name = detection[0]
            count = obj_counts.get(obj_name, 1)
            dist_text = ""
            formatted_dist = ""
            if len(detection) > 6 and detection[6] is not None:
                dist_m = detection[6] / 1000
                dist_text = f"{dist_m:.1f}m"
                if 1.0 <= dist_m <= 1.8: 
                    formatted_dist = f'<span style="color:red;">{dist_text}</span>'
                else:
                    formatted_dist = dist_text
            self.results_display.append(f'Object: {obj_name} Count: {count} {formatted_dist}')
            
    def automatic_start(self):
        print("Interface ready, starting detection")
        self.start_vision()

if __name__ == '__main__':
    vision_system = VisionSystem(model_file='yolo11s.pt', device_id='0') 
    app = QApplication(sys.argv)
    main_app = VisionApp(vision_system)
    main_app.show()
    sys.exit(app.exec_())
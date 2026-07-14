"""
PyQt5 GUI + 检测流程控制

MainWindow: 主界面（图像显示 + 结果列表 + 状态 + 启停按钮）
DetectWorker: 后台检测线程
检测流程: 多帧采集 → 投票 → 生成结果文件 → 发送裁判盒
"""

import os
import time
import glob
import threading
from collections import defaultdict, Counter

import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget,
                              QVBoxLayout, QHBoxLayout, QPushButton,
                              QLabel, QTextEdit)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QTimer, QThread, QObject, pyqtSignal

from config import (
    RoundConfig, TEAM_ID, TEAM_NAME, MIN_OCCURRENCES,
    LABELS_DIR, D_DIR, RESULT_DIR, DESKTOP_RESULT_DIR, VIDEO_DIR,
    DEPTH_MIN_MM, DEPTH_MAX_MM,
)
from network import JudgeBoxClient


class DetectWorker(QObject):
    """后台检测线程 — 循环取帧→推理→写txt"""
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
            os.makedirs(self.video_folder, exist_ok=True)
            _, first_frame = self.detector.inference_image()
            if first_frame is not None and not np.all(first_frame == 0):
                h, w = first_frame.shape[:2]
                self.out = cv2.VideoWriter(
                    os.path.join(self.video_folder, 'output_video.avi'),
                    cv2.VideoWriter_fourcc(*'XVID'), 15, (w, h))
            else:
                self.save_video = False

        while self.running:
            result_list, result_image = self.detector.inference_image()
            if result_image is None or np.all(result_image == 0):
                time.sleep(0.05)
                continue

            if any(r[1] >= self.detector.conf_thres for r in result_list):
                self.detector.write_results_to_txt(
                    result_list, self.frame_idx, self.txt_output_folder)
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
    """主GUI窗口"""

    def __init__(self, detector, round_config=None):
        super().__init__()
        self.detector = detector
        self.round_config = round_config or RoundConfig(round_num=2)
        self.worker_thread = None
        self.worker = None
        self.frame_idx = 0
        self.save_video = True

        # 网络客户端
        self.judge = JudgeBoxClient()

        self.init_ui()
        self.init_socket()

        # 自动启动定时器
        self.auto_start_timer = QTimer(self)
        self.auto_start_timer.timeout.connect(self.auto_start_detection)
        self.auto_start_timer.setSingleShot(True)
        self.auto_start_timer.start(1000)

    # ==================== UI ====================

    def init_ui(self):
        self.setWindowTitle('3D视觉识别 — 观薪 NEEPU-VF')
        self.setGeometry(100, 100, 960, 540)
        self.setStyleSheet("background-color:#f5f5f5;")

        # 主容器
        central = QWidget(self)
        h_layout = QHBoxLayout(central)
        h_layout.setContentsMargins(12, 12, 12, 12)
        h_layout.setSpacing(12)

        # ====== 左侧: 实时画面 ======
        left_panel = QVBoxLayout()
        cam_label = QLabel('📷 实时画面')
        cam_label.setFont(QFont('Arial', 12, QFont.Bold))
        cam_label.setStyleSheet("color:#333;")
        left_panel.addWidget(cam_label)

        self.image_label = QLabel()
        self.image_label.setFixedSize(640, 480)
        self.image_label.setStyleSheet(
            "border:2px solid #ccc; border-radius:6px; background-color:#fff;")
        self.image_label.setAlignment(Qt.AlignCenter)
        left_panel.addWidget(self.image_label)
        left_panel.addStretch()
        h_layout.addLayout(left_panel, stretch=3)

        # ====== 右侧: 结果面板 ======
        right_panel = QVBoxLayout()
        right_panel.setSpacing(10)

        # 状态标签
        self.status_label = QLabel('⏳ 等待启动...')
        self.status_label.setFont(QFont('Arial', 18, QFont.Bold))
        self.status_label.setFixedHeight(44)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet(
            "background-color:#e0e0e0; color:#333; "
            "border-radius:8px; padding:4px;")
        right_panel.addWidget(self.status_label)

        # 识别结果
        results_label = QLabel('📋 实时识别结果')
        results_label.setFont(QFont('Arial', 12, QFont.Bold))
        results_label.setStyleSheet("color:#333;")
        right_panel.addWidget(results_label)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setFont(QFont('Consolas', 12))
        self.result_text.setStyleSheet(
            "border:2px solid #ccc; border-radius:6px; "
            "background-color:#fff; padding:6px;")
        right_panel.addWidget(self.result_text, stretch=1)

        # 启动按钮
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        self.start_button = QPushButton('● 启动中')
        self.start_button.setFont(QFont('Arial', 14, QFont.Bold))
        self.start_button.setFixedSize(120, 120)
        self.start_button.setStyleSheet(
            "QPushButton{background-color:#FF9800; color:white; "
            "border-radius:60px; border:none;}"
            "QPushButton:pressed{background-color:#E65100;}"
            "QPushButton:disabled{background-color:#BDBDBD;}")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_detection)
        btn_layout.addWidget(self.start_button)
        btn_layout.addStretch()
        right_panel.addLayout(btn_layout)

        right_panel.addStretch()
        h_layout.addLayout(right_panel, stretch=2)

        self.setCentralWidget(central)

    # ==================== 网络 ====================

    def init_socket(self):
        self.judge.connect()

    # ==================== 检测流程 ====================

    def start_detection(self):
        """启动检测线程 + 后台流程控制"""
        self.status_label.setText('🔍 识别中...')
        self.status_label.setStyleSheet(
            "background-color:#FFF3E0; color:#E65100; "
            "border-radius:8px; padding:4px;")
        self.judge.send_start()  # DataType 0 = 队伍ID + 开始计时

        os.makedirs(LABELS_DIR, exist_ok=True)
        self.frame_idx = 0

        # 后台检测线程
        self.worker_thread = QThread()
        self.worker = DetectWorker(self.detector, txt_output_folder=LABELS_DIR,
                                   save_video=self.save_video, video_folder=VIDEO_DIR)
        self.worker.moveToThread(self.worker_thread)
        self.worker.result_ready.connect(self.on_result_ready)
        self.worker_thread.started.connect(self.worker.start)
        self.worker_thread.start()

        # 流程控制线程
        if self.round_config.round_num == 2:
            threading.Thread(target=self._run_round2, daemon=True).start()
        else:
            threading.Thread(target=self._run_round1, daemon=True).start()

    def _run_round1(self):
        """Round 1流程: 单桌台，相机固定"""
        output_path = f'{RESULT_DIR}/{TEAM_NAME}-R1.txt'
        self._detect_single_table(table_id=1, output_path=output_path)

    def _run_round2(self):
        """Round 2流程: 三桌台，相机旋转"""
        all_results = {}
        os.makedirs(RESULT_DIR, exist_ok=True)

        for table_id in self.round_config.tables:
            print(f"\n===== Round 2: Table {table_id} =====")
            if table_id > 1:
                self.judge.send_rotate(table_id)  # DataType 3 = 转台旋转信号

            # 清理上一桌台的临时文件
            JudgeBoxClient.delete_all_files_in_folder(LABELS_DIR)
            JudgeBoxClient.delete_all_files_in_folder(D_DIR)
            time.sleep(1)

            # 检测
            time.sleep(self.round_config.detect_time_per_table)
            JudgeBoxClient.move_txt_files(LABELS_DIR, D_DIR)
            time.sleep(0.5)

            # 处理
            stable = self._process_folder(D_DIR, min_occurrences=MIN_OCCURRENCES)
            for word, count in stable.items():
                all_results[f"{word}_T{table_id}"] = (word, count, table_id)

        # 合并写入
        final_path = f'{RESULT_DIR}/{TEAM_NAME}-R2.txt'
        with open(final_path, 'w') as f:
            f.write("START\n")
            for word, count, tid in all_results.values():
                f.write(f"Goal_ID={word};Num={count};Table={tid}\n")
            f.write("END\n")
        print(f"===== Round 2 完成: {final_path} =====")

        # send_result_file (DataType 1) 自动停止计时，无需单独发送结束信号
        self._finish(final_path)

    def _detect_single_table(self, table_id, output_path):
        """单桌台检测（Round 1 或 Round 2 单桌均可）"""
        JudgeBoxClient.delete_all_files_in_folder(LABELS_DIR)
        JudgeBoxClient.delete_all_files_in_folder(D_DIR)
        time.sleep(2)

        detect_time = self.round_config.detect_time_per_table
        time.sleep(detect_time)

        JudgeBoxClient.move_txt_files(LABELS_DIR, D_DIR)
        time.sleep(0.5)

        stable = self._process_folder(D_DIR, min_occurrences=MIN_OCCURRENCES)

        os.makedirs(RESULT_DIR, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write("START\n")
            for word, count in stable.items():
                f.write(f"Goal_ID={word};Num={count};Table={table_id}\n")
            f.write("END\n")
        print(f"{output_path} 已生成")

        # send_result_file (DataType 1) 自动停止计时，无需单独发送结束信号
        self._finish(output_path)

    def _finish(self, result_path):
        """发送结果 → 关闭连接 → 清理"""
        JudgeBoxClient.delete_all_files_in_folder(DESKTOP_RESULT_DIR)
        time.sleep(0.5)
        JudgeBoxClient.copy_txt_files(RESULT_DIR, DESKTOP_RESULT_DIR)
        self.judge.send_result_file(result_path)
        self.judge.close()
        time.sleep(3)
        JudgeBoxClient.delete_all_files_in_folder(RESULT_DIR)

    # ==================== 多帧投票 ====================

    def _process_folder(self, folder, min_occurrences=MIN_OCCURRENCES):
        """多帧投票: 统计所有帧中稳定出现的物品及众数数量"""
        txt_files = glob.glob(os.path.join(folder, '*.txt'))
        object_counts = defaultdict(list)
        frame_presence = defaultdict(int)

        for file in txt_files:
            with open(file, 'r') as fh:
                lines = fh.readlines()
            current_frame_counts = defaultdict(int)
            for line in lines:
                words = line.split()
                if not words or words[0] in ['Table', 'R_Table']:
                    continue
                if len(words) >= 8:
                    try:
                        depth = float(words[-1])
                        if DEPTH_MIN_MM <= depth <= DEPTH_MAX_MM:
                            current_frame_counts[words[0]] += 1
                    except ValueError:
                        pass
            for obj, count in current_frame_counts.items():
                object_counts[obj].append(count)
                frame_presence[obj] += 1

        result = {}
        for obj in object_counts:
            if frame_presence[obj] >= min_occurrences:
                result[obj] = Counter(object_counts[obj]).most_common(1)[0][0]
        return result

    # ==================== 显示回调 ====================

    def on_result_ready(self, result_image, result_list):
        if result_image is not None:
            self.display_image(result_image)
        if result_list is not None:
            self.display_results(result_list)
        self.frame_idx += 1

    def display_image(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
        h, w, ch = rgb.shape
        bytes_per_line = w * ch
        qt_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_img).scaled(
            self.image_label.width(), self.image_label.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)

    def display_results(self, result_list):
        self.result_text.clear()
        # 统计每类物品数量
        counts = {}
        for r in result_list:
            name = r[0]
            counts[name] = counts.get(name, 0) + 1

        for name, count in sorted(counts.items()):
            # 取该类的第一条记录获取距离
            dist_text = ""
            for r in result_list:
                if r[0] == name and len(r) > 6 and r[6] is not None:
                    dist_m = r[6] / 1000
                    if DEPTH_MIN_MM / 1000 <= dist_m <= DEPTH_MAX_MM / 1000:
                        dist_text = f"  ({dist_m:.2f}m)"
                    break
            self.result_text.append(f'{name} ×{count}{dist_text}')

    def stop_detection(self):
        if self.worker:
            self.worker.stop()
        if self.worker_thread:
            self.worker_thread.quit()
            self.worker_thread.wait()
        self.detector.close()
        cv2.destroyAllWindows()

    def auto_start_detection(self):
        print("自动启动中...")
        self.start_detection()

"""
YOLO + ORBBEC 相机检测器

整合:
  - ORBBEC Astra Pro Plus RGBD相机 (pyorbbecsdk)
  - YOLO物体检测 (ultralytics / NPU Ascend 310B4)
  - OCR文字识别 (ocr_module)
  - 深度时域滤波
"""

import os
import time
import cv2
import numpy as np
from ultralytics import YOLO
from pyorbbecsdk import Config, OBSensorType, Pipeline, AlignFilter, OBStreamType
from ocr_module import LightweightOCR
from config import (
    CONF_THRES, IOU_THRES, DEPTH_MIN_MM, DEPTH_MAX_MM,
    TEMPORAL_ALPHA, DEFAULT_WEIGHTS, USE_NPU,
)

# 延迟导入 NPU 模块，避免在没有 CANN 环境的机器上崩溃
try:
    from npu_detector import NPUDetector, decode_npu_output
    HAS_NPU = True
except ImportError:
    HAS_NPU = False


class TemporalFilter:
    """时域滤波器 — 平滑深度数据，减少帧间抖动"""
    def __init__(self, alpha=TEMPORAL_ALPHA):
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
    """YOLO检测器 + ORBBEC相机 + OCR — 支持 NPU/CPU 双路径"""

    def __init__(self, weights=DEFAULT_WEIGHTS, device='0', half=False):
        self.conf_thres = CONF_THRES
        self.device = device
        self.half = half
        self.use_npu = USE_NPU and HAS_NPU
        self.npu = None
        self.num_classes = 8  # 默认 8 类

        # ── 尝试 NPU 加载 ──
        if self.use_npu:
            om_path = weights.replace('.pt', '.om')
            if not os.path.exists(om_path):
                print(f"[检测器] .om 未找到 ({om_path})，回退 PyTorch CPU")
                self.use_npu = False
            else:
                try:
                    self.npu = NPUDetector(om_path)
                    # 用 PyTorch 模型读一下类别名
                    self.model = YOLO(weights)
                    self.names = self.model.names
                    self.num_classes = len(self.names)
                    print(f"[检测器] ★ NPU 推理已启用: {om_path}, 类别数={self.num_classes}")
                except Exception as e:
                    print(f"[检测器] NPU 初始化失败: {e}，回退 PyTorch CPU")
                    self.use_npu = False
                    self.npu = None

        # ── PyTorch CPU 路径 ──
        if not self.use_npu:
            try:
                self.model = YOLO(weights)
                self.names = self.model.names
                self.num_classes = len(self.names)
                print(f"[检测器] YOLO模型已加载(CPU): {weights}, 类别数={self.num_classes}")
            except Exception as e:
                print(f"[检测器] YOLO模型加载失败: {e}")
                raise

        # OCR
        self.ocr = LightweightOCR()
        self._w_class_ids = self._detect_w_class_ids()

        # 相机
        self.pipeline = None
        self.align = None
        self.depth_available = False
        self.temporal_filter = TemporalFilter()
        self.init_camera()

    # ==================== 相机初始化 ====================

    def init_camera(self):
        """初始化ORBBEC相机，先预热彩色流，再开启彩色+深度"""
        try:
            os.system('sudo sh -c "echo 2048 > /sys/module/usbcore/parameters/usbfs_memory_mb"')
        except:
            pass

        config = Config()
        self.pipeline = Pipeline()

        try:
            # 阶段1: 彩色流预热
            color_profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            config.enable_stream(color_profiles.get_default_video_stream_profile())
            self.pipeline.start(config)
            print("[相机] 预热中...")
            for _ in range(30):
                try:
                    frames = self.pipeline.wait_for_frames(200)
                except:
                    pass
                time.sleep(0.3)
            self.pipeline.stop()
            print("[相机] 预热完成")

            # 阶段2: 彩色+深度
            config = Config()
            config.enable_stream(color_profiles.get_default_video_stream_profile())
            try:
                depth_profiles = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
                config.enable_stream(depth_profiles.get_default_video_stream_profile())
                self.align = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
                try:
                    self.pipeline.enable_frame_sync()
                except:
                    pass
                self.depth_available = True
            except Exception as e:
                print(f"[相机] 深度流不可用: {e}, 仅用彩色模式")

            self.pipeline.start(config)
            mode = "彩色+深度" if self.depth_available else "彩色"
            print(f"[相机] 启动成功, 模式={mode}")

        except Exception as e:
            print(f"[相机] 启动失败: {e}")
            self.pipeline = None

    # ==================== 图像转换 ====================

    def frame_to_bgr_image(self, frame):
        """将ORBBEC帧转为OpenCV BGR格式"""
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
                except:
                    pass
                return np.zeros((h, w, 3), dtype=np.uint8)
            img = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3))
            return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"[检测器] 帧转换失败: {e}")
            return np.zeros((480, 640, 3), dtype=np.uint8)

    # ==================== OCR辅助 ====================

    def _detect_w_class_ids(self):
        """检测W类(class名以W开头)的ID列表"""
        w_ids = [cid for cid, name in self.names.items()
                 if isinstance(name, str) and name.startswith('W')]
        print(f"[OCR] W类物品ID: {w_ids}")
        return w_ids

    def _ocr_classify(self, image, bbox, yolo_name):
        """对W类物品进行OCR分类"""
        if not self._w_class_ids or not self.ocr:
            return yolo_name
        recognized = self.ocr.recognize(image, bbox)
        if recognized:
            subclass = self.ocr.classify_subclass(recognized)
            if subclass:
                return subclass
        return yolo_name

    # ==================== 推理 ====================

    def inference_image(self, opencv_image=None):
        """单帧推理: 获取相机帧 → YOLO检测 → OCR(W类) → 返回结果列表+图像"""
        result_list = []
        depth_data = None

        if self.pipeline is None:
            return [], (opencv_image if opencv_image is not None
                       else np.zeros((480, 640, 3), dtype=np.uint8))

        # 清空帧缓冲
        try:
            while self.pipeline.poll_for_frames():
                _ = self.pipeline.wait_for_frames(1)
        except:
            pass

        # 获取最新帧
        try:
            frames = self.pipeline.wait_for_frames(200)
            if frames is None:
                return [], opencv_image or np.zeros((480, 640, 3), dtype=np.uint8)

            if self.depth_available and self.align is not None:
                try:
                    frames = self.align.process(frames)
                except:
                    pass

            color_frame = frames.get_color_frame()
            if color_frame is None:
                return [], opencv_image or np.zeros((480, 640, 3), dtype=np.uint8)

            opencv_image = self.frame_to_bgr_image(color_frame)

            if self.depth_available:
                depth_frame = frames.get_depth_frame()
                if depth_frame is not None:
                    try:
                        raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
                        h, w = depth_frame.get_height(), depth_frame.get_width()
                        depth_data = (raw.reshape((h, w)).astype(np.float32)
                                      * depth_frame.get_depth_scale())
                        depth_data = self.temporal_filter.process(depth_data)
                    except Exception as e:
                        print(f"[检测器] 深度处理失败: {e}")
        except Exception as e:
            print(f"[检测器] 获取帧失败: {e}")
            return [], opencv_image or np.zeros((480, 640, 3), dtype=np.uint8)

        # YOLO推理 — NPU 或 CPU 双路径
        try:
            if self.use_npu and self.npu is not None:
                # ── NPU 推理 ──
                raw = self.npu.infer(opencv_image)
                ih, iw = opencv_image.shape[:2]
                detections = decode_npu_output(
                    raw, iw, ih,
                    num_classes=self.num_classes,
                    conf_thres=self.conf_thres,
                    iou_thres=IOU_THRES,
                )
                for det in detections:
                    x1, y1, x2, y2 = map(int, det[:4])
                    conf = float(det[4])
                    cls_id = int(det[5])
                    name = self.names.get(cls_id, f'W{cls_id:03d}')

                    # OCR 跳过 (NPU 路径暂不接 OCR，先跑通基础检测)
                    # 深度测距
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    distance = None
                    if depth_data is not None:
                        if (0 <= cy < depth_data.shape[0]
                                and 0 <= cx < depth_data.shape[1]):
                            depth_mm = depth_data[cy, cx]
                            if DEPTH_MIN_MM < depth_mm < DEPTH_MAX_MM:
                                distance = depth_mm

                    result_list.append([name, conf, x1, y1, x2, y2, distance])
                    cv2.rectangle(opencv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{name} {conf:.2f}"
                    if distance is not None:
                        label += f" {distance:.0f}mm"
                    cv2.putText(opencv_image, label, (x1 - 5, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                # ── PyTorch CPU 推理 ──
                results = self.model(opencv_image, conf=self.conf_thres, iou=IOU_THRES)
                if len(results) > 0 and hasattr(results[0], 'boxes') and len(results[0].boxes) > 0:
                    for box in results[0].boxes.data.cpu().numpy():
                        x1, y1, x2, y2, conf, cls_id = box
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                        cls_id = int(cls_id)
                        name = self.names[cls_id]

                        # OCR: W类物品读表面文字
                        if cls_id in self._w_class_ids and self.ocr.available:
                            ocr_name = self._ocr_classify(opencv_image, (x1, y1, x2, y2), name)
                            if ocr_name != name:
                                print(f"  [OCR] {name} → {ocr_name}")
                                name = ocr_name

                        # 深度测距
                        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                        distance = None
                        if depth_data is not None:
                            if (0 <= cy < depth_data.shape[0]
                                    and 0 <= cx < depth_data.shape[1]):
                                depth_mm = depth_data[cy, cx]
                                if DEPTH_MIN_MM < depth_mm < DEPTH_MAX_MM:
                                    distance = depth_mm

                        result_list.append([name, float(conf), x1, y1, x2, y2, distance])

                        # 可视化绘制
                        cv2.rectangle(opencv_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        label = f"{name},{conf:.2f}"
                        if distance is not None:
                            label += f",{distance:.0f}mm"
                        cv2.putText(opencv_image, label, (x1 - 5, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        except Exception as e:
            print(f"[检测器] YOLO推理失败: {e}")

        return result_list, opencv_image

    def inference_image_from_file(self, image_file):
        """从文件推理（测试用）"""
        img = cv2.imread(image_file)
        return self.inference_image(img)

    # ==================== 结果写入 ====================

    def write_results_to_txt(self, result_list, frame_idx, output_folder):
        """将单帧检测结果写入txt（供多帧投票用）"""
        filtered = [r for r in result_list if r[1] >= CONF_THRES]
        if not filtered:
            return
        os.makedirs(output_folder, exist_ok=True)
        with open(os.path.join(output_folder, f"frame_{frame_idx}.txt"), 'w') as f:
            for r in filtered:
                name, conf, xmin, ymin, xmax, ymax = r[:6]
                x = int((xmax - xmin) / 2 + xmin)
                y = int((ymax - ymin) / 2 + ymin)
                depth_str = str(int(r[6])) if len(r) > 6 and r[6] is not None else "0"
                f.write(f"{name} {x} {y} {xmin} {xmax} {ymin} {ymax} {conf:.2f} {depth_str}\n")

    def close(self):
        """释放资源"""
        if self.pipeline is not None:
            self.pipeline.stop()
            print("[相机] 已关闭")
        if self.npu is not None:
            self.npu.close()

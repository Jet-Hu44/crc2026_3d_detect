"""
NPU 推理模块 — Ascend 310B4 加速 YOLO 检测 (CANN 7.0)

用法:
    detector = NPUDetector('best4.om')
    output = detector.infer(image_np)  # raw float32 output
"""

import os
import sys
import numpy as np
import cv2

CANN_PYTHON = '/usr/local/Ascend/ascend-toolkit/latest/python/site-packages'
if CANN_PYTHON not in sys.path:
    sys.path.append(CANN_PYTHON)

import acl

# CANN 7.0 移除了大部分常量，硬编码
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2


def _safe_ret(ret):
    """CANN 7.0 有时返回 (value, code) 有时直接返回 code"""
    if isinstance(ret, tuple):
        return ret[-1]
    return ret


class NPUDetector:
    """Ascend 310B4 NPU 推理器"""

    def __init__(self, om_path, imgsz=640):
        if not os.path.exists(om_path):
            raise FileNotFoundError(f"模型文件不存在: {om_path}")

        self.imgsz = imgsz
        self.device_id = 0
        self._model_id = None
        self._model_desc = None
        self._input_buffer = None
        self._output_buffer = None
        self._input_dataset = None
        self._output_dataset = None
        self._stream = None
        self._input_size = 0
        self._output_size = 0

        # 1. 初始化 ACL
        ret = _safe_ret(acl.init(""))
        if ret != 0:
            raise RuntimeError(f"acl.init failed: {ret}")
        print("[NPU] ACL 初始化完成")

        # 2. 设置设备
        ret = _safe_ret(acl.rt.set_device(self.device_id))
        if ret != 0:
            raise RuntimeError(f"set_device failed: {ret}")

        # 3. 创建流
        stream_result = acl.rt.create_stream()
        if isinstance(stream_result, tuple):
            self._stream = stream_result[0]
        else:
            self._stream = stream_result

        # 4. 加载模型
        load_result = acl.mdl.load_from_file(om_path)
        if isinstance(load_result, tuple):
            self._model_id = load_result[0]
        else:
            self._model_id = load_result
        print(f"[NPU] 模型已加载: {os.path.basename(om_path)}")

        # 5. 获取模型描述
        self._model_desc = acl.mdl.create_desc()
        ret = _safe_ret(acl.mdl.get_desc(self._model_desc, self._model_id))
        if ret != 0:
            raise RuntimeError(f"get_desc failed: {ret}")

        # 6. 分配输入内存
        self._input_size = acl.mdl.get_input_size_by_index(self._model_desc, 0)
        malloc_result = acl.rt.malloc(self._input_size)
        if isinstance(malloc_result, tuple):
            self._input_buffer = malloc_result[0]
        else:
            self._input_buffer = malloc_result

        self._input_dataset = acl.mdl.create_dataset()
        input_buf = acl.create_data_buffer(self._input_buffer, self._input_size)
        acl.mdl.add_dataset_buffer(self._input_dataset, input_buf)

        # 7. 分配输出内存
        self._output_size = acl.mdl.get_output_size_by_index(self._model_desc, 0)
        malloc_result = acl.rt.malloc(self._output_size)
        if isinstance(malloc_result, tuple):
            self._output_buffer = malloc_result[0]
        else:
            self._output_buffer = malloc_result

        self._output_dataset = acl.mdl.create_dataset()
        output_buf = acl.create_data_buffer(self._output_buffer, self._output_size)
        acl.mdl.add_dataset_buffer(self._output_dataset, output_buf)

        print(f"[NPU] 输入: {self._input_size} bytes, 输出: {self._output_size} bytes")

        # 8. 预热
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        for _ in range(2):
            self.infer(dummy)
        print("[NPU] 预热完成")

    def _preprocess(self, image):
        """BGR → RGB → resize → HWC→CHW → float32[0,1] → NCHW"""
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.imgsz, self.imgsz))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return np.ascontiguousarray(img)

    def infer(self, image):
        """单帧推理 → 返回 raw float32 numpy 数组"""
        input_np = self._preprocess(image)
        nbytes = input_np.nbytes

        # H2D
        ret = _safe_ret(acl.rt.memcpy(self._input_buffer, self._input_size,
                                       input_np.tobytes(), nbytes,
                                       ACL_MEMCPY_HOST_TO_DEVICE))
        if ret != 0:
            raise RuntimeError(f"memcpy H2D failed: {ret}")

        # Execute
        ret = _safe_ret(acl.mdl.execute(self._model_id,
                                         self._input_dataset,
                                         self._output_dataset,
                                         self._stream))
        if ret != 0:
            raise RuntimeError(f"mdl.execute failed: {ret}")

        # Sync
        ret = _safe_ret(acl.rt.synchronize_stream(self._stream))
        if ret != 0:
            raise RuntimeError(f"synchronize_stream failed: {ret}")

        # D2H
        output_np = np.empty(self._output_size, dtype=np.byte)
        ret = _safe_ret(acl.rt.memcpy(output_np.ctypes.data, self._output_size,
                                       self._output_buffer, self._output_size,
                                       ACL_MEMCPY_DEVICE_TO_HOST))
        if ret != 0:
            raise RuntimeError(f"memcpy D2H failed: {ret}")

        return np.frombuffer(output_np.tobytes(), dtype=np.float32)

    def close(self):
        """释放资源"""
        try:
            if self._input_buffer is not None:
                acl.rt.free(self._input_buffer)
            if self._output_buffer is not None:
                acl.rt.free(self._output_buffer)
            if self._input_dataset is not None:
                acl.mdl.destroy_dataset(self._input_dataset)
            if self._output_dataset is not None:
                acl.mdl.destroy_dataset(self._output_dataset)
            if self._model_id is not None:
                acl.mdl.unload(self._model_id)
            if self._model_desc is not None:
                acl.mdl.destroy_desc(self._model_desc)
            if self._stream is not None:
                acl.rt.destroy_stream(self._stream)
        except:
            pass
        try:
            acl.rt.reset_device(self.device_id)
        except:
            pass
        try:
            acl.finalize()
        except:
            pass
        print("[NPU] 资源已释放")

    def __del__(self):
        self.close()


# ── YOLO 后处理 ────────────────────────────────────────────────────────

def nms_fast(boxes, scores, iou_thres=0.45):
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(), scores.tolist(), score_threshold=0.0,
        nms_threshold=iou_thres, top_k=300)
    if len(indices) == 0:
        return np.array([], dtype=np.int32)
    return indices.flatten().astype(np.int32)


def decode_npu_output(raw_output, img_w, img_h, num_classes=8,
                      conf_thres=0.5, iou_thres=0.45):
    """
    解码 NPU 输出 → 检测框 (N, 6): [x1, y1, x2, y2, conf, cls_id]
    """
    total = len(raw_output)
    nc = num_classes
    expected = 4 + nc  # cx,cy,w,h + class_scores

    if total % expected != 0:
        # 可能带 batch 维度
        if total % (expected * 1) == 0:
            pass  # OK
        else:
            print(f"[NPU解码] 无法解析输出: {total} 元素, 预期每anchor {expected}")
            return np.empty((0, 6))

    num_anchors = total // expected
    output = raw_output.reshape(1, expected, num_anchors)[0]

    boxes_raw = output[:4, :]   # (4, A): cx, cy, w, h
    scores_raw = output[4:, :]  # (nc, A)

    max_scores = scores_raw.max(axis=0)
    max_cls = scores_raw.argmax(axis=0)

    mask = max_scores > conf_thres
    if not mask.any():
        return np.empty((0, 6))

    boxes_raw = boxes_raw[:, mask]
    max_scores = max_scores[mask]
    max_cls = max_cls[mask]

    cx, cy, w, h = boxes_raw[0], boxes_raw[1], boxes_raw[2], boxes_raw[3]
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h

    boxes = np.stack([x1, y1, x2, y2], axis=1)

    keep = nms_fast(boxes, max_scores, iou_thres)
    if len(keep) == 0:
        return np.empty((0, 6))

    return np.column_stack([
        boxes[keep],
        max_scores[keep][:, None],
        max_cls[keep][:, None],
    ])


# ── 测试入口 ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    import time

    om_file = 'best4.om'
    if not os.path.exists(om_file):
        print(f"❌ {om_file} 未找到"); sys.exit(1)

    print(f"[NPU 测试] 加载 {om_file} ...")
    d = NPUDetector(om_file)

    dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    times = []
    for i in range(50):
        t0 = time.perf_counter()
        _ = d.infer(dummy)
        t = (time.perf_counter() - t0) * 1000
        if i >= 5:
            times.append(t)

    if times:
        print(f"[NPU 测试] 推理速度: {np.mean(times):.1f}ms "
              f"(min={np.min(times):.1f}, max={np.max(times):.1f})")
    d.close()
    print("[NPU 测试] ✅ 完成")

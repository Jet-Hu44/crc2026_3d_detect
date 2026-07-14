"""
NPU 推理模块 — Ascend 310B4 加速 YOLO 检测

用法:
    detector = NPUDetector('best4.om')
    boxes = detector.infer(image_np)  # → [(x1,y1,x2,y2,conf,cls), ...]
"""

import os
import sys
import numpy as np
import cv2

# CANN Python API 路径
CANN_PYTHON = '/usr/local/Ascend/ascend-toolkit/latest/python/site-packages'
if CANN_PYTHON not in sys.path:
    sys.path.append(CANN_PYTHON)

import acl


# ── ACL 错误码 ─────────────────────────────────────────────────────────
ACL_SUCCESS = 0
ACL_ERROR_NONE = 0


def _check_ret(msg, ret):
    """检查 ACL 返回值，非零抛异常"""
    if ret != ACL_SUCCESS:
        raise RuntimeError(f"{msg} failed, error={ret}")


# ── NPU 检测器 ─────────────────────────────────────────────────────────

class NPUDetector:
    """Ascend 310B4 NPU 推理器 — 加载 .om 模型，执行推理，返回检测框"""

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

        # 1. 初始化 ACL
        ret = acl.init("")
        _check_ret("acl.init", ret)
        print(f"[NPU] ACL 初始化完成")

        # 2. 设置设备
        ret = acl.rt.set_device(self.device_id)
        _check_ret("acl.rt.set_device", ret)

        # 3. 创建运行流
        self._stream, ret = acl.rt.create_stream()
        _check_ret("acl.rt.create_stream", ret)

        # 4. 加载模型
        self._model_id, ret = acl.mdl.load_from_file(om_path)
        _check_ret("acl.mdl.load_from_file", ret)
        print(f"[NPU] 模型已加载: {os.path.basename(om_path)}")

        # 5. 获取模型描述
        self._model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self._model_desc, self._model_id)
        _check_ret("acl.mdl.get_desc", ret)

        # 6. 分配输入内存
        self._input_size = acl.mdl.get_input_size_by_index(self._model_desc, 0)
        self._input_buffer, ret = acl.rt.malloc(
            self._input_size, acl.ACL_MEM_MALLOC_HUGE_FIRST)
        _check_ret("acl.rt.malloc(input)", ret)

        self._input_dataset = acl.mdl.create_dataset()
        input_ds_buf = acl.create_data_buffer(self._input_buffer, self._input_size)
        ret, = acl.mdl.add_dataset_buffer(self._input_dataset, input_ds_buf)
        _check_ret("add_dataset_buffer(input)", ret)

        # 7. 分配输出内存
        self._output_size = acl.mdl.get_output_size_by_index(self._model_desc, 0)
        self._output_buffer, ret = acl.rt.malloc(
            self._output_size, acl.ACL_MEM_MALLOC_HUGE_FIRST)
        _check_ret("acl.rt.malloc(output)", ret)

        self._output_dataset = acl.mdl.create_dataset()
        output_ds_buf = acl.create_data_buffer(self._output_buffer, self._output_size)
        ret, = acl.mdl.add_dataset_buffer(self._output_dataset, output_ds_buf)
        _check_ret("add_dataset_buffer(output)", ret)

        print(f"[NPU] 输入: {self._input_size} bytes, 输出: {self._output_size} bytes")

        # 8. 预热（避免首次推理慢）
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        for _ in range(2):
            self.infer(dummy)

    # ── 图像预处理 ──────────────────────────────────────────────────

    def _preprocess(self, image):
        """BGR → RGB → resize(640×640) → HWC→CHW → float32[0,1] → NCHW"""
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.imgsz, self.imgsz))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))   # HWC → CHW
        img = np.expand_dims(img, axis=0)     # → (1, 3, 640, 640)
        return np.ascontiguousarray(img)

    # ── NPU 推理 ────────────────────────────────────────────────────

    def infer(self, image):
        """单帧推理 → 返回原始输出 numpy (1, 4+nc, anchors)"""
        input_np = self._preprocess(image)

        # 数据拷贝到 NPU
        nbytes = input_np.nbytes
        ret = acl.rt.memcpy(self._input_buffer, self._input_size,
                            input_np.tobytes(), nbytes,
                            acl.ACL_MEMCPY_HOST_TO_DEVICE)
        _check_ret("memcpy H2D", ret)

        # 执行推理
        ret = acl.mdl.execute(self._model_id,
                              self._input_dataset,
                              self._output_dataset,
                              self._stream)
        _check_ret("mdl.execute", ret)

        # 同步等待完成
        ret = acl.rt.synchronize_stream(self._stream)
        _check_ret("synchronize_stream", ret)

        # 结果拷贝回 CPU
        output_np = np.empty(self._output_size, dtype=np.byte)
        ret = acl.rt.memcpy(output_np.ctypes.data, self._output_size,
                            self._output_buffer, self._output_size,
                            acl.ACL_MEMCPY_DEVICE_TO_HOST)
        _check_ret("memcpy D2H", ret)

        # 转 float32 并 reshape
        output = np.frombuffer(output_np.tobytes(), dtype=np.float32)
        return output

    # ── 资源释放 ────────────────────────────────────────────────────

    def close(self):
        if self._input_buffer:
            acl.rt.free(self._input_buffer)
        if self._output_buffer:
            acl.rt.free(self._output_buffer)
        if self._input_dataset:
            acl.mdl.destroy_dataset(self._input_dataset)
        if self._output_dataset:
            acl.mdl.destroy_dataset(self._output_dataset)
        if self._model_id:
            acl.mdl.unload(self._model_id)
        if self._model_desc:
            acl.mdl.destroy_desc(self._model_desc)
        if self._stream:
            acl.rt.destroy_stream(self._stream)
        acl.rt.reset_device(self.device_id)
        acl.finalize()
        print("[NPU] 资源已释放")

    def __del__(self):
        try:
            self.close()
        except:
            pass


# ── YOLO 后处理 ────────────────────────────────────────────────────────

def nms_fast(boxes, scores, iou_thres=0.45, max_dets=300):
    """OpenCV NMS — 在 CPU 上做，很快 (~1ms)"""
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)
    indices = cv2.dnn.NMSBoxes(
        boxes.tolist(), scores.tolist(), score_threshold=0.0,
        nms_threshold=iou_thres, top_k=max_dets)
    if len(indices) == 0:
        return np.array([], dtype=np.int32)
    return indices.flatten().astype(np.int32)


def decode_npu_output(raw_output, img_w, img_h, num_classes=8,
                      conf_thres=0.5, iou_thres=0.45):
    """
    解码 NPU 输出 → 检测框列表

    raw_output: np.float32, 来自 NPU 的原始 1D 数组
    实际 reshape 为 (1, 4+nc, 8400)

    返回: np.array shape (N, 6) 每行 [x1, y1, x2, y2, conf, cls_id]
    """
    total_elements = len(raw_output)
    nc = num_classes
    # 预期 anchors = total / (1 * (4+nc))
    expected_per_anchor = 4 + nc  # 4 bbox + nc scores
    num_anchors = total_elements // expected_per_anchor

    if total_elements % expected_per_anchor != 0:
        # 可能是 NMS 版本（输出 = (1, N, 6)）
        # 尝试 reshape 为 (N, 7) — [batch, x1, y1, x2, y2, conf, cls]
        if total_elements % 7 == 0:
            dets = raw_output.reshape(-1, 7)
            batch = dets[:, 0]
            x1, y1, x2, y2 = dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
            conf = dets[:, 5]
            cls_id = dets[:, 6].astype(np.int32)
            mask = conf > conf_thres
            boxes = np.stack([x1, y1, x2, y2], axis=1)[mask]
            conf = conf[mask]
            cls_id = cls_id[mask]
            return np.column_stack([boxes, conf[:, None], cls_id[:, None]])
        else:
            print(f"[NPU解码] 无法解析输出: {total_elements} 元素")
            return np.empty((0, 6))

    # 标准版本：reshape → 解码
    output = raw_output.reshape(1, expected_per_anchor, num_anchors)
    output = output[0]  # (4+nc, anchors)

    boxes_raw = output[:4, :]     # (4, anchors): cx, cy, w, h
    scores_raw = output[4:, :]    # (nc, anchors)

    # 每个 anchor 取最大置信度
    max_scores = scores_raw.max(axis=0)
    max_cls = scores_raw.argmax(axis=0)

    # 置信度过滤
    mask = max_scores > conf_thres
    if not mask.any():
        return np.empty((0, 6))

    boxes_raw = boxes_raw[:, mask]
    max_scores = max_scores[mask]
    max_cls = max_cls[mask]

    # cx,cy,w,h → x1,y1,x2,y2 (归一化坐标 → 像素)
    cx, cy, w, h = boxes_raw[0], boxes_raw[1], boxes_raw[2], boxes_raw[3]
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h

    boxes = np.stack([x1, y1, x2, y2], axis=1)

    # NMS
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

    print("[NPU 测试] 查找模型...")
    om_file = None
    for f in ['best4.om', 'best4_npu.om']:
        if os.path.exists(f):
            om_file = f
            break
    if om_file is None:
        print("[NPU 测试] ❌ 未找到 .om 文件，请先运行 export_onnx.py 和 ATC 转换")
        sys.exit(1)

    print(f"[NPU 测试] 加载 {om_file} ...")
    d = NPUDetector(om_file)

    # 计时测试
    dummy = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    times = []
    for i in range(50):
        t0 = time.perf_counter()
        _ = d.infer(dummy)
        t = (time.perf_counter() - t0) * 1000
        if i >= 5:  # 跳过预热
            times.append(t)

    if times:
        print(f"[NPU 测试] 推理速度: {np.mean(times):.1f}ms (min={np.min(times):.1f}, max={np.max(times):.1f})")
    d.close()
    print("[NPU 测试] ✅ 完成")

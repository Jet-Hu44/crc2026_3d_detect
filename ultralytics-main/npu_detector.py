"""
NPU 推理模块 — Ascend 310B4 + CANN 7.0

用法:
    detector = NPUDetector('best4.om')
    output = detector.infer(image_np)  # raw float32 output
"""

import os
import sys
import ctypes
import numpy as np
import cv2

sys.path.append('/usr/local/Ascend/ascend-toolkit/latest/python/site-packages')
import acl

# CANN 7.0 方向常量 (Python 层不暴露，硬编码)
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEMCPY_HOST_TO_HOST = 3
ACL_MALLOC_POLICY = 0  # acl.rt.malloc 第二个参数


def _get_ret(result):
    """CANN 7.0 函数返回 (value, code) 元组时取 code"""
    if isinstance(result, tuple):
        return result[-1]
    return result


def _get_val(result):
    """取 (value, code) 元组的 value"""
    if isinstance(result, tuple) and len(result) >= 2:
        return result[0]
    return result


class NPUDetector:
    """Ascend 310B4 NPU 推理器"""

    def __init__(self, om_path, imgsz=640):
        if not os.path.exists(om_path):
            raise FileNotFoundError(f"模型不存在: {om_path}")

        self.imgsz = imgsz
        self.device_id = 0

        # 1. init ACL
        ret = _get_ret(acl.init(""))
        assert ret == 0, f"acl.init: {ret}"
        print("[NPU] ACL init OK")

        # 2. set device
        ret = _get_ret(acl.rt.set_device(self.device_id))
        assert ret == 0, f"set_device: {ret}"

        # 3. create stream
        stream_result = acl.rt.create_stream()
        self._stream = _get_val(stream_result)
        ret = _get_ret(stream_result)
        assert ret == 0, f"create_stream: {ret}"

        # 4. load model
        load_result = acl.mdl.load_from_file(om_path)
        self._model_id = _get_val(load_result)
        ret = _get_ret(load_result)
        assert ret == 0, f"load_from_file: {ret}"
        print(f"[NPU] Model loaded: {os.path.basename(om_path)}")

        # 5. model desc
        self._model_desc = acl.mdl.create_desc()
        ret = _get_ret(acl.mdl.get_desc(self._model_desc, self._model_id))
        assert ret == 0, f"get_desc: {ret}"

        # 6. input dataset
        self._input_size = acl.mdl.get_input_size_by_index(self._model_desc, 0)
        self._dev_input, ret = acl.rt.malloc(self._input_size, ACL_MALLOC_POLICY)
        assert ret == 0, f"malloc(input): {ret}"

        self._input_ds = acl.mdl.create_dataset()
        in_buf = acl.create_data_buffer(self._dev_input, self._input_size)
        ret = _get_ret(acl.mdl.add_dataset_buffer(self._input_ds, in_buf))
        assert ret == 0, f"add_dataset_buffer(input): {ret}"

        # 7. output dataset
        self._output_size = acl.mdl.get_output_size_by_index(self._model_desc, 0)

        self._dev_output, ret = acl.rt.malloc(self._output_size, ACL_MALLOC_POLICY)
        assert ret == 0, f"malloc(output): {ret}"

        self._output_ds = acl.mdl.create_dataset()
        out_buf = acl.create_data_buffer(self._dev_output, self._output_size)
        ret = _get_ret(acl.mdl.add_dataset_buffer(self._output_ds, out_buf))
        assert ret == 0, f"add_dataset_buffer(output): {ret}"

        # 8. host memory for transfers
        self._host_input, ret = acl.rt.malloc_host(self._input_size)
        assert ret == 0, f"malloc_host(input): {ret}"
        self._host_output, ret = acl.rt.malloc_host(self._output_size)
        assert ret == 0, f"malloc_host(output): {ret}"

        print(f"[NPU] Input: {self._input_size}B, Output: {self._output_size}B")

        # 9. warmup
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(2):
            self.infer(dummy)
        print("[NPU] Warmup done")

    def _preprocess(self, image):
        """BGR → letterbox(640,640) → RGB → HWC→CHW → /255 → (1,3,640,640)

        返回 (input_tensor, ratio, (dw, dh)) 用于后续坐标还原
        """
        # Letterbox: 保持宽高比，黑边填充到 640x640
        h0, w0 = image.shape[:2]
        r = min(self.imgsz / h0, self.imgsz / w0)
        new_w, new_h = int(round(w0 * r)), int(round(h0 * r))
        dw = (self.imgsz - new_w) / 2
        dh = (self.imgsz - new_h) / 2

        img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=(114, 114, 114))

        # 转为模型输入格式
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return np.ascontiguousarray(img), r, (dw, dh)

    def infer(self, image):
        """单帧 NPU 推理 → (raw_float32, ratio, (dw, dh))"""
        input_np, ratio, (dw, dh) = self._preprocess(image)
        nbytes = input_np.nbytes

        # ① numpy → host memory (H2H)
        ctypes.memmove(self._host_input, input_np.ctypes.data, nbytes)

        # ② host → device (H2D)
        ret = _get_ret(acl.rt.memcpy(self._dev_input, self._input_size,
                                      self._host_input, nbytes,
                                      ACL_MEMCPY_HOST_TO_DEVICE))
        assert ret == 0, f"memcpy H2D: {ret}"

        # ③ execute (CANN 7.0: 3 args, no stream)
        ret = _get_ret(acl.mdl.execute(self._model_id,
                                        self._input_ds,
                                        self._output_ds))
        assert ret == 0, f"execute: {ret}"

        # ④ sync device (streamless execute)
        ret = _get_ret(acl.rt.synchronize_device())
        assert ret == 0, f"sync: {ret}"

        # ⑤ device → host (D2H)
        ret = _get_ret(acl.rt.memcpy(self._host_output, self._output_size,
                                      self._dev_output, self._output_size,
                                      ACL_MEMCPY_DEVICE_TO_HOST))
        assert ret == 0, f"memcpy D2H: {ret}"

        # ⑥ host → numpy — output is FP16 (ATC --output_type=FP16)
        output = np.ctypeslib.as_array(
            ctypes.cast(self._host_output, ctypes.POINTER(ctypes.c_uint16)),
            shape=(self._output_size // 2,))
        # FP16 → FP32
        return output.view(np.float16).astype(np.float32), ratio, (dw, dh)

    def close(self):
        if getattr(self, '_closed', False):
            return
        self._closed = True
        try:
            if hasattr(self, '_input_ds') and self._input_ds is not None:
                acl.mdl.destroy_dataset(self._input_ds)
            if hasattr(self, '_output_ds') and self._output_ds is not None:
                acl.mdl.destroy_dataset(self._output_ds)
            if hasattr(self, '_dev_input') and self._dev_input:
                acl.rt.free(self._dev_input)
            if hasattr(self, '_dev_output') and self._dev_output:
                acl.rt.free(self._dev_output)
            if hasattr(self, '_host_input') and self._host_input:
                acl.rt.free_host(self._host_input)
            if hasattr(self, '_host_output') and self._host_output:
                acl.rt.free_host(self._host_output)
            if hasattr(self, '_model_id') and self._model_id is not None:
                acl.mdl.unload(self._model_id)
            if hasattr(self, '_model_desc') and self._model_desc is not None:
                acl.mdl.destroy_desc(self._model_desc)
            if hasattr(self, '_stream') and self._stream is not None:
                acl.rt.destroy_stream(self._stream)
        except:
            pass
        try:
            acl.rt.reset_device(self.device_id)
            acl.finalize()
        except:
            pass


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


def decode_npu_output(raw_output, img_w, img_h, ratio=1.0, dw=0, dh=0,
                      num_classes=8, conf_thres=0.5, iou_thres=0.45):
    """
    解码 NPU 输出 → (N, 6): [x1,y1,x2,y2,conf,cls_id]

    ratio, dw, dh: letterbox 参数，用于还原到原始图像坐标
    """
    total = len(raw_output)
    nc = num_classes
    expected = 4 + nc

    if total % expected != 0:
        print(f"[NPUDecode] 输出元素 {total} 不匹配 4+{nc}={expected}")
        return np.empty((0, 6))

    num_anchors = total // expected
    output = raw_output.reshape(1, expected, num_anchors)[0]

    boxes_raw = output[:4, :]
    scores_raw = output[4:, :]

    max_scores = scores_raw.max(axis=0)
    max_cls = scores_raw.argmax(axis=0)

    mask = max_scores > conf_thres
    if not mask.any():
        return np.empty((0, 6))

    # 坐标映射：letterbox空间(640) → 原始图像空间
    cx = boxes_raw[0, mask]
    cy = boxes_raw[1, mask]
    bw = boxes_raw[2, mask]
    bh = boxes_raw[3, mask]

    # 先去归一化到 letterbox 像素
    x1 = (cx - bw / 2) * 640
    y1 = (cy - bh / 2) * 640
    x2 = (cx + bw / 2) * 640
    y2 = (cy + bh / 2) * 640

    # 还原 letterbox → 原始图像坐标
    x1 = (x1 - dw) / ratio
    y1 = (y1 - dh) / ratio
    x2 = (x2 - dw) / ratio
    y2 = (y2 - dh) / ratio

    boxes = np.stack([x1, y1, x2, y2], axis=1)
    scores = max_scores[mask]
    classes = max_cls[mask]

    keep = nms_fast(boxes, scores, iou_thres)
    if len(keep) == 0:
        return np.empty((0, 6))

    return np.column_stack([boxes[keep], scores[keep, None], classes[keep, None]])


# ── Test ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import time

    om_file = 'best4.om'
    assert os.path.exists(om_file), f"{om_file} not found"

    print(f"[Test] Loading {om_file} ...")
    d = NPUDetector(om_file)

    dummy = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    times = []
    for i in range(50):
        t0 = time.perf_counter()
        raw, r, (dw, dh) = d.infer(dummy)
        t = (time.perf_counter() - t0) * 1000
        if i >= 5:
            times.append(t)

    print(f"[Test] Inference: {np.mean(times):.1f}ms "
          f"(min={np.min(times):.1f}, max={np.max(times):.1f})")
    d.close()
    print("[Test] Done")

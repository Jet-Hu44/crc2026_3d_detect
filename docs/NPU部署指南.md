# 香橙派 AI Pro — NPU 部署指南

> 基于 OrangePi AI Pro 昇腾用户手册 v1.2（207页）关键章节提取
> 目标：将 YOLO PyTorch 模型转换为 Ascend NPU 可运行的 .om 格式

---

## 一、硬件确认

```bash
# 确认NPU存在
npu-smi info

# 查看NPU算力分配
npu-smi info -t cpu-num-cfg -i 0 -c 0
# 输出: Current AI CPU number : 1
#       Current control CPU number : 3
```

**规格**：
- INT8 算力：8 TOPS
- FP16 算力：4 TFLOPS
- 4核 ARM CPU + AI 处理器
- 预装 MindSpore 2.x + CANN 7.0

---

## 二、验证 NPU 环境

### 2.1 测试 MindSpore

```bash
# 先确保有 16GB swap（MindSpore 需要）
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# 测试
python -c "import mindspore;mindspore.set_context(device_target='Ascend');mindspore.run_check()"

# 期望输出:
# MindSpore version: 2.x.xx
# The result of multiplication calculation is correct,
# MindSpore has been installed on platform [Ascend] successfully!
```

### 2.2 确认 CANN 环境变量

```bash
# 检查是否存在
ls /usr/local/Ascend/ascend-toolkit/

# 如存在，source 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 检查 ATC 工具
atc --help
# 如果有输出说明 ATC 可用
```

### 2.3 验证硬件编解码

```bash
ffmpeg -hwaccels | grep ascend   # 应输出: ascend
```

---

## 三、模型转换流程

### 3.1 PyTorch → ONNX

在开发机或香橙派上：

```bash
cd /home/HwHiAiUser/ultralytics-main
python3 -c "
from ultralytics import YOLO
model = YOLO('best4.pt')
model.export(format='onnx', imgsz=640, simplify=True)
"
# 输出: best4.onnx
```

### 3.2 ONNX → Ascend OM（需要 ATC 工具）

```bash
# 在香橙派上执行（需先 source CANN 环境）
atc --model=best4.onnx \
    --framework=5 \              # 5=ONNX
    --output=yolo11n_crc2026 \   # 输出文件名
    --soc_version=Ascend310B4 \ # 橙派AI Pro 的芯片型号
    --input_shape="images:1,3,640,640" \
    --input_format=NCHW
```

> ⚠️ 如果 ATC 报算子不支持错误（常见），需要联系华为Ascend社区获取支持的算子列表，或者用 MindSpore Lite 替代 ATC 做转换。

### 3.3 备选路径：MindSpore Lite 转换

```bash
# ONNX → MindIR
python3 -c "
import mindspore_lite as mslite
converter = mslite.Converter()
converter.optimize = 'ascend_oriented'
converter.convert_fmk_type = mslite.FmkType.ONNX
converter.save_type = mslite.ModelType.MINDIR
converter.convert('best4.onnx', 'best4.mindir')
"

# MindIR → 部署推理
# 参考: https://www.mindspore.cn/lite
```

---

## 四、NPU 推理集成（规划）

转换成功后，在 `detector.py` 中添加 NPU 推理路径：

```python
# detector.py — YoloOrbbecDetector 中增加

class YoloOrbbecDetector:
    def __init__(self, weights='best4.pt', device='0', use_npu=False):
        self.use_npu = use_npu
        if use_npu:
            self._init_npu(weights)
        else:
            self._init_cpu(weights)

    def _init_npu(self, om_path):
        """NPU推理初始化"""
        import ascendcl
        self.npu_model = ascendcl.Model(om_path)
        print(f"[NPU] Ascend模型已加载: {om_path}")

    def inference_npu(self, image):
        """NPU推理 — 预期推理速度: 15-20ms (vs CPU 120ms)"""
        # 预处理: resize 640×640, normalize
        input_tensor = self._preprocess(image)
        # NPU推理
        output = self.npu_model.infer(input_tensor)
        # 后处理: 解析检测框
        return self._postprocess(output)
```

---

## 五、NPU 优化建议

### 5.1 INT8 量化

INT8 量化可将模型从 6MB 压缩至 ~1.5MB，推理速度提升 3-5 倍：

```bash
# 使用 ATC 做量化（需要校准数据集）
atc --model=best4.onnx \
    --framework=5 \
    --output=yolo11n_int8 \
    --soc_version=Ascend310B4 \
    --input_shape="images:1,3,640,640" \
    --precision_mode=allow_fp32_to_int8  # 自动INT8量化
```

### 5.2 模型加载时间优化

2025 新规：模型加载时间计入识别分。优化方法：

```bash
# 思路: 系统启动时预加载模型到共享内存
# 方案A: systemd 服务在开机时加载模型
# 方案B: start.sh 中提前 warmup（在裁判遮镜头时加载）

# 在 识别.sh 中加:
python3 -c "from detector import YoloOrbbecDetector; d=YoloOrbbecDetector()" &
# 后台预加载，GUI启动时模型已在内存
```

---

## 六、环境问题排查

| 问题 | 检查命令 | 解决方案 |
|------|----------|----------|
| ATC 找不到 | `which atc` | `source /usr/local/Ascend/ascend-toolkit/set_env.sh` |
| MindSpore 报错 | `python -c "import mindspore"` | `pip install --upgrade mindspore` |
| NPU 内存不足 | `npu-smi info` | 减小 batch_size 或模型尺寸 |
| 算子不支持 | ATC 报错日志 | 联系华为 Ascend 社区，或回退 CPU 推理 |
| Swap 不足 | `free -h` | `sudo fallocate -l 16G /swapfile` |

---

## 七、参考资源

- **橙派资料下载**: http://www.orangepi.cn/html/hardWare/computerAndMicrocontrollers/service-and-support/Orange-Pi-AIpro.html
- **华为 Ascend 社区**: https://www.hiascend.com/
- **MindSpore 文档**: https://www.mindspore.cn/
- **CANN 文档**: https://www.hiascend.com/document
- **ATC 工具指南**: https://www.hiascend.com/document/detail/zh/canncommercial/60RC1/devtools/atc/atc_0001.html

---

## 八、当前状态

| 步骤 | 状态 | 说明 |
|------|------|------|
| NPU 环境确认 | ✅ | MindSpore + CANN 已预装 |
| ONNX 导出 | ⚠️ 待训练后 | train.py 自动导出 |
| ATC 转换 | ⚠️ 待 ONNX | 需验证算子兼容性 |
| INT8 量化 | 📋 远期 | 先确保 FP16 能跑 |
| NPU 推理集成 | 📋 远期 | detector.py 预留了 use_npu 接口 |

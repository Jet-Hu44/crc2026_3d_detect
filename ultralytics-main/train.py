"""
3D识别 — YOLO模型训练脚本
============================
用法: python3 train.py
训练前: 1) 照片放入 dataset/images/train/ + dataset/images/val/
        2) 标签放入 dataset/labels/train/ + dataset/labels/val/
        3) 确认 crc2026_18class.yaml 中 nc 和 names 正确

模型选择: 改 YOLO() 里的 yaml 名字即可切换
  - yolo11n.yaml  轻量(2.6M参数, ARM CPU ~120ms)
  - yolo11s.yaml  均衡(9.4M参数, ARM CPU ~250ms, 精度更高)
  - yolov8n.yaml  经典(3.2M参数)
"""

import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # ============================================
    # 模型选择 — 改这里切换
    # ============================================
    model = YOLO('yolo11s.yaml')  # 推荐: 精度优先, 后续可蒸馏到n版本

    # 加载预训练权重（COCO预训练, 已含常见物体知识, 加速收敛）
    model.load('yolo11s.pt')

    # ============================================
    # 训练参数
    # ============================================
    model.train(
        data='crc2026_18class.yaml',

        # --- 图像 ---
        imgsz=640,           # 输入分辨率（比赛距离1.0-1.8m, 640足够）
        rect=False,          # 矩形训练（方形输入更通用）

        # --- 训练轮次 ---
        epochs=100,          # 总轮次（18类小数据集, 100轮通常收敛）
        patience=20,         # 早停: 20轮无提升则停止

        # --- 批次 ---
        batch=16,            # 批次大小（香橙派16GB内存可用16-32）
        workers=0,           # 数据加载进程数（ARM上设0避免卡死）

        # --- 优化器 ---
        optimizer='AdamW',   # AdamW比SGD收敛更快, 适合小数据集
        lr0=0.001,           # 初始学习率
        lrf=0.01,            # 最终lr = lr0 * lrf = 1e-5
        momentum=0.937,
        weight_decay=0.0005,

        # --- 数据增强（比赛关键） ---
        hsv_h=0.015,         # HSV色相抖动（模拟不同光源颜色）
        hsv_s=0.7,           # HSV饱和度抖动
        hsv_v=0.4,           # HSV明度抖动（模拟光照强弱变化）
        degrees=30.0,        # 随机旋转±30°（模拟物品不同摆放姿态）
        translate=0.1,       # 随机平移±10%
        scale=0.5,           # 随机缩放（模拟0.7-1.8m距离变化）
        shear=0.0,           # 不剪切（物品不会变形）
        perspective=0.0,     # 不透视变换
        flipud=0.0,          # 不上下翻转
        fliplr=0.5,          # 50%概率左右翻转
        mosaic=0.5,          # 50%概率Mosaic增强（多图拼接）
        mixup=0.1,           # 10%概率MixUp增强（图像混合）
        copy_paste=0.0,      # 不复制粘贴

        # --- 训练策略 ---
        single_cls=False,     # 多类别检测
        close_mosaic=10,      # 最后10轮关闭Mosaic（提升精度）
        cos_lr=True,          # 余弦学习率衰减
        amp=True,             # 自动混合精度（ARM上加速训练）

        # --- 设备 ---
        device='0',           # 训练设备

        # --- 保存 ---
        project='runs/crc2026',
        name='train',
        exist_ok=True,        # 覆盖同名目录
        save=True,
        save_period=10,       # 每10轮保存一次

        # --- 验证 ---
        val=True,             # 每轮结束后验证
        plots=True,           # 生成训练曲线图
    )

    # ============================================
    # 训练完成后导出最佳模型
    # ============================================
    print("\n===== 训练完成 =====")
    best_pt = 'runs/crc2026/train/weights/best.pt'
    print(f"最佳模型: {best_pt}")

    # 导出ONNX（为NPU转换做准备）
    print("导出ONNX...")
    try:
        best_model = YOLO(best_pt)
        best_model.export(format='onnx', imgsz=640)
        print("ONNX导出完成 → runs/crc2026/train/weights/best.onnx")
    except Exception as e:
        print(f"ONNX导出失败: {e}")

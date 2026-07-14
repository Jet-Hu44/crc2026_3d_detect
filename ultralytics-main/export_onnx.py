"""
ONNX 模型导出脚本 — 为 ATC 转换准备
用法: python3 export_onnx.py [--weights best4.pt]
"""
import argparse
import os
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='导出 YOLO 模型到 ONNX')
    parser.add_argument('--weights', type=str, default='best4.pt')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--opset', type=int, default=11)
    args = parser.parse_args()

    if not os.path.exists(args.weights):
        print(f"✗ 模型文件不存在: {args.weights}")
        sys.exit(1)

    print(f"加载模型: {args.weights} ...")
    from ultralytics import YOLO
    model = YOLO(args.weights)
    print(f"类别数: {len(model.names)}, 类别: {model.names}")

    # 导出 ONNX（不带 NMS，用标准算子）
    onnx_path = args.weights.replace('.pt', '.onnx')
    print(f"导出 ONNX → {onnx_path} ...")
    model.export(
        format='onnx',
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=True,   # 简化算子图
        nms=False,       # 不带 NMS，ATC 兼容性更好
        half=False,      # FP32（ATC 转 FP16 更稳定）
        device='cpu',
    )
    print(f"✓ 导出完成: {onnx_path}")

    # 验证
    import onnx
    onnx.checker.check_model(onnx_path)
    print(f"✓ ONNX 模型验证通过")

    size_mb = os.path.getsize(onnx_path) / 1024 / 1024
    print(f"  文件大小: {size_mb:.1f} MB")
    print(f"\n下一步: 运行 ATC 转换 → atc --model={onnx_path} ...")

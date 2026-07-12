"""
3D识别比赛主入口 — 中国机器人大赛 机器人先进视觉赛项
========================================================
硬件: OrangePi AI Pro (8T NPU) + ORBBEC Astra Pro Plus
用法: python3 huanshibest.py --round 1|2 [--weights best4.pt] [--device 0]
      或直接运行 识别.sh
========================================================
"""

import sys
import argparse
from PyQt5.QtWidgets import QApplication

from config import RoundConfig, DEFAULT_WEIGHTS
from detector import YoloOrbbecDetector
from gui import MainWindow


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='3D识别比赛程序')
    parser.add_argument('--round', type=int, choices=[1, 2], default=2,
                        help='比赛轮次: 1=单桌台无光源, 2=三桌台旋转+特定光源')
    parser.add_argument('--weights', type=str, default=DEFAULT_WEIGHTS,
                        help='模型权重文件路径')
    parser.add_argument('--device', type=str, default='0',
                        help='推理设备: 0=CPU')
    args = parser.parse_args()

    round_config = RoundConfig(round_num=args.round)
    print(f"===== 3D识别比赛程序 =====")
    print(f"轮次: Round {args.round}  |  桌台数: {round_config.num_tables}")
    print(f"旋转: {'是' if round_config.rotate else '否'}  |  "
          f"特定光源: {'是' if round_config.has_specific_light else '否'}")
    print(f"模型: {args.weights}  |  设备: {args.device}")

    detector = YoloOrbbecDetector(weights=args.weights, device=args.device)
    app = QApplication(sys.argv)
    main_window = MainWindow(detector, round_config=round_config)
    main_window.show()
    sys.exit(app.exec_())

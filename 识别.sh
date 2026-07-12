#!/bin/bash
# 3D识别比赛启动脚本
# 用法: ./识别.sh [轮次]   例如: ./识别.sh 1 (Round 1)   ./识别.sh 2 (Round 2, 默认)
ROUND=${1:-2}
sudo ifconfig eth0 192.168.1.67 netmask 255.255.255.0
export PYTHONPATH=$PYTHONPATH:/home/HwHiAiUser/ultralytics-main/install/lib/
source /usr/local/miniconda3/etc/profile.d/conda.sh
conda deactivate
cd /home/HwHiAiUser/ultralytics-main &&
python3 huanshibest.py --round $ROUND


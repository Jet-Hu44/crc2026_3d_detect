#!/bin/bash
# =============================================================================
# 3D识别比赛启动脚本 — 中国机器人大赛 机器人先进视觉赛项
# =============================================================================
# 硬件: OrangePi AI Pro (8T NPU) + ORBBEC Astra Pro Plus
# 网络: 香橙派 192.168.1.67  →  裁判盒(Windows PC) 192.168.1.66:6666
#
# 用法:
#   ./识别.sh          # Round 2 (默认，三桌台)
#   ./识别.sh 1        # Round 1 (单桌台)
#   ./识别.sh 2 best6  # Round 2 + 指定模型权重
#   ./识别.sh 1 best4  # Round 1 + 指定模型权重
# =============================================================================
set -e  # 遇到错误立即退出

# ── 解析参数 ──────────────────────────────────────────────────────────────
ROUND=${1:-2}
WEIGHTS=${2:-best4.pt}

echo "=============================================="
echo "  3D识别比赛程序 — 启动中"
echo "  轮次: Round ${ROUND}"
echo "  模型: ${WEIGHTS}"
echo "  目标裁判盒: 192.168.1.66:6666"
echo "=============================================="

# ── 1. 设置静态IP ──────────────────────────────────────────────────────────
echo "[1/5] 配置网络接口 eth0 → 192.168.1.67 ..."
sudo ifconfig eth0 192.168.1.67 netmask 255.255.255.0
echo "      本机IP: $(ifconfig eth0 2>/dev/null | grep 'inet ' | awk '{print $2}')"

# ── 2. 检查裁判盒连通性 ────────────────────────────────────────────────────
echo "[2/5] 检查裁判盒连通性..."
if ping -c 2 -W 2 192.168.1.66 > /dev/null 2>&1; then
    echo "      ✓ 裁判盒 192.168.1.66 可达"
else
    echo "      ⚠ 裁判盒 192.168.1.66 Ping 不通！"
    echo "      请确认:"
    echo "        1. Windows PC 已开机且 IP 为 192.168.1.66"
    echo "        2. 网线已连接交换机/路由器"
    echo "        3. judgeGui.exe 已启动"
    echo ""
    read -p "      是否继续运行？(y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# ── 3. 设置环境变量 ────────────────────────────────────────────────────────
echo "[3/5] 设置 Python 环境..."
export PYTHONPATH=$PYTHONPATH:/home/HwHiAiUser/Desktop/Round/ultralytics-main/install/lib/

# 激活 conda 环境（如果存在）
if [ -f /usr/local/miniconda3/etc/profile.d/conda.sh ]; then
    source /usr/local/miniconda3/etc/profile.d/conda.sh
    conda deactivate 2>/dev/null || true
    echo "      conda 环境已就绪"
else
    echo "      conda 未安装，使用系统 Python"
fi

# ── 4. 切换工作目录 ────────────────────────────────────────────────────────
echo "[4/5] 切换到工作目录..."
cd /home/HwHiAiUser/Desktop/Round/ultralytics-main
echo "      当前目录: $(pwd)"

# 检查模型文件
if [ ! -f "${WEIGHTS}" ]; then
    echo "      ✗ 模型文件 ${WEIGHTS} 不存在！"
    echo "      可用模型: $(ls *.pt 2>/dev/null | tr '\n' ' ')"
    exit 1
fi
echo "      ✓ 模型文件 ${WEIGHTS} 已找到"

# ── 5. 启动识别 ────────────────────────────────────────────────────────────
echo "[5/5] 启动识别程序..."
echo "=============================================="
echo "  程序运行中..."
echo "  GUI 将自动弹出 → 1秒后自动开始检测"
echo "  检测完成后自动发送结果到裁判盒"
echo "  按 Ctrl+C 可手动中止"
echo "=============================================="
echo ""

python3 huanshibest.py --round ${ROUND} --weights ${WEIGHTS}

# ── 完成 ───────────────────────────────────────────────────────────────────
EXIT_CODE=$?
echo ""
echo "=============================================="
if [ ${EXIT_CODE} -eq 0 ]; then
    echo "  识别程序正常退出"
else
    echo "  识别程序异常退出 (exit code: ${EXIT_CODE})"
fi
echo "=============================================="
exit ${EXIT_CODE}

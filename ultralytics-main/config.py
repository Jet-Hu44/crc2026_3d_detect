"""
比赛配置常量 + RoundConfig 类
"""

class RoundConfig:
    """两轮比赛配置"""
    def __init__(self, round_num=2):
        self.round_num = round_num
        if round_num == 1:
            self.num_tables = 1
            self.tables = [1]
            self.rotate = False
            self.has_specific_light = False
            self.detect_time_per_table = 20
        else:
            self.num_tables = 3
            self.tables = [1, 2, 3]
            self.rotate = True
            self.has_specific_light = True
            self.detect_time_per_table = 10


# ========== 网络配置 ==========
JUDGE_HOST = '192.168.1.66'  # Windows PC 运行裁判盒 judgeGui.exe
JUDGE_PORT = 6666
LOCAL_IP = '192.168.1.67'
NETMASK = '255.255.255.0'

# ========== 队伍信息 ==========
TEAM_ID = 'Y2507T1892934'
TEAM_NAME = 'NEEPU-HS'  # 东北电力大学-幻视

# ========== 路径配置（香橙派上） ==========
HOME_DIR = '/home/HwHiAiUser/Desktop/Round/ultralytics-main'
DESKTOP_RESULT_DIR = '/home/HwHiAiUser/Desktop/result_r'
RUNSS_DIR = '/home/HwHiAiUser/Desktop/Round/ultralytics-main/runss'
LABELS_DIR = '/home/HwHiAiUser/Desktop/Round/ultralytics-main/runss/labels'
D_DIR = '/home/HwHiAiUser/Desktop/Round/ultralytics-main/runss/d'
RESULT_DIR = '/home/HwHiAiUser/Desktop/Round/ultralytics-main/runss/result'
VIDEO_DIR = '/home/HwHiAiUser/Desktop/Round/ultralytics-main/runss'

# ========== 检测参数 ==========
CONF_THRES = 0.50
IOU_THRES = 0.45
DEPTH_MIN_MM = 100
DEPTH_MAX_MM = 1600
MIN_OCCURRENCES = 5
TEMPORAL_ALPHA = 0.7

# ========== 模型 ==========
DEFAULT_WEIGHTS = 'best4.pt'

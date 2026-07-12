"""
裁判盒网络通讯模块

协议格式（二进制）:
  [4字节 datatype big-endian][4字节 data_length big-endian][data bytes]

数据类型:
  0 — 队伍ID字符串
  1 — 结果文件
  2 — 控制指令（旋转、开始、结束信号）
"""

import socket
import struct
import time
import os
import shutil
import glob
from config import JUDGE_HOST, JUDGE_PORT, TEAM_ID, DESKTOP_RESULT_DIR, RESULT_DIR


class JudgeBoxClient:
    """裁判盒TCP客户端 — 处理所有与裁判盒的通讯"""

    def __init__(self, host=JUDGE_HOST, port=JUDGE_PORT):
        self.host = host
        self.port = port
        self.socket = None

    def connect(self):
        """建立TCP连接"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.socket.connect((self.host, self.port))
            print(f"[网络] 已连接裁判盒 {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"[网络] 连接裁判盒失败: {e}")
            return False

    def _send_message(self, datatype, data_bytes):
        """发送二进制消息（底层方法）"""
        if self.socket is None:
            print("[网络] 未连接，跳过发送")
            return
        header = struct.pack('>II', datatype, len(data_bytes))
        self.socket.sendall(header)
        self.socket.sendall(data_bytes)

    def _send_string(self, datatype, text):
        """发送字符串消息"""
        self._send_message(datatype, text.encode())

    def send_team_id(self):
        """发送队伍ID — 类型0"""
        print(f"[网络] 发送队伍ID: {TEAM_ID}")
        self._send_string(0, TEAM_ID)

    def send_signal_start(self):
        """发送开始识别信号 — 类型2, 内容START

        规则要求: 程序启动后立即发送，裁判盒以此开始计时。
        若超3秒未收到，裁判将手动计时并扣10%分数。
        """
        print("[网络] 发送 START 识别开始信号")
        self._send_string(2, "START")

    def send_signal_end(self):
        """发送识别结束信号 — 类型2, 内容END

        规则要求: 识别完成后发送，裁判盒以此停止计时。
        """
        print("[网络] 发送 END 识别结束信号")
        self._send_string(2, "END")

    def send_rotate_command(self, table_id):
        """发送相机旋转指令 — 类型2, 内容ROTATE:{table_id}

        裁判盒收到后控制电机旋转相机到指定桌台位置。
        协议格式待QQ群确认后调整。
        """
        msg = f"ROTATE:{table_id}"
        print(f"[网络] 发送旋转指令: {msg}")
        self._send_string(2, msg)
        time.sleep(3)  # 等待旋转完成（保守值，按实际转速调整）

    def send_result_file(self, file_path):
        """发送结果文件 — 类型1

        等待文件生成后读取并发送。
        """
        while not os.path.exists(file_path):
            print(f"[网络] 等待文件: {file_path}")
            time.sleep(0.1)
        with open(file_path, 'rb') as f:
            data = f.read()
        self._send_message(1, data)
        print(f"[网络] 结果文件已发送: {file_path} ({len(data)} bytes)")

    def close(self):
        """关闭连接"""
        if self.socket:
            try:
                self.socket.close()
                print("[网络] 连接已关闭")
            except:
                pass
            self.socket = None

    # ========== 文件操作辅助（从MainWindow移出） ==========

    @staticmethod
    def delete_all_files_in_folder(folder):
        if not os.path.exists(folder):
            return
        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            try:
                os.remove(fpath)
            except:
                pass

    @staticmethod
    def move_txt_files(src, dest):
        if not os.path.exists(dest):
            os.makedirs(dest)
        for fname in os.listdir(src):
            if fname.endswith('.txt'):
                shutil.move(os.path.join(src, fname), os.path.join(dest, fname))

    @staticmethod
    def copy_txt_files(src, dest):
        if not os.path.exists(dest):
            os.makedirs(dest)
        for fpath in glob.glob(os.path.join(src, '*.txt')):
            shutil.copy(fpath, os.path.join(dest, os.path.basename(fpath)))

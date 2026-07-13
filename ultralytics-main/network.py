"""
裁判盒网络通讯模块 (judgeGui_240328 协议)

协议格式（二进制，大端）:
  [4字节 DataType][4字节 DataLength][Data bytes]

数据类型（与 judgeGui_240328 一致）:
  0 — 队伍ID + 开始计时（裁判盒收到后立即开始计时）
  1 — 结果文件 + 停止计时（裁判盒收到后停止计时并计算得分）
  2 — 结果文件（工业测量，得分始终为0，3D识别不使用）
  3 — 转台旋转信号（Data="0000"）
"""

import socket
import struct
import time
import os
import shutil
import glob
from config import JUDGE_HOST, JUDGE_PORT, TEAM_ID, DESKTOP_RESULT_DIR, RESULT_DIR


class JudgeBoxClient:
    """裁判盒 TCP 客户端 — 处理所有与裁判盒的通讯"""

    def __init__(self, host=JUDGE_HOST, port=JUDGE_PORT):
        self.host = host
        self.port = port
        self.socket = None
        self._connected = False

    # ==================== 连接管理 ====================

    def connect(self):
        """建立 TCP 连接"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(5)
        try:
            self.socket.connect((self.host, self.port))
            self._connected = True
            print(f"[网络] 已连接裁判盒 {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"[网络] 连接裁判盒失败: {e}")
            self._connected = False
            return False

    def close(self):
        """关闭连接"""
        if self.socket:
            try:
                self.socket.close()
                print("[网络] 连接已关闭")
            except:
                pass
            self.socket = None
            self._connected = False

    @property
    def is_connected(self):
        return self._connected and self.socket is not None

    # ==================== 底层发送 ====================

    def _send_message(self, datatype, data_bytes):
        """发送二进制消息（大端 int32 header + data）"""
        if not self.is_connected:
            print(f"[网络] 未连接，跳过 DataType={datatype}")
            return False
        try:
            header = struct.pack('>II', datatype, len(data_bytes))
            self.socket.sendall(header)
            self.socket.sendall(data_bytes)
            return True
        except Exception as e:
            print(f"[网络] 发送 DataType={datatype} 失败: {e}")
            self._connected = False
            return False

    def _send_string(self, datatype, text):
        """发送字符串消息"""
        return self._send_message(datatype, text.encode())

    # ==================== 比赛信号（与 judgeGui_240328 协议一一对应） ====================

    def send_start(self):
        """DataType 0 — 发送队伍ID，裁判盒开始计时

        规则: 此信号一经发出，裁判盒立即开始计时。
              若3秒内未发出，裁判手动计时并扣10%分数。
        """
        print(f"[网络] >>> DataType 0: 开始计时 (TeamID={TEAM_ID})")
        return self._send_string(0, TEAM_ID)

    def send_result_and_stop(self, file_path):
        """DataType 1 — 发送结果文件，裁判盒停止计时

        阻塞等待文件生成 → 读取 → 发送 → 裁判盒停止计时并计分。
        """
        print(f"[网络] >>> DataType 1: 发送结果文件 + 停止计时")

        # 等待文件生成
        waited = 0
        while not os.path.exists(file_path):
            if waited % 10 == 0:
                print(f"[网络]   等待结果文件 ({waited}s)...")
            time.sleep(0.1)
            waited += 0.1

        with open(file_path, 'rb') as f:
            data = f.read()

        ok = self._send_message(1, data)
        if ok:
            print(f"[网络]   已发送 {file_path} ({len(data)} bytes) → 计时停止")
        return ok

    def send_rotate(self, table_id=None):
        """DataType 3 — 转台旋转信号

        通知裁判盒相机将旋转到下一张桌台。
        Data 固定为 "0000"（裁判盒不实际控制旋转，仅记录时序）。
        实际的旋转动作由 send_rotate_command() 直接发串口指令给电机。
        """
        print(f"[网络] >>> DataType 3: 转台旋转信号 (table={table_id or 'next'})")
        return self._send_string(3, "0000")

    def send_industrial_result(self, file_path):
        """DataType 2 — 发送工业测量结果（3D识别不使用）"""
        print(f"[网络] >>> DataType 2: 工业测量结果")
        with open(file_path, 'rb') as f:
            data = f.read()
        return self._send_message(2, data)

    # ==================== 兼容旧接口（不推荐，新代码用上面3个） ====================

    def send_team_id(self):
        """[已废弃] 等价于 send_start()"""
        return self.send_start()

    def send_signal_start(self):
        """[已废弃] 等价于 send_start() — DataType 0 本身就含开始计时"""
        return self.send_start()

    def send_signal_end(self):
        """[已废弃] 发送结束信号 — 现在由 send_result_and_stop() 自动完成"""
        print("[网络] ⚠️ send_signal_end 已废弃，请用 send_result_and_stop()")

    def send_rotate_command(self, table_id):
        """[已废弃] 等效 send_rotate(table_id)"""
        return self.send_rotate(table_id)

    def send_result_file(self, file_path):
        """[已废弃] 等价于 send_result_and_stop(file_path)"""
        return self.send_result_and_stop(file_path)

    # ==================== 文件操作辅助 ====================

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
            os.makedirs(dest, exist_ok=True)
        for fname in os.listdir(src):
            if fname.endswith('.txt'):
                shutil.move(os.path.join(src, fname), os.path.join(dest, fname))

    @staticmethod
    def copy_txt_files(src, dest):
        if not os.path.exists(dest):
            os.makedirs(dest, exist_ok=True)
        for fpath in glob.glob(os.path.join(src, '*.txt')):
            shutil.copy(fpath, os.path.join(dest, os.path.basename(fpath)))

    # ==================== 调试辅助 ====================

    def test_connection(self):
        """验证与裁判盒的连通性（ping 检查 + socket 尝试）"""
        import subprocess
        print(f"[网络] 测试连接 {self.host}:{self.port} ...")

        # Ping 检查
        try:
            subprocess.run(['ping', '-c', '2', '-W', '1', self.host],
                          capture_output=True, timeout=3)
            print(f"[网络] Ping {self.host} 可达")
        except:
            print(f"[网络] ⚠️ Ping {self.host} 不可达，检查网线和IP配置")
            return False

        # Socket 检查
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        try:
            sock.connect((self.host, self.port))
            sock.close()
            print(f"[网络] 端口 {self.port} 已开放")
            return True
        except:
            print(f"[网络] ⚠️ 端口 {self.port} 未开放，裁判盒是否已启动？")
            return False

    def send_full_sequence(self, result_file_path, tables=None):
        """一键完整时序: 开始→(旋转→检测→旋转)×N→结束

        用于 Round 2 多桌台场景。
        tables: 桌台编号列表，如 [1, 2, 3]，Round 1 时用 [1]
        """
        if tables is None:
            tables = [1]

        # 1. 开始计时
        if not self.send_start():
            return False
        time.sleep(0.5)

        # 2. 每张桌台检测 → 旋转
        for i, table in enumerate(tables):
            print(f"\n[网络] === 桌台 {table} 检测中 ({i+1}/{len(tables)}) ===")

            if i < len(tables) - 1:
                # 非最后一张桌台：发送旋转信号
                self.send_rotate(table + 1)
                time.sleep(3)  # 等待旋转完成

        # 3. 最后一张桌台完成后发送结果 + 停止计时
        return self.send_result_and_stop(result_file_path)

@echo off
chcp 65001 >nul
title 裁判盒 judgeGui — 3D识别评分

echo ==============================================
echo   裁判盒评分程序 — 启动中
echo   中国机器人大赛 机器人先进视觉赛项
echo ==============================================
echo.
echo   当前 Windows PC 网络配置:
echo.

:: 显示网络配置
ipconfig | findstr /C:"IPv4" /C:"子网掩码"

echo.
echo   预期配置:
echo     IP 地址:   192.168.1.66
echo     子网掩码:   255.255.255.0
echo     监听端口:   6666
echo.
echo   香橙派设备:   192.168.1.67
echo ==============================================
echo.

:: 检查 IP 是否正确
ipconfig | findstr "192.168.1.66" >nul
if %errorlevel% neq 0 (
    echo   [警告] 未检测到 IP 192.168.1.66！
    echo   请确认网络配置:
    echo     控制面板 → 网络和共享中心 → 以太网 → 属性
    echo     → Internet 协议版本 4 (TCP/IPv4) → 属性
    echo     → IP地址: 192.168.1.66
    echo     → 子网掩码: 255.255.255.0
    echo.
    choice /C YN /M "是否仍要启动裁判盒？"
    if errorlevel 2 exit /b 1
)

echo   [OK] 网络配置检查通过
echo.

:: 检查必要文件
if not exist "judgeGui.exe" (
    echo   [错误] judgeGui.exe 未找到！
    echo   请确认此 bat 文件放在 judgeGui_240328 目录下运行
    pause
    exit /b 1
)

:: 检查 Excel 模板
if not exist "Template_excel\Distinguish_real.xlsx" (
    echo   [警告] Excel 评分模板缺失，计分可能出错
)

echo   [启动] 正在启动裁判盒...
echo   ==============================================
echo.
echo   重要提醒:
echo   1. 裁判盒启动后会监听 TCP 端口 6666
echo   2. 等待香橙派连接后开始计时
echo   3. 比赛流程由香橙派端 识别.sh 控制
echo   4. 关闭此窗口即可停止裁判盒
echo.
echo   ==============================================
echo.

start "" "judgeGui.exe"

echo   裁判盒已启动！
echo   等待香橙派 (192.168.1.67) 连接...
echo.
echo   按任意键关闭此窗口（不影响裁判盒运行）...
pause >nul

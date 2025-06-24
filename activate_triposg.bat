@echo off
echo ========================================
echo 激活TripoSG conda环境
echo ========================================

REM 设置conda路径
set CONDA_PATH=C:\Users\david\miniconda3\Scripts\conda.exe

REM 检查conda是否存在
if not exist "%CONDA_PATH%" (
    echo 错误: 找不到conda可执行文件
    echo 路径: %CONDA_PATH%
    pause
    exit /b 1
)

REM 激活环境
echo 正在激活TripoSG环境...
call "%CONDA_PATH%" activate TripoSG

REM 显示环境信息
echo.
echo 环境信息:
python -c "import torch; print('PyTorch版本:', torch.__version__); print('CUDA可用:', torch.cuda.is_available()); print('GPU设备:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '无')"

echo.
echo ========================================
echo TripoSG环境已激活！
echo 您现在可以运行TripoSG相关的Python脚本了
echo ========================================
echo.

REM 保持命令行窗口打开
cmd /k

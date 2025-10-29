@echo off
setlocal enabledelayedexpansion

:: 获取当前批处理文件所在目录的父目录作为 PYTHONPATH
set "SCRIPT_DIR=%~dp0"
for %%I in ("%SCRIPT_DIR%\..") do set "PARENT_DIR=%%~fI"
set "PYTHONPATH=%PARENT_DIR%"
echo PYTHONPATH set to: %PYTHONPATH%

:: 设置模型和输入路径
set "model_dir=pretrained_models/SoulX-Podcast-1.7B"
set "input_file=example/podcast_script/script_mandarin.json"

:: 运行 Python 脚本
python cli/podcast.py ^
    --json_path "%input_file%" ^
    --model_path "%model_dir%" ^
    --output_path outputs/mandarin.wav ^
    --seed 7

endlocal
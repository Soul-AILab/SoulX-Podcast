#!/bin/bash
# =============================
# 修复 VLLM 安装（从当前状态继续）
# =============================

set -e

# 初始化 conda 环境
source /root/anaconda3/etc/profile.d/conda.sh

# 激活环境
conda activate soulxpodcast

# 进入项目目录（确保不在临时目录中）
cd /ygq/rag/workspace/my-Soul-Podcast/SoulX-Podcast

# 临时目录路径（根据日志）
TEMP_DIR="/tmp/tmp.aXt4nU03Ph"
SOURCE_DIR="$TEMP_DIR/vllm"

# 获取系统安装的 vllm 路径（在项目目录中执行，避免导入临时目录的 vllm）
echo "查找系统安装的 vllm 路径..."
# 使用 python -c 但确保 PYTHONPATH 不包含临时目录
VLLM_PATH=$(PYTHONPATH="" python3 -c "import sys; sys.path = [p for p in sys.path if 'tmp' not in p]; import vllm; import os; print(os.path.dirname(vllm.__file__))" 2>/dev/null)

# 如果上面的方法不行，尝试直接查找 conda 环境的 site-packages
if [ -z "$VLLM_PATH" ] || [[ "$VLLM_PATH" == *"tmp"* ]]; then
    echo "尝试从 conda 环境查找 vllm..."
    CONDA_ENV_PATH=$(conda info --envs | grep soulxpodcast | awk '{print $NF}')
    if [ -n "$CONDA_ENV_PATH" ]; then
        VLLM_PATH="$CONDA_ENV_PATH/lib/python*/site-packages/vllm"
        VLLM_PATH=$(python3 -c "import glob; paths = glob.glob('$VLLM_PATH'); print(paths[0] if paths else '')" 2>/dev/null)
    fi
fi

# 如果还是找不到，使用 pip show 来查找
if [ -z "$VLLM_PATH" ] || [[ "$VLLM_PATH" == *"tmp"* ]]; then
    echo "使用 pip show 查找 vllm 安装位置..."
    VLLM_LOCATION=$(pip show vllm | grep Location | awk '{print $2}')
    if [ -n "$VLLM_LOCATION" ]; then
        VLLM_PATH="$VLLM_LOCATION/vllm"
    fi
fi

if [ -z "$VLLM_PATH" ] || [[ "$VLLM_PATH" == *"tmp"* ]] || [ ! -d "$VLLM_PATH" ]; then
    echo "错误: 无法找到系统安装的 vllm 路径"
    echo "请手动运行: python3 -c 'import vllm; import os; print(os.path.dirname(vllm.__file__))'"
    exit 1
fi

echo "找到系统 VLLM 路径: $VLLM_PATH"

# 检查源文件是否存在
if [ ! -f "${SOURCE_DIR}/vllm/model_executor/layers/sampler.py" ]; then
    echo "错误: 源文件不存在: ${SOURCE_DIR}/vllm/model_executor/layers/sampler.py"
    exit 1
fi

# 替换修改版的文件
echo "替换修改版文件..."
cp "${SOURCE_DIR}/vllm/model_executor/layers/sampler.py" "${VLLM_PATH}/model_executor/layers/sampler.py"
cp "${SOURCE_DIR}/vllm/model_executor/layers/utils.py" "${VLLM_PATH}/model_executor/layers/utils.py"
cp "${SOURCE_DIR}/vllm/model_executor/sampling_metadata.py" "${VLLM_PATH}/model_executor/sampling_metadata.py"
cp "${SOURCE_DIR}/vllm/sampling_params.py" "${VLLM_PATH}/sampling_params.py"

echo "✓ 文件替换完成"

# 清理临时目录
echo "清理临时文件..."
rm -rf "$TEMP_DIR"

# 验证安装
echo "验证安装..."
cd /ygq/rag/workspace/my-Soul-Podcast/SoulX-Podcast
python3 -c "from vllm import LLM; print('VLLM 安装成功!')" && echo "✓ VLLM 安装完成并验证通过" || echo "✗ VLLM 验证失败"

echo ""
echo "安装完成! 现在可以使用 --llm_engine vllm 参数了。"

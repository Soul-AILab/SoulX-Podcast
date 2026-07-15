#!/bin/bash
# =============================
# 安装 SoulX-Podcast 修改版 VLLM
# =============================

set -e

echo "开始安装 VLLM (修改版 v0.10.1.1-soulxpodcast)..."

# 初始化 conda 环境
source /root/anaconda3/etc/profile.d/conda.sh

# 激活环境
conda activate soulxpodcast

# 进入项目目录
cd /ygq/rag/workspace/my-Soul-Podcast/SoulX-Podcast

# 1. 首先安装基础版本的 vllm 0.10.1
echo "步骤 1: 安装基础 vllm 0.10.1..."
pip install vllm==0.10.1

# 2. 获取已安装的 vllm 路径（在克隆仓库之前，避免路径冲突）
echo "步骤 2: 查找已安装的 vllm 路径..."
VLLM_PATH=$(python3 -c "import vllm; import os; print(os.path.dirname(vllm.__file__))" 2>/dev/null)

if [ -z "$VLLM_PATH" ]; then
    echo "错误: 无法找到已安装的 vllm 路径"
    exit 1
fi

echo "找到系统 VLLM 路径: $VLLM_PATH"

# 3. 克隆修改版的 vllm 仓库到临时目录
echo "步骤 3: 克隆修改版 vllm 仓库..."
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"
git clone https://github.com/Soul-AILab/vllm.git
cd vllm
git checkout v0.10.1.1-soulxpodcast

# 4. 替换修改版的文件（从克隆的仓库复制到系统安装的路径）
echo "步骤 4: 替换修改版文件..."
SOURCE_DIR="$TEMP_DIR/vllm"
cp "${SOURCE_DIR}/vllm/model_executor/layers/sampler.py" "${VLLM_PATH}/model_executor/layers/sampler.py"
cp "${SOURCE_DIR}/vllm/model_executor/layers/utils.py" "${VLLM_PATH}/model_executor/layers/utils.py"
cp "${SOURCE_DIR}/vllm/model_executor/sampling_metadata.py" "${VLLM_PATH}/model_executor/sampling_metadata.py"
cp "${SOURCE_DIR}/vllm/sampling_params.py" "${VLLM_PATH}/sampling_params.py"

# 5. 清理临时目录
echo "步骤 5: 清理临时文件..."
cd /ygq/rag/workspace/my-Soul-Podcast/SoulX-Podcast
rm -rf "$TEMP_DIR"

# 6. 验证安装
echo "步骤 6: 验证安装..."
python3 -c "from vllm import LLM; print('VLLM 安装成功!')" && echo "✓ VLLM 安装完成并验证通过" || echo "✗ VLLM 验证失败"

echo ""
echo "安装完成! 现在可以使用 --llm_engine vllm 参数了。"

#!/bin/bash

################################################################################
# SoulX-Podcast WSL Ubuntu 24.04 自动初始化脚本
#
# 用途: 在全新的WSL Ubuntu 24.04环境中自动安装所有依赖并配置项目
# 使用方法: bash setup_wsl.sh
#
# 作者: Claude
# 日期: 2025-11-01
################################################################################

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 打印标题
print_banner() {
    echo ""
    echo "=================================================================="
    echo "  SoulX-Podcast WSL Ubuntu 24.04 自动初始化脚本"
    echo "=================================================================="
    echo ""
}

# 检查是否在WSL环境中
check_wsl() {
    log_info "检查WSL环境..."
    if grep -qEi "(Microsoft|WSL)" /proc/version &> /dev/null ; then
        log_success "检测到WSL环境"
    else
        log_warning "未检测到WSL环境，但脚本仍可继续运行"
    fi
}

# 检查Ubuntu版本
check_ubuntu_version() {
    log_info "检查Ubuntu版本..."
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        log_info "操作系统: $NAME $VERSION"
        if [[ "$VERSION_ID" != "24.04" ]]; then
            log_warning "检测到Ubuntu版本为 $VERSION_ID，建议使用24.04"
        fi
    fi
}

# 更新系统包
update_system() {
    log_info "更新系统包列表..."
    sudo apt-get update -y
    log_success "系统包列表已更新"
}

# 安装基础工具
install_basic_tools() {
    log_info "安装基础工具..."

    local tools=(
        "curl"
        "wget"
        "git"
        "build-essential"
        "libssl-dev"
        "zlib1g-dev"
        "libbz2-dev"
        "libreadline-dev"
        "libsqlite3-dev"
        "libncursesw5-dev"
        "xz-utils"
        "tk-dev"
        "libxml2-dev"
        "libxmlsec1-dev"
        "libffi-dev"
        "liblzma-dev"
        "ca-certificates"
    )

    for tool in "${tools[@]}"; do
        if ! dpkg -l | grep -q "^ii  $tool"; then
            log_info "安装 $tool..."
            sudo apt-get install -y "$tool"
        else
            log_info "$tool 已安装，跳过"
        fi
    done

    log_success "基础工具安装完成"
}

# 安装Miniconda
install_miniconda() {
    log_info "检查Conda安装..."

    if command -v conda &> /dev/null; then
        log_info "Conda已安装: $(conda --version)"
        return 0
    fi

    log_info "开始安装Miniconda..."

    local MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    local MINICONDA_INSTALLER="/tmp/miniconda_installer.sh"

    # 下载Miniconda
    log_info "下载Miniconda安装包..."
    if [ -f "$MINICONDA_INSTALLER" ]; then
        log_info "安装包已存在，跳过下载"
    else
        wget -q --show-progress "$MINICONDA_URL" -O "$MINICONDA_INSTALLER"
    fi

    # 安装Miniconda
    log_info "安装Miniconda到 $HOME/miniconda3..."
    bash "$MINICONDA_INSTALLER" -b -p "$HOME/miniconda3"

    # 初始化conda
    log_info "初始化Conda..."
    "$HOME/miniconda3/bin/conda" init bash

    # 加载conda
    source "$HOME/.bashrc" || true
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)" || true

    # 清理安装包
    rm -f "$MINICONDA_INSTALLER"

    log_success "Miniconda安装完成"
}

# 配置Conda镜像源
configure_conda_mirrors() {
    log_info "配置Conda官方源..."

    # 清空可能存在的自定义channels
    log_info "清除自定义channels配置..."
    conda config --remove-key channels 2>/dev/null || true

    # 移除可能存在的镜像配置
    conda config --remove-key channel_alias 2>/dev/null || true
    conda config --remove-key default_channels 2>/dev/null || true

    # 加回官方源
    log_info "添加官方源（defaults + conda-forge）..."
    conda config --add channels defaults
    conda config --add channels conda-forge

    # 设置通道优先级
    conda config --set channel_priority strict
    conda config --set show_channel_urls yes

    # 清理索引缓存
    log_info "清理索引缓存..."
    conda clean -i -y 2>/dev/null || true

    log_success "Conda官方源配置完成"
}

# 创建Conda环境
create_conda_env() {
    log_info "创建Conda环境: soulxpodcast..."

    # 确保conda可用
    if ! command -v conda &> /dev/null; then
        eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    fi

    # 检查环境是否已存在
    if conda env list | grep -q "^soulxpodcast "; then
        log_warning "环境 soulxpodcast 已存在"
        read -p "是否删除并重新创建？[y/N]: " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "删除旧环境..."
            conda env remove -n soulxpodcast -y
        else
            log_info "使用现有环境"
            return 0
        fi
    fi

    log_info "创建Python 3.11环境..."
    conda create -n soulxpodcast python=3.11 -y

    log_success "Conda环境创建完成"
}

# 安装CUDA工具包（如果有NVIDIA GPU）
install_cuda() {
    log_info "检查NVIDIA GPU..."

    if command -v nvidia-smi &> /dev/null; then
        log_success "检测到NVIDIA GPU:"
        nvidia-smi --query-gpu=name --format=csv,noheader

        log_info "CUDA工具包将通过PyTorch自动安装"
    else
        log_warning "未检测到NVIDIA GPU，将使用CPU模式"
        log_warning "注意：CPU模式下推理速度会非常慢！"
    fi
}

# 克隆或更新项目代码
setup_project() {
    log_info "设置项目代码..."

    local PROJECT_DIR="$HOME/SoulX-Podcast"

    if [ -d "$PROJECT_DIR" ]; then
        log_warning "项目目录已存在: $PROJECT_DIR"
        read -p "是否使用现有目录？[Y/n]: " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Nn]$ ]]; then
            log_info "删除旧目录..."
            rm -rf "$PROJECT_DIR"
        else
            cd "$PROJECT_DIR"
            log_info "使用现有项目目录"
            return 0
        fi
    fi

    log_info "克隆项目代码..."
    git clone https://github.com/Soul-AILab/SoulX-Podcast.git "$PROJECT_DIR"
    cd "$PROJECT_DIR"

    log_success "项目代码设置完成"
}

# 安装Python依赖
install_dependencies() {
    log_info "安装Python依赖..."

    # 激活conda环境
    eval "$(conda shell.bash hook)"
    conda activate soulxpodcast

    # 检测是否配置pip镜像源
    read -p "是否使用阿里云PyPI镜像源（加速下载）？[y/N]: " -n 1 -r
    echo

    local PIP_INDEX=""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        PIP_INDEX="-i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com"
        log_info "使用阿里云镜像源"
    fi

    # 升级pip
    log_info "升级pip..."
    pip install --upgrade pip $PIP_INDEX

    # 安装PyTorch（CUDA版本）
    log_info "安装PyTorch（这可能需要较长时间）..."
    if command -v nvidia-smi &> /dev/null; then
        log_info "安装GPU版本PyTorch..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    else
        log_warning "安装CPU版本PyTorch..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi

    # 安装项目依赖
    log_info "安装项目依赖..."
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt $PIP_INDEX
    else
        log_error "未找到requirements.txt文件"
        exit 1
    fi

    # 安装API依赖
    log_info "安装API依赖..."
    if [ -f "api/requirements.txt" ]; then
        pip install -r api/requirements.txt $PIP_INDEX
    else
        log_warning "未找到api/requirements.txt文件"
    fi

    log_success "Python依赖安装完成"
}

# 下载模型
download_models() {
    log_info "下载模型文件..."

    # 激活conda环境
    eval "$(conda shell.bash hook)"
    conda activate soulxpodcast

    # 安装huggingface_hub库
    log_info "安装huggingface_hub库..."
    pip install -U huggingface_hub

    # 选择模型
    echo ""
    echo "请选择要下载的模型:"
    echo "  1) 基础模型 (SoulX-Podcast-1.7B) - 约7GB"
    echo "  2) 方言模型 (SoulX-Podcast-1.7B-dialect) - 约7GB"
    echo "  3) 两者都下载 - 约14GB"
    echo "  4) 跳过下载（稍后手动下载）"
    echo ""
    read -p "请输入选项 [1-4]: " -n 1 -r
    echo

    local MODEL_DIR="pretrained_models"
    mkdir -p "$MODEL_DIR"

    case $REPLY in
        1)
            log_info "下载基础模型..."
            python -c "from huggingface_hub import snapshot_download; snapshot_download('Soul-AILab/SoulX-Podcast-1.7B', local_dir='$MODEL_DIR/SoulX-Podcast-1.7B', resume_download=True)"
            log_success "基础模型下载完成"
            ;;
        2)
            log_info "下载方言模型..."
            python -c "from huggingface_hub import snapshot_download; snapshot_download('Soul-AILab/SoulX-Podcast-1.7B-dialect', local_dir='$MODEL_DIR/SoulX-Podcast-1.7B-dialect', resume_download=True)"
            log_success "方言模型下载完成"
            ;;
        3)
            log_info "下载基础模型..."
            python -c "from huggingface_hub import snapshot_download; snapshot_download('Soul-AILab/SoulX-Podcast-1.7B', local_dir='$MODEL_DIR/SoulX-Podcast-1.7B', resume_download=True)"
            log_info "下载方言模型..."
            python -c "from huggingface_hub import snapshot_download; snapshot_download('Soul-AILab/SoulX-Podcast-1.7B-dialect', local_dir='$MODEL_DIR/SoulX-Podcast-1.7B-dialect', resume_download=True)"
            log_success "所有模型下载完成"
            ;;
        4)
            log_warning "跳过模型下载"
            log_info "稍后可使用以下命令手动下载:"
            echo ""
            echo "  python -c \"from huggingface_hub import snapshot_download; snapshot_download('Soul-AILab/SoulX-Podcast-1.7B', local_dir='pretrained_models/SoulX-Podcast-1.7B', resume_download=True)\""
            echo ""
            ;;
        *)
            log_error "无效选项，跳过下载"
            ;;
    esac
}

# 创建启动脚本
create_start_script() {
    log_info "创建快速启动脚本..."

    cat > start_api.sh << 'EOF'
#!/bin/bash

# SoulX-Podcast API 快速启动脚本

# 颜色定义
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}=================================================================="
echo -e "  启动 SoulX-Podcast API 服务"
echo -e "==================================================================${NC}"
echo ""

# 激活conda环境
eval "$(conda shell.bash hook)"
conda activate soulxpodcast

# 检查模型是否存在
if [ ! -d "pretrained_models/SoulX-Podcast-1.7B" ] && [ ! -d "pretrained_models/SoulX-Podcast-1.7B-dialect" ]; then
    echo -e "${RED}[ERROR]${NC} 未找到模型文件！"
    echo "请先下载模型:"
    echo "  python -c \"from huggingface_hub import snapshot_download; snapshot_download('Soul-AILab/SoulX-Podcast-1.7B', local_dir='pretrained_models/SoulX-Podcast-1.7B', resume_download=True)\""
    exit 1
fi

# 启动API
echo -e "${GREEN}[INFO]${NC} 启动API服务..."
python run_api.py "$@"
EOF

    chmod +x start_api.sh
    log_success "启动脚本已创建: start_api.sh"
}

# 创建环境配置文件
create_env_file() {
    log_info "创建环境配置文件..."

    cat > .env.example << 'EOF'
# SoulX-Podcast API 环境配置

# 模型路径
MODEL_PATH=pretrained_models/SoulX-Podcast-1.7B
# MODEL_PATH=pretrained_models/SoulX-Podcast-1.7B-dialect

# LLM引擎 (hf 或 vllm)
LLM_ENGINE=hf

# Flow模型精度 (true=FP16更快, false=FP32更准确)
FP16_FLOW=false

# API服务配置
API_HOST=0.0.0.0
API_PORT=8000

# 最大并发任务数（根据GPU显存调整）
MAX_CONCURRENT_TASKS=2
EOF

    log_success "环境配置示例已创建: .env.example"
}

# 运行测试
run_tests() {
    log_info "是否运行快速测试？"
    read -p "运行测试需要启动模型（可能需要几分钟）[y/N]: " -n 1 -r
    echo

    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "跳过测试"
        return 0
    fi

    log_info "运行Python环境测试..."

    # 激活conda环境
    eval "$(conda shell.bash hook)"
    conda activate soulxpodcast

    python << 'PYEOF'
import sys
print("Python版本:", sys.version)

try:
    import torch
    print("PyTorch版本:", torch.__version__)
    print("CUDA可用:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA版本:", torch.version.cuda)
        print("GPU设备:", torch.cuda.get_device_name(0))
    print("\n✓ 环境测试通过！")
except Exception as e:
    print(f"\n✗ 环境测试失败: {e}")
    sys.exit(1)
PYEOF

    if [ $? -eq 0 ]; then
        log_success "测试通过"
    else
        log_error "测试失败，请检查安装"
    fi
}

# 打印完成信息
print_completion() {
    echo ""
    echo "=================================================================="
    echo -e "${GREEN}  安装完成！${NC}"
    echo "=================================================================="
    echo ""
    echo "下一步操作:"
    echo ""
    echo "1. 激活conda环境:"
    echo "   ${BLUE}conda activate soulxpodcast${NC}"
    echo ""
    echo "2. 启动API服务:"
    echo "   ${BLUE}python run_api.py${NC}"
    echo "   或使用快捷脚本:"
    echo "   ${BLUE}./start_api.sh${NC}"
    echo ""
    echo "3. 访问API文档:"
    echo "   ${BLUE}http://localhost:8000/docs${NC}"
    echo ""
    echo "4. 测试API:"
    echo "   ${BLUE}python api/test_client.py${NC}"
    echo ""
    echo "配置文件:"
    echo "  - 查看 .env.example 了解环境变量配置"
    echo "  - 查看 README_API.md 了解API使用方法"
    echo ""
    echo "=================================================================="
    echo ""
}

# 主函数
main() {
    print_banner

    # 检查环境
    check_wsl
    check_ubuntu_version

    # 系统更新和基础工具
    update_system
    install_basic_tools

    # 安装Conda
    install_miniconda
    # configure_conda_mirrors  # 已禁用：使用conda默认配置

    # 创建Python环境
    create_conda_env

    # 检查GPU
    install_cuda

    # 设置项目
    setup_project

    # 安装依赖
    install_dependencies

    # 下载模型
    download_models

    # 创建辅助脚本
    create_start_script
    create_env_file

    # 测试
    run_tests

    # 完成
    print_completion
}

# 执行主函数
main

# WSL Ubuntu 24.04 快速部署指南

本指南帮助您在全新的Windows WSL Ubuntu 24.04环境中快速部署SoulX-Podcast API服务。

## 前置要求

- Windows 10/11 with WSL2
- Ubuntu 24.04 LTS (WSL)
- 至少 20GB 可用磁盘空间
- （推荐）NVIDIA GPU with CUDA support

## 快速开始

### 方法1: 使用自动安装脚本（推荐）

**1. 在Windows中打开WSL终端**

```bash
# 启动WSL
wsl
```

**2. 复制脚本到WSL**

如果项目已在Windows中：
```bash
# 假设项目在 Windows 的 D:\SoulX-Podcast
cd ~
cp /mnt/d/SoulX-Podcast/setup_wsl.sh .
```

或直接下载：
```bash
cd ~
wget https://raw.githubusercontent.com/Soul-AILab/SoulX-Podcast/main/setup_wsl.sh
```

**3. 运行安装脚本**

```bash
bash setup_wsl.sh
```

脚本会自动完成以下步骤：
- ✅ 检查WSL环境
- ✅ 更新系统包
- ✅ 安装基础工具（curl, git等）
- ✅ 安装Miniconda
- ✅ 配置镜像源（可选）
- ✅ 创建Python 3.11环境
- ✅ 克隆项目代码
- ✅ 安装PyTorch和依赖
- ✅ 下载模型文件
- ✅ 创建启动脚本

**预计耗时**: 30-60分钟（取决于网络速度）

### 方法2: 手动安装

详见下方的[手动安装步骤](#手动安装步骤)

## 启动服务

安装完成后：

```bash
# 进入项目目录
cd ~/SoulX-Podcast

# 激活环境
conda activate soulxpodcast

# 启动API（方式1）
python run_api.py

# 或使用快捷脚本（方式2）
./start_api.sh
```

访问 http://localhost:8000/docs 查看API文档

## 常见问题

### 1. 如何在Windows访问WSL中的API服务？

WSL2会自动转发端口，直接在Windows浏览器访问：
```
http://localhost:8000/docs
```

### 2. 网络下载很慢怎么办？

脚本会询问是否使用国内镜像源（阿里云）：
- Conda镜像源：加速Conda包下载
- PyPI镜像源：加速pip包下载
- HuggingFace镜像：可使用 hf-mirror.com

### 3. GPU不可用怎么办？

**检查WSL CUDA支持**：
```bash
# 检查NVIDIA驱动
nvidia-smi

# 检查PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

**安装CUDA Toolkit for WSL**：
参考：https://docs.nvidia.com/cuda/wsl-user-guide/index.html

### 4. 模型下载失败

手动下载模型：
```bash
cd ~/SoulX-Podcast

# 激活环境
conda activate soulxpodcast

# 使用镜像站下载（国内用户）
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download Soul-AILab/SoulX-Podcast-1.7B \
    --local-dir pretrained_models/SoulX-Podcast-1.7B

# 或使用官方源
huggingface-cli download --resume-download Soul-AILab/SoulX-Podcast-1.7B \
    --local-dir pretrained_models/SoulX-Podcast-1.7B
```

### 5. 端口被占用

修改端口：
```bash
python run_api.py --port 8080
```

### 6. 显存不足

减少并发任务数：
```bash
python run_api.py --max-tasks 1
```

或使用FP16精度：
```bash
python run_api.py --fp16-flow
```

## 手动安装步骤

如果自动脚本失败，可以手动执行以下步骤：

### 1. 更新系统
```bash
sudo apt-get update
sudo apt-get upgrade -y
```

### 2. 安装基础工具
```bash
sudo apt-get install -y curl wget git build-essential \
    libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
    libsqlite3-dev libncursesw5-dev xz-utils tk-dev \
    libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
```

### 3. 安装Miniconda
```bash
# 下载Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# 安装
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3

# 初始化
$HOME/miniconda3/bin/conda init bash
source ~/.bashrc
```

### 4. 创建Python环境
```bash
conda create -n soulxpodcast python=3.11 -y
conda activate soulxpodcast
```

### 5. 克隆项目
```bash
git clone https://github.com/Soul-AILab/SoulX-Podcast.git
cd SoulX-Podcast
```

### 6. 安装依赖
```bash
# 安装PyTorch (CUDA 12.1)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 或CPU版本
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 安装项目依赖
pip install -r requirements.txt
pip install -r api/requirements.txt
```

### 7. 下载模型
```bash
pip install -U huggingface_hub

huggingface-cli download --resume-download Soul-AILab/SoulX-Podcast-1.7B \
    --local-dir pretrained_models/SoulX-Podcast-1.7B
```

### 8. 启动服务
```bash
python run_api.py
```

## 性能优化

### WSL配置优化

创建 `C:\Users\你的用户名\.wslconfig`：
```ini
[wsl2]
# 内存限制（根据实际情况调整）
memory=16GB

# CPU核心数
processors=8

# 交换空间
swap=8GB
```

重启WSL生效：
```powershell
# 在Windows PowerShell中执行
wsl --shutdown
wsl
```

### GPU显存优化

如果显存不足，编辑配置：
```bash
# 设置环境变量
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 或启动时指定
MAX_CONCURRENT_TASKS=1 FP16_FLOW=true python run_api.py
```

## 测试API

```bash
# 运行测试脚本
python api/test_client.py

# 健康检查
curl http://localhost:8000/health

# 查看API文档
# 浏览器访问: http://localhost:8000/docs
```

## 卸载

```bash
# 删除conda环境
conda deactivate
conda env remove -n soulxpodcast

# 删除项目
rm -rf ~/SoulX-Podcast

# 删除Miniconda（可选）
rm -rf ~/miniconda3
```

## 国内用户加速技巧

### 1. 使用镜像站

**Conda镜像**：
```bash
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --set show_channel_urls yes
```

**PyPI镜像**：
```bash
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

**HuggingFace镜像**：
```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download Soul-AILab/SoulX-Podcast-1.7B \
    --local-dir pretrained_models/SoulX-Podcast-1.7B
```

### 2. Git加速

```bash
# 使用镜像克隆
git clone https://ghproxy.com/https://github.com/Soul-AILab/SoulX-Podcast.git
```

## 故障排查

### WSL无法访问GPU

1. 确保Windows驱动已更新（版本 >= 510.60.02）
2. 重启WSL：
   ```powershell
   wsl --shutdown
   wsl
   ```
3. 检查CUDA：
   ```bash
   nvidia-smi
   ```

### 端口无法访问

检查防火墙设置，或使用WSL IP：
```bash
# 获取WSL IP
hostname -I

# 使用WSL IP访问
# http://<WSL_IP>:8000/docs
```

### 内存不足

增加WSL内存限制（见性能优化）

## 参考链接

- [WSL官方文档](https://docs.microsoft.com/en-us/windows/wsl/)
- [WSL CUDA支持](https://docs.nvidia.com/cuda/wsl-user-guide/)
- [SoulX-Podcast项目主页](https://github.com/Soul-AILab/SoulX-Podcast)
- [API使用文档](README_API.md)

## 支持

如遇问题，请：
1. 查看日志输出
2. 检查 GitHub Issues
3. 联系项目维护者

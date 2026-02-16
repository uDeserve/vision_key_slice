#!/bin/bash

# Vision-SR1 RL训练环境安装脚本
# 此脚本用于设置RL训练的conda环境
# 参考了deepeyesv2_rl环境的配置，但适配了Vision-SR1的特定要求（Python 3.11等）

set -e  # 遇到错误立即退出

echo "=========================================="
echo "Vision-SR1 RL训练环境安装脚本"
echo "=========================================="
echo "注意: 本脚本参考了deepeyesv2_rl环境的配置"
echo "但使用Python 3.11和Vision-SR1要求的特定版本"
echo ""
echo "💡 提示: pip和conda会自动使用缓存"
echo "   相同版本的包会从缓存安装，节省下载时间"
echo "   环境之间仍然是隔离的，不会混乱"
echo "=========================================="

# 检测CUDA版本
echo ""
echo "正在检测CUDA版本..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
    echo "检测到CUDA版本: $CUDA_VERSION"
else
    echo "警告: 未检测到nvcc，尝试从nvidia-smi获取CUDA版本..."
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | sed 's/.*CUDA Version: \([0-9]\+\.[0-9]\+\).*/\1/' || echo "unknown")
        echo "检测到CUDA版本: $CUDA_VERSION"
    else
        echo "错误: 无法检测CUDA版本，请确保已安装NVIDIA驱动和CUDA工具包"
        exit 1
    fi
fi

# 确定PyTorch的CUDA版本
# setup.sh中使用的是cu124，但CUDA 12.4+通常可以使用cu124的PyTorch
# 使用简单的数值比较（不依赖bc）
CUDA_MAJOR=$(echo "$CUDA_VERSION" | cut -d. -f1)
CUDA_MINOR=$(echo "$CUDA_VERSION" | cut -d. -f2)

if [ "$CUDA_MAJOR" -gt 12 ] || ([ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 4 ]); then
    # CUDA 12.4+ 使用 cu124
    PYTORCH_CUDA="cu124"
    echo "将使用PyTorch CUDA版本: $PYTORCH_CUDA"
elif [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 1 ]; then
    # CUDA 12.1-12.3 使用 cu121
    PYTORCH_CUDA="cu121"
    echo "将使用PyTorch CUDA版本: $PYTORCH_CUDA"
elif [ "$CUDA_MAJOR" -gt 11 ] || ([ "$CUDA_MAJOR" -eq 11 ] && [ "$CUDA_MINOR" -ge 8 ]); then
    # CUDA 11.8-12.0 使用 cu118
    PYTORCH_CUDA="cu118"
    echo "将使用PyTorch CUDA版本: $PYTORCH_CUDA"
else
    echo "警告: CUDA版本 $CUDA_VERSION 可能不受支持，将尝试使用cu124"
    PYTORCH_CUDA="cu124"
fi

# 检测Python版本
PYTHON_VERSION=$(python --version 2>&1 | grep -oE '[0-9]+\.[0-9]+' | head -1)
echo "Python版本: $PYTHON_VERSION"

# 检查是否在conda环境中
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo ""
    echo "警告: 未检测到激活的conda环境"
    echo "请先创建并激活conda环境:"
    echo "  conda env create -f environment.yml"
    echo "  conda activate vision-sr1-rl"
    read -p "是否继续? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "当前conda环境: $CONDA_DEFAULT_ENV"
fi

echo ""
echo "开始安装依赖..."
echo ""

# 安装PyTorch（第一次安装）
echo "步骤 1/8: 安装PyTorch 2.6.0..."
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/${PYTORCH_CUDA}

# 安装flash-attention
echo ""
echo "步骤 2/8: 安装flash-attention 2.7.4..."
# 根据Python版本选择wheel文件
PYTHON_MAJOR_MINOR=$(echo $PYTHON_VERSION | cut -d. -f1,2 | tr -d '.')
if [ "$PYTHON_MAJOR_MINOR" = "311" ]; then
    FLASH_ATTN_WHL="flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
elif [ "$PYTHON_MAJOR_MINOR" = "310" ]; then
    FLASH_ATTN_WHL="flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
elif [ "$PYTHON_MAJOR_MINOR" = "312" ]; then
    FLASH_ATTN_WHL="flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp312-cp312-linux_x86_64.whl"
else
    echo "警告: Python版本 $PYTHON_VERSION 可能没有预编译的flash-attention wheel"
    echo "将尝试安装通用版本..."
    FLASH_ATTN_WHL="flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
fi

pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/${FLASH_ATTN_WHL} || {
    echo "警告: flash-attention安装失败，可能需要从源码编译"
    echo "尝试安装可用的版本..."
    pip install flash-attn>=2.4.3
}

# 安装vllm（第一次）
echo ""
echo "步骤 3/8: 安装vllm 0.8.4..."
pip install "vllm==0.8.4"

# 重新安装PyTorch（确保版本一致）
echo ""
echo "步骤 4/8: 重新安装PyTorch以确保版本一致..."
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/${PYTORCH_CUDA}

# 安装transformers特定版本
echo ""
echo "步骤 5/8: 安装transformers特定版本..."
pip install git+https://github.com/huggingface/transformers.git@1931a351408dbd1d0e2c4d6d7ee0eb5e8807d7bf

# 安装项目本身
echo ""
echo "步骤 6/8: 安装Vision-SR1项目..."
pip install -e .

# 重新安装torch和flash-attn（解决可能的依赖冲突）
echo ""
echo "步骤 7/8: 重新安装torch和flash-attn以解决依赖冲突..."
pip uninstall -y torch flash-attn || true
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/${PYTORCH_CUDA}
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/${FLASH_ATTN_WHL} || pip install flash-attn>=2.4.3

# 重新安装vllm
echo ""
echo "步骤 8/8: 重新安装vllm..."
pip uninstall -y vllm || true
pip install "vllm==0.8.4"

# 重新安装transformers并固定版本
echo ""
echo "最终步骤: 固定transformers版本为4.49.0..."
pip install git+https://github.com/huggingface/transformers.git@1931a351408dbd1d0e2c4d6d7ee0eb5e8807d7bf
pip install transformers==4.49.0

# 安装requirements.txt中的其他依赖
echo ""
echo "安装requirements.txt中的其他依赖..."
pip install -r requirements.txt

# 检查wandb登录状态
echo ""
echo "检查wandb登录状态..."
if ! wandb status 2>/dev/null | grep -q "Logged in"; then
    echo "⚠️  您尚未登录W&B (Weights & Biases)"
    echo "W&B用于训练过程的可视化和日志记录"
    read -p "是否现在登录W&B? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        read -rs -p "请输入您的W&B API key: " WANDB_KEY
        echo
        wandb login "$WANDB_KEY"
    else
        echo "您可以稍后使用 'wandb login' 命令登录"
    fi
else
    echo "✓ W&B已登录"
fi

echo ""
echo "=========================================="
echo "安装完成！"
echo "=========================================="
echo ""
echo "环境信息:"
echo "  - Conda环境: ${CONDA_DEFAULT_ENV:-未设置}"
echo "  - Python版本: $PYTHON_VERSION"
echo "  - CUDA版本: $CUDA_VERSION"
echo "  - PyTorch CUDA: $PYTORCH_CUDA"
echo ""
echo "下一步:"
echo "  1. 等待学姐发送SFT训练好的模型文件"
echo "  2. 将模型文件放置在指定目录"
echo "  3. 运行RL训练脚本，例如:"
echo "     bash ./train_examples/2-7b_selfReward_train.sh"
echo ""
echo "如有问题，请参考 RL_ENV_SETUP.md 文档"
echo ""


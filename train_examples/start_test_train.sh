#!/bin/bash
# 非交互式训练启动脚本（用于自动化测试）

set -e

# 获取脚本所在目录（项目根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# 自动检测conda路径
if [ -z "$CONDA_PREFIX" ]; then
    # 尝试常见的conda安装路径
    if [ -f "/opt/miniconda3/etc/profile.d/conda.sh" ]; then
        source /opt/miniconda3/etc/profile.d/conda.sh
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
        source /opt/conda/etc/profile.d/conda.sh
    else
        echo "错误: 未找到conda，请手动激活conda环境"
        echo "或者设置CONDA_PREFIX环境变量"
        exit 1
    fi
fi

# 激活conda环境
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    conda activate vision-sr1-rl || {
        echo "错误: 无法激活conda环境 vision-sr1-rl"
        echo "请确保环境已创建: conda create -n vision-sr1-rl python=3.11"
        exit 1
    }
fi

# 设置环境变量
export PYTHONUNBUFFERED=1
# 设置HuggingFace镜像站（不使用代理，直接连接镜像站）
export HF_ENDPOINT=https://hf-mirror.com
# 不使用代理（代理有SSL问题，镜像站可以直接访问）
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy

# 设置Ray临时目录（自动检测有足够空间的位置）
if [ -z "$RAY_TMPDIR" ]; then
    # 优先使用项目目录下的ray_tmp
    if [ -d "$PROJECT_ROOT" ] && [ -w "$PROJECT_ROOT" ]; then
        export RAY_TMPDIR="$PROJECT_ROOT/ray_tmp"
    # 其次使用用户home目录
    elif [ -d "$HOME" ] && [ -w "$HOME" ]; then
        export RAY_TMPDIR="$HOME/ray_tmp"
    # 最后使用系统临时目录
    else
        export RAY_TMPDIR="/tmp/ray_tmp"
    fi
    mkdir -p "$RAY_TMPDIR"
fi

# 检测可用GPU数量
AVAILABLE_GPUS=8  # 使用8卡
NUM_GPUS=${1:-$AVAILABLE_GPUS}  # 可以通过参数指定GPU数量（默认8卡）

# 创建日志目录
mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/train_${TIMESTAMP}.log"
PID_FILE="logs/train_${TIMESTAMP}.pid"

echo "=========================================="
echo "启动Qwen3-VL-8B测试训练"
echo "=========================================="
echo "时间: $(date)"
echo "使用GPU数量: $NUM_GPUS"
echo "日志文件: $LOG_FILE"
echo "=========================================="
echo ""

# 启动训练（后台运行）
nohup python3 -m verl.trainer.main \
    config=train_examples/test_qwen3vl_8b_config.yaml \
    trainer.n_gpus_per_node=$NUM_GPUS \
    > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
echo $TRAIN_PID > "$PID_FILE"

echo "训练已启动！"
echo "  PID: $TRAIN_PID"
echo "  日志: $LOG_FILE"
echo "  PID文件: $PID_FILE"
echo ""
echo "查看日志: tail -f $LOG_FILE"
echo "停止训练: kill $TRAIN_PID"


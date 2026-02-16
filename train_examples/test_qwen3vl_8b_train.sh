#!/bin/bash

# Qwen3-VL-8B测试数据RL训练启动脚本
# 支持conda环境检查、GPU检测、后台运行

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 配置
CONDA_ENV="vision-sr1-rl"
# 获取脚本所在目录（项目根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/test_qwen3vl_8b_config.yaml"
LOG_DIR="${PROJECT_ROOT}/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"
PID_FILE="${LOG_DIR}/train_${TIMESTAMP}.pid"

# 创建日志目录
mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "Qwen3-VL-8B 测试数据 RL 训练启动脚本"
echo "=========================================="
echo ""

# 1. 检查conda环境
echo -e "${YELLOW}[1/5] 检查conda环境...${NC}"
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo -e "${RED}错误: 未检测到激活的conda环境${NC}"
    echo "请先激活conda环境:"
    echo "  conda activate ${CONDA_ENV}"
    exit 1
fi

if [ "$CONDA_DEFAULT_ENV" != "$CONDA_ENV" ]; then
    echo -e "${YELLOW}警告: 当前conda环境是 '$CONDA_DEFAULT_ENV'，期望是 '$CONDA_ENV'${NC}"
    read -p "是否继续? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "${GREEN}✓ 当前conda环境: $CONDA_DEFAULT_ENV${NC}"
fi

# 2. 检查GPU可用性
echo -e "${YELLOW}[2/5] 检查GPU可用性...${NC}"
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}错误: 未找到nvidia-smi命令${NC}"
    exit 1
fi

# 获取可用GPU数量
AVAILABLE_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo -e "${GREEN}✓ 检测到 $AVAILABLE_GPUS 块GPU${NC}"

# 显示GPU状态
echo ""
echo "GPU状态:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader | head -8

# 询问使用的GPU数量
echo ""
read -p "使用多少块GPU? (默认: $AVAILABLE_GPUS) " NUM_GPUS
NUM_GPUS=${NUM_GPUS:-$AVAILABLE_GPUS}

if [ "$NUM_GPUS" -gt "$AVAILABLE_GPUS" ]; then
    echo -e "${RED}错误: 请求的GPU数量 ($NUM_GPUS) 超过可用数量 ($AVAILABLE_GPUS)${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 将使用 $NUM_GPUS 块GPU${NC}"

# 3. 检查配置文件
echo -e "${YELLOW}[3/5] 检查配置文件...${NC}"
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}错误: 配置文件不存在: $CONFIG_FILE${NC}"
    exit 1
fi
echo -e "${GREEN}✓ 配置文件存在: $CONFIG_FILE${NC}"

# 4. 检查数据文件
echo -e "${YELLOW}[4/5] 检查数据文件...${NC}"
TRAIN_FILE=$(grep "train_files:" "$CONFIG_FILE" | awk '{print $2}' | tr -d '"')
VAL_FILE=$(grep "val_files:" "$CONFIG_FILE" | awk '{print $2}' | tr -d '"')

if [ ! -f "$TRAIN_FILE" ]; then
    echo -e "${YELLOW}警告: 训练数据文件不存在: $TRAIN_FILE${NC}"
    echo "请先运行数据转换脚本:"
    echo "  python scripts/convert_test_data.py --input_dir <解压后的数据目录> --output_jsonl $TRAIN_FILE"
    read -p "是否继续? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "${GREEN}✓ 训练数据文件存在: $TRAIN_FILE${NC}"
fi

if [ ! -f "$VAL_FILE" ]; then
    echo -e "${YELLOW}警告: 验证数据文件不存在: $VAL_FILE${NC}"
else
    echo -e "${GREEN}✓ 验证数据文件存在: $VAL_FILE${NC}"
fi

# 5. 设置环境变量
echo -e "${YELLOW}[5/5] 设置环境变量...${NC}"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
echo -e "${GREEN}✓ 环境变量设置完成${NC}"
echo "  PYTHONUNBUFFERED=1"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# 切换到项目根目录
cd "$PROJECT_ROOT"

# 6. 构建训练命令
TRAIN_CMD="python3 -m verl.trainer.main \
    config=$CONFIG_FILE \
    trainer.n_gpus_per_node=$NUM_GPUS"

echo ""
echo "=========================================="
echo "训练命令:"
echo "$TRAIN_CMD"
echo "=========================================="
echo ""

# 询问是否后台运行
read -p "是否后台运行? (y/n, 默认: y) " -n 1 -r
echo
BACKGROUND=${REPLY:-y}

if [[ $BACKGROUND =~ ^[Yy]$ ]]; then
    echo -e "${GREEN}将在后台运行训练...${NC}"
    echo "日志文件: $LOG_FILE"
    echo "PID文件: $PID_FILE"
    
    # 后台运行
    nohup bash -c "$TRAIN_CMD" > "$LOG_FILE" 2>&1 &
    TRAIN_PID=$!
    
    # 保存PID
    echo $TRAIN_PID > "$PID_FILE"
    
    echo ""
    echo -e "${GREEN}训练已启动！${NC}"
    echo "  PID: $TRAIN_PID"
    echo "  日志: $LOG_FILE"
    echo "  PID文件: $PID_FILE"
    echo ""
    echo "查看日志:"
    echo "  tail -f $LOG_FILE"
    echo ""
    echo "停止训练:"
    echo "  kill $TRAIN_PID"
    echo "  或: kill \$(cat $PID_FILE)"
else
    echo -e "${GREEN}在前台运行训练...${NC}"
    echo "按 Ctrl+C 停止训练"
    echo ""
    # 前台运行
    $TRAIN_CMD
fi








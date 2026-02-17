#!/bin/bash

# GPU显存监控脚本
# 实时监控GPU显存使用情况，记录到日志文件

# 配置
INTERVAL=5  # 监控间隔（秒）
LOG_FILE="${1:-./gpu_monitor.log}"  # 日志文件路径（可通过参数指定）

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "=========================================="
echo "GPU显存监控脚本"
echo "=========================================="
echo "监控间隔: ${INTERVAL}秒"
echo "日志文件: $LOG_FILE"
echo "按 Ctrl+C 停止监控"
echo "=========================================="
echo ""

# 检查nvidia-smi
if ! command -v nvidia-smi &> /dev/null; then
    echo "错误: 未找到nvidia-smi命令"
    exit 1
fi

# 创建日志目录
LOG_DIR=$(dirname "$LOG_FILE")
if [ ! -z "$LOG_DIR" ] && [ "$LOG_DIR" != "." ]; then
    mkdir -p "$LOG_DIR"
fi

# 写入日志头部
{
    echo "=========================================="
    echo "GPU显存监控日志"
    echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    echo ""
} >> "$LOG_FILE"

# 监控循环
while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    # 获取GPU信息
    GPU_INFO=$(nvidia-smi --query-gpu=index,name,memory.used,memory.total,memory.free,utilization.gpu,utilization.memory,temperature.gpu --format=csv,noheader,nounits)
    
    # 输出到控制台（带颜色）
    echo -e "${GREEN}[$TIMESTAMP]${NC} GPU状态:"
    echo "$GPU_INFO" | while IFS=',' read -r index name mem_used mem_total mem_free util_gpu util_mem temp; do
        # 计算显存使用百分比
        mem_used=$(echo "$mem_used" | xargs)
        mem_total=$(echo "$mem_total" | xargs)
        mem_percent=$(awk "BEGIN {printf \"%.1f\", ($mem_used/$mem_total)*100}")
        
        # 格式化输出
        printf "  GPU %s: %s | 显存: %s/%s MB (%.1f%%) | GPU利用率: %s%% | 温度: %s°C\n" \
            "$(echo $index | xargs)" \
            "$(echo $name | xargs)" \
            "$mem_used" \
            "$mem_total" \
            "$mem_percent" \
            "$(echo $util_gpu | xargs)" \
            "$(echo $temp | xargs)"
    done
    echo ""
    
    # 写入日志文件
    {
        echo "[$TIMESTAMP] GPU状态:"
        echo "$GPU_INFO" | while IFS=',' read -r index name mem_used mem_total mem_free util_gpu util_mem temp; do
            mem_used=$(echo "$mem_used" | xargs)
            mem_total=$(echo "$mem_total" | xargs)
            mem_percent=$(awk "BEGIN {printf \"%.1f\", ($mem_used/$mem_total)*100}")
            
            printf "  GPU %s: %s | 显存: %s/%s MB (%.1f%%) | GPU利用率: %s%% | 温度: %s°C\n" \
                "$(echo $index | xargs)" \
                "$(echo $name | xargs)" \
                "$mem_used" \
                "$mem_total" \
                "$mem_percent" \
                "$(echo $util_gpu | xargs)" \
                "$(echo $temp | xargs)"
        done
        echo ""
    } >> "$LOG_FILE"
    
    # 等待指定间隔
    sleep "$INTERVAL"
done












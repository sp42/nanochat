#!/bin/bash

# See speedrun.sh for more comments
# 更多注释请参阅 speedrun.sh
# Usage: ./miniseries.sh [series_name]
# 用法：./miniseries.sh [series_name]
# Example: ./miniseries.sh jan11
# 示例：./miniseries.sh jan11
# Default series name is today's date (e.g., jan11)
# 默认系列名称是今天的日期（例如 jan11）

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# Setup (skip with SKIP_SETUP=1)
# 设置（使用 SKIP_SETUP=1 跳过）
if [ -z "$SKIP_SETUP" ]; then
    # uv
    command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
    [ -d ".venv" ] || uv venv
    uv sync --extra gpu
    source .venv/bin/activate

    # Tokenizer, download 1000 shards for pretraining
    # 分词器，下载 1000 个分片用于预训练
    # (probably this can be reduced but it's tricky to determine the exact right number, TODO).
    # （可能可以减少，但很难确定确切的正确数字，TODO）。
    python -m nanochat.dataset -n 1000
    python -m scripts.tok_train --max-chars=2000000000 --vocab-size=32768
else
    source .venv/bin/activate
fi

# Series name: from arg, env var, or default to today's date (e.g., jan11)
# 系列名称：来自参数、环境变量，或默认为今天的日期（例如 jan11）
SERIES_NAME="${1:-${SERIES_NAME:-$(date +%b%d | tr '[:upper:]' '[:lower:]')}}"
# Depths to train (the "miniseries")
# 要训练的深度（"miniseries"）
DEPTHS=(12 14 16 18 20 22 24 26)
# Hardware
# 硬件
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
# Logging
# 日志
WANDB_RUN="${WANDB_RUN:-${SERIES_NAME}_miniseries}"

RESULTS_DIR="$NANOCHAT_BASE_DIR/${SERIES_NAME}_miniseries_results"
mkdir -p "$RESULTS_DIR"
RESULTS_FILE="$RESULTS_DIR/results.csv"

# Write CSV header only if file doesn't exist
# 仅在文件不存在时写入 CSV 标题
if [ ! -f "$RESULTS_FILE" ]; then
    echo "depth,model_dim,num_params,num_scaling_params,num_iterations,tokens_trained,param_data_ratio,val_bpb,core_score,train_time_sec" > "$RESULTS_FILE"
fi

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "=============================================="
log "${SERIES_NAME} Miniseries Training"
log "${SERIES_NAME} Miniseries 训练"
log "=============================================="

for d in "${DEPTHS[@]}"; do
    log "Training d=$d..."
    log "训练 d=$d..."

    TAG="${SERIES_NAME}_miniseries_d${d}"
    START_TIME=$(date +%s)

    # Reduce --device-batch-size to avoid OOM at larger depths
# 减少 --device-batch-size 以避免在较大深度时 OOM
    if [ $d -ge 28 ]; then
        DEVICE_BATCH_SIZE_ARG="--device-batch-size=8"
    elif [ $d -ge 20 ]; then
        DEVICE_BATCH_SIZE_ARG="--device-batch-size=16"
    else
        DEVICE_BATCH_SIZE_ARG="--device-batch-size=32"
    fi

    torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- \
        --depth=$d \
        --run="${WANDB_RUN}_d${d}" \
        --model-tag="${TAG}" \
        --core-metric-every=999999 \
        --core-metric-max-per-task=-1 \
        --sample-every=-1 \
        --save-every=-1 \
        $DEVICE_BATCH_SIZE_ARG \
        2>&1 | tee "$RESULTS_DIR/${TAG}_train.log"

    END_TIME=$(date +%s)
    TRAIN_TIME=$((END_TIME - START_TIME))

    # Extract stats from log
    # 从日志中提取统计信息
    LOG_FILE="$RESULTS_DIR/${TAG}_train.log"
    NUM_PARAMS=$(grep "Number of parameters:" "$LOG_FILE" | tail -1 | grep -oP '[\d,]+' | head -1 | tr -d ',')
    NUM_SCALING_PARAMS=$(grep "Number of parameters:" "$LOG_FILE" | tail -1 | grep -oP 'scaling: [\d,]+' | grep -oP '[\d,]+' | tr -d ',')
    NUM_ITERS=$(grep "Calculated number of iterations" "$LOG_FILE" | tail -1 | sed 's/.*: //' | tr -d ',')
    TOKENS_TRAINED=$((NUM_ITERS * 524288))
    PARAM_DATA_RATIO=$(python -c "print(f'{$TOKENS_TRAINED / $NUM_SCALING_PARAMS:.2f}')")
    MODEL_DIM=$((d * 64))
    VAL_BPB=$(grep "Validation bpb:" "$LOG_FILE" | tail -1 | grep -oP '[\d.]+$')
    CORE_SCORE=$(grep "CORE metric:" "$LOG_FILE" | tail -1 | awk '{print $NF}')

    if [ -z "$CORE_SCORE" ]; then
        CORE_SCORE="0.0"
    fi

    log "  d=$d: params=$NUM_PARAMS, scaling=$NUM_SCALING_PARAMS, ratio=$PARAM_DATA_RATIO, bpb=$VAL_BPB, CORE=$CORE_SCORE, time=${TRAIN_TIME}s"

    # Append to CSV
    # 追加到 CSV
    echo "$d,$MODEL_DIM,$NUM_PARAMS,$NUM_SCALING_PARAMS,$NUM_ITERS,$TOKENS_TRAINED,$PARAM_DATA_RATIO,$VAL_BPB,$CORE_SCORE,$TRAIN_TIME" >> "$RESULTS_FILE"
done

log "=============================================="
log "${SERIES_NAME} Miniseries Complete!"
log "${SERIES_NAME} Miniseries 完成！"
log "=============================================="
log "Results saved to: $RESULTS_FILE"
log "结果保存到：$RESULTS_FILE"
echo ""
echo "Results:"
echo "结果："
column -t -s',' "$RESULTS_FILE"

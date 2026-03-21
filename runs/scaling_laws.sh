#!/bin/bash

LABEL="jan26"

FLOPS_BUDGETS=(
    1e18
    2.15e18
    4.64e18
    1e19
)
DEPTHS=(8 10 12 14 16 18 20)

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
WANDB_RUN="${WANDB_RUN:-scaling_${LABEL}}"
EVAL_TOKENS=$((100 * 524288))  # ~100M tokens for final eval (default is ~10M)
                                # ~100M token 用于最终评估（默认为 ~10M）

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
source .venv/bin/activate

RESULTS_DIR="$NANOCHAT_BASE_DIR/scaling_laws_results_${LABEL}"
mkdir -p "$RESULTS_DIR"
RESULTS_FILE="$RESULTS_DIR/results.csv"

# Write CSV header only if file doesn't exist
# 仅在文件不存在时写入 CSV 标题
if [ ! -f "$RESULTS_FILE" ]; then
    echo "flops_budget,depth,model_dim,params_wte,params_value_embeds,params_lm_head,params_transformer,params_scalars,params_total,num_iterations,tokens_trained,val_bpb,core_score,train_time_sec" > "$RESULTS_FILE"
fi

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Check if a run already exists in results
# 检查运行是否已存在于结果中
run_exists() {
    local flops=$1
    local depth=$2
    grep -q "^${flops},${depth}," "$RESULTS_FILE" 2>/dev/null
}

# =============================================================================
# Main Loop
# 主循环
# =============================================================================

for flops in "${FLOPS_BUDGETS[@]}"; do
    log "=============================================="
    log "Compute budget: $flops FLOPs"
    log "计算预算：$flops FLOPs"
    log "=============================================="

    for d in "${DEPTHS[@]}"; do

        # Skip if already completed
        # 如果已完成则跳过
        if run_exists "$flops" "$d"; then
            log "Skipping d=$d at $flops FLOPs (already in results)"
            log "跳过 d=$d 在 $flops FLOPs（已在结果中）"
            continue
        fi

        log "Training d=$d at $flops FLOPs..."
        log "训练 d=$d 在 $flops FLOPs..."

        # Unique tag for this run
        # 此运行的唯一标签
        TAG="scaling_${flops}_d${d}"

        # Record start time
        # 记录开始时间
        START_TIME=$(date +%s)

        # Train the model with fixed flops budget
        # 使用固定 flops 预算训练模型
        # The script will auto-calculate num_iterations to hit target_flops
        # 脚本将自动计算 num_iterations 以达到 target_flops
        # CORE eval happens once at the end (999999 ensures only final step)
        # CORE 评估在结束时进行一次（999999 确保只在最后一步）
        torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- \
            --depth=$d \
            --target-flops=$flops \
            --target-param-data-ratio=-1 \
            --run="${WANDB_RUN}_${TAG}" \
            --model-tag="${TAG}" \
            --eval-tokens=$EVAL_TOKENS \
            --core-metric-every=999999 \
            --core-metric-max-per-task=-1 \
            --sample-every=-1 \
            --save-every=-1 \
            2>&1 | tee "$RESULTS_DIR/${TAG}_train.log"

        END_TIME=$(date +%s)
        TRAIN_TIME=$((END_TIME - START_TIME))

        # Extract training stats from the log
        # 从日志中提取训练统计信息
        LOG_FILE="$RESULTS_DIR/${TAG}_train.log"

        # Extract detailed parameter counts (for scaling law analysis with different conventions)
        # 提取详细的参数计数（用于不同约定的缩放定律分析）
        # Note: the log format is padded, e.g. "wte                     : 25,165,824"
        # 注意：日志格式是填充的，例如 "wte                     : 25,165,824"
        # so we grep for "^key " (key at start of line followed by space) to avoid false matches
        # 所以我们 grep "^key "（行首的 key 后跟空格）以避免错误匹配
        PARAMS_WTE=$(grep "^wte " "$LOG_FILE" | tail -1 | grep -oP '[\d,]+' | tr -d ',')
        PARAMS_VE=$(grep "^value_embeds " "$LOG_FILE" | tail -1 | grep -oP '[\d,]+' | tr -d ',')
        PARAMS_LM=$(grep "^lm_head " "$LOG_FILE" | tail -1 | grep -oP '[\d,]+' | tr -d ',')
        PARAMS_TRANSFORMER=$(grep "^transformer_matrices " "$LOG_FILE" | tail -1 | grep -oP '[\d,]+' | tr -d ',')
        PARAMS_SCALARS=$(grep "^scalars " "$LOG_FILE" | tail -1 | grep -oP '[\d,]+' | tr -d ',')
        PARAMS_TOTAL=$(grep "^total " "$LOG_FILE" | tail -1 | grep -oP '[\d,]+' | tr -d ',')

        NUM_ITERS=$(grep "Calculated number of iterations" "$LOG_FILE" | tail -1 | sed 's/.*: //' | tr -d ',')
        # Calculate tokens trained (iterations * batch_size, default 524288)
        # 计算训练的 token 数（迭代次数 * batch_size，默认 524288）
        TOKENS_TRAINED=$((NUM_ITERS * 524288))
        # Model dim
        # 模型维度
        MODEL_DIM=$((d * 64))
        # Val BPB from final eval
        # 最终评估的 Val BPB
        VAL_BPB=$(grep "Validation bpb:" "$LOG_FILE" | tail -1 | grep -oP '[\d.]+$')

        # Extract CORE score from training log (evaluated on final step)
        # 从训练日志中提取 CORE 分数（在最后一步评估）
        CORE_SCORE=$(grep "CORE metric:" "$LOG_FILE" | tail -1 | awk '{print $NF}')
        if [ -z "$CORE_SCORE" ]; then
            log "WARNING: Could not extract CORE score for d=$d"
            log "警告：无法提取 d=$d 的 CORE 分数"
            CORE_SCORE="0.0"
        fi

        log "  Params: $PARAMS_TOTAL (transformer: $PARAMS_TRANSFORMER), Iters: $NUM_ITERS, Val BPB: $VAL_BPB, CORE: $CORE_SCORE"
        log "  参数：$PARAMS_TOTAL（transformer：$PARAMS_TRANSFORMER），迭代：$NUM_ITERS，Val BPB：$VAL_BPB，CORE：$CORE_SCORE"

        # Append to CSV
        # 追加到 CSV
        echo "$flops,$d,$MODEL_DIM,$PARAMS_WTE,$PARAMS_VE,$PARAMS_LM,$PARAMS_TRANSFORMER,$PARAMS_SCALARS,$PARAMS_TOTAL,$NUM_ITERS,$TOKENS_TRAINED,$VAL_BPB,$CORE_SCORE,$TRAIN_TIME" >> "$RESULTS_FILE"
    done
done

log "=============================================="
log "Scaling Laws Sweep Complete"
log "缩放定律扫描完成"
log "=============================================="
log "Results saved to: $RESULTS_FILE"
log "结果保存到：$RESULTS_FILE"
echo ""
echo "Results:"
echo "结果："
column -t -s',' "$RESULTS_FILE"

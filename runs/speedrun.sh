#!/bin/bash

# This script is configured to train your own GPT-2 grade LLM (pretraining + finetuning)
# 此脚本配置用于训练您自己的 GPT-2 级别 LLM（预训练 + 微调）
# It is designed to run on a blank 8XH100 GPU node and takes approximately 3 hours to complete.
# 设计用于在全新的 8XH100 GPU 节点上运行，大约需要 3 小时完成。

# 1) Example launch (simplest):
# 1) 示例启动（最简单）：
# bash runs/speedrun.sh
# 2) Example launch in a screen session (because the run takes ~3 hours):
# 2) 在 screen 会话中启动示例（因为运行需要约 3 小时）：
# screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh
# 3) Example launch with wandb logging, but see below for setting up wandb first:
# 3) 使用 wandb 日志记录的启动示例，但请先参阅下面设置 wandb：
# WANDB_RUN=speedrun screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh

# Default intermediate artifacts directory is in ~/.cache/nanochat
# 默认中间产物目录在 ~/.cache/nanochat
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# -----------------------------------------------------------------------------
# Python venv setup with uv
# 使用 uv 设置 Python 虚拟环境

# install uv (if not already installed)
# 安装 uv（如果尚未安装）
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# create a .venv local virtual environment (if it doesn't exist)
# 创建 .venv 本地虚拟环境（如果不存在）
[ -d ".venv" ] || uv venv
# install the repo dependencies
# 安装仓库依赖
uv sync --extra gpu
# activate venv so that `python` uses the project's venv instead of system python
# 激活 venv，使 `python` 使用项目的 venv 而不是系统 python
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup
# wandb 设置
# If you wish to use wandb for logging (it's nice!, recommended).
# 如果您想使用 wandb 进行日志记录（它很好！推荐）。
# 1) Make sure to first log in to wandb, e.g. run:
# 1) 确保先登录 wandb，例如运行：
#    `wandb login`
# 2) Set the WANDB_RUN environment variable when running this script, e.g.:
# 2) 运行此脚本时设置 WANDB_RUN 环境变量，例如：
#    `WANDB_RUN=d26 bash speedrun.sh`
if [ -z "$WANDB_RUN" ]; then
    # by default use "dummy" : it's handled as a special case, skips logging to wandb
    # 默认使用 "dummy"：它作为特殊情况处理，跳过 wandb 日志记录
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
# 在运行过程中，我们将把 markdown 报告写入基础目录中的 report/ 目录。
# 此命令清除它并写入标题部分，包含一堆系统信息和标记运行开始的时间戳。
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer
# 分词器

# Download the first ~2B characters of pretraining dataset
# 下载预训练数据集的前约 2B 个字符
# each data shard is ~250M chars
# 每个数据分片约 250M 个字符
# so we download 2e9 / 250e6 = 8 data shards at this point
# 所以此时我们下载 2e9 / 250e6 = 8 个数据分片
# each shard is ~100MB of text (compressed), so this is about ~800MB of data on disk
# 每个分片约 100MB 文本（压缩后），所以磁盘上约 800MB 数据
# look at dev/repackage_data_reference.py for details on how this data was prepared
# 有关此数据如何准备的详细信息，请参阅 dev/repackage_data_reference.py
python -m nanochat.dataset -n 8
# Immediately also kick off downloading more shards in the background while tokenizer trains
# 同时在后台立即开始下载更多分片，同时分词器训练
# Approximately 150 shards are needed for GPT-2 capability pretraining, add 20 for padding.
# GPT-2 能力预训练大约需要 150 个分片，添加 20 个作为填充。
# The maximum total number of shards available in the entire dataset is 6542.
# 整个数据集中可用的最大总分片数为 6542。
python -m nanochat.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!
# train the tokenizer with vocab size 2**15 = 32768 on ~2B characters of data
# 在约 2B 个字符的数据上训练词汇量为 2**15 = 32768 的分词器
python -m scripts.tok_train
# evaluate the tokenizer (report compression ratio etc.)
# 评估分词器（报告压缩比等）
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)
# 基础模型（预训练）
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

# d24 model (slightly undertrained to beat GPT-2 => decrease data:params ratio from compute optimal 10.5 (default) to 8)
# d24 模型（略微欠训练以击败 GPT-2 => 将 data:params 比率从计算最优 10.5（默认）降低到 8）
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=24 --target-param-data-ratio=8 --device-batch-size=16 --fp8 --run=$WANDB_RUN
# evaluate the model: CORE metric, BPB on train/val, and draw samples
# 评估模型：CORE 指标、训练/验证集上的 BPB，并生成样本
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- --device-batch-size=16

# -----------------------------------------------------------------------------
# SFT (teach the model conversation special tokens, tool use, multiple choice)
# SFT（教模型对话特殊 token、工具使用、多项选择）

# download 2.3MB of synthetic identity conversations to impart a personality to nanochat
# 下载 2.3MB 的合成身份对话，为 nanochat 赋予个性
# see dev/gen_synthetic_data.py for details on how this data was prepared and to get a sense of how you can easily tune it
# 有关此数据如何准备的详细信息以及如何轻松调整它，请参阅 dev/gen_synthetic_data.py
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# run SFT and eval the model
# 运行 SFT 并评估模型
torchrun --standalone --nproc_per_node=8 -m scripts.chat_sft -- --device-batch-size=16 --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft

# chat with the model over CLI! Leave out the -p to chat interactively
# 通过 CLI 与模型聊天！省略 -p 以交互式聊天
# python -m scripts.chat_cli -p "Why is the sky blue?"

# even better, chat with your model over a pretty WebUI ChatGPT style
# 更好的是，通过漂亮的 WebUI ChatGPT 风格与您的模型聊天
# python -m scripts.chat_web

# -----------------------------------------------------------------------------
# Generate the full report by putting together all the sections
# 通过组合所有部分生成完整报告
# report.md is the output and will be copied to current directory for convenience
# report.md 是输出，将被复制到当前目录以方便使用
python -m nanochat.report generate

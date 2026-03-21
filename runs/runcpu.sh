#!/bin/bash

# Showing an example run for exercising some of the code paths on the CPU (or MPS on Macbooks)
# 展示在 CPU（或 Macbook 上的 MPS）上运行一些代码路径的示例
# This script was last updated/tuned on Jan 17, 2026.
# 此脚本最后更新/调整于 2026 年 1 月 17 日。

# Run as:
# 运行方式：
# bash runs/runcpu.sh

# NOTE: Training LLMs requires GPU compute and $$$. You will not get far on your Macbook.
# 注意：训练 LLM 需要 GPU 计算和 $$$。你在 Macbook 上走不了多远。
# Think of this run as educational/fun demo, not something you should expect to work well.
# 把这次运行当作教育/有趣的演示，而不是你应该期望运行良好的东西。
# You may also want to run this script manually and one by one, copy pasting commands into your terminal.
# 您可能还想手动运行此脚本，逐个将命令复制粘贴到终端中。

# all the setup stuff
# 所有设置内容
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra cpu
source .venv/bin/activate
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

# train tokenizer on ~2B characters (~34 seconds on my MacBook Pro M3 Max)
# 在约 2B 个字符上训练分词器（在我的 MacBook Pro M3 Max 上约 34 秒）
python -m nanochat.dataset -n 8
python -m scripts.tok_train --max-chars=2000000000
python -m scripts.tok_eval

# train a small 4 layer model
# 训练一个小的 4 层模型
# I tuned this run to complete in about 30 minutes on my MacBook Pro M3 Max.
# 我调整了这次运行，使其在我的 MacBook Pro M3 Max 上大约 30 分钟内完成。
# To get better results, try increasing num_iterations, or get other ideas from your favorite LLM.
# 要获得更好的结果，尝试增加 num_iterations，或从您喜欢的 LLM 获取其他想法。
python -m scripts.base_train \
    --depth=6 \
    --head-dim=64 \
    --window-pattern=L \
    --max-seq-len=512 \
    --device-batch-size=32 \
    --total-batch-size=16384 \
    --eval-every=100 \
    --eval-tokens=524288 \
    --core-metric-every=-1 \
    --sample-every=100 \
    --num-iterations=5000 \
    --run=$WANDB_RUN
python -m scripts.base_eval --device-batch-size=1 --split-tokens=16384 --max-per-task=16

# SFT (~10 minutes on my MacBook Pro M3 Max)
# SFT（在我的 MacBook Pro M3 Max 上约 10 分钟）
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl
python -m scripts.chat_sft \
    --max-seq-len=512 \
    --device-batch-size=32 \
    --total-batch-size=16384 \
    --eval-every=200 \
    --eval-tokens=524288 \
    --num-iterations=1500 \
    --run=$WANDB_RUN

# Chat with the model over CLI
# 通过 CLI 与模型聊天
# The model should be able to say that it is Paris.
# 模型应该能够说它是巴黎。
# It might even know that the color of the sky is blue.
# 它甚至可能知道天空的颜色是蓝色的。
# Sometimes the model likes it if you first say Hi before you ask it questions.
# 有时如果你在问问题之前先说 Hi，模型会喜欢。
# python -m scripts.chat_cli -p "What is the capital of France?"

# Chat with the model over a pretty WebUI ChatGPT style
# 通过漂亮的 WebUI ChatGPT 风格与模型聊天
# python -m scripts.chat_web

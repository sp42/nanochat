"""
Reinforcement learning on GSM8K via "GRPO".
在GSM8K上通过"GRPO"进行强化学习。

I put GRPO in quotes because we actually end up with something a lot
simpler and more similar to just REINFORCE:
我把GRPO放在引号里，因为我们实际上最终得到了更简单、更类似于REINFORCE的东西：

1) Delete trust region, so there is no KL regularization to a reference model
1) 删除信任区域，因此没有对参考模型的KL正则化
   (the original GRPO paper uses a trust region to prevent the model from
   deviating too far from the reference model, but we don't use this)
   （原始GRPO论文使用信任区域来防止模型偏离参考模型太远，但我们不使用这个）
2) Delete group aspect, so we only sample a single response per prompt
2) 删除组方面，因此我们每个提示只采样单个响应
   (the original GRPO paper samples multiple responses per prompt and compares
   them, but we only sample one)
   （原始GRPO论文每个提示采样多个响应并进行比较，但我们只采样一个）
3) Delete clipping, so we don't clip the importance sampling ratio
3) 删除裁剪，因此我们不裁剪重要性采样比率
   (the original GRPO paper clips the ratio to prevent too large updates,
   but we don't use this)
   （原始GRPO论文裁剪比率以防止过大的更新，但我们不使用这个）

The result is just REINFORCE with a simple baseline that is the mean reward
of the batch. This is a very simple and effective algorithm.
结果只是带有简单基线的REINFORCE，该基线是批次的平均奖励。这是一个非常简单有效的算法。

Example usage:
用法示例：
torchrun --nproc_per_node=8 -m scripts.chat_rl
"""

import argparse
import os
import itertools
import wandb
import torch
import torch.distributed as dist
from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, DummyWandb, autodetect_device_type
from nanochat.checkpoint_manager import save_checkpoint, load_model
from nanochat.engine import Engine
from tasks.gsm8k import GSM8K

# -----------------------------------------------------------------------------
# CLI arguments
# 命令行参数
parser = argparse.ArgumentParser(description="Reinforcement learning on GSM8K")
# Logging
# 日志
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
parser.add_argument("--run", type=str, default="dummy", help="wandb 运行名称（'dummy' 禁用 wandb 日志）")
# Runtime
# 运行时
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps（空 = 自动检测）")
# Model loading
# 模型加载
parser.add_argument("--model-tag", type=str, default=None, help="model tag to load from")
parser.add_argument("--model-tag", type=str, default=None, help="要加载的模型标签")
parser.add_argument("--model-step", type=int, default=None, help="model step to load from")
parser.add_argument("--model-step", type=int, default=None, help="要加载的模型步数")
# Training horizon
# 训练周期
parser.add_argument("--num-epochs", type=int, default=1, help="number of epochs over GSM8K")
parser.add_argument("--num-epochs", type=int, default=1, help="在 GSM8K 上的训练轮数")
# Batch sizes / sampling
# 批量大小 / 采样
parser.add_argument("--device-batch-size", type=int, default=8, help="max batch size per forward pass")
parser.add_argument("--device-batch-size", type=int, default=8, help="每次前向传播的最大批量大小")
parser.add_argument("--examples-per-step", type=int, default=16, help="total examples per optimization step across all ranks")
parser.add_argument("--examples-per-step", type=int, default=16, help="所有 rank 每个优化步骤的总示例数")
parser.add_argument("--num-samples", type=int, default=16, help="number of samples per example/question")
parser.add_argument("--num-samples", type=int, default=16, help="每个示例/问题的采样数")
# Generation
# 生成
parser.add_argument("--max-new-tokens", type=int, default=256, help="max tokens to generate per sample")
parser.add_argument("--max-new-tokens", type=int, default=256, help="每个样本生成的最大 token 数")
parser.add_argument("--temperature", type=float, default=1.0, help="sampling temperature")
parser.add_argument("--temperature", type=float, default=1.0, help="采样温度")
parser.add_argument("--top-k", type=int, default=50, help="top-k sampling (0 = disabled)")
parser.add_argument("--top-k", type=int, default=50, help="top-k 采样（0 = 禁用）")
# Optimization
# 优化
parser.add_argument("--embedding-lr", type=float, default=0.2, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--embedding-lr", type=float, default=0.2, help="嵌入参数的学习率（Adam）")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="反嵌入参数的学习率（Adam）")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="矩阵参数的学习率（Muon）")
parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay for embedding/unembedding parameters (Adam)")
parser.add_argument("--weight-decay", type=float, default=0.0, help="嵌入/反嵌入参数的权重衰减（Adam）")
parser.add_argument("--init-lr-frac", type=float, default=0.05, help="initial LR as fraction of base LR")
parser.add_argument("--init-lr-frac", type=float, default=0.05, help="初始学习率占基础学习率的比例")
# Evaluation / checkpointing
# 评估 / 检查点
parser.add_argument("--eval-every", type=int, default=60, help="evaluate pass@k every N steps")
parser.add_argument("--eval-every", type=int, default=60, help="每 N 步评估 pass@k")
parser.add_argument("--eval-examples", type=int, default=400, help="number of examples for pass@k evaluation")
parser.add_argument("--eval-examples", type=int, default=400, help="pass@k 评估的示例数")
parser.add_argument("--save-every", type=int, default=60, help="save checkpoint every N steps")
parser.add_argument("--save-every", type=int, default=60, help="每 N 步保存检查点")
args = parser.parse_args()
user_config = vars(args).copy()
# -----------------------------------------------------------------------------

# Init compute/precision
# 初始化计算/精度
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
                               # 此进程将进行日志记录、检查点等操作

# wandb logging init
# wandb 日志初始化
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-rl", name=args.run, config=user_config)

# Init model and tokenizer
# 初始化模型和分词器
model, tokenizer, meta = load_model("sft", device, phase="eval", model_tag=args.model_tag, step=args.model_step)
engine = Engine(model, tokenizer) # for sampling rollouts
                                   # 用于采样轨迹

# -----------------------------------------------------------------------------
# Rollout / sampling generator loop that yields batches of examples for training
# 生成训练样本批次的轨迹/采样生成器循环
train_task = GSM8K(subset="main", split="train")
val_task = GSM8K(subset="main", split="test")
num_steps = (len(train_task) // args.examples_per_step) * args.num_epochs
print0(f"Calculated number of steps: {num_steps}")
print0(f"计算得到的步数: {num_steps}")

@torch.no_grad()
def get_batch():
    assistant_end = tokenizer.encode_special("<|assistant_end|>") # ok to use this token, it's only for padding and isn't used in the loss.
                                                                  # 可以使用此 token，它仅用于填充，不参与损失计算
    rank_indices = range(ddp_rank, len(train_task), ddp_world_size) # each rank is responsible for different examples in the training data
                                                                    # 每个 rank 负责训练数据中的不同示例
    for example_idx in itertools.cycle(rank_indices):

        # First get the full conversation of both user and assistant messages
        # 首先获取用户和助手消息的完整对话
        conversation = train_task[example_idx]

        # Tokenize the conversation, deleting the last Assistant message and priming the Assistant for a completion instead
        # 对对话进行分词，删除最后的助手消息，并为助手准备补全
        # (i.e. keep the <|assistant_start|>, but delete everything after it)
        # （即保留 <|assistant_start|>，但删除其后的所有内容）
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)

        # Generate num_samples samples using batched generation, use loop to avoid OOMs
        # 使用批量生成生成 num_samples 个样本，使用循环避免 OOM
        model.eval() # ensure the model is in eval mode
                     # 确保模型处于评估模式
        generated_token_sequences = []
        masks = []
        num_sampling_steps = args.num_samples // args.device_batch_size # go sequentially to prevent OOMs
                                                                         # 顺序执行以防止 OOM
        for sampling_step in range(num_sampling_steps):
            seed = hash((step, example_idx, sampling_step)) & 0x7FFFFFFF # positive half of int32
                                                                         # int32 的正半部分
            generated_token_sequences_batch, masks_batch = engine.generate_batch(
                tokens,
                num_samples=args.device_batch_size,
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                seed=seed, # must make sure to change the seed for each sampling step
                           # 必须确保每个采样步骤更改种子
            )
            generated_token_sequences.extend(generated_token_sequences_batch)
            masks.extend(masks_batch)

        # Calculate the rewards for each sample
        # 计算每个样本的奖励
        rewards = []
        for sample_tokens in generated_token_sequences:
            # Get just the generated tokens (after the prompt)
            # 仅获取生成的 token（提示之后）
            generated_tokens = sample_tokens[prefix_length:]
            # Decode the generated response
            # 解码生成的响应
            generated_text = tokenizer.decode(generated_tokens)
            # Calculate the reward
            # 计算奖励
            reward = train_task.reward(conversation, generated_text)
            rewards.append(reward)

        # Pad the sequences so that their lengths (in time) match
        # 填充序列使其长度（时间上）匹配
        max_length = max(len(seq) for seq in generated_token_sequences)
        padded_generated_token_sequences = [seq + [assistant_end] * (max_length - len(seq)) for seq in generated_token_sequences]
        padded_masks = [mask + [0] * (max_length - len(mask)) for mask in masks]
        # Stack up the sequences and masks into PyTorch tensors
        # 将序列和掩码堆叠成 PyTorch 张量
        ids = torch.tensor(padded_generated_token_sequences, dtype=torch.long, device=device)
        mask_ids = torch.tensor(padded_masks, dtype=torch.long, device=device)
        # Generate autoregressive inputs and targets to the Transformer
        # 生成 Transformer 的自回归输入和目标
        inputs = ids[:, :-1]
        targets = ids[:, 1:].clone() # clone to avoid in-place modification:
                                     # 克隆以避免原地修改
        targets[mask_ids[:, 1:] == 0] = -1 # <-- inplace modification right here. -1 is the ignore index
                                           # <-- 这里进行原地修改。-1 是忽略索引
        # NOTE also that the Engine returns mask=0 for BOTH the prompt tokens AND the tool use tokens.
        # 注意：Engine 对提示 token 和工具使用 token 都返回 mask=0。
        # So we will (correctly) end up not training on the prompt tokens, or the tool use forced tokens.
        # 因此我们将（正确地）不在提示 token 或工具使用强制 token 上进行训练。
        rewards = torch.tensor(rewards, dtype=torch.float, device=device)
        # Calculate the advantages by simply subtracting the mean (instead of z-score (x-mu)/sigma)
        # 通过简单减去均值来计算优势（而不是 z-score (x-mu)/sigma）
        mu = rewards.mean()
        advantages = rewards - mu
        # yield inputs/targets as (B, T) of ids and rewards as (B,) of floats
        # 将输入/目标作为 (B, T) 的 id 和奖励作为 (B,) 的浮点数 yield
        yield generated_token_sequences, inputs, targets, rewards, advantages

# -----------------------------------------------------------------------------
# Simple evaluation loop for GSM8K pass@k
# GSM8K pass@k 的简单评估循环
def run_gsm8k_eval(task, tokenizer, engine,
    max_examples=None,
    num_samples=1,
    max_completion_tokens=256,
    temperature=0.0,
    top_k=50
):
    """
    Evaluates GSM8K task and returns a list of records of evaluation outcomes.
    评估 GSM8K 任务并返回评估结果记录列表。
    In a distributed setting, all ranks cooperate but this function will NOT
    do the reduction across ranks. This is the responsibility of the caller.
    在分布式设置中，所有 rank 协作，但此函数不会跨 rank 进行归约。这是调用者的责任。
    Because the evaluation can take a while, this function will yield records one by one.
    由于评估可能需要一些时间，此函数将逐个 yield 记录。
    """
    max_examples = min(max_examples, len(task)) if max_examples is not None else len(task)
    for idx in range(ddp_rank, max_examples, ddp_world_size):
        conversation = task[idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)
        # Generate k samples using batched generation inside the Engine
        # 使用 Engine 内的批量生成生成 k 个样本
        assert num_samples <= args.device_batch_size # usually this is true. we can add a loop if not...
                                                     # 通常这是真的。如果不是，我们可以添加一个循环...
        generated_token_sequences, masks = engine.generate_batch(
            tokens,
            num_samples=num_samples,
            max_tokens=max_completion_tokens,
            temperature=temperature,
            top_k=top_k
        )
        # Check each sample for correctness
        # 检查每个样本的正确性
        outcomes = []
        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            is_correct = task.evaluate(conversation, generated_text)
            outcomes.append({
                "is_correct": is_correct
            })
        # A bit bloated because I wanted to do more complex logging at one point.
        # 有点臃肿，因为我曾经想做更复杂的日志记录。
        record = {
            "idx": idx,
            "outcomes": outcomes,
        }
        yield record

# -----------------------------------------------------------------------------
# Training loop
# 训练循环

# Init the optimizer
# 初始化优化器
optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=args.weight_decay,
)

# Set the initial learning rate as a fraction of the base learning rate
# 将初始学习率设置为基础学习率的一部分
for group in optimizer.param_groups:
    group["lr"] = group["lr"] * args.init_lr_frac
    group["initial_lr"] = group["lr"]

# Learning rate scheduler: simple rampdown to zero over num_steps
# 学习率调度器：在 num_steps 内简单递减到零
def get_lr_multiplier(it):
    lrm = 1.0 - it / num_steps
    return lrm

# Calculate the number of examples each rank handles to achieve the desired examples_per_step
# 计算每个 rank 处理的示例数以达到所需的 examples_per_step
print0(f"Total sequences per step: {args.examples_per_step * args.num_samples}") # total batch size in sequences/step
                                                                                  # 每步序列的总批量大小
assert args.examples_per_step % ddp_world_size == 0, "Desired examples per step must be divisible by the number of ranks"
assert args.examples_per_step % ddp_world_size == 0, "每步期望的示例数必须能被 rank 数整除"
examples_per_rank = args.examples_per_step // ddp_world_size # per GPU
                                                             # 每个 GPU
print0(f"Calculated examples per rank: {examples_per_rank}")
print0(f"计算得到的每个 rank 示例数: {examples_per_rank}")

# Kick off the training loop
# 启动训练循环
batch_iterator = get_batch()
for step in range(num_steps):

    # Evaluate the model once in a while and log to wandb
    # 定期评估模型并记录到 wandb
    if step % args.eval_every == 0:
        model.eval()
        passk = torch.zeros(args.device_batch_size, device=device) # pass@k for k=1..device_batch_size
                                                                   # k=1..device_batch_size 的 pass@k
        records_iter = run_gsm8k_eval(val_task, tokenizer, engine, num_samples=args.device_batch_size, max_examples=args.eval_examples, temperature=1.0)
        records = list(records_iter) # collect all records
                                     # 收集所有记录
        for k in range(1, args.device_batch_size + 1):
            passk[k - 1] = sum(any(o["is_correct"] for o in r["outcomes"][:k]) for r in records)
        num_records = torch.tensor(len(records), dtype=torch.long, device=device)
        if ddp:
            dist.all_reduce(num_records, op=dist.ReduceOp.SUM)
            dist.all_reduce(passk, op=dist.ReduceOp.SUM)
        passk = passk / num_records.item() # normalize by the total number of records
                                           # 按总记录数归一化
        print_passk = [f"Pass@{k}: {passk[k - 1].item():.4f}" for k in range(1, args.device_batch_size + 1)]
        print0(f"Step {step} | {', '.join(print_passk)}")
        log_passk = {f"pass@{k}": passk[k - 1].item() for k in range(1, args.device_batch_size + 1)}
        wandb_run.log({
            "step": step,
            **log_passk,
        })

    # Forward/Backward on rollouts over multiple examples in the dataset
    # 在数据集的多个示例上进行轨迹的前向/反向传播
    rewards_list = []
    sequence_lengths = []
    for example_step in range(examples_per_rank):
        # Get one batch corresponding to one example in the training dataset
        # 获取训练数据集中一个示例对应的一个批次
        sequences_all, inputs_all, targets_all, rewards_all, advantages_all = next(batch_iterator)
        # Evaluate the loss and gradients
        # 计算损失和梯度
        model.train() # ensure the model is in train mode
                      # 确保模型处于训练模式
        # We need one more loop because we can never exceed the device_batch_size
        # 我们需要再一个循环，因为我们不能超过 device_batch_size
        assert inputs_all.size(0) % args.device_batch_size == 0
        num_passes = inputs_all.size(0) // args.device_batch_size
        for pass_idx in range(num_passes):
            # Pluck out the batch for this pass
            # 提取此次传递的批次
            b0, b1 = pass_idx * args.device_batch_size, (pass_idx + 1) * args.device_batch_size
            inputs = inputs_all[b0:b1]
            targets = targets_all[b0:b1]
            rewards = rewards_all[b0:b1]
            advantages = advantages_all[b0:b1]
            # Calculate log probabilities. Note that the loss calculates NLL = -logp, so we negate
            # 计算对数概率。注意损失计算 NLL = -logp，所以我们取负
            logp = -model(inputs, targets, loss_reduction='none').view_as(inputs) # (B, T)
            # Calculate the PG objective. Note that ignore_index=-1 ensures that invalid tokens have loss 0.
            # 计算策略梯度目标。注意 ignore_index=-1 确保无效 token 的损失为 0。
            pg_obj = (logp * advantages.unsqueeze(-1)).sum()
            # normalize by the number of valid tokens, number of passes, and examples_per_rank
            # 按有效 token 数、传递数和 examples_per_rank 归一化
            num_valid = (targets >= 0).sum().clamp(min=1)
            pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)
            # Note, there is no need to add PPO ratio+clip because we are on policy
            # 注意，不需要添加 PPO ratio+clip，因为我们是同策略的
            # Finally, formulate the loss that we want to minimize (instead of objective we wish to maximize)
            # 最后，制定我们要最小化的损失（而不是我们希望最大化的目标）
            loss = -pg_obj
            loss.backward()
            print0(f"Step {step}/{num_steps} | Example step {example_step} | Pass {pass_idx} | loss: {loss.item():.6f} | Average reward: {rewards.mean().item()}")
        # For logging
        # 用于日志记录
        rewards_list.append(rewards_all.mean().item())
        sequence_lengths.extend(len(seq) for seq in sequences_all)

    # A bunch of logging for how the rollouts went this step
    # 关于这一步轨迹如何的一系列日志记录
    mean_reward = sum(rewards_list) / len(rewards_list)
    mean_sequence_length = sum(sequence_lengths) / len(sequence_lengths)
    if ddp: # aggregate across ranks
            # 跨 rank 聚合
        mean_reward_tensor = torch.tensor(mean_reward, dtype=torch.float, device=device)
        mean_sequence_length_tensor = torch.tensor(mean_sequence_length, dtype=torch.float, device=device)
        dist.all_reduce(mean_reward_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(mean_sequence_length_tensor, op=dist.ReduceOp.AVG)
        mean_reward = mean_reward_tensor.item()
        mean_sequence_length = mean_sequence_length_tensor.item()
    print0(f"Step {step}/{num_steps} | Average reward: {mean_reward} | Average sequence length: {mean_sequence_length:.2f}")
    wandb_run.log({
        "step": step,
        "reward": mean_reward,
        "sequence_length": mean_sequence_length,
    })

    # Update the model parameters
    # 更新模型参数
    lrm = get_lr_multiplier(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
    optimizer.step()
    model.zero_grad(set_to_none=True)
    wandb_run.log({
        "step": step,
        "lrm": lrm,
    })

    # Master process saves the model once in a while. Skip first step. Save last step.
    # 主进程定期保存模型。跳过第一步。保存最后一步。
    if master_process and ((step > 0 and step % args.save_every == 0) or step == num_steps - 1):
        base_dir = get_base_dir()
        depth = model.config.n_layer
        output_dirname = args.model_tag if args.model_tag else f"d{depth}" # base the model tag on the depth of the base model
                                                                           # 根据基础模型的深度确定模型标签
        checkpoint_dir = os.path.join(base_dir, "chatrl_checkpoints", output_dirname)
        model_config_kwargs = model.config.__dict__ # slightly naughty, abusing the simplicity of GPTConfig, TODO nicer
                                                    # 有点不规范，利用了 GPTConfig 的简单性，TODO 改进
        save_checkpoint(
            checkpoint_dir,
            step,
            model.state_dict(),
            None, # note: we don't bother to save the optimizer state
                   # 注意：我们不保存优化器状态
            {
                "model_config": model_config_kwargs,
            }
        )
        print(f"✅ Saved model checkpoint to {checkpoint_dir}")
        print(f"✅ 模型检查点已保存到 {checkpoint_dir}")

# Log to report
# 记录到报告
from nanochat.report import get_report
get_report().log(section="Chat RL", data=[
    user_config, # CLI args
                  # 命令行参数
])

wandb_run.finish() # wandb run finish
                   # wandb 运行结束
compute_cleanup()

"""
A number of functions that help with evaluating a base model.
一些帮助评估基础模型的函数。
"""
import math
import torch
import torch.distributed as dist

@torch.no_grad()
def evaluate_bpb(model, batches, steps, token_bytes):
    """
    Instead of the naive 'mean loss', this function returns the bits per byte (bpb),
    此函数返回每字节位数（bpb），而不是朴素的"平均损失"，
    which is a tokenization vocab size-independent metric, meaning you are still comparing
    这是一个与分词器词表大小无关的指标，意味着即使你改变
    apples:apples if you change the vocab size. The way this works is that instead of just
    词表大小，你仍然在进行同类比较。其工作方式是，不是简单地
    calculating the average loss as usual, you calculate the sum loss, and independently
    像往常一样计算平均损失，而是计算损失总和，并独立地
    also the sum bytes (of all the target tokens), and divide. This normalizes the loss by
    计算所有目标 token 的字节总和，然后相除。这通过
    the number of bytes that the target tokens represent.
    目标 token 表示的字节数来归一化损失。

    The added complexity is so that:
    增加的复杂性是为了：
    1) All "normal" tokens are normalized by the length of the token in bytes
    1) 所有"正常" token 按其字节长度归一化
    2) No special tokens (e.g. <|bos|>) are included in the metric - they are masked out.
    2) 不包括特殊 token（例如 <|bos|>）— 它们被屏蔽。
    3) No actively masked tokens (using ignore_index of e.g. -1) are included in the metric.
    3) 不包括主动屏蔽的 token（使用例如 -1 的 ignore_index）。

    In addition to evaluate_loss, we need the token_bytes tensor:
    除了 evaluate_loss，我们需要 token_bytes 张量：
    It is a 1D tensor of shape (vocab_size,), indicating the number of bytes for
    它是一个形状为 (vocab_size,) 的 1D 张量，指示
    each token id, or 0 if the token is to not be counted (e.g. special tokens).
    每个 token id 的字节数，如果不计数则为 0（例如特殊 token）。
    """
    # record the losses
    # 记录损失
    total_nats = torch.tensor(0.0, dtype=torch.float32, device=model.get_device())
    total_bytes = torch.tensor(0, dtype=torch.int64, device=model.get_device())
    batch_iter = iter(batches)
    for _ in range(steps):
        x, y = next(batch_iter)
        loss2d = model(x, y, loss_reduction='none') # (B, T)
        loss2d = loss2d.view(-1) # flatten
                                  # 展平
        y = y.view(-1) # flatten
                       # 展平
        if (y.int() < 0).any(): # mps does not currently have kernel for < 0 for int64, only int32
                                 # mps 目前没有 int64 < 0 的内核，只有 int32
            # slightly more complex code path if some target tokens are ignore_index (e.g. -1)
            # 如果某些目标 token 是 ignore_index（例如 -1），代码路径稍微复杂
            # any target token < 0 is to be ignored: do NOT index token_bytes with negatives
            # 任何 < 0 的目标 token 都要忽略：不要用负数索引 token_bytes
            valid = y >= 0
            y_safe = torch.where(valid, y, torch.zeros_like(y))
            # map valid targets to their byte length; ignored targets contribute 0 bytes
            # 将有效目标映射到其字节长度；被忽略的目标贡献 0 字节
            num_bytes2d = torch.where(
                valid,
                token_bytes[y_safe],
                torch.zeros_like(y, dtype=token_bytes.dtype)
            )
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()
        else:
            # fast path: no ignored targets, safe to index directly
            # 快速路径：没有忽略的目标，可以安全地直接索引
            num_bytes2d = token_bytes[y]
            total_nats += (loss2d * (num_bytes2d > 0)).sum()
            total_bytes += num_bytes2d.sum()
    # sum reduce across all ranks
    # 跨所有 rank 求和归约
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size > 1:
        dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)
    # move both to cpu, calculate bpb and return
    # 将两者移到 cpu，计算 bpb 并返回
    total_nats = total_nats.item()
    total_bytes = total_bytes.item()
    if total_bytes == 0:
        return float('inf')
    bpb = total_nats / (math.log(2) * total_bytes)
    return bpb

"""
Distributed dataloaders for pretraining.
用于预训练的分布式数据加载器。

BOS-aligned bestfit:
BOS 对齐的最佳适应：
   - Every row starts with BOS token
   - 每行以 BOS token 开头
   - Documents packed using best-fit algorithm to minimize cropping
   - 使用最佳适应算法打包文档以最小化裁剪
   - When no document fits remaining space, crops a document to fill exactly
   - 当没有文档适合剩余空间时，裁剪文档以精确填充
   - 100% utilization (no padding), ~35% tokens cropped at T=2048
   - 100% 利用率（无填充），在 T=2048 时约 35% token 被裁剪

Compared to the original tokenizing_distributed_data_loader:
与原始 tokenizing_distributed_data_loader 相比：
BOS-aligned loses ~35% of tokens to cropping, but ensures that
BOS 对齐会损失约 35% 的 token 到裁剪，但确保
there are fewer "confusing" tokens in the train/val batches as every token can
训练/验证批次中有更少的"令人困惑"的 token，因为每个 token 都
now attend back to the BOS token and sees the full context of the document.
可以回溯到 BOS token 并看到文档的完整上下文。

Fallback to the original if you have very limited data AND long documents:
如果您数据非常有限且文档很长，请回退到原始版本：
https://github.com/karpathy/nanochat/blob/3c3a3d7/nanochat/dataloader.py#L78-L117
"""

import torch
import pyarrow.parquet as pq

from nanochat.common import get_dist_info
from nanochat.dataset import list_parquet_files

def _document_batches(split, resume_state_dict, tokenizer_batch_size):
    """
    Infinite iterator over document batches (list of text strings) from parquet files.
    从 parquet 文件无限迭代文档批次（文本字符串列表）。

    Handles DDP sharding and approximate resume. Each yield is (text_batch, (pq_idx, rg_idx, epoch))
    处理 DDP 分片和近似恢复。每次产生 (text_batch, (pq_idx, rg_idx, epoch))
    where text_batch is a list of document strings, indices track position for resumption,
    其中 text_batch 是文档字符串列表，索引跟踪恢复位置，
    and epoch counts how many times we've cycled through the dataset (starts at 1).
    epoch 计算我们循环数据集的次数（从 1 开始）。
    """
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()

    warn_on_legacy = ddp_rank == 0 and split == "train" # rank 0 on train split will warn on legacy
                                                        # rank 0 在训练集上将警告旧版
    parquet_paths = list_parquet_files(warn_on_legacy=warn_on_legacy)
    assert len(parquet_paths) != 0, "No dataset parquet files found, did you run dataset.py?"
    parquet_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]

    resume_pq_idx = resume_state_dict["pq_idx"] if resume_state_dict is not None else 0
    resume_rg_idx = resume_state_dict["rg_idx"] if resume_state_dict is not None else None
    resume_epoch = resume_state_dict.get("epoch", 1) if resume_state_dict is not None else 1
    first_pass = True
    pq_idx = resume_pq_idx
    epoch = resume_epoch

    while True:  # iterate infinitely (multi-epoch)
                 # 无限迭代（多轮）
        pq_idx = resume_pq_idx if first_pass else 0
        while pq_idx < len(parquet_paths):
            filepath = parquet_paths[pq_idx]
            pf = pq.ParquetFile(filepath)
            # Start from resume point if resuming on same file, otherwise from DDP rank
            # 如果在同一文件上恢复则从恢复点开始，否则从 DDP rank 开始
            if first_pass and (resume_rg_idx is not None) and (pq_idx == resume_pq_idx):
                base_idx = resume_rg_idx // ddp_world_size
                base_idx += 1  # advance by 1 so we don't repeat data after resuming
                               # 前进 1 以便恢复后不重复数据
                rg_idx = base_idx * ddp_world_size + ddp_rank
                if rg_idx >= pf.num_row_groups:
                    pq_idx += 1
                    continue
                resume_rg_idx = None  # only do this once
                                      # 只执行一次
            else:
                rg_idx = ddp_rank
            while rg_idx < pf.num_row_groups:
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], (pq_idx, rg_idx, epoch)
                rg_idx += ddp_world_size
            pq_idx += 1
        first_pass = False
        epoch += 1


def tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer, B, T, split,
    tokenizer_threads=4, tokenizer_batch_size=128,
    device="cuda", resume_state_dict=None,
    buffer_size=1000
):
    """
    BOS-aligned dataloader with Best-Fit Cropping.
    BOS 对齐的数据加载器，使用最佳适应裁剪。

    Reduces token waste compared to simple greedy cropping by searching a buffer
    通过搜索缓冲区中适合的文档，相比简单贪婪裁剪减少 token 浪费
    for documents that fit well, while maintaining 100% utilization (no padding).
    同时保持 100% 利用率（无填充）。

    Algorithm for each row:
    每行的算法：
    1. From buffered docs, pick the LARGEST doc that fits entirely
    1. 从缓冲的文档中，选择能完全适合的最大文档
    2. Repeat until no doc fits
    2. 重复直到没有文档适合
    3. When nothing fits, crop a doc to fill remaining space exactly
    3. 当没有文档适合时，裁剪文档以精确填充剩余空间

    Key properties:
    关键属性：
    - Every row starts with BOS
    - 每行以 BOS 开头
    - 100% utilization (no padding, every token is trained on)
    - 100% 利用率（无填充，每个 token 都被训练）
    - Approximately 35% of all tokens are discarded due to cropping
    - 大约 35% 的 token 因裁剪而被丢弃
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    row_capacity = T + 1
    batches = _document_batches(split, resume_state_dict, tokenizer_batch_size)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    pq_idx, rg_idx, epoch = 0, 0, 1

    def refill_buffer():
        nonlocal pq_idx, rg_idx, epoch
        doc_batch, (pq_idx, rg_idx, epoch) = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
        for tokens in token_lists:
            doc_buffer.append(tokens)

    # Pre-allocate buffers once: layout is [inputs (B*T) | targets (B*T)]
    # 预分配缓冲区一次：布局是 [inputs (B*T) | targets (B*T)]
    # This gives us contiguous views and a single HtoD transfer
    # 这给我们连续的视图和单个 HtoD 传输
    use_cuda = device == "cuda"
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long) # for building rows without creating Python lists
                                                                   # 用于构建行而不创建 Python 列表
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda) # staging area (CPU)
                                                                                 # 暂存区（CPU）
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device) # on-device buffer
                                                                          # 设备上的缓冲区
    cpu_inputs = cpu_buffer[:B * T].view(B, T) # a few views into these buffers just for convenience
                                                # 这些缓冲区的几个视图，只是为了方便
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                # Ensure buffer has documents
                # 确保缓冲区有文档
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                # Find largest doc that fits entirely
                # 找到能完全适合的最大文档
                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    doc_len = len(doc)
                    row_buffer[row_idx, pos:pos + doc_len] = torch.tensor(doc, dtype=torch.long)
                    pos += doc_len
                else:
                    # No doc fits - crop shortest in buffer to fill remaining and minimize waste
                    # 没有文档适合 - 裁剪缓冲区中最短的文档以填充剩余空间并最小化浪费
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        # Copy to pinned CPU buffer, then single HtoD transfer
        # 复制到固定 CPU 缓冲区，然后单个 HtoD 传输
        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])

        state_dict = {"pq_idx": pq_idx, "rg_idx": rg_idx, "epoch": epoch}

        # Single HtoD copy into persistent GPU buffer and yield
        # 单个 HtoD 复制到持久 GPU 缓冲区并产生
        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        yield inputs, targets, state_dict

def tokenizing_distributed_data_loader_bos_bestfit(*args, **kwargs):
    """Helper that omits state_dict from yields."""
    """省略 state_dict 的辅助函数。"""
    for inputs, targets, state_dict in tokenizing_distributed_data_loader_with_state_bos_bestfit(*args, **kwargs):
        yield inputs, targets

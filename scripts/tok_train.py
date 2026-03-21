"""
Train a tokenizer using our own BPE Tokenizer library.
使用我们自己的 BPE 分词器库训练分词器。
In the style of GPT-4 tokenizer.
采用 GPT-4 分词器的风格。
"""
import os
import time
import argparse
import torch
from nanochat.tokenizer import RustBPETokenizer
from nanochat.common import get_base_dir
from nanochat.dataset import parquets_iter_batched

# -----------------------------------------------------------------------------
# Parse command line arguments
# 解析命令行参数

parser = argparse.ArgumentParser(description='Train a BPE tokenizer')
parser.add_argument('--max-chars', type=int, default=2_000_000_000, help='Maximum characters to train on (default: 10B)')
parser.add_argument('--doc-cap', type=int, default=10_000, help='Maximum characters per document (default: 10,000)')
parser.add_argument('--vocab-size', type=int, default=32768, help='Vocabulary size (default: 32768 = 2^15)')
args = parser.parse_args()
print(f"max_chars: {args.max_chars:,}")
print(f"doc_cap: {args.doc_cap:,}")
print(f"vocab_size: {args.vocab_size:,}")

# -----------------------------------------------------------------------------
# Text iterator
# 文本迭代器

def text_iterator():
    """
    1) Flatten the batches into a single iterator
    1) 将批次展平为单个迭代器
    2) Crop every document to args.doc_cap characters
    2) 将每个文档裁剪到 args.doc_cap 个字符
    3) Break when we've seen args.max_chars characters
    3) 当我们看到 args.max_chars 个字符时中断
    """
    nchars = 0
    for batch in parquets_iter_batched(split="train"):
        for doc in batch:
            doc_text = doc
            if len(doc_text) > args.doc_cap:
                doc_text = doc_text[:args.doc_cap]
            nchars += len(doc_text)
            yield doc_text
            if nchars > args.max_chars:
                return
text_iter = text_iterator()

# -----------------------------------------------------------------------------
# Train the tokenizer
# 训练分词器
t0 = time.time()
tokenizer = RustBPETokenizer.train_from_iterator(text_iter, args.vocab_size)
t1 = time.time()
train_time = t1 - t0
print(f"Training time: {train_time:.2f}s")

# -----------------------------------------------------------------------------
# Save the tokenizer to disk
# 将分词器保存到磁盘
base_dir = get_base_dir()
tokenizer_dir = os.path.join(base_dir, "tokenizer")
tokenizer.save(tokenizer_dir)

# -----------------------------------------------------------------------------
# Quick inline sanity check
# 快速内联健全性检查
test_text = """Hello world! This is a test.
Numbers: 123, 4567, 89
Contractions: I'm, you're, it's
Special chars: @#$%^&*()
Unicode: 你好世界 🌍"""
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
assert decoded == test_text

# -----------------------------------------------------------------------------
# One more thing: we wish to cache a mapping from token id to number of bytes of that token
# 还有一件事：我们希望缓存从 token id 到该 token 字节数的映射
# for efficient evaluation of bits per byte. Unlike the typical mean loss, this
# 以便高效评估每字节位数。与典型的平均损失不同，这
# allows us to report a loss that is invariant to the vocab size of the tokenizer.
# 允许我们报告一个与分词器词表大小无关的损失。
# The bits per byte on the validation set is then one of the primary metrics we care about.
# 验证集上的每字节位数是我们关心的主要指标之一。
vocab_size = tokenizer.get_vocab_size()
special_set = set(tokenizer.get_special_tokens())
token_strings = [tokenizer.decode([token_id]) for token_id in range(vocab_size)]
token_bytes = []
for token_id in range(vocab_size):
    token_str = token_strings[token_id] # the Python string representation of this token
                                        # 此 token 的 Python 字符串表示
    if token_str in special_set:
        token_bytes.append(0) # special characters are not counted
                              # 特殊字符不计入
    else:
        id_bytes = len(token_str.encode("utf-8")) # number of bytes that make up this token
                                                   # 组成此 token 的字节数
        token_bytes.append(id_bytes)
token_bytes = torch.tensor(token_bytes, dtype=torch.int32, device='cpu')
token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
with open(token_bytes_path, "wb") as f:
    torch.save(token_bytes, f)
print(f"Saved token_bytes to {token_bytes_path}")

# Log to report
# 记录到报告
from nanochat.report import get_report
token_bytes_nonzero = (token_bytes[token_bytes > 0]).to(dtype=torch.float32)
get_report().log(section="Tokenizer training", data=[
    vars(args), # argparse command line arguments
                # argparse 命令行参数
    {"train_time": train_time},
    {"num_special_tokens": len(special_set)},
    {
        "token_bytes_min": int(token_bytes_nonzero.min().item()),
        "token_bytes_max": int(token_bytes_nonzero.max().item()),
        "token_bytes_mean": token_bytes_nonzero.mean().item(),
        "token_bytes_std": token_bytes_nonzero.std().item(),
    }
])

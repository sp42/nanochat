"""
Test Engine class. Example run:
测试 Engine 类。示例运行：

python -m pytest tests/test_engine.py -v
"""

import torch
from nanochat.engine import KVCache, Engine
from dataclasses import dataclass


# -----------------------------------------------------------------------------
# Mock classes for testing Engine without loading a real model
# 用于测试 Engine 的 Mock 类，无需加载真实模型
# -----------------------------------------------------------------------------

@dataclass
class MockConfig:
    """Minimal config for Engine tests."""
    """Engine 测试的最小配置。"""
    n_kv_head: int = 4
    n_head: int = 4
    n_embd: int = 64
    n_layer: int = 2
    sequence_len: int = 128


class MockModel:
    """
    Mock model that returns uniform logits over the vocab.
    返回词表上均匀分布 logits 的 Mock 模型。
    This ensures that with temperature > 0, different samples should
    这确保当 temperature > 0 时，不同的样本应该
    (with very high probability) produce different tokens.
    （以极大概率）产生不同的 token。
    """
    def __init__(self, vocab_size=262):  # 256 bytes + 6 special tokens
                                            # 256 字节 + 6 个特殊 token
        self.vocab_size = vocab_size
        self.config = MockConfig()
        self._device = torch.device("cpu")

    def get_device(self):
        return self._device

    def forward(self, ids, kv_cache=None):
        """Return uniform logits so sampling is spread across vocab."""
        """返回均匀分布的 logits，使采样分布在词表上。"""
        B, T = ids.shape
        # With FA3, flash_attn_with_kvcache updates cache in-place and we advance position
        # 使用 FA3 时，flash_attn_with_kvcache 原地更新缓存，我们推进位置
        if kv_cache is not None:
            kv_cache.advance(T)
        # Uniform logits -> equal probability for all tokens
        # 均匀 logits -> 所有 token 概率相等
        logits = torch.zeros(B, T, self.vocab_size)
        return logits


class ByteTokenizer:
    """
    Simple byte-level tokenizer for testing.
    用于测试的简单字节级分词器。
    Tokens 0-255 are raw bytes, 256+ are special tokens.
    Token 0-255 是原始字节，256+ 是特殊 token。
    """
    def __init__(self):
        # Special tokens start at 256
        # 特殊 token 从 256 开始
        self._special_tokens = {
            "<|python_start|>": 256,
            "<|python_end|>": 257,
            "<|output_start|>": 258,
            "<|output_end|>": 259,
            "<|assistant_end|>": 260,
            "<|bos|>": 261,
        }
        self._bos = 261

    def encode_special(self, s):
        return self._special_tokens[s]

    def get_bos_token_id(self):
        return self._bos

    def encode(self, s, prepend=None):
        tokens = list(s.encode("utf-8"))  # bytes 0-255
                                            # 字节 0-255
        if prepend is not None:
            tokens = [prepend] + tokens
        return tokens

    def decode(self, tokens):
        # Filter out special tokens before decoding
        # 解码前过滤掉特殊 token
        byte_tokens = [t for t in tokens if t < 256]
        return bytes(byte_tokens).decode("utf-8", errors="replace")

def test_kv_cache_basic():
    """Test basic KVCache functionality for FA3."""
    """测试 FA3 的基本 KVCache 功能。"""
    batch_size = 2
    num_heads = 3
    seq_len = 64
    head_dim = 5
    num_layers = 6

    kv_cache = KVCache(
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        num_layers=num_layers,
        device="cpu",
        dtype=torch.float32,
    )

    # Check initial state
    # 检查初始状态
    assert kv_cache.get_pos() == 0
    assert kv_cache.k_cache.shape == (num_layers, batch_size, seq_len, num_heads, head_dim)
    assert kv_cache.v_cache.shape == (num_layers, batch_size, seq_len, num_heads, head_dim)

    # Test advance
    # 测试推进
    kv_cache.advance(10)
    assert kv_cache.get_pos() == 10

    kv_cache.advance(5)
    assert kv_cache.get_pos() == 15

    # Test reset
    # 测试重置
    kv_cache.reset()
    assert kv_cache.get_pos() == 0

    # Test get_layer_cache returns correct views
    # 测试 get_layer_cache 返回正确的视图
    k_layer0, v_layer0 = kv_cache.get_layer_cache(0)
    assert k_layer0.shape == (batch_size, seq_len, num_heads, head_dim)
    assert v_layer0.shape == (batch_size, seq_len, num_heads, head_dim)


def test_kv_cache_prefill():
    """Test KVCache.prefill() copies data correctly."""
    """测试 KVCache.prefill() 正确复制数据。"""
    batch_size = 1
    num_heads = 4
    head_dim = 8
    num_layers = 2

    # Create source cache and advance it
    # 创建源缓存并推进
    src_cache = KVCache(
        batch_size=batch_size, num_heads=num_heads, seq_len=32,
        head_dim=head_dim, num_layers=num_layers, device="cpu", dtype=torch.float32,
    )
    # Write some data to source cache
    # 向源缓存写入一些数据
    src_cache.k_cache[0, 0, :16, :, :] = 1.0
    src_cache.v_cache[0, 0, :16, :, :] = 2.0
    src_cache.advance(16)

    # Create destination cache with larger seq_len
    # 创建具有更大 seq_len 的目标缓存
    dst_cache = KVCache(
        batch_size=batch_size, num_heads=num_heads, seq_len=64,
        head_dim=head_dim, num_layers=num_layers, device="cpu", dtype=torch.float32,
    )

    # Prefill
    # 预填充
    dst_cache.prefill(src_cache)

    # Check position was copied
    # 检查位置已复制
    assert dst_cache.get_pos() == 16

    # Check data was copied
    # 检查数据已复制
    assert (dst_cache.k_cache[0, 0, :16, :, :] == 1.0).all()
    assert (dst_cache.v_cache[0, 0, :16, :, :] == 2.0).all()


def test_multi_sample_first_token_diversity():
    """
    Test that when generating multiple samples, each sample gets an independently
    测试生成多个样本时，每个样本获得独立
    sampled first token (not a broadcast of the same token to all rows).
    采样的第一个 token（而不是将相同 token 广播到所有行）。

    Previously, the first token after prefill was sampled once and broadcast to all
    以前，预填充后的第一个 token 被采样一次并广播到所有
    rows, causing all samples to start identically. The fix expands the prefill logits
    行，导致所有样本以相同方式开始。修复将预填充 logits 扩展
    to num_samples and samples independently for each row.
    到 num_samples 并为每行独立采样。

    With uniform logits over 262 tokens and 16 samples, the probability that all
    在 262 个 token 上均匀分布 logits 和 16 个样本的情况下，所有
    samples independently pick the same token is (1/262)^15 ≈ 10^-36. So if they're
    样本独立选择相同 token 的概率是 (1/262)^15 ≈ 10^-36。所以如果它们
    all identical, it indicates tokens are being broadcast instead of independently sampled.
    全部相同，则表示 token 正在被广播而不是独立采样。
    """
    model = MockModel(vocab_size=262)
    tokenizer = ByteTokenizer()
    engine = Engine(model, tokenizer)

    # Generate 16 samples with temperature=1.0 (stochastic sampling)
    # 使用 temperature=1.0（随机采样）生成 16 个样本
    prompt_tokens = [261, 72, 101, 108, 108, 111]  # <bos> + "Hello"
    num_samples = 16

    # Collect the first generated token from each sample
    # 收集每个样本的第一个生成 token
    first_tokens = []
    gen = engine.generate(
        prompt_tokens,
        num_samples=num_samples,
        max_tokens=1,  # We only need the first token
                       # 我们只需要第一个 token
        temperature=1.0,
        seed=42,
    )
    for token_column, token_masks in gen:
        first_tokens = token_column  # This is the first (and only) yield
                                     # 这是第一个（也是唯一一个）yield

    # With uniform distribution and 16 samples, they should NOT all be identical
    # 在均匀分布和 16 个样本的情况下，它们不应该全部相同
    # If they are all identical, the bug exists (broadcasting instead of sampling)
    # 如果它们全部相同，则存在 bug（广播而不是采样）
    unique_tokens = set(first_tokens)
    assert len(unique_tokens) > 1, (
        f"All {num_samples} samples got the same first token ({first_tokens[0]}). "
        f"所有 {num_samples} 个样本获得了相同的第一个 token ({first_tokens[0]})。"
        f"With uniform logits, this is statistically impossible (~10^-36 probability) "
        f"在均匀 logits 下，这在统计上是不可能的（约 10^-36 概率）"
        f"unless tokens are being broadcast instead of independently sampled."
        f"除非 token 正在被广播而不是独立采样。"
    )


def test_seed_reproducibility():
    """Same seed must produce identical output."""
    """相同的种子必须产生相同的输出。"""
    model = MockModel()
    engine = Engine(model, ByteTokenizer())
    prompt = [261, 72, 101, 108, 108, 111]  # <bos> + "Hello"

    for seed in [1, 42, 123, 999]:
        r1, _ = engine.generate_batch(prompt, max_tokens=5, seed=seed)
        r2, _ = engine.generate_batch(prompt, max_tokens=5, seed=seed)
        r3, _ = engine.generate_batch(prompt, max_tokens=5, seed=seed)
        assert r1 == r2 == r3, "Same seed must produce identical output for the same prompt."
        assert r1 == r2 == r3, "相同的种子必须为相同的提示产生相同的输出。"


def test_temperature_zero_determinism():
    """Temperature=0 is deterministic regardless of seed."""
    """Temperature=0 无论种子如何都是确定性的。"""
    model = MockModel()
    engine = Engine(model, ByteTokenizer())
    prompt = [261, 72, 101, 108, 108, 111]

    r1, _ = engine.generate_batch(prompt, temperature=0.0, max_tokens=5, seed=1)
    r2, _ = engine.generate_batch(prompt, temperature=0.0, max_tokens=5, seed=42)
    r3, _ = engine.generate_batch(prompt, temperature=0.0, max_tokens=5, seed=123)
    assert r1 == r2 == r3, "Temperature=0 must result in the same output for the same prompt regardless of seed."
    assert r1 == r2 == r3, "Temperature=0 必须为相同的提示产生相同的输出，无论种子如何。"


def test_max_tokens_respected():
    """Generation stops at max_tokens limit."""
    """生成在 max_tokens 限制处停止。"""
    model = MockModel()
    engine = Engine(model, ByteTokenizer())
    prompt = [261, 72, 101, 108, 108, 111]

    for max_tokens in [1, 4, 16, 64]:
        results, _ = engine.generate_batch(prompt, max_tokens=max_tokens)
        num_generated_tokens = len(results[0]) - len(prompt)
        assert num_generated_tokens <= max_tokens, f"Generated {num_generated_tokens} tokens, expected max_tokens={max_tokens} or less."
        assert num_generated_tokens <= max_tokens, f"生成了 {num_generated_tokens} 个 token，预期 max_tokens={max_tokens} 或更少。"


def test_num_samples_count():
    """num_samples=N produces exactly N sequences."""
    """num_samples=N 产生恰好 N 个序列。"""
    model = MockModel()
    engine = Engine(model, ByteTokenizer())
    prompt = [261, 72, 101, 108, 108, 111]

    for num_samples in [1, 4, 16, 64]:
        results, _ = engine.generate_batch(prompt, num_samples=num_samples, max_tokens=3)
        assert len(results) == num_samples, f"Expected {num_samples} sequences from {num_samples} samples, got {len(results)}"
        assert len(results) == num_samples, f"预期 {num_samples} 个序列来自 {num_samples} 个样本，得到 {len(results)}"


def test_different_seeds_introduce_variation_when_temperature_nonzero():
    """With temperature > 0, different seeds should introduce sampling variation."""
    """当 temperature > 0 时，不同的种子应该引入采样变化。"""
    model = MockModel()
    engine = Engine(model, ByteTokenizer())
    prompt = [261, 72, 101, 108, 108, 111]  # <bos> + "Hello"

    outputs = set()

    for seed in [1, 42, 123, 999, 1000, 1001, 1002, 1003, 1004, 1005]:
        results, _ = engine.generate_batch(
            prompt,
            temperature=1.0,
            max_tokens=5,
            seed=seed,
        )
        outputs.add(tuple(results[0]))

    # Sanity check: sampling actually introduces variation
    # 健全性检查：采样确实引入了变化
    assert len(outputs) > 1, "All seeds produced the same output which is statistically highly improbable."
    assert len(outputs) > 1, "所有种子产生了相同的输出，这在统计上极不可能。"

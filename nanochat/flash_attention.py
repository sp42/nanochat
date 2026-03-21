"""
Unified Flash Attention interface with automatic FA3/SDPA switching.
统一的 Flash Attention 接口，支持 FA3/SDPA 自动切换。

Exports `flash_attn` module that matches the FA3 API exactly, but falls back
导出与 FA3 API 完全匹配的 `flash_attn` 模块，但在非 Hopper GPU
to PyTorch SDPA on non-Hopper GPUs (including Blackwell), MPS, and CPU.
（包括 Blackwell）、MPS 和 CPU 上回退到 PyTorch SDPA。

Usage (drop-in replacement for FA3):
用法（FA3 的直接替代）：
    from nanochat.flash_attention import flash_attn

    # Training (no KV cache)
    # 训练（无 KV 缓存）
    y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)

    # Inference (with KV cache)
    # 推理（带 KV 缓存）
    y = flash_attn.flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, ...)
"""
import torch
import torch.nn.functional as F


# =============================================================================
# Detection: Try to load FA3 on Hopper+ GPUs
# 检测：尝试在 Hopper+ GPU 上加载 FA3
# =============================================================================
def _load_flash_attention_3():
    """Try to load Flash Attention 3 (requires Hopper GPU, sm90)."""
    """尝试加载 Flash Attention 3（需要 Hopper GPU，sm90）。"""
    if not torch.cuda.is_available():
        return None
    try:
        major, _ = torch.cuda.get_device_capability()
        # FA3 kernels are compiled for Hopper (sm90) only
        # FA3 内核仅针对 Hopper (sm90) 编译
        # Ada (sm89), Blackwell (sm100) need SDPA fallback until FA3 is recompiled
        # Ada (sm89)、Blackwell (sm100) 需要 SDPA 回退，直到 FA3 重新编译
        if major != 9:
            return None
        import os
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        from kernels import get_kernel
        return get_kernel('varunneal/flash-attention-3').flash_attn_interface
    except Exception:
        return None


_fa3 = _load_flash_attention_3()
HAS_FA3 = _fa3 is not None

# Override for testing: set to 'fa3', 'sdpa', or None (auto)
# 测试覆盖：设置为 'fa3'、'sdpa' 或 None（自动）
_override_impl = None


def _resolve_use_fa3():
    """Decide once whether to use FA3, based on availability, override, and dtype."""
    """根据可用性、覆盖和数据类型决定是否使用 FA3。"""
    if _override_impl == 'fa3':
        assert HAS_FA3, "Cannot override to FA3: not available on this hardware"
        return True
    if _override_impl == 'sdpa':
        return False
    if HAS_FA3:
        # FA3 Hopper kernels only support bf16 and fp8; fp16/fp32 must use SDPA fallback
        # FA3 Hopper 内核仅支持 bf16 和 fp8；fp16/fp32 必须使用 SDPA 回退
        from nanochat.common import COMPUTE_DTYPE
        if COMPUTE_DTYPE == torch.bfloat16:
            return True
        return False
    return False

USE_FA3 = _resolve_use_fa3()


# =============================================================================
# SDPA helpers
# SDPA 辅助函数
# =============================================================================
def _sdpa_attention(q, k, v, window_size, enable_gqa):
    """
    SDPA attention with sliding window support.
    支持滑动窗口的 SDPA 注意力。
    q, k, v are (B, H, T, D) format.
    q, k, v 是 (B, H, T, D) 格式。
    """
    Tq = q.size(2)
    Tk = k.size(2)
    window = window_size[0]

    # Full context, same length
    # 完整上下文，相同长度
    if (window < 0 or window >= Tq) and Tq == Tk:
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=enable_gqa)

    # Single token generation
    # 单 token 生成
    if Tq == 1:
        if window >= 0 and window < Tk:
            # window is "left" tokens we need to include (window + 1) keys total
            # window 是我们需要包含的"左侧" token（总共 window + 1 个键）
            start = max(0, Tk - (window + 1))
            k = k[:, :, start:, :]
            v = v[:, :, start:, :]
        return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=enable_gqa)

    # Need explicit mask for sliding window/chunk inference
    # 滑动窗口/分块推理需要显式掩码
    device = q.device
    # For chunk inference (Tq != Tk), is_causal is not aligned to cache position => build an explicit bool mask
    # 对于分块推理（Tq != Tk），is_causal 未与缓存位置对齐 => 构建显式布尔掩码
    row_idx = (Tk - Tq) + torch.arange(Tq, device=device).unsqueeze(1)
    col_idx = torch.arange(Tk, device=device).unsqueeze(0)
    mask = col_idx <= row_idx

    # sliding window (left)
    # 滑动窗口（左侧）
    if window >= 0 and window < Tk:
        mask = mask & ((row_idx - col_idx) <= window)

    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=enable_gqa)

# =============================================================================
# Public API: Same interface as FA3
# 公共 API：与 FA3 相同的接口
# =============================================================================
def flash_attn_func(q, k, v, causal=False, window_size=(-1, -1)):
    """
    Flash Attention for training (no KV cache).
    用于训练的 Flash Attention（无 KV 缓存）。

    Args:
    参数：
        q, k, v: Tensors of shape (B, T, H, D)
        q, k, v: 形状为 (B, T, H, D) 的张量
        causal: Whether to use causal masking
        causal: 是否使用因果掩码
        window_size: (left, right) sliding window. -1 means unlimited.
        window_size: (left, right) 滑动窗口。-1 表示无限制。

    Returns:
    返回：
        Output tensor of shape (B, T, H, D)
        形状为 (B, T, H, D) 的输出张量
    """
    if USE_FA3:
        return _fa3.flash_attn_func(q, k, v, causal=causal, window_size=window_size)

    # SDPA fallback: transpose (B, T, H, D) -> (B, H, T, D)
    # SDPA 回退：转置 (B, T, H, D) -> (B, H, T, D)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    enable_gqa = q.size(1) != k.size(1)
    y = _sdpa_attention(q, k, v, window_size, enable_gqa)
    return y.transpose(1, 2)  # back to (B, T, H, D)
                              # 返回 (B, T, H, D)


def flash_attn_with_kvcache(q, k_cache, v_cache, k=None, v=None, cache_seqlens=None,
                            causal=False, window_size=(-1, -1)):
    """
    Flash Attention with KV cache for inference.
    带 KV 缓存的 Flash Attention 用于推理。

    FA3 updates k_cache/v_cache in-place. Our SDPA fallback does the same.
    FA3 原地更新 k_cache/v_cache。我们的 SDPA 回退也是如此。

    Args:
    参数：
        q: Queries, shape (B, T_new, H, D)
        q: 查询，形状 (B, T_new, H, D)
        k_cache, v_cache: Pre-allocated cache tensors, shape (B, T_max, H_kv, D)
        k_cache, v_cache: 预分配的缓存张量，形状 (B, T_max, H_kv, D)
        k, v: New keys/values to insert, shape (B, T_new, H_kv, D)
        k, v: 要插入的新键/值，形状 (B, T_new, H_kv, D)
        cache_seqlens: Current position in cache, shape (B,) int32
        cache_seqlens: 缓存中的当前位置，形状 (B,) int32
        causal: Whether to use causal masking
        causal: 是否使用因果掩码
        window_size: (left, right) sliding window. -1 means unlimited.
        window_size: (left, right) 滑动窗口。-1 表示无限制。

    Returns:
    返回：
        Output tensor of shape (B, T_new, H, D)
        形状为 (B, T_new, H, D) 的输出张量
    """
    if USE_FA3:
        return _fa3.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens,
            causal=causal, window_size=window_size
        )

    # SDPA fallback: manually manage KV cache
    # SDPA 回退：手动管理 KV 缓存
    B, T_new, H, D = q.shape
    pos = cache_seqlens[0].item()  # assume uniform position across batch
                                   # 假设批次中位置一致

    # Insert new k, v into cache (in-place, matching FA3 behavior)
    # 将新 k, v 插入缓存（原地，匹配 FA3 行为）
    if k is not None and v is not None:
        k_cache[:, pos:pos+T_new, :, :] = k
        v_cache[:, pos:pos+T_new, :, :] = v

    # Get full cache up to current position + new tokens
    # 获取直到当前位置 + 新 token 的完整缓存
    end_pos = pos + T_new
    k_full = k_cache[:, :end_pos, :, :]
    v_full = v_cache[:, :end_pos, :, :]

    # Transpose to SDPA layout: (B, T, H, D) -> (B, H, T, D)
    # 转置为 SDPA 布局：(B, T, H, D) -> (B, H, T, D)
    q_sdpa = q.transpose(1, 2)
    k_sdpa = k_full.transpose(1, 2)
    v_sdpa = v_full.transpose(1, 2)

    enable_gqa = q_sdpa.size(1) != k_sdpa.size(1)
    y_sdpa = _sdpa_attention(q_sdpa, k_sdpa, v_sdpa, window_size, enable_gqa)

    return y_sdpa.transpose(1, 2)  # back to (B, T, H, D)
                                   # 返回 (B, T, H, D)


# =============================================================================
# Export: flash_attn module interface (drop-in replacement for FA3)
# 导出：flash_attn 模块接口（FA3 的直接替代）
# =============================================================================
from types import SimpleNamespace
flash_attn = SimpleNamespace(
    flash_attn_func=flash_attn_func,
    flash_attn_with_kvcache=flash_attn_with_kvcache,
)

"""
Test Flash Attention unified interface - verify FA3 and SDPA produce identical results.
测试 Flash Attention 统一接口 - 验证 FA3 和 SDPA 产生相同的结果。

Run: python -m pytest tests/test_attention_fallback.py -v -s
运行：python -m pytest tests/test_attention_fallback.py -v -s

Note on test structure:
    Tests are split into two classes due to dtype/device constraints:
关于测试结构的说明：
    由于数据类型/设备限制，测试分为两个类：

    1. TestFA3VsSDPA: Comparison tests that run both FA3 and SDPA on the same inputs
       and verify they produce identical results. These require a Hopper GPU (FA3 only
       works on sm90+) and use bfloat16 (FA3 doesn't support float32).
    1. TestFA3VsSDPA：比较测试，在相同输入上运行 FA3 和 SDPA，
       并验证它们产生相同的结果。这些测试需要 Hopper GPU（FA3 仅在 sm90+ 上工作）
       并使用 bfloat16（FA3 不支持 float32）。

    2. TestSDPAOnly: Tests that only exercise the SDPA fallback path. These can run
       on any device (CUDA, CPU, MPS) with the appropriate dtype for that device.
    2. TestSDPAOnly：仅测试 SDPA 回退路径的测试。这些测试可以在任何设备
       （CUDA、CPU、MPS）上运行，使用该设备的适当数据类型。
"""
import torch
import pytest
import nanochat.flash_attention as fa_module
from nanochat.flash_attention import flash_attn, HAS_FA3
from nanochat.engine import KVCache


def set_impl(impl):
    """Set the implementation override ('fa3', 'sdpa', or None for auto) and re-resolve USE_FA3."""
    """设置实现覆盖（'fa3'、'sdpa' 或 None 表示自动），并重新解析 USE_FA3。"""
    fa_module._override_impl = impl
    fa_module.USE_FA3 = fa_module._resolve_use_fa3()


def run_both_impls(fn):
    """Run a function with both FA3 and SDPA, return both outputs."""
    """使用 FA3 和 SDPA 运行函数，返回两个输出。"""
    set_impl('fa3')
    out_fa3 = fn()
    set_impl('sdpa')
    out_sdpa = fn()
    set_impl(None)  # reset
                     # 重置
    return out_fa3, out_sdpa


def assert_close(t1, t2, name, atol=1e-2, rtol=1e-2):
    """Assert two tensors are close, with helpful error message."""
    """断言两个张量接近，带有有用的错误消息。"""
    max_diff = (t1 - t2).abs().max().item()
    mean_diff = (t1 - t2).abs().mean().item()
    assert torch.allclose(t1, t2, atol=atol, rtol=rtol), \
        f"{name}: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}"
    return max_diff, mean_diff


# =============================================================================
# FA3 vs SDPA comparison tests (require Hopper GPU)
# FA3 与 SDPA 比较测试（需要 Hopper GPU）
# =============================================================================
@pytest.mark.skipif(not HAS_FA3, reason="FA3 required to compare implementations")
class TestFA3VsSDPA:
    """Compare FA3 and SDPA produce identical results. Requires Hopper GPU."""
    """比较 FA3 和 SDPA 是否产生相同的结果。需要 Hopper GPU。"""

    DEVICE = "cuda"
    DTYPE = torch.bfloat16

    def test_basic_causal(self):
        """Basic causal attention."""
        """基本因果注意力。"""
        B, T, H, D = 2, 64, 4, 32
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            return flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(T, 0))

        y_fa3, y_sdpa = run_both_impls(run)
        max_diff, mean_diff = assert_close(y_fa3, y_sdpa, "basic_causal")
        print(f"basic_causal: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_full_context(self):
        """Full context (window_size=-1)."""
        """完整上下文（window_size=-1）。"""
        B, T, H, D = 2, 128, 4, 32
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            return flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(-1, -1))

        y_fa3, y_sdpa = run_both_impls(run)
        max_diff, mean_diff = assert_close(y_fa3, y_sdpa, "full_context")
        print(f"full_context: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_sliding_window(self):
        """Sliding window attention."""
        """滑动窗口注意力。"""
        B, T, H, D = 2, 128, 4, 32
        window = 32
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            return flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(window, 0))

        y_fa3, y_sdpa = run_both_impls(run)
        max_diff, mean_diff = assert_close(y_fa3, y_sdpa, "sliding_window")
        print(f"sliding_window: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_gqa(self):
        """Group Query Attention (fewer KV heads than Q heads)."""
        """分组查询注意力（KV 头数少于 Q 头数）。"""
        B, T, D = 2, 64, 32
        n_heads = 8
        n_kv_heads = 2

        q = torch.randn(B, T, n_heads, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, n_kv_heads, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, n_kv_heads, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            return flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(T, 0))

        y_fa3, y_sdpa = run_both_impls(run)
        max_diff, mean_diff = assert_close(y_fa3, y_sdpa, "gqa")
        print(f"gqa: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_larger_model(self):
        """Larger dimensions closer to real model."""
        """更接近真实模型的较大维度。"""
        B, T, H, D = 4, 256, 12, 64
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            return flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(-1, -1))

        y_fa3, y_sdpa = run_both_impls(run)
        max_diff, mean_diff = assert_close(y_fa3, y_sdpa, "larger_model")
        print(f"larger_model: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_kvcache_prefill(self):
        """Test prefill (inserting multiple tokens into empty cache)."""
        """测试预填充（将多个 token 插入空缓存）。"""
        B, T_max, H, D = 2, 64, 4, 32
        T_prefill = 16

        q = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            k_cache = torch.zeros(B, T_max, H, D, device=self.DEVICE, dtype=self.DTYPE)
            v_cache = torch.zeros(B, T_max, H, D, device=self.DEVICE, dtype=self.DTYPE)
            cache_seqlens = torch.zeros(B, dtype=torch.int32, device=self.DEVICE)
            return flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache, k=k, v=v,
                cache_seqlens=cache_seqlens,
                causal=True, window_size=(T_max, 0)
            )

        y_fa3, y_sdpa = run_both_impls(run)
        max_diff, mean_diff = assert_close(y_fa3, y_sdpa, "prefill")
        print(f"prefill: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_kvcache_single_token(self):
        """Test single token generation (cache already has content)."""
        """测试单 token 生成（缓存已有内容）。"""
        B, T_max, H, D = 2, 64, 4, 32
        T_prefill = 16

        k_init = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v_init = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        q_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            k_cache = torch.zeros(B, T_max, H, D, device=self.DEVICE, dtype=self.DTYPE)
            v_cache = torch.zeros(B, T_max, H, D, device=self.DEVICE, dtype=self.DTYPE)
            k_cache[:, :T_prefill, :, :] = k_init
            v_cache[:, :T_prefill, :, :] = v_init
            cache_seqlens = torch.full((B,), T_prefill, dtype=torch.int32, device=self.DEVICE)
            return flash_attn.flash_attn_with_kvcache(
                q_single, k_cache, v_cache, k=k_single, v=v_single,
                cache_seqlens=cache_seqlens,
                causal=True, window_size=(T_max, 0)
            )

        y_fa3, y_sdpa = run_both_impls(run)
        max_diff, mean_diff = assert_close(y_fa3, y_sdpa, "single_token")
        print(f"single_token: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_kvcache_single_token_sliding_window(self):
        """Test single token decode with sliding window smaller than cache size.
        测试滑动窗口小于缓存大小时的单 token 解码。

        This catches the bug where SDPA ignores window_size during Tq=1 decode.
        这可以捕获 SDPA 在 Tq=1 解码时忽略 window_size 的错误。
        When window < Tk, FA3 only attends to the last (window+1) tokens,
        当 window < Tk 时，FA3 仅关注最后 (window+1) 个 token，
        but SDPA was attending to all cached tokens.
        但 SDPA 关注所有缓存的 token。
        """
        B, T_max, H, D = 2, 64, 4, 32
        T_prefill = 32  # Enough tokens to exceed window
                        # 足够多的 token 以超过窗口
        window = 8      # Window SMALLER than cache size
                        # 窗口小于缓存大小

        k_init = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v_init = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        q_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            k_cache = torch.zeros(B, T_max, H, D, device=self.DEVICE, dtype=self.DTYPE)
            v_cache = torch.zeros(B, T_max, H, D, device=self.DEVICE, dtype=self.DTYPE)
            k_cache[:, :T_prefill, :, :] = k_init
            v_cache[:, :T_prefill, :, :] = v_init
            cache_seqlens = torch.full((B,), T_prefill, dtype=torch.int32, device=self.DEVICE)
            return flash_attn.flash_attn_with_kvcache(
                q_single, k_cache, v_cache, k=k_single, v=v_single,
                cache_seqlens=cache_seqlens,
                causal=True, window_size=(window, 0)  # window=8 < Tk=33
            )

        y_fa3, y_sdpa = run_both_impls(run)
        max_diff, mean_diff = assert_close(y_fa3, y_sdpa, "single_token_sliding_window")
        print(f"single_token_sliding_window: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

    def test_backward_gradients_match(self):
        """Verify gradients are similar between FA3 and SDPA."""
        """验证 FA3 和 SDPA 的梯度相似。"""
        B, T, H, D = 2, 32, 4, 16

        q_data = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k_data = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v_data = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        def run():
            q = q_data.clone().requires_grad_(True)
            k = k_data.clone().requires_grad_(True)
            v = v_data.clone().requires_grad_(True)
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(T, 0))
            loss = y.sum()
            loss.backward()
            return y.detach(), q.grad.detach(), k.grad.detach(), v.grad.detach()

        set_impl('fa3')
        y_fa3, q_grad_fa3, k_grad_fa3, v_grad_fa3 = run()
        set_impl('sdpa')
        y_sdpa, q_grad_sdpa, k_grad_sdpa, v_grad_sdpa = run()
        set_impl(None)

        max_diff, mean_diff = assert_close(y_fa3, y_sdpa, "backward_output")
        print(f"backward_output: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

        max_diff, mean_diff = assert_close(q_grad_fa3, q_grad_sdpa, "q_grad", atol=0.05, rtol=0.05)
        print(f"q_grad: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

        max_diff, mean_diff = assert_close(k_grad_fa3, k_grad_sdpa, "k_grad", atol=0.05, rtol=0.05)
        print(f"k_grad: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")

        max_diff, mean_diff = assert_close(v_grad_fa3, v_grad_sdpa, "v_grad", atol=0.05, rtol=0.05)
        print(f"v_grad: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")


# =============================================================================
# SDPA-only tests (run on any device)
# 仅 SDPA 测试（可在任何设备上运行）
# =============================================================================
class TestSDPAOnly:
    """Test SDPA fallback works correctly. Runs on any device."""
    """测试 SDPA 回退是否正常工作。可在任何设备上运行。"""

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    def test_basic_forward(self):
        """Test SDPA forward pass produces valid output."""
        """测试 SDPA 前向传播产生有效输出。"""
        set_impl('sdpa')
        B, T, H, D = 2, 64, 4, 32
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE)

        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(T, 0))

        assert y.shape == (B, T, H, D)
        assert not torch.isnan(y).any(), "Output contains NaN"
        set_impl(None)

    def test_backward(self):
        """Test gradients flow through SDPA."""
        """测试梯度流经 SDPA。"""
        set_impl('sdpa')
        B, T, H, D = 2, 32, 4, 16
        q = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE, requires_grad=True)
        k = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE, requires_grad=True)
        v = torch.randn(B, T, H, D, device=self.DEVICE, dtype=self.DTYPE, requires_grad=True)

        y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=(T, 0))
        loss = y.sum()
        loss.backward()

        assert q.grad is not None, "No gradient for q"
        assert k.grad is not None, "No gradient for k"
        assert v.grad is not None, "No gradient for v"
        assert not torch.isnan(q.grad).any(), "NaN in q gradient"
        set_impl(None)

    def test_kvcache(self):
        """Test SDPA with KV cache."""
        """测试带 KV 缓存的 SDPA。"""
        set_impl('sdpa')
        B, T_max, H, D = 2, 64, 4, 32
        n_layers = 1

        cache = KVCache(
            batch_size=B, num_heads=H, seq_len=T_max, head_dim=D,
            num_layers=n_layers, device=self.DEVICE, dtype=self.DTYPE
        )
        k_cache, v_cache = cache.get_layer_cache(0)

        # Prefill
        # 预填充
        T_prefill = 16
        q = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v = torch.randn(B, T_prefill, H, D, device=self.DEVICE, dtype=self.DTYPE)

        y = flash_attn.flash_attn_with_kvcache(
            q, k_cache, v_cache, k=k, v=v,
            cache_seqlens=cache.cache_seqlens,
            causal=True, window_size=(T_max, 0)
        )
        cache.advance(T_prefill)

        assert y.shape == (B, T_prefill, H, D)
        assert cache.get_pos() == T_prefill

        # Generate single token
        # 生成单个 token
        q_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)
        k_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)
        v_single = torch.randn(B, 1, H, D, device=self.DEVICE, dtype=self.DTYPE)

        y_single = flash_attn.flash_attn_with_kvcache(
            q_single, k_cache, v_cache, k=k_single, v=v_single,
            cache_seqlens=cache.cache_seqlens,
            causal=True, window_size=(T_max, 0)
        )
        cache.advance(1)

        assert y_single.shape == (B, 1, H, D)
        assert cache.get_pos() == T_prefill + 1
        set_impl(None)


# =============================================================================
# Override mechanism tests
# 覆盖机制测试
# =============================================================================
class TestOverrideMechanism:
    """Test that the override mechanism works correctly."""
    """测试覆盖机制是否正常工作。"""

    @pytest.mark.skipif(not HAS_FA3, reason="FA3 required")
    def test_override_fa3(self):
        """Test that override='fa3' uses FA3."""
        """测试 override='fa3' 使用 FA3。"""
        set_impl('fa3')
        assert fa_module.USE_FA3 == True
        set_impl(None)

    def test_override_sdpa(self):
        """Test that override='sdpa' uses SDPA."""
        """测试 override='sdpa' 使用 SDPA。"""
        set_impl('sdpa')
        assert fa_module.USE_FA3 == False
        set_impl(None)

    def test_override_auto(self):
        """Test that override=None uses auto-detection."""
        """测试 override=None 使用自动检测。"""
        set_impl(None)
        assert fa_module.USE_FA3 == HAS_FA3


if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        major, minor = torch.cuda.get_device_capability()
        print(f"Compute capability: {major}.{minor}")
    print(f"HAS_FA3: {HAS_FA3}")
    print()

    pytest.main([__file__, "-v", "-s"])

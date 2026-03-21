"""
GPT model (rewrite, a lot simpler)
GPT 模型（重写，更加简洁）
Notable features:
主要特性：
- rotary embeddings (and no positional embeddings)
- 旋转嵌入（无位置嵌入）
- QK norm
- QK 归一化
- untied weights for token embedding and lm_head
- token embedding 和 lm_head 使用独立权重
- relu^2 activation in MLP
- MLP 中使用 relu^2 激活函数
- norm after token embedding
- token embedding 后进行归一化
- no learnable params in rmsnorm
- rmsnorm 中无可学习参数
- no bias in linear layers
- 线性层无偏置
- Group-Query Attention (GQA) support for more efficient inference
- 支持分组查询注意力（GQA）以实现更高效的推理
- Flash Attention 3 integration
- Flash Attention 3 集成
"""

from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW, DistMuonAdamW

# Our custom Flash Attention module that automatically uses FA3 on Hopper+ and SDPA fallback elsewhere
# 我们自定义的 Flash Attention 模块，在 Hopper+ 上自动使用 FA3，其他平台回退到 SDPA
from nanochat.flash_attention import flash_attn

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of query heads
                    # 查询头数量
    n_kv_head: int = 6 # number of key/value heads (GQA)
                       # 键/值头数量（GQA）
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # 滑动窗口注意力模式字符串，在各层间平铺。最后一层始终为 L。
    # Characters: L=long (full context), S=short (quarter context)
    # 字符：L=长（完整上下文），S=短（四分之一上下文）
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    # 示例："L"=全部完整上下文，"SL"=交替，"SSL"=两个短后跟一个长
    window_pattern: str = "SSSL"


def norm(x):
    return F.rms_norm(x, (x.size(-1),)) # note that this will run in bf16, seems ok
                                        # 注意这将在 bf16 下运行，看起来没问题

class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    在 forward 中将权重转换为匹配输入数据类型的 nn.Linear。
    Replaces autocast: master weights stay fp32 for optimizer precision,
    替代 autocast：主权重保持 fp32 以保证优化器精度，
    but matmuls run in the activation dtype (typically bf16 from embeddings).
    但矩阵乘法在激活数据类型（通常是来自 embedding 的 bf16）下运行。"""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    """如果 GPT 层应该有值嵌入则返回 True（交替，最后一层始终包含）。"""
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
                        # 多头注意力
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
                                    # 将最后一维分成两半
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
                             # 旋转成对的维度
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 12
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        # 将输入投影得到查询、键和值
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        # 形状：(B, T, H, D) - FA3 的原生布局，无需转置！
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        # 值残差（ResFormer）：使用每个头的输入相关门控混合值嵌入
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
                                                                                  # (B, T, n_kv_head)，范围 (0, 3)
            v = v + gate.unsqueeze(-1) * ve

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        # 对查询和键应用旋转嵌入以获得相对位置编码
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK norm
                                 # QK 归一化
        q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
                     # 更锐利的注意力（在 Q 和 K 之间分配缩放），TODO 需要更好地考虑
        k = k * 1.2

        # Flash Attention (FA3 on Hopper+, PyTorch SDPA fallback elsewhere)
        # Flash Attention（Hopper+ 上使用 FA3，其他平台回退到 PyTorch SDPA）
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        # window_size 是 (left, right) 元组：(N, 0) 表示因果，(-1, 0) 表示完整上下文
        if kv_cache is None:
            # Training: causal attention with optional sliding window
            # 训练：带可选滑动窗口的因果注意力
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            # 推理：使用 flash_attn_with_kvcache 处理缓存管理
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            # Advance position after last layer processes
            # 最后一层处理完后推进位置
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # Re-assemble the heads and project back to residual stream
        # 重新组装头并投影回残差流
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        注意一个主要陷阱：此 __init__ 函数在 meta 设备上下文中运行！！
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        因此，这里的任何计算只是形状和数据类型，没有实际数据。
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        => 我们实际上在 init_weights() 中初始化所有数据（参数、缓冲区等）。
        """
        super().__init__()
        self.config = config
        # Compute per-layer window sizes for sliding window attention
        # 计算滑动窗口注意力的每层窗口大小
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        # window_size 是 (left, right) 元组：(-1, 0) 表示完整上下文，(N, 0) 表示滑动窗口
        self.window_sizes = self._compute_window_sizes(config)
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # 为了效率填充词表（DDP、张量核心）。这只是优化 - 输出在 forward() 中被裁剪。
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # 每层可学习标量（灵感来自 modded-nanogpt）
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # resid_lambdas：缩放每层的残差流（初始值 1.0 = 中性）
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # x0_lambdas：在每层混合初始嵌入（初始值 0.0 = 禁用）
        # Separate parameters so they can have different optimizer treatment
        # 分离参数以便它们可以有不同的优化器处理
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
                                                                        # 假初始化，真正初始化在 init_weights() 中
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
                                                                        # 假初始化，真正初始化在 init_weights() 中
        # Smear: mix previous token's embedding into current token (cheap bigram-like info)
        # Smear：将前一个 token 的嵌入混合到当前 token（廉价的双字类信息）
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        # Backout: subtract cached mid-layer residual before final norm to remove low-level features
        # Backout：在最终归一化前减去缓存的中间层残差以移除低级特征
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        # 值嵌入（ResFormer 风格）：交替层，最后一层始终包含
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # 为了支持 meta 设备初始化，我们在这里初始化旋转嵌入，但这只是"假"的 meta 张量。
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # 至于 rotary_seq_len，这些旋转嵌入在内存中很小/廉价，
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # 所以我们只是多计算 10 倍，但如果达到该数量则断言失败。
        # In the future we can dynamically grow the cache, for now it's fine.
        # 将来我们可以动态增长缓存，现在这样就可以了。
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
                                                       # 10 倍过度计算应该足够，TODO 改进？
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
                                                           # persistent=False 表示它不会保存到检查点
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.
        在这一个函数中初始化完整模型以获得最大的清晰度。

        wte (embedding):     normal, std=1.0
        wte（嵌入）：         正态分布，std=1.0
        lm_head:             normal, std=0.001
        lm_head：            正态分布，std=0.001
        for each block:
        对于每个块：
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_q：       均匀分布，std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_k：       均匀分布，std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_v：       均匀分布，std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            attn.c_proj：    零
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_fc：       均匀分布，std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
            mlp.c_proj：     零
        """

        # Embedding and unembedding
        # 嵌入和反嵌入
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        # Transformer 块：均匀初始化，边界 = sqrt(3) * std（与正态分布的标准差相同）
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
                                   # sqrt(3) 乘数确保均匀分布达到与正态分布相同的 std
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # weights use Uniform to avoid outliers
                                                                  # 权重使用均匀分布以避免异常值
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero
                                                           # 投影为零
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)  # 0.4x init scale for c_fc
                                                                              # c_fc 使用 0.4x 初始化缩放
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Per-layer scalars
        # 每层标量
        # Per-layer resid init: stronger residual at early layers, weaker at deep layers
        # 每层残差初始化：早期层残差较强，深层残差较弱
        n_layer = self.config.n_layer
        for i in range(n_layer):
            self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
        # Decaying x0 init: earlier layers get more input embedding blending
        # 衰减的 x0 初始化：早期层获得更多输入嵌入混合
        for i in range(n_layer):
            self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))

        # Value embeddings (init like c_v: uniform with same std)
        # 值嵌入（像 c_v 一样初始化：具有相同 std 的均匀分布）
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init with small positive values so gates start slightly above neutral
        # 门权重用小的正值初始化，以便门从略高于中性的位置开始
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)

        # Rotary embeddings
        # 旋转嵌入
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to COMPUTE_DTYPE: optimizer can tolerate reduced-precision
        # 将嵌入转换为 COMPUTE_DTYPE：优化器可以容忍降低精度
        # embeddings and it saves memory. Exception: fp16 requires fp32 embeddings
        # 嵌入，这节省内存。例外：fp16 需要 fp32 嵌入
        # because GradScaler cannot unscale fp16 gradients.
        # 因为 GradScaler 无法取消缩放 fp16 梯度。
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for ve in self.value_embeds.values():
                ve.to(dtype=COMPUTE_DTYPE)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # TODO：增加基础 theta？例如 100K 最近更常见
        # autodetect the device from model embeddings
        # 从模型嵌入自动检测设备
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        # 步进通道
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        # 步进时间步
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        # 计算每个（时间，通道）对的旋转频率
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
                                                                 # 添加批次和头维度以便稍后广播
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.
        计算滑动窗口注意力的每层窗口大小。

        Returns list of (left, right) tuples for FA3's window_size parameter:
        返回 FA3 的 window_size 参数的 (left, right) 元组列表：
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - left：当前位置之前要关注的 token 数量（-1 = 无限制）
        - right: how many tokens after current position to attend to (0 for causal)
        - right：当前位置之后要关注的 token 数量（0 表示因果）

        Pattern string is tiled across layers. Final layer always gets L (full context).
        模式字符串在各层间平铺。最后一层始终获得 L（完整上下文）。
        Characters: L=long (full context), S=short (quarter context)
        字符：L=长（完整上下文），S=短（四分之一上下文）
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        # 将字符映射到窗口大小
        long_window = config.sequence_len
        short_window = -(-long_window // 4 // 128) * 128  # ceil to FA3 tile size (2048 -> 768)
                                                          # 向上取整到 FA3 块大小（2048 -> 768）
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        # 在各层间平铺模式
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        # 最后一层始终获得完整上下文
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        返回模型每个 token 的估计 FLOPs（前向 + 反向）。
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        每个矩阵乘法权重参数在前向中贡献 2 FLOPs（乘法 *，累加 +），在反向中是其 2 倍 => 2+4=6。
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        最清晰的解释：https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        此外，12 * h * q * effective_seq_len 计算注意力内部的 key @ query 矩阵乘法 flops。
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        使用滑动窗口时，effective_seq_len 因层而异（受窗口大小限制）。
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        参考：https://arxiv.org/abs/2204.02311（PaLM 论文）。
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        这与 Chinchilla 论文的精确公式相差约 1%，区别在于：
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla 将嵌入层算作 flops（？奇怪，它只是查找 => 我们忽略）
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        - Chinchilla 将注意力 softmax 中的 exp/sum/divide 算作 flops（有点可疑且很小 => 我们忽略）
        """
        nparams = sum(p.numel() for p in self.parameters())
        # Exclude non-matmul params: embeddings and per-layer scalars
        # 排除非矩阵乘法参数：嵌入和每层标量
        value_embeds_numel = sum(ve.weight.numel() for ve in self.value_embeds.values())
        nparams_exclude = (self.transformer.wte.weight.numel() + value_embeds_numel +
                          self.resid_lambdas.numel() + self.x0_lambdas.numel() +
                          self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel())
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window
        # 对每层的注意力 FLOPs 求和，考虑滑动窗口
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
                                     # (left, right) 元组，我们使用 left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * (nparams - nparams_exclude) + attn_flops
        return num_flops_per_token

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        返回用于缩放定律分析的详细参数计数。
        Different papers use different conventions:
        不同的论文使用不同的约定：
        - Kaplan et al. excluded embedding parameters
        - Kaplan 等人排除了嵌入参数
        - Chinchilla included all parameters
        - Chinchilla 包含所有参数
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        参考：https://arxiv.org/abs/2203.15556（Chinchilla 论文）
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)
        参考：https://arxiv.org/abs/2001.08361（Kaplan 等人原始缩放定律论文）

        Returns a dict with counts for each parameter group, so downstream analysis
        返回包含每个参数组计数的字典，以便下游分析
        can experiment with which combination gives the cleanest scaling laws.
        可以实验哪种组合给出最干净的缩放定律。
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        # 分别计算每个组（镜像 setup_optimizers 中的分组）
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel() + self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Separate out all parameters into groups
        # 将所有参数分成组
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        smear_params = [self.smear_gate.weight, self.smear_lambda, self.backout_lambda]
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params) + len(smear_params)

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        # 按 ∝1/√dmodel 缩放 AdamW 参数的学习率（针对 768 维模型调优）
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        # 构建显式包含所有必需字段的 param_groups
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            # AdamW 组（嵌入、lm_head、标量）
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),  # higher beta1 for x0
            dict(kind='adamw', params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        # Muon groups (matrix params, grouped by shape for stacking)
        # Muon 组（矩阵参数，按形状分组以便堆叠）
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        # 获取当前序列长度的旋转嵌入（形状为 (1, seq_len, 1, head_dim/2)）
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == COMPUTE_DTYPE, f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        # 如果 kv 缓存存在，我们需要将旋转嵌入偏移到缓存中的当前位置
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length
                                                             # 将缓存截断到当前序列长度

        # Embed the tokens
        # 嵌入 token
        x = self.transformer.wte(idx) # embed current token
                                      # 嵌入当前 token
        x = x.to(COMPUTE_DTYPE) # ensure activations are in compute dtype (no-op usually, but active for fp16 code path)
                                # 确保激活在计算数据类型中（通常是无操作，但对 fp16 代码路径有效）
        x = norm(x)

        # Smear: mix previous token's embedding into current position (cheap bigram info)
        # Smear：将前一个 token 的嵌入混合到当前位置（廉价的双字信息）
        if kv_cache is None:
            # Training / naive generate: full sequence available, use fast slice
            # 训练 / 朴素生成：完整序列可用，使用快速切片
            assert T > 1, "Training forward pass should have T > 1"
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        else:
            # KV cache inference: read prev embedding from cache, store current for next step
            # KV 缓存推理：从缓存读取前一个嵌入，存储当前的供下一步使用
            x_pre_smear = kv_cache.prev_embedding
            kv_cache.prev_embedding = x[:, -1:, :]
            if T > 1:
                # Prefill: apply smear to positions 1+, same as training
                # 预填充：对位置 1+ 应用 smear，与训练相同
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
                x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
            elif x_pre_smear is not None:
                # Decode: single token, use cached prev embedding
                # 解码：单个 token，使用缓存的前一个嵌入
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :, :24]))
                x = x + gate * x_pre_smear

        # Forward the trunk of the Transformer
        # 前向传播 Transformer 的主干
        x0 = x  # save initial normalized embedding for x0 residual
                # 保存初始归一化嵌入用于 x0 残差
        n_layer = self.config.n_layer
        backout_layer = n_layer // 2  # cache at halfway point
                                       # 在中间点缓存
        x_backout = None
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
            if i == backout_layer:
                x_backout = x
        # Subtract mid-layer residual to remove low-level features before logit projection
        # 在 logit 投影前减去中间层残差以移除低级特征
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = norm(x)

        # Forward the lm_head (compute logits)
        # 前向传播 lm_head（计算 logits）
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
                     # 平滑地将 logits 限制在 [-softcap, softcap] 范围内
        logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
                                 # (B, T, padded_vocab_size) <- 非常大的张量，大量内存
        logits = logits[..., :self.config.vocab_size] # slice to remove padding
                                                      # 切片以移除填充
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
                                # 切换到 fp32 用于 logit softcap 和损失计算
        logits = softcap * torch.tanh(logits / softcap) # squash the logits
                                                        # 压缩 logits

        if targets is not None:
            # training: given the targets, compute and return the loss
            # 训练：给定目标，计算并返回损失
            # TODO experiment with chunked cross-entropy?
            # TODO 实验分块交叉熵？
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            # inference: just return the logits directly
            # 推理：直接返回 logits
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        朴素的自回归流式推理。
        To make it super simple, let's assume:
        简单起见假设：
        - batch size is 1
        - 批大小为 1
        - ids and the yielded tokens are simple Python lists and ints
        - ids 和生成的 token 是简单的 Python 列表和整数
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
                                                                      # 添加批次维度
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token

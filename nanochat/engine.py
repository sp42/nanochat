"""
Engine for efficient inference of our models.
用于模型高效推理的引擎。

Everything works around token sequences:
一切围绕 token 序列工作：
- The user can send token sequences to the engine
- 用户可以向引擎发送 token 序列
- The engine returns the next token
- 引擎返回下一个 token

Notes:
注意：
- The engine knows nothing about tokenization, it's purely token id sequences.
- 引擎对分词一无所知，它纯粹是 token id 序列。

The whole thing is made as efficient as possible.
整个系统尽可能高效。
"""

import torch
import torch.nn.functional as F
import signal
import warnings
from contextlib import contextmanager
from collections import deque
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model

# -----------------------------------------------------------------------------
# Calculator tool helpers
# 计算器工具辅助函数
@contextmanager
def timeout(duration, formula):
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)

def eval_with_timeout(formula, max_time=3):
    try:
        with timeout(max_time, formula):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return eval(formula, {"__builtins__": {}}, {})
    except Exception as e:
        signal.alarm(0)
        # print(f"Warning: Failed to eval {formula}, exception: {e}") # it's ok ignore wrong calculator usage
                                                                     # 可以忽略错误的计算器使用
        return None

def use_calculator(expr):
    """
    Evaluate a Python expression safely.
    安全地评估 Python 表达式。
    Supports both math expressions and string operations like .count()
    支持数学表达式和字符串操作如 .count()
    """
    # Remove commas from numbers
    # 从数字中移除逗号
    expr = expr.replace(",", "")

    # Check if it's a pure math expression (old behavior)
    # 检查是否是纯数学表达式（旧行为）
    if all([x in "0123456789*+-/.() " for x in expr]):
        if "**" in expr:  # disallow power operator
                           # 禁止幂运算符
            return None
        return eval_with_timeout(expr)

    # Check if it's a string operation we support
    # 检查是否是我们支持的字符串操作
    # Allow: strings (single/double quotes), .count(), letters, numbers, spaces, parens
    # 允许：字符串（单引号/双引号）、.count()、字母、数字、空格、括号
    allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\"()._ "
    if not all([x in allowed_chars for x in expr]):
        return None

    # Disallow dangerous patterns
    # 禁止危险模式
    dangerous_patterns = ['__', 'import', 'exec', 'eval', 'compile', 'open', 'file',
                         'input', 'raw_input', 'globals', 'locals', 'vars', 'dir',
                         'getattr', 'setattr', 'delattr', 'hasattr']
    expr_lower = expr.lower()
    if any(pattern in expr_lower for pattern in dangerous_patterns):
        return None

    # Only allow .count() method for now (can expand later)
    # 目前只允许 .count() 方法（以后可以扩展）
    if '.count(' not in expr:
        return None

    # Evaluate with timeout
    # 带超时评估
    return eval_with_timeout(expr)

# -----------------------------------------------------------------------------
class KVCache:
    """
    KV Cache designed for Flash Attention 3's flash_attn_with_kvcache API.
    为 Flash Attention 3 的 flash_attn_with_kvcache API 设计的 KV 缓存。

    Key differences from FA2-style cache:
    与 FA2 风格缓存的主要区别：
    - Tensors are (B, T, H, D) not (B, H, T, D)
    - 张量是 (B, T, H, D) 而不是 (B, H, T, D)
    - FA3 updates the cache in-place during flash_attn_with_kvcache
    - FA3 在 flash_attn_with_kvcache 期间就地更新缓存
    - Position tracked per batch element via cache_seqlens tensor
    - 通过 cache_seqlens 张量跟踪每个批次元素的位置
    """

    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers, device, dtype):
        self.batch_size = batch_size
        self.max_seq_len = seq_len
        self.n_layers = num_layers
        self.n_heads = num_heads
        self.head_dim = head_dim
        # Pre-allocate cache tensors: (n_layers, B, T, H, D)
        # 预分配缓存张量：(n_layers, B, T, H, D)
        self.k_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        self.v_cache = torch.zeros(num_layers, batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        # Current sequence length per batch element (FA3 needs int32)
        # 每个批次元素的当前序列长度（FA3 需要 int32）
        self.cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)
        # Previous token's normalized embedding for smear (set by model forward pass)
        # 前一个 token 的归一化嵌入用于 smear（由模型前向传播设置）
        self.prev_embedding = None

    def reset(self):
        """Reset cache to empty state."""
        """将缓存重置为空状态。"""
        self.cache_seqlens.zero_()
        self.prev_embedding = None

    def get_pos(self):
        """Get current position (assumes all batch elements at same position)."""
        """获取当前位置（假设所有批次元素在同一位置）。"""
        return self.cache_seqlens[0].item()

    def get_layer_cache(self, layer_idx):
        """Return (k_cache, v_cache) views for a specific layer."""
        """返回特定层的 (k_cache, v_cache) 视图。"""
        return self.k_cache[layer_idx], self.v_cache[layer_idx]

    def advance(self, num_tokens):
        """Advance the cache position by num_tokens."""
        """将缓存位置推进 num_tokens。"""
        self.cache_seqlens += num_tokens

    def prefill(self, other):
        """
        Copy cached KV from another cache into this one.
        将另一个缓存的 KV 复制到这个缓存中。
        Used when we do batch=1 prefill and then want to generate multiple samples in parallel.
        用于当我们进行 batch=1 预填充然后想要并行生成多个样本时。
        """
        assert self.get_pos() == 0, "Cannot prefill a non-empty KV cache"
        assert self.n_layers == other.n_layers and self.n_heads == other.n_heads and self.head_dim == other.head_dim
        assert self.max_seq_len >= other.max_seq_len
        other_pos = other.get_pos()
        self.k_cache[:, :, :other_pos, :, :] = other.k_cache[:, :, :other_pos, :, :]
        self.v_cache[:, :, :other_pos, :, :] = other.v_cache[:, :, :other_pos, :, :]
        self.cache_seqlens.fill_(other_pos)
        # Copy smear state: expand batch=1 prev_embedding to num_samples
        # 复制 smear 状态：将 batch=1 prev_embedding 扩展到 num_samples
        if other.prev_embedding is not None:
            self.prev_embedding = other.prev_embedding.expand(self.batch_size, -1, -1).clone()

# -----------------------------------------------------------------------------
@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None):
    """Sample a single next token from given logits of shape (B, vocab_size). Returns (B, 1)."""
    """从给定形状为 (B, vocab_size) 的 logits 中采样单个下一个 token。返回 (B, 1)。"""
    assert temperature >= 0.0, "temperature must be non-negative"
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        vals, idx = torch.topk(logits, k, dim=-1)
        vals = vals / temperature
        probs = F.softmax(vals, dim=-1)
        choice = torch.multinomial(probs, num_samples=1, generator=rng)
        return idx.gather(1, choice)
    else:
        logits = logits / temperature
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=rng)

# -----------------------------------------------------------------------------

class RowState:
    # Per-row state tracking during generation
    # 生成期间每行状态跟踪
    def __init__(self, current_tokens=None):
        self.current_tokens = current_tokens or [] # Current token sequence for this row
                                                   # 此行的当前 token 序列
        self.forced_tokens = deque() # Queue of tokens to force inject
                                     # 强制注入的 token 队列
        self.in_python_block = False # Whether we are inside a python block
                                     # 是否在 python 块内
        self.python_expr_tokens = [] # Tokens of the current python expression
                                     # 当前 python 表达式的 token
        self.completed = False # Whether this row has completed generation
                               # 此行是否已完成生成

class Engine:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer # needed for tool use

    @torch.inference_mode()
    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, seed=42):
        """Same as generate, but does single prefill and then clones the KV cache."""
        """与 generate 相同，但执行单次预填充然后克隆 KV 缓存。"""
        assert isinstance(tokens, list) and isinstance(tokens[0], int), "expecting list of ints"
        device = self.model.get_device()
        # NOTE: setting the dtype here and in this way is an ugly hack.
        # 注意：在这里以这种方式设置 dtype 是一个丑陋的 hack。
        # Currently the repo assumes that cuda -> bfloat16 and everything else -> float32.
        # 目前仓库假设 cuda -> bfloat16，其他 -> float32。
        # We need to know the dtype here to call __init__ on KVCache and pre-allocate its tensors.
        # 我们需要在这里知道 dtype 以调用 KVCache 的 __init__ 并预分配其张量。
        # As a quick hack, we're making generate() function inherit and know about this repo-wise assumption.
        # 作为一个快速的 hack，我们让 generate() 函数继承并知道这个仓库范围的假设。
        # I think there has to be a bigger refactor to deal with device/dtype tracking across the codebase.
        # 我认为需要进行更大的重构来处理整个代码库中的设备/dtype 跟踪。
        # In particular, the KVCache should allocate its tensors lazily
        # 特别是，KVCache 应该延迟分配其张量
        dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        # Get the special tokens we need to coordinate the tool use state machine
        # 获取我们需要协调工具使用状态机的特殊 token
        get_special = lambda s: self.tokenizer.encode_special(s)
        python_start = get_special("<|python_start|>")
        python_end = get_special("<|python_end|>")
        output_start = get_special("<|output_start|>")
        output_end = get_special("<|output_end|>")
        assistant_end = get_special("<|assistant_end|>") # if sampled, ends row
                                                         # 如果采样到，结束行
        bos = self.tokenizer.get_bos_token_id() # if sampled, ends row
                                                 # 如果采样到，结束行

        # 1) Run a batch 1 prefill of the prompt tokens
        # 1) 对提示 token 运行批次 1 预填充
        m = self.model.config
        kv_model_kwargs = {"num_heads": m.n_kv_head, "head_dim": m.n_embd // m.n_head, "num_layers": m.n_layer}
        kv_cache_prefill = KVCache(
            batch_size=1,
            seq_len=len(tokens),
            device=device,
            dtype=dtype,
            **kv_model_kwargs,
        )
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
        logits = logits[:, -1, :].expand(num_samples, -1)  # (num_samples, vocab_size)

        # 2) Replicate the KV cache for each sample/row
        # 2) 为每个样本/行复制 KV 缓存
        kv_length_hint = (len(tokens) + max_tokens) if max_tokens is not None else self.model.config.sequence_len
        kv_cache_decode = KVCache(
            batch_size=num_samples,
            seq_len=kv_length_hint,
            device=device,
            dtype=dtype,
            **kv_model_kwargs,
        )
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill # no need to keep this memory around
                             # 不需要保留这个内存

        # 3) Initialize states for each sample
        # 3) 初始化每个样本的状态
        row_states = [RowState(tokens.copy()) for _ in range(num_samples)]

        # 4) Main generation loop
        # 4) 主生成循环
        num_generated = 0
        while True:
            # Stop condition: we've reached max tokens
            # 停止条件：已达到最大 token 数
            if max_tokens is not None and num_generated >= max_tokens:
                break
            # Stop condition: all rows are completed
            # 停止条件：所有行已完成
            if all(state.completed for state in row_states):
                break

            # Sample the next token for each row
            # 为每行采样下一个 token
            next_ids = sample_next_token(logits, rng, temperature, top_k)  # (B, 1)
            sampled_tokens = next_ids[:, 0].tolist()

            # Process each row: choose the next token, update state, optional tool use
            # 处理每行：选择下一个 token，更新状态，可选工具使用
            token_column = [] # contains the next token id along each row
                              # 包含每行的下一个 token id
            token_masks = [] # contains the mask (was it sampled (1) or forced (0)?) along each row
                             # 包含每行的掩码（是采样的 (1) 还是强制的 (0)？）
            for i, state in enumerate(row_states):
                # Select the next token in this row
                # 选择此行的下一个 token
                is_forced = len(state.forced_tokens) > 0 # are there tokens waiting to be forced in deque?
                                                         # deque 中是否有等待强制注入的 token？
                token_masks.append(0 if is_forced else 1) # mask is 0 if forced, 1 if sampled
                                                          # 如果强制则掩码为 0，如果采样则为 1
                next_token = state.forced_tokens.popleft() if is_forced else sampled_tokens[i]
                token_column.append(next_token)
                # Update the state of this row to include the next token
                # 更新此行状态以包含下一个 token
                state.current_tokens.append(next_token)
                # On <|assistant_end|> or <|bos|>, mark the row as completed
                # 在 <|assistant_end|> 或 <|bos|> 时，将行标记为已完成
                if next_token == assistant_end or next_token == bos:
                    state.completed = True
                # Handle tool logic
                # 处理工具逻辑
                if next_token == python_start:
                    state.in_python_block = True
                    state.python_expr_tokens = []
                elif next_token == python_end and state.in_python_block:
                    state.in_python_block = False
                    if state.python_expr_tokens:
                        expr = self.tokenizer.decode(state.python_expr_tokens)
                        result = use_calculator(expr)
                        if result is not None:
                            result_tokens = self.tokenizer.encode(str(result))
                            state.forced_tokens.append(output_start)
                            state.forced_tokens.extend(result_tokens)
                            state.forced_tokens.append(output_end)
                    state.python_expr_tokens = []
                elif state.in_python_block:
                    state.python_expr_tokens.append(next_token)

            # Yield the token column
            # 生成 token 列
            yield token_column, token_masks
            num_generated += 1

            # Prepare logits for next iteration
            # 为下一次迭代准备 logits
            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)
            logits = self.model.forward(ids, kv_cache=kv_cache_decode)[:, -1, :]  # (B, vocab_size)

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        """
        Non-streaming batch generation that just returns the final token sequences.
        非流式批量生成，只返回最终的 token 序列。
        Returns a list of token sequences (list of lists of ints).
        返回 token 序列列表（整数列表的列表）。
        Terminal tokens (assistant_end, bos) are not included in the results.
        终止 token（assistant_end、bos）不包含在结果中。
        """
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id()
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token == assistant_end or token == bos:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            # Stop if all rows are completed
            if all(completed):
                break
        return results, masks


if __name__ == "__main__":
    """
    Quick inline test to make sure that the naive/slow model.generate function
    快速内联测试，确保朴素/慢速的 model.generate 函数
    is equivalent to the faster Engine.generate function here.
    与这里更快的 Engine.generate 函数等效。
    """
    import time
    # init compute
    # 初始化计算
    device_type = autodetect_device_type()
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
    # load the model and tokenizer
    # 加载模型和分词器
    model, tokenizer, meta = load_model("base", device, phase="eval")
    bos_token_id = tokenizer.get_bos_token_id()
    # common hyperparameters
    # 通用超参数
    kwargs = dict(max_tokens=64, temperature=0.0)
    # set the starting prompt
    # 设置起始提示
    prompt_tokens = tokenizer.encode("The chemical formula of water is", prepend=bos_token_id)
    # generate the reference sequence using the model.generate() function
    # 使用 model.generate() 函数生成参考序列
    generated_tokens = []
    torch.cuda.synchronize()
    t0 = time.time()
    stream = model.generate(prompt_tokens, **kwargs)
    for token in stream:
        generated_tokens.append(token)
        chunk = tokenizer.decode([token])
        print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Reference time: {t1 - t0:.2f}s")
    reference_ids = generated_tokens
    # generate tokens with Engine
    # 使用 Engine 生成 token
    generated_tokens = []
    engine = Engine(model, tokenizer)
    stream = engine.generate(prompt_tokens, num_samples=1, **kwargs) # note: runs in fp32
                                                                     # 注意：在 fp32 下运行
    torch.cuda.synchronize()
    t0 = time.time()
    for token_column, token_masks in stream:
        token = token_column[0] # only print out the first row
                                # 只打印第一行
        generated_tokens.append(token)
        chunk = tokenizer.decode([token])
        print(chunk, end="", flush=True)
    print()
    torch.cuda.synchronize()
    t1 = time.time()
    print(f"Engine time: {t1 - t0:.2f}s")
    # compare the two sequences
    # 比较两个序列
    for i in range(len(reference_ids)):
        if reference_ids[i] != generated_tokens[i]:
            print(f"Mismatch at {i}: {reference_ids[i]} != {generated_tokens[i]}")
            break
    print(f"Match: {reference_ids == generated_tokens}")

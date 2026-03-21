"""Minimal FP8 training for nanochat — tensorwise dynamic scaling only.
nanochat 的最小 FP8 训练 — 仅支持张量级动态缩放。

Drop-in replacement for torchao's Float8Linear (~2000 lines) with ~150 lines.
torchao Float8Linear（约 2000 行）的直接替代，用约 150 行实现。
We only need the "tensorwise" recipe (one scalar scale per tensor), not the full
我们只需要"张量级"方案（每个张量一个标量缩放因子），不需要 torchao 的完整
generality of torchao (rowwise scaling, FSDP float8 all-gather, DTensor, tensor
通用性（行级缩放、FSDP float8 全收集、DTensor、张量
subclass dispatch tables, etc.)
子类分发表等）

How FP8 training works
FP8 训练如何工作
======================
A standard Linear layer does one matmul in forward and two in backward:
标准 Linear 层在前向传播中执行一次矩阵乘法，在反向传播中执行两次：
  forward:      output     = input      @ weight.T
  backward:     grad_input = grad_output @ weight
                grad_weight= grad_output.T @ input

FP8 training wraps each of these three matmuls with:
FP8 训练为这三个矩阵乘法分别包装：
  1. Compute scale = FP8_MAX / max(|tensor|)  for each operand
     计算缩放因子 = FP8_MAX / max(|张量|)  为每个操作数
  2. Quantize: fp8_tensor = clamp(tensor * scale, -FP8_MAX, FP8_MAX).to(fp8)
     量化：fp8_tensor = clamp(张量 * 缩放因子, -FP8_MAX, FP8_MAX).to(fp8)
  3. Matmul via torch._scaled_mm (cuBLAS FP8 kernel, ~2x faster than bf16)
     通过 torch._scaled_mm 执行矩阵乘法（cuBLAS FP8 内核，比 bf16 快约 2 倍）
  4. Dequantize: _scaled_mm handles this internally using the inverse scales
     反量化：_scaled_mm 使用逆缩放因子在内部处理

The key insight: torch._scaled_mm and the float8 dtypes are PyTorch built-ins.
关键洞察：torch._scaled_mm 和 float8 数据类型是 PyTorch 内置的。
torchao is just orchestration around these primitives. We can call them directly.
torchao 只是围绕这些原语的编排。我们可以直接调用它们。

FP8 dtype choice
FP8 数据类型选择
================
There are two FP8 formats. We use both, following the standard convention:
有两种 FP8 格式。我们两者都用，遵循标准惯例：
  - float8_e4m3fn: 4-bit exponent, 3-bit mantissa, range [-448, 448]
    float8_e4m3fn：4 位指数，3 位尾数，范围 [-448, 448]
    Higher precision (more mantissa bits), used for input and weight.
    更高精度（更多尾数位），用于输入和权重。
  - float8_e5m2:   5-bit exponent, 2-bit mantissa, range [-57344, 57344]
    float8_e5m2：5 位指数，2 位尾数，范围 [-57344, 57344]
    Wider range (more exponent bits), used for gradients which can be large.
    更宽范围（更多指数位），用于可能很大的梯度。

torch._scaled_mm layout requirements
torch._scaled_mm 布局要求
=====================================
The cuBLAS FP8 kernel requires specific memory layouts:
cuBLAS FP8 内核需要特定的内存布局：
  - First argument (A):  must be row-major (contiguous)
    第一个参数 (A)：必须是行主序（连续）
  - Second argument (B): must be column-major (B.t().contiguous().t())
    第二个参数 (B)：必须是列主序（B.t().contiguous().t()）
If B is obtained by transposing a contiguous tensor (e.g. weight.t()), it is
如果 B 是通过转置连续张量获得的（例如 weight.t()），它
already column-major — no copy needed. Otherwise we use _to_col_major().
已经是列主序 — 无需复制。否则我们使用 _to_col_major()。

How this differs from torchao's approach
这与 torchao 方法的区别
========================================
torchao uses a "tensor subclass" architecture: Float8TrainingTensor is a subclass
torchao 使用"张量子类"架构：Float8TrainingTensor 是 torch.Tensor 的子类，
of torch.Tensor that bundles FP8 data + scale + metadata. It implements
捆绑了 FP8 数据 + 缩放因子 + 元数据。它实现了
__torch_dispatch__ with a dispatch table that intercepts every aten op (mm, t,
带有分发表的 __torch_dispatch__，拦截每个 aten 操作（mm、t、
reshape, clone, ...) and handles it in FP8-aware fashion. When you call
reshape、clone 等）并以 FP8 感知的方式处理。当你调用
  output = input @ weight.T
@ 运算符分发到 aten.mm，被拦截并路由到
the @ operator dispatches to aten.mm, which gets intercepted and routed to
幕后的 torch._scaled_mm。这大约需要 2000 行代码，因为需要
torch._scaled_mm behind the scenes. This is ~2000 lines of code because you need
为每个可能触及 FP8 张量的张量操作提供处理器。
a handler for every tensor operation that might touch an FP8 tensor.

We take a simpler approach: a single autograd.Function (_Float8Matmul) that takes
我们采用更简单的方法：一个单独的 autograd.Function（_Float8Matmul），接收
full-precision inputs, quantizes to FP8 internally, calls _scaled_mm, and returns
全精度输入，在内部量化为 FP8，调用 _scaled_mm，并返回
full-precision outputs. Marked @allow_in_graph so torch.compile treats it as one
全精度输出。标记 @allow_in_graph 使 torch.compile 将其视为一个
opaque node rather than trying to trace inside.
不透明节点，而不是尝试追踪内部。

The trade-off is in how torch.compile sees the two approaches:
权衡在于 torch.compile 如何看待这两种方法：
  - torchao: compile decomposes the tensor subclass (via __tensor_flatten__) and
    torchao：compile 分解张量子类（通过 __tensor_flatten__）并
    sees every individual op (amax, scale, cast, _scaled_mm) as separate graph
    将每个单独的操作（amax、scale、cast、_scaled_mm）视为单独的图
    nodes. Inductor can fuse these with surrounding operations (e.g. fuse the
    节点。Inductor 可以将它们与周围操作融合（例如将 amax 计算
    amax computation with the preceding layer's activation function).
    与前一层的激活函数融合）。
  - ours: compile sees a single opaque call. It can optimize everything around
    我们的：compile 看到单个不透明调用。它可以优化 FP8 线性层周围的
    the FP8 linear (attention, norms, etc.) but cannot fuse across the boundary.
    所有内容（注意力、归一化等），但无法跨边界融合。

Both call the exact same cuBLAS _scaled_mm kernel — the GPU matmul is identical.
两者都调用完全相同的 cuBLAS _scaled_mm 内核 — GPU 矩阵乘法是相同的。
The difference is only in the "glue" ops (amax, scale, cast) which are tiny
区别仅在于"粘合"操作（amax、scale、cast），它们与矩阵乘法相比
compared to the matmul. In practice this means our version is slightly faster
非常小。实际上这意味着我们的版本稍快一些
(less compilation overhead, no tensor subclass dispatch cost) but can produce
（更少的编译开销，没有张量子类分发成本），但在 torch.compile 下
subtly different floating-point rounding paths under torch.compile, since Inductor
可能产生微妙不同的浮点舍入路径，因为 Inductor
generates a different graph. Numerics are bitwise identical in eager mode.
生成不同的图。在 eager 模式下数值是逐位相同的。
"""

import torch
import torch.nn as nn

from nanochat.common import COMPUTE_DTYPE

# Avoid division by zero when computing scale from an all-zeros tensor
# 避免在从全零张量计算缩放时除以零
EPS = 1e-12


@torch.no_grad()
def _to_fp8(x, fp8_dtype):
    """Dynamically quantize a tensor to FP8 using tensorwise scaling.
    使用张量级缩放动态量化张量为 FP8。

    "Tensorwise" means one scalar scale for the entire tensor (as opposed to
    "张量级"意味着整个张量使用一个标量缩放（相对于
    "rowwise" which computes a separate scale per row). Tensorwise is faster
    "行级"为每行计算单独的缩放）。张量级更快
    because cuBLAS handles the scaling; rowwise needs the CUTLASS kernel.
    因为 cuBLAS 处理缩放；行级需要 CUTLASS 内核。

    Returns (fp8_data, inverse_scale) for use with torch._scaled_mm.
    返回 (fp8_data, inverse_scale) 用于 torch._scaled_mm。
    """
    fp8_max = torch.finfo(fp8_dtype).max
    # Compute the max absolute value across the entire tensor
    # 计算整个张量的最大绝对值
    amax = x.float().abs().max()
    # Scale maps [0, amax] -> [0, fp8_max]. Use float64 for the division to
    # 缩放将 [0, amax] 映射到 [0, fp8_max]。使用 float64 进行除法以
    # ensure consistent numerics between torch.compile and eager mode.
    # 确保 torch.compile 和 eager 模式之间的一致数值。
    # (torchao does the same upcast — without it, compile/eager can diverge)
    # （torchao 也做同样的上转型 — 没有它，compile/eager 可能会分歧）
    scale = fp8_max / amax.double().clamp(min=EPS)
    scale = scale.float()
    # Quantize: scale into FP8 range, saturate (clamp prevents overflow when
    # 量化：缩放到 FP8 范围，饱和（clamp 防止转换时溢出 —
    # casting — PyTorch's default is to wrap, not saturate), then cast to FP8
    # PyTorch 默认是回绕，不是饱和），然后转换为 FP8
    x_scaled = x.float() * scale
    x_clamped = x_scaled.clamp(-fp8_max, fp8_max)
    x_fp8 = x_clamped.to(fp8_dtype)
    # _scaled_mm expects the *inverse* of our scale (it multiplies by this to
    # _scaled_mm 期望我们缩放的*逆*（它乘以此值以
    # convert FP8 values back to the original range during the matmul)
    # 在矩阵乘法期间将 FP8 值转换回原始范围）
    inv_scale = scale.reciprocal()
    return x_fp8, inv_scale


def _to_col_major(x):
    """Rearrange a 2D tensor's memory to column-major layout.
    将 2D 张量的内存重新排列为列优先布局。

    torch._scaled_mm requires its second operand in column-major layout.
    torch._scaled_mm 要求其第二个操作数为列优先布局。
    The trick: transpose -> contiguous (forces a copy in transposed order)
    技巧：转置 -> 连续（强制以转置顺序复制）
    -> transpose back. The result has the same logical shape but column-major
    -> 转置回来。结果具有相同的逻辑形状但列优先
    strides, e.g. a [M, N] tensor gets strides (1, M) instead of (N, 1).
    步幅，例如 [M, N] 张量获得步幅 (1, M) 而不是 (N, 1)。
    """
    return x.t().contiguous().t()


# allow_in_graph tells torch.compile to treat this as an opaque operation —
# allow_in_graph 告诉 torch.compile 将此视为不透明操作 —
# dynamo won't try to decompose it into smaller ops. See the module docstring
# dynamo 不会尝试将其分解为更小的操作。参见模块文档字符串
# for how this differs from torchao's tensor subclass approach.
# 了解这与 torchao 张量子类方法有何不同。
@torch._dynamo.allow_in_graph
class _Float8Matmul(torch.autograd.Function):
    """Custom autograd for the three FP8 GEMMs of a Linear layer.
    Linear 层三个 FP8 GEMM 的自定义 autograd。

    The forward quantizes input and weight to FP8 and saves
    前向将输入和权重量化为 FP8 并保存
    the quantized tensors + scales for backward.
    量化张量 + 缩放用于反向传播。
    """

    @staticmethod
    def forward(ctx, input_2d, weight):
        # Quantize both operands to e4m3 (higher precision format)
        # 将两个操作数量化为 e4m3（更高精度格式）
        input_fp8, input_inv = _to_fp8(input_2d, torch.float8_e4m3fn)
        weight_fp8, weight_inv = _to_fp8(weight, torch.float8_e4m3fn)
        ctx.save_for_backward(input_fp8, input_inv, weight_fp8, weight_inv)

        # output = input @ weight.T
        # 输出 = 输入 @ 权重.T
        # input_fp8 is [B, K] contiguous = row-major (good for first arg)
        # input_fp8 是 [B, K] 连续 = 行优先（适合第一个参数）
        # weight_fp8 is [N, K] contiguous, so weight_fp8.t() is [K, N] with
        # weight_fp8 是 [N, K] 连续，所以 weight_fp8.t() 是 [K, N]
        # strides (1, K) = column-major (good for second arg, no copy needed!)
        # 步幅 (1, K) = 列优先（适合第二个参数，无需复制！）
        output = torch._scaled_mm(
            input_fp8,
            weight_fp8.t(),
            scale_a=input_inv,
            scale_b=weight_inv,
            out_dtype=input_2d.dtype,
            # use_fast_accum=True accumulates the dot products in lower precision.
            # use_fast_accum=True 以较低精度累加点积。
            # Slightly less accurate but measurably faster. Standard practice for
            # 稍微不太准确但明显更快。标准做法用于
            # the forward pass; we use False in backward for more precise gradients.
            # 前向传播；我们在反向传播中使用 False 以获得更精确的梯度。
            use_fast_accum=True,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        in_fp8, in_inv, w_fp8, w_inv = ctx.saved_tensors

        # === GEMM 1: grad_input = grad_output @ weight ===
        # === GEMM 1: 输入梯度 = 输出梯度 @ 权重 ===
        # Shapes: [B, N] @ [N, K] -> [B, K]
        # 形状：[B, N] @ [N, K] -> [B, K]
        # Gradients use e5m2 (wider range), weights use e4m3 (higher precision)
        # 梯度使用 e5m2（更宽范围），权重使用 e4m3（更高精度）
        go_fp8, go_inv = _to_fp8(grad_output, torch.float8_e5m2)
        # go_fp8 is [B, N] contiguous = row-major, good for first arg
        # go_fp8 是 [B, N] 连续 = 行优先，适合第一个参数
        # w_fp8 is [N, K] contiguous = row-major, need column-major for second arg
        # w_fp8 是 [N, K] 连续 = 行优先，第二个参数需要列优先
        w_col = _to_col_major(w_fp8)
        grad_input = torch._scaled_mm(
            go_fp8,
            w_col,
            scale_a=go_inv,
            scale_b=w_inv,
            out_dtype=grad_output.dtype,
            use_fast_accum=False,
        )

        # === GEMM 2: grad_weight = grad_output.T @ input ===
        # === GEMM 2: 权重梯度 = 输出梯度.T @ 输入 ===
        # Shapes: [N, B] @ [B, K] -> [N, K]
        # 形状：[N, B] @ [B, K] -> [N, K]
        # go_fp8 is [B, N] contiguous, we need go.T = [N, B] as first arg.
        # go_fp8 是 [B, N] 连续，我们需要 go.T = [N, B] 作为第一个参数。
        # Transposing gives column-major, but first arg needs row-major,
        # 转置得到列优先，但第一个参数需要行优先，
        # so we must call .contiguous() to physically rearrange the memory.
        # 所以我们必须调用 .contiguous() 来物理重新排列内存。
        go_T = go_fp8.t().contiguous()  # [N, B] row-major
                                        # [N, B] 行优先
        in_col = _to_col_major(in_fp8)    # [B, K] column-major
                                          # [B, K] 列优先
        grad_weight = torch._scaled_mm(
            go_T,
            in_col,
            scale_a=go_inv,
            scale_b=in_inv,
            out_dtype=grad_output.dtype,
            use_fast_accum=False,
        )

        return grad_input, grad_weight


class Float8Linear(nn.Linear):
    """Drop-in nn.Linear replacement that does FP8 compute.
    nn.Linear 的直接替代，执行 FP8 计算。

    Weights and biases remain in their original precision (e.g. fp32/bf16).
    权重和偏置保持原始精度（例如 fp32/bf16）。
    Only the matmul is performed in FP8 via the _Float8Matmul autograd function.
    只有矩阵乘法通过 _Float8Matmul autograd 函数以 FP8 执行。
    """

    def forward(self, input):
        # Cast input to COMPUTE_DTYPE (typically bf16) since _scaled_mm expects
        # 将输入转换为 COMPUTE_DTYPE（通常是 bf16），因为 _scaled_mm 期望
        # reduced precision input, and we no longer rely on autocast to do this.
        # 降低精度的输入，我们不再依赖 autocast 来做这个。
        input = input.to(COMPUTE_DTYPE)
        # _scaled_mm only works on 2D tensors, so flatten batch dimensions
        # _scaled_mm 仅适用于 2D 张量，因此展平批次维度
        orig_shape = input.shape
        input_2d = input.reshape(-1, orig_shape[-1])
        output = _Float8Matmul.apply(input_2d, self.weight)
        output = output.reshape(*orig_shape[:-1], output.shape[-1])
        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output

    @classmethod
    def from_float(cls, mod):
        """Create Float8Linear from nn.Linear, sharing the same weight and bias.
        从 nn.Linear 创建 Float8Linear，共享相同的权重和偏置。

        Uses meta device to avoid allocating a temporary weight tensor — we
        使用 meta 设备避免分配临时权重张量 — 我们
        create the module shell on meta (shapes/dtypes only, no memory), then
        在 meta 上创建模块外壳（仅形状/数据类型，无内存），然后
        point .weight and .bias to the original module's parameters.
        将 .weight 和 .bias 指向原始模块的参数。
        """
        with torch.device("meta"):
            new_mod = cls(mod.in_features, mod.out_features, bias=False)
        new_mod.weight = mod.weight
        new_mod.bias = mod.bias
        return new_mod


class Float8LinearConfig:
    """Minimal config matching torchao's API. Only tensorwise recipe is supported."""
    """匹配 torchao API 的最小配置。仅支持张量级配方。"""

    @staticmethod
    def from_recipe_name(recipe_name):
        if recipe_name != "tensorwise":
            raise ValueError(
                f"Only 'tensorwise' recipe is supported, got '{recipe_name}'. "
                f"仅支持 'tensorwise' 配方，得到 '{recipe_name}'。"
                f"Rowwise/axiswise recipes require the full torchao library."
                f"行级/轴级配方需要完整的 torchao 库。"
            )
        return Float8LinearConfig()


def convert_to_float8_training(module, *, config=None, module_filter_fn=None):
    """Replace nn.Linear layers with Float8Linear throughout a module.
    将模块中的 nn.Linear 层替换为 Float8Linear。

    Walks the module tree in post-order (children before parents) and swaps
    以后序遍历模块树（子节点在父节点之前）并交换
    each nn.Linear that passes the optional filter. The new Float8Linear shares
    每个通过可选过滤器的 nn.Linear。新的 Float8Linear 共享
    the original weight and bias tensors — no copies, no extra memory.
    原始权重和偏置张量 — 无复制，无额外内存。

    Args:
    参数：
        module: Root module to convert.
        module: 要转换的根模块。
        config: Float8LinearConfig (accepted for API compat, only tensorwise supported).
        config: Float8LinearConfig（为 API 兼容性接受，仅支持 tensorwise）。
        module_filter_fn: Optional filter(module, fqn) -> bool. Only matching Linears
        module_filter_fn: 可选过滤器(module, fqn) -> bool。只有匹配的 Linear
            are converted. Common use: skip layers with dims not divisible by 16
            被转换。常见用途：跳过维度不能被 16 整除的层
            (hardware requirement for FP8 matmuls on H100).
            （H100 上 FP8 矩阵乘法的硬件要求）。
    """
    def _convert(mod, prefix=""):
        for name, child in mod.named_children():
            fqn = f"{prefix}.{name}" if prefix else name
            _convert(child, fqn)
            if isinstance(child, nn.Linear) and not isinstance(child, Float8Linear):
                if module_filter_fn is None or module_filter_fn(child, fqn):
                    setattr(mod, name, Float8Linear.from_float(child))

    _convert(module)
    return module

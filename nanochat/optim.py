"""
A nice and efficient mixed AdamW/Muon Combined Optimizer.
一个高效且优雅的混合 AdamW/Muon 组合优化器。
Usually the embeddings and scalars go into AdamW, and the matrix parameters go into Muon.
通常嵌入层和标量参数使用 AdamW，矩阵参数使用 Muon。
Two versions are provided (MuonAdamW, DistMuonAdamW), for single GPU and distributed.
提供两个版本（MuonAdamW, DistMuonAdamW），分别用于单 GPU 和分布式。

Addapted from: https://github.com/KellerJordan/modded-nanogpt
改编自：https://github.com/KellerJordan/modded-nanogpt
Further contributions from @karpathy and @chrisjmccormick.
进一步贡献来自 @karpathy 和 @chrisjmccormick。
"""

import torch
import torch.distributed as dist
from torch import Tensor

# -----------------------------------------------------------------------------
"""
Good old AdamW optimizer, fused kernel.
经典的 AdamW 优化器，融合内核。
https://arxiv.org/abs/1711.05101
"""

@torch.compile(dynamic=False, fullgraph=True)
def adamw_step_fused(
    p: Tensor,              # (32768, 768) - parameter tensor
                             # (32768, 768) - 参数张量
    grad: Tensor,           # (32768, 768) - gradient, same shape as p
                             # (32768, 768) - 梯度，与 p 形状相同
    exp_avg: Tensor,        # (32768, 768) - first moment, same shape as p
                             # (32768, 768) - 一阶矩，与 p 形状相同
    exp_avg_sq: Tensor,     # (32768, 768) - second moment, same shape as p
                             # (32768, 768) - 二阶矩，与 p 形状相同
    step_t: Tensor,         # () - 0-D CPU tensor, step count
                             # () - 0-D CPU 张量，步数计数
    lr_t: Tensor,           # () - 0-D CPU tensor, learning rate
                             # () - 0-D CPU 张量，学习率
    beta1_t: Tensor,        # () - 0-D CPU tensor, beta1
                             # () - 0-D CPU 张量，beta1
    beta2_t: Tensor,        # () - 0-D CPU tensor, beta2
                             # () - 0-D CPU 张量，beta2
    eps_t: Tensor,          # () - 0-D CPU tensor, epsilon
                             # () - 0-D CPU 张量，epsilon
    wd_t: Tensor,           # () - 0-D CPU tensor, weight decay
                             # () - 0-D CPU 张量，权重衰减
) -> None:
    """
    Fused AdamW step: weight_decay -> momentum_update -> bias_correction -> param_update
    融合 AdamW 步骤：权重衰减 -> 动量更新 -> 偏差校正 -> 参数更新
    All in one compiled graph to eliminate Python overhead between ops.
    全部在一个编译图中完成，以消除操作之间的 Python 开销。
    The 0-D CPU tensors avoid recompilation when hyperparameter values change.
    0-D CPU 张量避免了超参数值更改时的重新编译。
    """
    # Weight decay (decoupled, applied before the update)
    # 权重衰减（解耦，在更新之前应用）
    p.mul_(1 - lr_t * wd_t)
    # Update running averages (lerp_ is cleaner and fuses well)
    # 更新运行平均值（lerp_ 更简洁且融合效果好）
    exp_avg.lerp_(grad, 1 - beta1_t)
    exp_avg_sq.lerp_(grad.square(), 1 - beta2_t)
    # Bias corrections
    # 偏差校正
    bias1 = 1 - beta1_t ** step_t
    bias2 = 1 - beta2_t ** step_t
    # Compute update and apply
    # 计算更新并应用
    denom = (exp_avg_sq / bias2).sqrt() + eps_t
    step_size = lr_t / bias1
    p.add_(exp_avg / denom, alpha=-step_size)

# -----------------------------------------------------------------------------
"""
Muon optimizer adapted and simplified from modded-nanogpt.
Muon 优化器改编并简化自 modded-nanogpt。
https://github.com/KellerJordan/modded-nanogpt

Background:
背景：
Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
Newton-Schulz 迭代计算 G 的零次幂/正交化。我们选择使用
quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
五次迭代，其系数选择以最大化零处的斜率。为了
of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
最小化步数，经验证明持续增加零处的斜率是有效的
zero even beyond the point where the iteration no longer converges all the way to one everywhere
即使超过迭代不再完全收敛到区间各处的一
on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
这个迭代因此不产生 UV^T 而是类似 US'V^T
where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
其中 S' 是对角矩阵，S_{ii}' ~ Uniform(0.5, 1.5)，这实际上不影响模型
performance at all relative to UV^T, where USV^T = G is the SVD.
性能相对于 UV^T，其中 USV^T = G 是 SVD。

Here, an alternative to Newton-Schulz iteration with potentially better convergence properties:
这里是一个替代 Newton-Schulz 迭代的可能具有更好收敛特性的方法：
Polar Express Sign Method for orthogonalization.
用于正交化的 Polar Express 符号方法。
https://arxiv.org/pdf/2505.16932
by Noah Amsel, David Persson, Christopher Musco, Robert M. Gower.
作者 Noah Amsel, David Persson, Christopher Musco, Robert M. Gower。

NorMuon variance reduction: per-neuron/column adaptive learning rate that normalizes
NorMuon 方差缩减：逐神经元/列自适应学习率，用于归一化
update scales after orthogonalization (Muon's output has non-uniform scales across neurons).
正交化后的更新尺度（Muon 的输出在神经元间具有不均匀的尺度）。
https://arxiv.org/pdf/2510.05491

Some of the changes in nanochat implementation:
nanochat 实现中的一些更改：
- Uses a simpler, more general approach to parameter grouping and stacking
- 使用更简单、更通用的参数分组和堆叠方法
- Uses a single fused kernel for the momentum -> polar_express -> variance_reduction -> update step
- 使用单个融合内核完成动量 -> polar_express -> 方差缩减 -> 更新步骤
- Makes no assumptions about model architecture (e.g. that attention weights are fused into QKVO format)
- 不对模型架构做假设（例如注意力权重融合为 QKVO 格式）
"""

# Coefficients for Polar Express (computed for num_iters=5, safety_factor=2e-2, cushion=2)
# Polar Express 的系数（为 num_iters=5, safety_factor=2e-2, cushion=2 计算）
# From https://arxiv.org/pdf/2505.16932
# 来自 https://arxiv.org/pdf/2505.16932
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
]

@torch.compile(dynamic=False, fullgraph=True)
def muon_step_fused(
    stacked_grads: Tensor,          # (12, 768, 3072) - stacked gradients
                                    # (12, 768, 3072) - 堆叠的梯度
    stacked_params: Tensor,         # (12, 768, 3072) - stacked parameters
                                    # (12, 768, 3072) - 堆叠的参数
    momentum_buffer: Tensor,        # (12, 768, 3072) - first moment buffer
                                    # (12, 768, 3072) - 一阶矩缓冲区
    second_momentum_buffer: Tensor, # (12, 768, 1) or (12, 1, 3072) - factored second moment
                                    # (12, 768, 1) 或 (12, 1, 3072) - 因子化二阶矩
    momentum_t: Tensor,             # () - 0-D CPU tensor, momentum coefficient
                                    # () - 0-D CPU 张量，动量系数
    lr_t: Tensor,                   # () - 0-D CPU tensor, learning rate
                                    # () - 0-D CPU 张量，学习率
    wd_t: Tensor,                   # () - 0-D CPU tensor, weight decay
                                    # () - 0-D CPU 张量，权重衰减
    beta2_t: Tensor,                # () - 0-D CPU tensor, beta2 for second moment
                                    # () - 0-D CPU 张量，二阶矩的 beta2
    ns_steps: int,                  # 5 - number of Newton-Schulz/Polar Express iterations
                                    # 5 - Newton-Schulz/Polar Express 迭代次数
    red_dim: int,                   # -1 or -2 - reduction dimension for variance
                                    # -1 或 -2 - 方差的归约维度
) -> None:
    """
    Fused Muon step: momentum -> polar_express -> variance_reduction -> cautious_update
    融合 Muon 步骤：动量 -> polar_express -> 方差缩减 -> 谨慎更新
    All in one compiled graph to eliminate Python overhead between ops.
    全部在一个编译图中完成，以消除操作之间的 Python 开销。
    Some of the constants are 0-D CPU tensors to avoid recompilation when values change.
    部分常量是 0-D CPU 张量，以避免值更改时的重新编译。
    """

    # Nesterov momentum
    # Nesterov 动量
    momentum = momentum_t.to(stacked_grads.dtype)
    momentum_buffer.lerp_(stacked_grads, 1 - momentum)
    g = stacked_grads.lerp_(momentum_buffer, momentum)

    # Polar express
    # Polar express 正交化
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.01 + 1e-6)
    if g.size(-2) > g.size(-1): # Tall matrix
                                # 高矩阵
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
    else: # Wide matrix (original math)
          # 宽矩阵（原始数学）
        for a, b, c in polar_express_coeffs[:ns_steps]:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    g = X

    # Variance reduction
    # 方差缩减
    beta2 = beta2_t.to(g.dtype)
    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
    red_dim_size = g.size(red_dim)
    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size
    v_norm = v_norm_sq.sqrt()
    second_momentum_buffer.lerp_(v_mean.to(dtype=second_momentum_buffer.dtype), 1 - beta2)
    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
    scaled_sq_sum = (v_mean * red_dim_size) * step_size.float().square()
    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()
    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))
    g = g * final_scale.to(g.dtype)

    # Cautious weight decay + parameter update
    # 谨慎权重衰减 + 参数更新
    lr = lr_t.to(g.dtype)
    wd = wd_t.to(g.dtype)
    mask = (g * stacked_params) >= 0
    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)

# -----------------------------------------------------------------------------
# Single GPU version of the MuonAdamW optimizer.
# MuonAdamW 优化器的单 GPU 版本。
# Used mostly for reference, debugging and testing.
# 主要用于参考、调试和测试。

class MuonAdamW(torch.optim.Optimizer):
    """
    Combined optimizer: Muon for 2D matrix params, AdamW for others, single GPU version.
    组合优化器：Muon 用于 2D 矩阵参数，AdamW 用于其他参数，单 GPU 版本。

    AdamW - Fused AdamW optimizer step.
    AdamW - 融合 AdamW 优化器步骤。

    Muon - MomentUm Orthogonalized by Newton-schulz
    Muon - 通过 Newton-schulz 正交化的动量优化器
    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    Muon 内部运行标准 SGD-动量，然后执行正交化后处理
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    步骤，其中每个 2D 参数的更新被替换为最近的正交
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    矩阵。为了高效地正交化每个更新，我们使用 Newton-Schulz 迭代，它具有
    the advantage that it can be stably run in bfloat16 on the GPU.
    可以在 GPU 上以 bfloat16 稳定运行的优势。

    Some warnings:
    一些警告：
    - The Muon optimizer should not be used for the embedding layer, the final fully connected layer,
    - Muon 优化器不应用于嵌入层、最后的全连接层
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    或任何 {0,1}-D 参数；这些都应该用标准方法（如 AdamW）优化。
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - 要将其用于 4D 卷积滤波器，只需展平其最后 3 个维度即可。

    Arguments:
    参数：
        param_groups: List of dicts, each containing:
        param_groups: 字典列表，每个包含：
            - 'params': List of parameters
            - 'params': 参数列表
            - 'kind': 'adamw' or 'muon'
            - 'kind': 'adamw' 或 'muon'
            - For AdamW groups: 'lr', 'betas', 'eps', 'weight_decay'
            - 对于 AdamW 组：'lr', 'betas', 'eps', 'weight_decay'
            - For Muon groups: 'lr', 'momentum', 'ns_steps', 'beta2', 'weight_decay'
            - 对于 Muon 组：'lr', 'momentum', 'ns_steps', 'beta2', 'weight_decay'
    """
    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        # 0-D CPU 张量以避免值更改时 torch.compile 重新编译
        # AdamW tensors
        # AdamW 张量
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        # Muon tensors
        # Muon 张量
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _step_adamw(self, group: dict) -> None:
        """
        AdamW update for each param in the group individually.
        对组中每个参数单独进行 AdamW 更新。
        Lazy init the state, fill in all 0-D tensors, call the fused kernel.
        延迟初始化状态，填充所有 0-D 张量，调用融合内核。
        """
        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]

            # State init
            # 状态初始化
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)
            exp_avg = state['exp_avg']
            exp_avg_sq = state['exp_avg_sq']
            state['step'] += 1

            # Fill 0-D tensors with current values
            # 用当前值填充 0-D 张量
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])

            # Fused update: weight_decay -> momentum -> bias_correction -> param_update
            # 融合更新：权重衰减 -> 动量 -> 偏差校正 -> 参数更新
            adamw_step_fused(
                p, grad, exp_avg, exp_avg_sq,
                self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t,
            )

    def _step_muon(self, group: dict) -> None:
        """
        Muon update for all params in the group (stacked for efficiency).
        对组中所有参数进行 Muon 更新（堆叠以提高效率）。
        Lazy init the state, fill in all 0-D tensors, call the fused kernel.
        延迟初始化状态，填充所有 0-D 张量，调用融合内核。
        """
        params: list[Tensor] = group['params']
        if not params:
            return

        # Get or create group-level buffers (stored in first param's state for convenience)
        # 获取或创建组级缓冲区（为方便存储在第一个参数的状态中）
        p = params[0]
        state = self.state[p]
        num_params = len(params)
        shape, device, dtype = p.shape, p.device, p.dtype

        # Momentum for every individual parameter
        # 每个参数的动量
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(num_params, *shape, dtype=dtype, device=device)
        momentum_buffer = state["momentum_buffer"]

        # Second momentum buffer is factored, either per-row or per-column
        # 二阶矩缓冲区是因子化的，可以是每行或每列
        if "second_momentum_buffer" not in state:
            state_shape = (num_params, shape[-2], 1) if shape[-2] >= shape[-1] else (num_params, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        second_momentum_buffer = state["second_momentum_buffer"]
        red_dim = -1 if shape[-2] >= shape[-1] else -2

        # Stack grads and params (NOTE: this assumes all params have the same shape)
        # 堆叠梯度和参数（注意：这假设所有参数具有相同的形状）
        stacked_grads = torch.stack([p.grad for p in params])
        stacked_params = torch.stack(params)

        # Fill all the 0-D tensors with current values
        # 用当前值填充所有 0-D 张量
        self._muon_momentum_t.fill_(group["momentum"])
        self._muon_beta2_t.fill_(group["beta2"] if group["beta2"] is not None else 0.0)
        self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
        self._muon_wd_t.fill_(group["weight_decay"])

        # Single fused kernel: momentum -> polar_express -> variance_reduction -> update
        # 单个融合内核：动量 -> polar_express -> 方差缩减 -> 更新
        muon_step_fused(
            stacked_grads,
            stacked_params,
            momentum_buffer,
            second_momentum_buffer,
            self._muon_momentum_t,
            self._muon_lr_t,
            self._muon_wd_t,
            self._muon_beta2_t,
            group["ns_steps"],
            red_dim,
        )

        # Copy back to original params
        # 复制回原始参数
        torch._foreach_copy_(params, list(stacked_params.unbind(0)))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                self._step_adamw(group)
            elif group['kind'] == 'muon':
                self._step_muon(group)
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")

# -----------------------------------------------------------------------------
# Distributed version of the MuonAdamW optimizer.
# MuonAdamW 优化器的分布式版本。
# Used for training on multiple GPUs.
# 用于多 GPU 训练。

class DistMuonAdamW(torch.optim.Optimizer):
    """
    Combined distributed optimizer: Muon for 2D matrix params, AdamW for others.
    组合分布式优化器：Muon 用于 2D 矩阵参数，AdamW 用于其他参数。

    See MuonAdamW for the algorithmic details of each optimizer. This class adds
    distributed communication to enable multi-GPU training without PyTorch DDP.

    Design Goals:
    - Overlap communication with computation (async ops)
    - Minimize memory by sharding optimizer states across ranks (ZeRO-2 style)
    - Batch small tensors into single comm ops where possible

    Communication Pattern (3-phase async):
    We use a 3-phase structure to maximize overlap between communication and compute:

        Phase 1: Launch all async reduce ops
            - Kick off all reduce_scatter/all_reduce operations
            - Don't wait - let them run in background while we continue

        Phase 2: Wait for reduces, compute updates, launch gathers
            - For each group: wait for its reduce, compute the update, launch gather
            - By processing groups in order, earlier gathers run while later computes happen

        Phase 3: Wait for gathers, copy back
            - Wait for all gathers to complete
            - Copy updated params back to original tensors (Muon only)

    AdamW Communication (ZeRO-2 style):
    - Small params (<1024 elements): all_reduce gradients, update full param on each rank.
      Optimizer state is replicated but these params are tiny (scalars, biases).
    - Large params: reduce_scatter gradients so each rank gets 1/N of the grad, update
      only that slice, then all_gather the updated slices. Optimizer state (exp_avg,
      exp_avg_sq) is sharded - each rank only stores state for its slice.
      Requires param.shape[0] divisible by world_size.

    Muon Communication (stacked + chunked):
    - All params in a Muon group must have the same shape (caller's responsibility).
    - Stack all K params into a single (K, *shape) tensor for efficient comm.
    - Divide K params across N ranks: each rank "owns" ceil(K/N) params.
    - reduce_scatter the stacked grads so each rank gets its chunk.
    - Each rank computes Muon update only for params it owns.
    - all_gather the updated params back to all ranks.
    - Optimizer state (momentum_buffer, second_momentum_buffer) is sharded by chunk.
    - Padding: if K doesn't divide evenly, we zero-pad to (ceil(K/N) * N) for comm,
      then ignore the padding when copying back.

    Buffer Reuse:
    - For Muon, we allocate stacked_grads for reduce_scatter input, then reuse the
      same buffer as the output for all_gather (stacked_params). This saves memory
      since we don't need both buffers simultaneously.

    Arguments:
        param_groups: List of dicts, each containing:
            - 'params': List of parameters
            - 'kind': 'adamw' or 'muon'
            - For AdamW groups: 'lr', 'betas', 'eps', 'weight_decay'
            - For Muon groups: 'lr', 'momentum', 'ns_steps', 'beta2', 'weight_decay'
    """
    def __init__(self, param_groups: list[dict]):
        super().__init__(param_groups, defaults={})
        # 0-D CPU tensors to avoid torch.compile recompilation when values change
        self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta1_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_eps_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._adamw_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_momentum_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_lr_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_wd_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
        self._muon_beta2_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")

    def _reduce_adamw(self, group: dict, world_size: int) -> dict:
        """Launch async reduce ops for AdamW group. Returns info dict with per-param infos."""
        param_infos = {}
        for p in group['params']:
            grad = p.grad
            if p.numel() < 1024:
                # Small params: all_reduce (no scatter/gather needed)
                future = dist.all_reduce(grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                param_infos[p] = dict(future=future, grad_slice=grad, is_small=True)
            else:
                # Large params: reduce_scatter
                assert grad.shape[0] % world_size == 0, f"AdamW reduce_scatter requires shape[0] ({grad.shape[0]}) divisible by world_size ({world_size})"
                rank_size = grad.shape[0] // world_size
                grad_slice = torch.empty_like(grad[:rank_size])
                future = dist.reduce_scatter_tensor(grad_slice, grad, op=dist.ReduceOp.AVG, async_op=True).get_future()
                param_infos[p] = dict(future=future, grad_slice=grad_slice, is_small=False)
        return dict(param_infos=param_infos)

    def _reduce_muon(self, group: dict, world_size: int) -> dict:
        """Launch async reduce op for Muon group. Returns info dict."""
        params = group['params']
        chunk_size = (len(params) + world_size - 1) // world_size
        padded_num_params = chunk_size * world_size
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype

        # Stack grads and zero-pad to padded_num_params
        grad_stack = torch.stack([p.grad for p in params])
        stacked_grads = torch.empty(padded_num_params, *shape, dtype=dtype, device=device)
        stacked_grads[:len(params)].copy_(grad_stack)
        if len(params) < padded_num_params:
            stacked_grads[len(params):].zero_()

        # Reduce_scatter to get this rank's chunk
        grad_chunk = torch.empty(chunk_size, *shape, dtype=dtype, device=device)
        future = dist.reduce_scatter_tensor(grad_chunk, stacked_grads, op=dist.ReduceOp.AVG, async_op=True).get_future()

        return dict(future=future, grad_chunk=grad_chunk, stacked_grads=stacked_grads, chunk_size=chunk_size)

    def _compute_adamw(self, group: dict, info: dict, gather_list: list, rank: int, world_size: int) -> None:
        """Wait for reduce, compute AdamW updates, launch gathers for large params."""
        param_infos = info['param_infos']
        for p in group['params']:
            pinfo = param_infos[p]
            pinfo['future'].wait()
            grad_slice = pinfo['grad_slice']
            state = self.state[p]

            # For small params, operate on full param; for large, operate on slice
            if pinfo['is_small']:
                p_slice = p
            else:
                rank_size = p.shape[0] // world_size
                p_slice = p[rank * rank_size:(rank + 1) * rank_size]

            # State init
            if not state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p_slice)
                state['exp_avg_sq'] = torch.zeros_like(p_slice)
            state['step'] += 1

            # Fill 0-D tensors and run fused kernel
            self._adamw_step_t.fill_(state['step'])
            self._adamw_lr_t.fill_(group['lr'])
            self._adamw_beta1_t.fill_(group['betas'][0])
            self._adamw_beta2_t.fill_(group['betas'][1])
            self._adamw_eps_t.fill_(group['eps'])
            self._adamw_wd_t.fill_(group['weight_decay'])
            adamw_step_fused(
                p_slice, grad_slice, state['exp_avg'], state['exp_avg_sq'],
                self._adamw_step_t, self._adamw_lr_t, self._adamw_beta1_t,
                self._adamw_beta2_t, self._adamw_eps_t, self._adamw_wd_t,
            )

            # Large params need all_gather
            if not pinfo['is_small']:
                future = dist.all_gather_into_tensor(p, p_slice, async_op=True).get_future()
                gather_list.append(dict(future=future, params=None))

    def _compute_muon(self, group: dict, info: dict, gather_list: list, rank: int) -> None:
        """Wait for reduce, compute Muon updates, launch gather."""
        info['future'].wait()
        params = group['params']
        chunk_size = info['chunk_size']
        grad_chunk = info['grad_chunk']
        p = params[0]
        shape, device, dtype = p.shape, p.device, p.dtype

        # How many params does this rank own?
        start_idx = rank * chunk_size
        num_owned = min(chunk_size, max(0, len(params) - start_idx))

        # Get or create group-level state
        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros(chunk_size, *shape, dtype=dtype, device=device)
        if "second_momentum_buffer" not in state:
            state_shape = (chunk_size, shape[-2], 1) if shape[-2] >= shape[-1] else (chunk_size, 1, shape[-1])
            state["second_momentum_buffer"] = torch.zeros(state_shape, dtype=dtype, device=device)
        red_dim = -1 if shape[-2] >= shape[-1] else -2

        # Build output buffer for all_gather
        updated_params = torch.empty(chunk_size, *shape, dtype=dtype, device=device)

        if num_owned > 0:
            owned_params = [params[start_idx + i] for i in range(num_owned)]
            stacked_owned = torch.stack(owned_params)

            # Fill 0-D tensors and run fused kernel
            self._muon_momentum_t.fill_(group["momentum"])
            self._muon_beta2_t.fill_(group["beta2"])
            self._muon_lr_t.fill_(group["lr"] * max(1.0, shape[-2] / shape[-1])**0.5)
            self._muon_wd_t.fill_(group["weight_decay"])
            muon_step_fused(
                grad_chunk[:num_owned], stacked_owned,
                state["momentum_buffer"][:num_owned], state["second_momentum_buffer"][:num_owned],
                self._muon_momentum_t, self._muon_lr_t, self._muon_wd_t, self._muon_beta2_t,
                group["ns_steps"], red_dim,
            )
            updated_params[:num_owned].copy_(stacked_owned)

        if num_owned < chunk_size:
            updated_params[num_owned:].zero_()

        # Reuse stacked_grads buffer for all_gather output
        stacked_params = info["stacked_grads"]
        future = dist.all_gather_into_tensor(stacked_params, updated_params, async_op=True).get_future()
        gather_list.append(dict(future=future, stacked_params=stacked_params, params=params))

    def _finish_gathers(self, gather_list: list) -> None:
        """Wait for all gathers and copy Muon params back."""
        for info in gather_list:
            info["future"].wait()
            if info["params"] is not None:
                # Muon: copy from stacked buffer back to individual params
                torch._foreach_copy_(info["params"], list(info["stacked_params"][:len(info["params"])].unbind(0)))

    @torch.no_grad()
    def step(self):
        rank = dist.get_rank()
        world_size = dist.get_world_size()

        # Phase 1: launch all async reduce ops
        reduce_infos: list[dict] = []
        for group in self.param_groups:
            if group['kind'] == 'adamw':
                reduce_infos.append(self._reduce_adamw(group, world_size))
            elif group['kind'] == 'muon':
                reduce_infos.append(self._reduce_muon(group, world_size))
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")

        # Phase 2: wait for reduces, compute updates, launch gathers
        gather_list: list[dict] = []
        for group, info in zip(self.param_groups, reduce_infos):
            if group['kind'] == 'adamw':
                self._compute_adamw(group, info, gather_list, rank, world_size)
            elif group['kind'] == 'muon':
                self._compute_muon(group, info, gather_list, rank)
            else:
                raise ValueError(f"Unknown optimizer kind: {group['kind']}")

        # Phase 3: wait for gathers, copy back
        self._finish_gathers(gather_list)

"""
Common utilities for nanochat.
nanochat 的通用工具。
"""

import os
import re
import logging
import urllib.request
import torch
import torch.distributed as dist
from filelock import FileLock

# The dtype used for compute (matmuls, activations). Master weights stay fp32 for optimizer precision.
# 用于计算的数据类型（矩阵乘法、激活）。主权重保持 fp32 以保证优化器精度。
# Linear layers cast their weights to this dtype in forward, replacing torch.amp.autocast.
# 线性层在 forward 中将其权重转换为此数据类型，替代 torch.amp.autocast。
# Override with NANOCHAT_DTYPE env var: "bfloat16", "float16", "float32"
# 通过 NANOCHAT_DTYPE 环境变量覆盖："bfloat16"、"float16"、"float32"
_DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
def _detect_compute_dtype():
    env = os.environ.get("NANOCHAT_DTYPE")
    if env is not None:
        return _DTYPE_MAP[env], f"set via NANOCHAT_DTYPE={env}"
    if torch.cuda.is_available():
        # bf16 requires SM 80+ (Ampere: A100, A10, etc.)
        # bf16 需要 SM 80+（Ampere：A100、A10 等）
        # Older GPUs like V100 (SM 70) and T4 (SM 75) only have fp16 tensor cores
        # 较旧的 GPU 如 V100（SM 70）和 T4（SM 75）只有 fp16 张量核心
        capability = torch.cuda.get_device_capability()
        if capability >= (8, 0):
            return torch.bfloat16, f"auto-detected: CUDA SM {capability[0]}{capability[1]} (bf16 supported)"
        # fp16 training requires GradScaler (not yet implemented), so fall back to fp32.
        # fp16 训练需要 GradScaler（尚未实现），因此回退到 fp32。
        # Users can still force fp16 via NANOCHAT_DTYPE=float16 if they know what they're doing.
        # 如果用户知道自己在做什么，仍然可以通过 NANOCHAT_DTYPE=float16 强制使用 fp16。
        return torch.float32, f"auto-detected: CUDA SM {capability[0]}{capability[1]} (pre-Ampere, bf16 not supported, using fp32)"
    return torch.float32, "auto-detected: no CUDA (CPU/MPS)"
COMPUTE_DTYPE, COMPUTE_DTYPE_REASON = _detect_compute_dtype()

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages."""
    """自定义格式化器，为日志消息添加颜色。"""
    # ANSI color codes
    # ANSI 颜色代码
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
                                # 青色
        'INFO': '\033[32m',     # Green
                                # 绿色
        'WARNING': '\033[33m',  # Yellow
                                # 黄色
        'ERROR': '\033[31m',    # Red
                                # 红色
        'CRITICAL': '\033[35m', # Magenta
                                # 品红色
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    def format(self, record):
        # Add color to the level name
        # 为级别名称添加颜色
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
        # Format the message
        # 格式化消息
        message = super().format(record)
        # Add color to specific parts of the message
        # 为消息的特定部分添加颜色
        if levelname == 'INFO':
            # Highlight numbers and percentages
            # 高亮数字和百分比
            message = re.sub(r'(\d+\.?\d*\s*(?:GB|MB|%|docs))', rf'{self.BOLD}\1{self.RESET}', message)
            message = re.sub(r'(Shard \d+)', rf'{self.COLORS["INFO"]}{self.BOLD}\1{self.RESET}', message)
        return message

def setup_default_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler]
    )

setup_default_logging()
logger = logging.getLogger(__name__)

def get_base_dir():
    # co-locate nanochat intermediates with other cached data in ~/.cache (by default)
    # 默认将 nanochat 中间文件与其他缓存数据一起放在 ~/.cache 中
    if os.environ.get("NANOCHAT_BASE_DIR"):
        nanochat_dir = os.environ.get("NANOCHAT_BASE_DIR")
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        nanochat_dir = os.path.join(cache_dir, "nanochat")
    os.makedirs(nanochat_dir, exist_ok=True)
    return nanochat_dir

def download_file_with_lock(url, filename, postprocess_fn=None):
    """
    Downloads a file from a URL to a local path in the base directory.
    从 URL 下载文件到基础目录中的本地路径。
    Uses a lock file to prevent concurrent downloads among multiple ranks.
    使用锁文件防止多个 rank 之间并发下载。
    """
    base_dir = get_base_dir()
    file_path = os.path.join(base_dir, filename)
    lock_path = file_path + ".lock"

    if os.path.exists(file_path):
        return file_path

    with FileLock(lock_path):
        # Only a single rank can acquire this lock
        # 只有一个 rank 可以获取此锁
        # All other ranks block until it is released
        # 所有其他 rank 阻塞直到它被释放

        # Recheck after acquiring lock
        # 获取锁后重新检查
        if os.path.exists(file_path):
            return file_path

        # Download the content as bytes
        # 下载内容为字节
        print(f"Downloading {url}...")
        with urllib.request.urlopen(url) as response:
            content = response.read() # bytes
                                      # 字节

        # Write to local file
        # 写入本地文件
        with open(file_path, 'wb') as f:
            f.write(content)
        print(f"Downloaded to {file_path}")

        # Run the postprocess function if provided
        # 如果提供了后处理函数则运行
        if postprocess_fn is not None:
            postprocess_fn(file_path)

    return file_path

def print0(s="",**kwargs):
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)

def print_banner():
    # Cool DOS Rebel font ASCII banner made with https://manytools.org/hacker-tools/ascii-banner/
    # 使用 https://manytools.org/hacker-tools/ascii-banner/ 制作的酷炫 DOS Rebel 字体 ASCII 横幅
    banner = """
                                                       █████                █████
                                                      ░░███                ░░███
     ████████    ██████   ████████    ██████   ██████  ░███████    ██████  ███████
    ░░███░░███  ░░░░░███ ░░███░░███  ███░░███ ███░░███ ░███░░███  ░░░░░███░░░███░
     ░███ ░███   ███████  ░███ ░███ ░███ ░███░███ ░░░  ░███ ░███   ███████  ░███
     ░███ ░███  ███░░███  ░███ ░███ ░███ ░███░███  ███ ░███ ░███  ███░░███  ░███ ███
     ████ █████░░████████ ████ █████░░██████ ░░██████  ████ █████░░███████  ░░█████
    ░░░░ ░░░░░  ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░░░   ░░░░░
    """
    print0(banner)

def is_ddp_requested() -> bool:
    """
    True if launched by torchrun (env present), even before init.
    如果由 torchrun 启动（环境变量存在）则返回 True，即使在初始化之前。
    Used to decide whether we *should* initialize a PG.
    用于决定我们是否应该初始化进程组。
    """
    return all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))

def is_ddp_initialized() -> bool:
    """
    True if torch.distributed is available and the process group is initialized.
    如果 torch.distributed 可用且进程组已初始化则返回 True。
    Used at cleanup to avoid destroying a non-existent PG.
    在清理时使用，以避免销毁不存在的进程组。
    """
    return dist.is_available() and dist.is_initialized()

def get_dist_info():
    if is_ddp_requested():
        # We rely on torchrun's env to decide if we SHOULD init.
        # (Initialization itself happens in compute init.)
        assert all(var in os.environ for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1

def autodetect_device_type():
    # prefer to use CUDA if available, otherwise use MPS, otherwise fallback on CPU
    # 优先使用 CUDA（如果可用），否则使用 MPS，否则回退到 CPU
    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"
    print0(f"Autodetected device type: {device_type}")
    return device_type

def compute_init(device_type="cuda"): # cuda|cpu|mps
    """Basic initialization that we keep doing over and over, so make common."""
    """我们反复进行的基本初始化，所以设为通用。"""

    assert device_type in ["cuda", "mps", "cpu"], "Invalid device type atm"
    if device_type == "cuda":
        assert torch.cuda.is_available(), "Your PyTorch installation is not configured for CUDA but device_type is 'cuda'"
    if device_type == "mps":
        assert torch.backends.mps.is_available(), "Your PyTorch installation is not configured for MPS but device_type is 'mps'"

    # Reproducibility
    # 可重现性
    # Note that we set the global seeds here, but most of the code uses explicit rng objects.
    # 注意我们在这里设置全局种子，但大多数代码使用显式的 rng 对象。
    # The only place where global rng might be used is nn.Module initialization of the model weights.
    # 可能使用全局 rng 的唯一地方是模型权重的 nn.Module 初始化。
    torch.manual_seed(42)
    if device_type == "cuda":
        torch.cuda.manual_seed(42)
    # skipping full reproducibility for now, possibly investigate slowdown later
    # 暂时跳过完全可重现性，可能稍后调查减速问题
    # torch.use_deterministic_algorithms(True)

    # Precision
    # 精度
    if device_type == "cuda":
        torch.set_float32_matmul_precision("high") # uses tf32 instead of fp32 for matmuls, see https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
                                                   # 矩阵乘法使用 tf32 而不是 fp32，参见 https://docs.pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html

    # Distributed setup: Distributed Data Parallel (DDP), optional, and requires CUDA
    # 分布式设置：分布式数据并行（DDP），可选，需要 CUDA
    is_ddp_requested, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if is_ddp_requested and device_type == "cuda":
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device)  # make "cuda" default to this device
                                        # 使 "cuda" 默认为此设备
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    else:
        device = torch.device(device_type) # mps|cpu

    if ddp_rank == 0:
        logger.info(f"Distributed world size: {ddp_world_size}")

    return is_ddp_requested, ddp_rank, ddp_local_rank, ddp_world_size, device

def compute_cleanup():
    """Companion function to compute_init, to clean things up before script exit"""
    """compute_init 的伴随函数，用于脚本退出前清理"""
    if is_ddp_initialized():
        dist.destroy_process_group()

class DummyWandb:
    """Useful if we wish to not use wandb but have all the same signatures"""
    """如果我们不想使用 wandb 但要有相同的签名，这很有用"""
    def __init__(self):
        pass
    def log(self, *args, **kwargs):
        pass
    def finish(self):
        pass

# hardcoded BF16 peak flops for various GPUs
# 各种 GPU 的硬编码 BF16 峰值 flops
# inspired by torchtitan: https://github.com/pytorch/torchtitan/blob/main/torchtitan/tools/utils.py
# 灵感来自 torchtitan：https://github.com/pytorch/torchtitan/blob/main/torchtitan/tools/utils.py
# and PR: https://github.com/karpathy/nanochat/pull/147
# 和 PR：https://github.com/karpathy/nanochat/pull/147
def get_peak_flops(device_name: str) -> float:
    name = device_name.lower()

    # Table order matters: more specific patterns first.
    # 表顺序很重要：更具体的模式在前。
    _PEAK_FLOPS_TABLE = (
        # NVIDIA Blackwell
        (["gb200"], 2.5e15),
        (["grace blackwell"], 2.5e15),
        (["b200"], 2.25e15),
        (["b100"], 1.8e15),
        # NVIDIA Hopper
        (["h200", "nvl"], 836e12),
        (["h200", "pcie"], 836e12),
        (["h200"], 989e12),
        (["h100", "nvl"], 835e12),
        (["h100", "pcie"], 756e12),
        (["h100"], 989e12),
        (["h800", "nvl"], 989e12),
        (["h800"], 756e12),
        # NVIDIA Ampere data center
        # NVIDIA Ampere 数据中心
        (["a100"], 312e12),
        (["a800"], 312e12),
        (["a40"], 149.7e12),
        (["a30"], 165e12),
        # NVIDIA Ada data center
        # NVIDIA Ada 数据中心
        (["l40s"], 362e12),
        (["l40-s"], 362e12),
        (["l40 s"], 362e12),
        (["l4"], 121e12),
        # AMD CDNA accelerators
        # AMD CDNA 加速器
        (["mi355"], 2.5e15),
        (["mi325"], 1.3074e15),
        (["mi300x"], 1.3074e15),
        (["mi300a"], 980.6e12),
        (["mi250x"], 383e12),
        (["mi250"], 362.1e12),
        # Consumer RTX
        # 消费级 RTX
        (["5090"], 209.5e12),
        (["4090"], 165.2e12),
        (["3090"], 71e12),
    )
    for patterns, flops in _PEAK_FLOPS_TABLE:
        if all(p in name for p in patterns):
            return flops
    if "data center gpu max 1550" in name:
        # Ponte Vecchio (PVC) - dynamic based on compute units
        # Ponte Vecchio (PVC) - 基于计算单元动态计算
        max_comp_units = torch.xpu.get_device_properties("xpu").max_compute_units
        return 512 * max_comp_units * 1300 * 10**6

    # Unknown GPU - return inf so MFU shows as 0% rather than a wrong guess
    # 未知 GPU - 返回 inf 以便 MFU 显示为 0% 而不是错误的猜测
    logger.warning(f"Peak flops undefined for: {device_name}, MFU will show as 0%")
    return float('inf')

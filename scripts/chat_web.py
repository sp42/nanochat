#!/usr/bin/env python3
"""
Unified web chat server - serves both UI and API from a single FastAPI instance.
统一 Web 聊天服务器 - 从单个 FastAPI 实例同时提供 UI 和 API。

Uses data parallelism to distribute requests across multiple GPUs. Each GPU loads
a full copy of the model, and incoming requests are distributed to available workers.
使用数据并行在多个 GPU 之间分发请求。每个 GPU 加载完整的模型副本，
传入的请求被分发到可用的工作器。

Launch examples:
启动示例：

- single available GPU (default)
- 单个可用 GPU（默认）
python -m scripts.chat_web

- 4 GPUs
- 4 个 GPU
python -m scripts.chat_web --num-gpus 4

To chat, open the URL printed in the console. (If on cloud box, make sure to use public IP)
要聊天，打开控制台中打印的 URL。（如果在云服务器上，请确保使用公共 IP）

Endpoints:
端点：
  GET  /           - Chat UI
  GET  /           - 聊天 UI
  POST /chat/completions - Chat API (streaming only)
  POST /chat/completions - 聊天 API（仅流式）
  GET  /health     - Health check with worker pool status
  GET  /health     - 健康检查，包含工作池状态
  GET  /stats      - Worker pool statistics and GPU utilization
  GET  /stats      - 工作池统计和 GPU 利用率

Abuse Prevention:
滥用防护：
  - Maximum 500 messages per request
  - 每个请求最多 500 条消息
  - Maximum 8000 characters per message
  - 每条消息最多 8000 个字符
  - Maximum 32000 characters total conversation length
  - 对话总长度最多 32000 个字符
  - Temperature clamped to 0.0-2.0
  - 温度限制在 0.0-2.0
  - Top-k clamped to 0-200 (0 disables top-k filtering, using full vocabulary)
  - Top-k 限制在 0-200（0 禁用 top-k 过滤，使用完整词表）
  - Max tokens clamped to 1-4096
  - 最大 token 数限制在 1-4096
"""

import argparse
import json
import os
import torch
import asyncio
import logging
import random
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import List, Optional, AsyncGenerator
from dataclasses import dataclass
from nanochat.common import compute_init, autodetect_device_type
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

# Abuse prevention limits
# 滥用防护限制
MAX_MESSAGES_PER_REQUEST = 500
MAX_MESSAGE_LENGTH = 8000
MAX_TOTAL_CONVERSATION_LENGTH = 32000
MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 2.0
MIN_TOP_K = 0 # 0 disables top-k filtering, using full vocabulary
              # 0 禁用 top-k 过滤，使用完整词表
MAX_TOP_K = 200
MIN_MAX_TOKENS = 1
MAX_MAX_TOKENS = 4096

parser = argparse.ArgumentParser(description='NanoChat Web Server')
parser.add_argument('-n', '--num-gpus', type=int, default=1, help='Number of GPUs to use (default: 1)')
parser.add_argument('-i', '--source', type=str, default="sft", help="Source of the model: sft|rl")
parser.add_argument('-t', '--temperature', type=float, default=0.8, help='Default temperature for generation')
parser.add_argument('-k', '--top-k', type=int, default=50, help='Default top-k sampling parameter')
parser.add_argument('-m', '--max-tokens', type=int, default=512, help='Default max tokens for generation')
parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load')
parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
parser.add_argument('-p', '--port', type=int, default=8000, help='Port to run the server on')
parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='Device type for evaluation: cuda|cpu|mps. empty => autodetect')
parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind the server to')
args = parser.parse_args()

# Configure logging for conversation traffic
# 配置对话流量的日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

@dataclass
class Worker:
    """A worker with a model loaded on a specific GPU."""
    """在特定 GPU 上加载模型的工作器。"""
    gpu_id: int
    device: torch.device
    engine: Engine
    tokenizer: object

class WorkerPool:
    """Pool of workers, each with a model replica on a different GPU."""
    """工作器池，每个工作器在不同 GPU 上有模型副本。"""

    def __init__(self, num_gpus: Optional[int] = None):
        if num_gpus is None:
            if device_type == "cuda":
                num_gpus = torch.cuda.device_count()
            else:
                num_gpus = 1 # e.g. cpu|mps
                             # 例如 cpu|mps
        self.num_gpus = num_gpus
        self.workers: List[Worker] = []
        self.available_workers: asyncio.Queue = asyncio.Queue()

    async def initialize(self, source: str, model_tag: Optional[str] = None, step: Optional[int] = None):
        """Load model on each GPU."""
        """在每个 GPU 上加载模型。"""
        print(f"Initializing worker pool with {self.num_gpus} GPUs...")
        if self.num_gpus > 1:
            assert device_type == "cuda", "Only CUDA supports multiple workers/GPUs. cpu|mps does not."
            assert device_type == "cuda", "只有 CUDA 支持多个工作器/GPU。cpu|mps 不支持。"

        for gpu_id in range(self.num_gpus):

            if device_type == "cuda":
                device = torch.device(f"cuda:{gpu_id}")
                print(f"Loading model on GPU {gpu_id}...")
                print(f"正在 GPU {gpu_id} 上加载模型...")
            else:
                device = torch.device(device_type) # e.g. cpu|mps
                                                    # 例如 cpu|mps
                print(f"Loading model on {device_type}...")
                print(f"正在 {device_type} 上加载模型...")

            model, tokenizer, _ = load_model(source, device, phase="eval", model_tag=model_tag, step=step)
            engine = Engine(model, tokenizer)
            worker = Worker(
                gpu_id=gpu_id,
                device=device,
                engine=engine,
                tokenizer=tokenizer,
            )
            self.workers.append(worker)
            await self.available_workers.put(worker)

        print(f"All {self.num_gpus} workers initialized!")
        print(f"所有 {self.num_gpus} 个工作器已初始化！")

    async def acquire_worker(self) -> Worker:
        """Get an available worker from the pool."""
        """从池中获取可用的工作器。"""
        return await self.available_workers.get()

    async def release_worker(self, worker: Worker):
        """Return a worker to the pool."""
        """将工作器归还到池中。"""
        await self.available_workers.put(worker)

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_k: Optional[int] = None

def validate_chat_request(request: ChatRequest):
    """Validate chat request to prevent abuse."""
    """验证聊天请求以防止滥用。"""
    # Check number of messages
    # 检查消息数量
    if len(request.messages) == 0:
        raise HTTPException(status_code=400, detail="At least one message is required")
        raise HTTPException(status_code=400, detail="至少需要一条消息")
    if len(request.messages) > MAX_MESSAGES_PER_REQUEST:
        raise HTTPException(
            status_code=400,
            detail=f"Too many messages. Maximum {MAX_MESSAGES_PER_REQUEST} messages allowed per request"
            detail=f"消息过多。每个请求最多允许 {MAX_MESSAGES_PER_REQUEST} 条消息"
        )

    # Check individual message lengths and total conversation length
    # 检查单条消息长度和对话总长度
    total_length = 0
    for i, message in enumerate(request.messages):
        if not message.content:
            raise HTTPException(status_code=400, detail=f"Message {i} has empty content")
            raise HTTPException(status_code=400, detail=f"消息 {i} 内容为空")

        msg_length = len(message.content)
        if msg_length > MAX_MESSAGE_LENGTH:
            raise HTTPException(
                status_code=400,
                detail=f"Message {i} is too long. Maximum {MAX_MESSAGE_LENGTH} characters allowed per message"
                detail=f"消息 {i} 过长。每条消息最多允许 {MAX_MESSAGE_LENGTH} 个字符"
            )
        total_length += msg_length

    if total_length > MAX_TOTAL_CONVERSATION_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Total conversation is too long. Maximum {MAX_TOTAL_CONVERSATION_LENGTH} characters allowed"
            detail=f"对话总长度过长。最多允许 {MAX_TOTAL_CONVERSATION_LENGTH} 个字符"
        )

    # Validate role values
    # 验证角色值
    for i, message in enumerate(request.messages):
        if message.role not in ["user", "assistant"]:
            raise HTTPException(
                status_code=400,
                detail=f"Message {i} has invalid role. Must be 'user', 'assistant', or 'system'"
                detail=f"消息 {i} 角色无效。必须是 'user'、'assistant' 或 'system'"
            )

    # Validate temperature
    # 验证温度
    if request.temperature is not None:
        if not (MIN_TEMPERATURE <= request.temperature <= MAX_TEMPERATURE):
            raise HTTPException(
                status_code=400,
                detail=f"Temperature must be between {MIN_TEMPERATURE} and {MAX_TEMPERATURE}"
                detail=f"温度必须在 {MIN_TEMPERATURE} 和 {MAX_TEMPERATURE} 之间"
            )

    # Validate top_k
    # 验证 top_k
    if request.top_k is not None:
        if not (MIN_TOP_K <= request.top_k <= MAX_TOP_K):
            raise HTTPException(
                status_code=400,
                detail=f"top_k must be between {MIN_TOP_K} and {MAX_TOP_K}"
                detail=f"top_k 必须在 {MIN_TOP_K} 和 {MAX_TOP_K} 之间"
            )

    # Validate max_tokens
    # 验证 max_tokens
    if request.max_tokens is not None:
        if not (MIN_MAX_TOKENS <= request.max_tokens <= MAX_MAX_TOKENS):
            raise HTTPException(
                status_code=400,
                detail=f"max_tokens must be between {MIN_MAX_TOKENS} and {MAX_MAX_TOKENS}"
                detail=f"max_tokens 必须在 {MIN_MAX_TOKENS} 和 {MAX_MAX_TOKENS} 之间"
            )

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on all GPUs on startup."""
    """启动时在所有 GPU 上加载模型。"""
    print("Loading nanochat models across GPUs...")
    print("正在跨 GPU 加载 nanochat 模型...")
    app.state.worker_pool = WorkerPool(num_gpus=args.num_gpus)
    await app.state.worker_pool.initialize(args.source, model_tag=args.model_tag, step=args.step)
    print(f"Server ready at http://localhost:{args.port}")
    print(f"服务器就绪：http://localhost:{args.port}")
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Serve the chat UI."""
    """提供聊天 UI。"""
    ui_html_path = os.path.join("nanochat", "ui.html")
    with open(ui_html_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    # Replace the API_URL to use the same origin
    # 替换 API_URL 以使用相同的源
    html_content = html_content.replace(
        "const API_URL = `http://${window.location.hostname}:8000`;",
        "const API_URL = '';"
    )
    return HTMLResponse(content=html_content)


@app.get("/logo.svg")
async def logo():
    """Serve the NanoChat logo for favicon and header."""
    """提供 NanoChat logo 用于 favicon 和头部。"""
    logo_path = os.path.join("nanochat", "logo.svg")
    return FileResponse(logo_path, media_type="image/svg+xml")

async def generate_stream(
    worker: Worker,
    tokens,
    temperature=None,
    max_new_tokens=None,
    top_k=None
) -> AsyncGenerator[str, None]:
    """Generate assistant response with streaming."""
    """生成助手响应并流式传输。"""
    temperature = temperature if temperature is not None else args.temperature
    max_new_tokens = max_new_tokens if max_new_tokens is not None else args.max_tokens
    top_k = top_k if top_k is not None else args.top_k

    assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")
    bos = worker.tokenizer.get_bos_token_id()

    # Accumulate tokens to properly handle multi-byte UTF-8 characters (like emojis)
    # 累积 token 以正确处理多字节 UTF-8 字符（如表情符号）
    accumulated_tokens = []
    # Track the last complete UTF-8 string (without replacement characters)
    # 跟踪最后一个完整的 UTF-8 字符串（无替换字符）
    last_clean_text = ""

    for token_column, token_masks in worker.engine.generate(
        tokens,
        num_samples=1,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        seed=random.randint(0, 2**31 - 1)
    ):
        token = token_column[0]

        # Stopping criteria
        # 停止条件
        if token == assistant_end or token == bos:
            break

        # Append the token to sequence
        # 将 token 追加到序列
        accumulated_tokens.append(token)
        # Decode all accumulated tokens to get proper UTF-8 handling
        # Note that decode is a quite efficient operation, basically table lookup and string concat
        # 解码所有累积的 token 以获得正确的 UTF-8 处理
        # 注意 decode 是一个非常高效的操作，基本上是表查找和字符串连接
        current_text = worker.tokenizer.decode(accumulated_tokens)
        # Only emit text if it doesn't end with a replacement character
        # This ensures we don't emit incomplete UTF-8 sequences
        # 只有当文本不以替换字符结尾时才发出
        # 这确保我们不会发出不完整的 UTF-8 序列
        if not current_text.endswith('�'):
            # Extract only the new text since last clean decode
            # 仅提取自上次干净解码以来的新文本
            new_text = current_text[len(last_clean_text):]
            if new_text:  # Only yield if there's new content
                           # 只有有新内容时才 yield
                yield f"data: {json.dumps({'token': new_text, 'gpu': worker.gpu_id}, ensure_ascii=False)}\n\n"
                last_clean_text = current_text

    yield f"data: {json.dumps({'done': True})}\n\n"

@app.post("/chat/completions")
async def chat_completions(request: ChatRequest):
    """Chat completion endpoint (streaming only) - uses worker pool for multi-GPU."""
    """聊天完成端点（仅流式）- 使用工作池支持多 GPU。"""

    # Basic validation to prevent abuse
    # 基本验证以防止滥用
    validate_chat_request(request)

    # Log incoming conversation to console
    # 将传入的对话记录到控制台
    logger.info("="*20)
    for i, message in enumerate(request.messages):
        logger.info(f"[{message.role.upper()}]: {message.content}")
    logger.info("-"*20)

    # Acquire a worker from the pool (will wait if all are busy)
    # 从池中获取工作器（如果都忙则等待）
    worker_pool = app.state.worker_pool
    worker = await worker_pool.acquire_worker()

    try:
        # Build conversation tokens
        # 构建对话 token
        bos = worker.tokenizer.get_bos_token_id()
        user_start = worker.tokenizer.encode_special("<|user_start|>")
        user_end = worker.tokenizer.encode_special("<|user_end|>")
        assistant_start = worker.tokenizer.encode_special("<|assistant_start|>")
        assistant_end = worker.tokenizer.encode_special("<|assistant_end|>")

        conversation_tokens = [bos]
        for message in request.messages:
            if message.role == "user":
                conversation_tokens.append(user_start)
                conversation_tokens.extend(worker.tokenizer.encode(message.content))
                conversation_tokens.append(user_end)
            elif message.role == "assistant":
                conversation_tokens.append(assistant_start)
                conversation_tokens.extend(worker.tokenizer.encode(message.content))
                conversation_tokens.append(assistant_end)

        conversation_tokens.append(assistant_start)

        # Streaming response with worker release after completion
        # 完成后释放工作器的流式响应
        response_tokens = []
        async def stream_and_release():
            try:
                async for chunk in generate_stream(
                    worker,
                    conversation_tokens,
                    temperature=request.temperature,
                    max_new_tokens=request.max_tokens,
                    top_k=request.top_k
                ):
                    # Accumulate response for logging
                    # 累积响应用于日志记录
                    chunk_data = json.loads(chunk.replace("data: ", "").strip())
                    if "token" in chunk_data:
                        response_tokens.append(chunk_data["token"])
                    yield chunk
            finally:
                # Log the assistant response to console
                # 将助手响应记录到控制台
                full_response = "".join(response_tokens)
                logger.info(f"[ASSISTANT] (GPU {worker.gpu_id}): {full_response}")
                logger.info("="*20)
                # Release worker back to pool after streaming is done
                # 流式传输完成后将工作器归还到池中
                await worker_pool.release_worker(worker)

        return StreamingResponse(
            stream_and_release(),
            media_type="text/event-stream"
        )
    except Exception as e:
        # Make sure to release worker even on error
        # 确保即使出错也释放工作器
        await worker_pool.release_worker(worker)
        raise e

@app.get("/health")
async def health():
    """Health check endpoint."""
    """健康检查端点。"""
    worker_pool = getattr(app.state, 'worker_pool', None)
    return {
        "status": "ok",
        "ready": worker_pool is not None and len(worker_pool.workers) > 0,
        "num_gpus": worker_pool.num_gpus if worker_pool else 0,
        "available_workers": worker_pool.available_workers.qsize() if worker_pool else 0
    }

@app.get("/stats")
async def stats():
    """Get worker pool statistics."""
    """获取工作池统计信息。"""
    worker_pool = app.state.worker_pool
    return {
        "total_workers": len(worker_pool.workers),
        "available_workers": worker_pool.available_workers.qsize(),
        "busy_workers": len(worker_pool.workers) - worker_pool.available_workers.qsize(),
        "workers": [
            {
                "gpu_id": w.gpu_id,
                "device": str(w.device)
            } for w in worker_pool.workers
        ]
    }

if __name__ == "__main__":
    import uvicorn
    print(f"Starting NanoChat Web Server")
    print(f"正在启动 NanoChat Web 服务器")
    print(f"Temperature: {args.temperature}, Top-k: {args.top_k}, Max tokens: {args.max_tokens}")
    uvicorn.run(app, host=args.host, port=args.port)

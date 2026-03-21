"""
New and upgraded chat mode because a lot of the code has changed since the last one.
新的升级版聊天模式，因为自上次以来很多代码已经改变。

Intended to be run single GPU only atm:
目前仅支持单 GPU 运行：
python -m scripts.chat_cli
"""
import argparse
import torch
from nanochat.common import compute_init, autodetect_device_type
from nanochat.engine import Engine
from nanochat.checkpoint_manager import load_model

parser = argparse.ArgumentParser(description='Chat with the model')
parser.add_argument('-i', '--source', type=str, default="sft", help="Source of the model: sft|rl")
parser.add_argument('-i', '--source', type=str, default="sft", help="模型来源: sft|rl")
parser.add_argument('-g', '--model-tag', type=str, default=None, help='Model tag to load')
parser.add_argument('-g', '--model-tag', type=str, default=None, help='要加载的模型标签')
parser.add_argument('-s', '--step', type=int, default=None, help='Step to load')
parser.add_argument('-s', '--step', type=int, default=None, help='要加载的步数')
parser.add_argument('-p', '--prompt', type=str, default='', help='Prompt the model, get a single response back')
parser.add_argument('-p', '--prompt', type=str, default='', help='提示模型，获取单个响应')
parser.add_argument('-t', '--temperature', type=float, default=0.6, help='Temperature for generation')
parser.add_argument('-t', '--temperature', type=float, default=0.6, help='生成的温度')
parser.add_argument('-k', '--top-k', type=int, default=50, help='Top-k sampling parameter')
parser.add_argument('-k', '--top-k', type=int, default=50, help='Top-k 采样参数')
parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='Device type for evaluation: cuda|cpu|mps. empty => autodetect')
parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps'], help='评估的设备类型: cuda|cpu|mps。空 => 自动检测')
args = parser.parse_args()

# Init the model and tokenizer
# 初始化模型和分词器

device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
model, tokenizer, meta = load_model(args.source, device, phase="eval", model_tag=args.model_tag, step=args.step)

# Special tokens for the chat state machine
# 聊天状态机的特殊 token
bos = tokenizer.get_bos_token_id()
user_start, user_end = tokenizer.encode_special("<|user_start|>"), tokenizer.encode_special("<|user_end|>")
assistant_start, assistant_end = tokenizer.encode_special("<|assistant_start|>"), tokenizer.encode_special("<|assistant_end|>")

# Create Engine for efficient generation
# 创建 Engine 用于高效生成
engine = Engine(model, tokenizer)

print("\nNanoChat Interactive Mode")
print("\nNanoChat 交互模式")
print("-" * 50)
print("Type 'quit' or 'exit' to end the conversation")
print("输入 'quit' 或 'exit' 结束对话")
print("Type 'clear' to start a new conversation")
print("输入 'clear' 开始新对话")
print("-" * 50)

conversation_tokens = [bos]

while True:

    if args.prompt:
        # Get the prompt from the launch command
        # 从启动命令获取提示
        user_input = args.prompt
    else:
        # Get the prompt interactively from the console
        # 从控制台交互式获取提示
        try:
            user_input = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            print("\n再见！")
            break

    # Handle special commands
    # 处理特殊命令
    if user_input.lower() in ['quit', 'exit']:
        print("Goodbye!")
        print("再见！")
        break

    if user_input.lower() == 'clear':
        conversation_tokens = [bos]
        print("Conversation cleared.")
        print("对话已清除。")
        continue

    if not user_input:
        continue

    # Add User message to the conversation
    # 将用户消息添加到对话中
    conversation_tokens.append(user_start)
    conversation_tokens.extend(tokenizer.encode(user_input))
    conversation_tokens.append(user_end)

    # Kick off the assistant
    # 启动助手
    conversation_tokens.append(assistant_start)
    generate_kwargs = {
        "num_samples": 1,
        "max_tokens": 256,
        "temperature": args.temperature,
        "top_k": args.top_k,
    }
    response_tokens = []
    print("\nAssistant: ", end="", flush=True)
    for token_column, token_masks in engine.generate(conversation_tokens, **generate_kwargs):
        token = token_column[0] # pop the batch dimension (num_samples=1)
                                # 弹出批量维度 (num_samples=1)
        response_tokens.append(token)
        token_text = tokenizer.decode([token])
        print(token_text, end="", flush=True)
    print()
    # we have to ensure that the assistant end token is the last token
    # so even if generation ends due to max tokens, we have to append it to the end
    # 我们必须确保助手结束 token 是最后一个 token
    # 所以即使生成因达到最大 token 数而结束，我们也必须在末尾追加它
    if response_tokens[-1] != assistant_end:
        response_tokens.append(assistant_end)
    conversation_tokens.extend(response_tokens)

    # In the prompt mode, we only want a single response and exit
    # 在提示模式下，我们只需要单个响应然后退出
    if args.prompt:
        break

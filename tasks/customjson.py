"""
CustomJSON task for loading conversations from JSONL files.
用于从 JSONL 文件加载对话的 CustomJSON 任务。
Each line in the JSONL file should be a JSON array of messages.
JSONL 文件中的每一行应该是一个消息的 JSON 数组。
"""

import os
import json
from tasks.common import Task

class CustomJSON(Task):
    """
    Load conversations from a JSONL file.
    从 JSONL 文件加载对话。
    Each line should be a JSON array of message objects with 'role' and 'content' fields.
    每行应该是一个包含 'role' 和 'content' 字段的消息对象的 JSON 数组。
    Example line: [{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello"}]
    示例行：[{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello"}]
    """

    def __init__(self, filepath, **kwargs):
        super().__init__(**kwargs)
        self.filepath = filepath
        self.conversations = []

        # Load all conversations from the JSONL file
        # 从 JSONL 文件加载所有对话
        if not os.path.exists(filepath):
            # Helpful error message due to recent change. Will be removed in the future.
            # 由于最近更改的有用错误消息。将来会删除。
            print("-" * 80)
            print(f"Warning: File {filepath} does not exist")
            print(f"警告：文件 {filepath} 不存在")
            print("HINT (Oct 21 2025)")
            print("提示（2025年10月21日）")
            print("If you recently did a git pull and suddenly see this, it might be due to the new addition of identity conversations")
            print("如果您最近做了 git pull 并突然看到这个，可能是由于新添加的身份对话")
            print("See this discussion for more details: https://github.com/karpathy/nanochat/discussions/139")
            print("有关更多详细信息，请参阅此讨论：https://github.com/karpathy/nanochat/discussions/139")
            print("Quick fix: simply run the following command to download the file and you're done:")
            print("快速修复：只需运行以下命令下载文件即可：")
            print(f"curl -L -o {filepath} https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl")
            print("-" * 80)

        else:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:  # skip empty lines
                                     # 跳过空行
                        continue
                    messages = json.loads(line)
                    # Validate the conversation structure
                    # 验证对话结构
                    assert isinstance(messages, list), f"Expected list of messages, got {type(messages)}"
                    assert isinstance(messages, list), f"期望消息列表，得到 {type(messages)}"
                    assert len(messages) >= 2, f"Conversation must have at least 2 messages, got {len(messages)}"
                    assert len(messages) >= 2, f"对话必须至少有 2 条消息，得到 {len(messages)}"
                    # Validate message structure and alternating roles
                    # 验证消息结构和交替角色
                    for i, message in enumerate(messages):
                        assert "role" in message, f"Message {i} missing 'role' field"
                        assert "role" in message, f"消息 {i} 缺少 'role' 字段"
                        assert "content" in message, f"Message {i} missing 'content' field"
                        assert "content" in message, f"消息 {i} 缺少 'content' 字段"
                        expected_role = "user" if i % 2 == 0 else "assistant"
                        assert message["role"] == expected_role, f"Message {i} has role {message['role']} but should be {expected_role}"
                        assert message["role"] == expected_role, f"消息 {i} 的角色是 {message['role']} 但应该是 {expected_role}"
                        assert isinstance(message["content"], str), f"Message {i} content must be a string"
                        assert isinstance(message["content"], str), f"消息 {i} 的内容必须是字符串"

                    self.conversations.append(messages)

        self.length = len(self.conversations)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        messages = self.conversations[index]
        conversation = {
            "messages": messages,
        }
        return conversation


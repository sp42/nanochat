"""
SmolTalk by HuggingFace. Good "general" conversational dataset.
HuggingFace 的 SmolTalk。良好的"通用"对话数据集。
https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk
We use the "smol" version, which is more appropriate for smaller models.
我们使用 "smol" 版本，它更适合较小的模型。
"""

from datasets import load_dataset
from tasks.common import Task

class SmolTalk(Task):
    """ smol-smoltalk dataset. train is 460K rows, test is 24K rows. """
    """ smol-smoltalk 数据集。训练集 460K 行，测试集 24K 行。 """

    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"], "SmolTalk split must be train|test"
        assert split in ["train", "test"], "SmolTalk split 必须是 train|test"
        self.ds = load_dataset("HuggingFaceTB/smol-smoltalk", split=split).shuffle(seed=42)
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        messages = row["messages"]
        # ---------------------------------------------------------------------
        # sanity checking asserts here
        # 这里的健全性检查断言
        # TODO: we could remove these asserts later, for now just don't want any footguns
        # TODO: 我们可以稍后删除这些断言，现在只是不想有任何陷阱
        # there is an optional system message at the beginning
        # 开头有一个可选的系统消息
        assert len(messages) >= 1
        first_message = messages[0]
        if first_message["role"] == "system":
            rest_messages = messages[1:] # optional system message is OK
                                         # 可选的系统消息是可以的
        else:
            rest_messages = messages
        assert len(rest_messages) >= 2, "SmolTalk messages must have at least 2 messages"
        assert len(rest_messages) >= 2, "SmolTalk 消息必须至少有 2 条消息"
        for i, message in enumerate(rest_messages):
            # user and assistant alternate as user,assistant,user,assistant,...
            # 用户和助手交替为 user,assistant,user,assistant,...
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == expected_role, f"Message {i} has role {message['role']} but should be {expected_role}"
            assert message["role"] == expected_role, f"消息 {i} 的角色是 {message['role']} 但应该是 {expected_role}"
            assert isinstance(message["content"], str), "Content must be a string"
            assert isinstance(message["content"], str), "内容必须是字符串"
        # ---------------------------------------------------------------------
        # create and return the Conversation object (ok to emit the system message too)
        # 创建并返回 Conversation 对象（也可以发出系统消息）
        conversation = {
            "messages": messages,
        }
        return conversation

"""
GSM8K evaluation.
GSM8K 评估。
https://huggingface.co/datasets/openai/gsm8k

Example problem instance:
示例问题实例：

Question:
问题：
Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Answer:
答案：
Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.
Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.
#### 10

Notice that GSM8K uses tool calls inside << >> tags.
注意 GSM8K 在 << >> 标签内使用工具调用。
"""

import re
from datasets import load_dataset
from tasks.common import Task


GSM_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
def extract_answer(completion):
    """
    Extract the numerical answer after #### marker.
    提取 #### 标记后的数字答案。
    Follows official code for normalization:
    遵循官方代码进行归一化：
    https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/dataset.py#L28
    """
    match = GSM_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    return None


class GSM8K(Task):

    def __init__(self, subset, split, **kwargs):
        super().__init__(**kwargs)
        assert subset in ["main", "socratic"], "GSM8K subset must be main|socratic"
        assert split in ["train", "test"], "GSM8K split must be train|test"
        self.ds = load_dataset("openai/gsm8k", subset, split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        """ Get a single problem from the dataset. """
        """ 从数据集中获取单个问题。 """
        row = self.ds[index]
        question = row['question'] # string of the question prompt
                                   # 问题提示的字符串
        answer = row['answer'] # string of the full solution and the answer after #### marker
                               # 完整解决方案的字符串和 #### 标记后的答案
        # Create and return the Conversation object
        # 创建并返回 Conversation 对象
        # This is tricky because GSM8K uses tool calls, which we need to parse here.
        # 这很棘手，因为 GSM8K 使用工具调用，我们需要在这里解析。
        assistant_message_parts = []
        parts = re.split(r'(<<[^>]+>>)', answer)
        for part in parts:
            if part.startswith('<<') and part.endswith('>>'):
                # This is a calculator tool call
                # 这是一个计算器工具调用
                inner = part[2:-2]  # Remove << >>
                                    # 移除 << >>
                # Split on = to get expression and result
                # 在 = 上分割以获取表达式和结果
                if '=' in inner:
                    expr, result = inner.rsplit('=', 1)
                else:
                    expr, result = inner, ""
                # Add the tool call as a part
                # 将工具调用作为一部分添加
                assistant_message_parts.append({"type": "python", "text": expr})
                # Add the result as a part
                # 将结果作为一部分添加
                assistant_message_parts.append({"type": "python_output", "text": result})
            else:
                # Regular text in between tool calls
                # 工具调用之间的常规文本
                assistant_message_parts.append({"type": "text", "text": part})
        # Now put it all together
        # 现在把它们放在一起
        messages = [
            {"role": "user", "content": question}, # note: simple string
                                                   # 注意：简单字符串
            {"role": "assistant", "content": assistant_message_parts}, # note: list of parts (as dicts)
                                                                       # 注意：部分列表（作为字典）
        ]
        conversation = {
            "messages": messages,
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        """
        Given (conversation, completion), return evaluation outcome (0 = wrong, 1 = correct)
        给定 (conversation, completion)，返回评估结果（0 = 错误，1 = 正确）
        Note that:
        注意：
        - the conversation has both user AND assistant message (containing the ground truth answer)
        - conversation 有用户和助手消息（包含真实答案）
        - the assistant_response is usually the alternative assistant message achieved via sampling
        - assistant_response 通常是通过采样获得的替代助手消息

        TODO: Technically, assistant_response should be a Message (either a string or a list of parts)
        TODO: 从技术上讲，assistant_response 应该是一个 Message（字符串或部分列表）
              We can handle this later possibly. For now just assume string.
              我们可以稍后处理这个问题。目前只假设字符串。
        """
        assert isinstance(assistant_response, str), "Assuming simple string response for now"
        # First extract the ground truth answer
        # 首先提取真实答案
        assistant_message = conversation['messages'][-1]
        assert assistant_message['role'] == "assistant", "Last message must be from the Assistant"
        assert isinstance(assistant_message['content'], list), "This is expected to be a list of parts"
        last_text_part = assistant_message['content'][-1]['text'] # this contains the final answer in GSM8K
                                                                  # 这包含 GSM8K 中的最终答案
        # Extract both the ground truth answer and the predicted answer
        # 提取真实答案和预测答案
        ref_num = extract_answer(last_text_part)
        pred_num = extract_answer(assistant_response)
        # Compare and return the success as int
        # 比较并返回成功作为整数
        is_correct = int(pred_num == ref_num)
        return is_correct

    def reward(self, conversation, assistant_response):
        """
        Used during RL. To keep things simple, just re-use the evaluation above.
        在 RL 期间使用。为了简单起见，只需重用上面的评估。
        Later this could be made more complex (e.g. format matching etc.)
        以后这可以变得更复杂（例如格式匹配等）
        """
        is_correct = self.evaluate(conversation, assistant_response)
        is_correct_float = float(is_correct)
        return is_correct_float

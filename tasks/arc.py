"""
The ARC dataset from Allen AI.
Allen AI 的 ARC 数据集。
https://huggingface.co/datasets/allenai/ai2_arc
"""

from datasets import load_dataset
from tasks.common import Task, render_mc

class ARC(Task):

    def __init__(self, subset, split, **kwargs):
        super().__init__(**kwargs)
        assert subset in ["ARC-Easy", "ARC-Challenge"], "ARC subset must be ARC-Easy or ARC-Challenge"
        assert subset in ["ARC-Easy", "ARC-Challenge"], "ARC subset 必须是 ARC-Easy 或 ARC-Challenge"
        assert split in ["train", "validation", "test"], "ARC split must be train|validation|test"
        assert split in ["train", "validation", "test"], "ARC split 必须是 train|validation|test"
        self.ds = load_dataset("allenai/ai2_arc", subset, split=split).shuffle(seed=42)

    @property
    def eval_type(self):
        return 'categorical'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        question = row["question"] # the question text
                                   # 问题文本
        choices = row["choices"]["text"] # the text of each choice
                                         # 每个选项的文本
        answer_string = row["answerKey"] # e.g. "A", "B", "C", "D"
                                         # 例如 "A"、"B"、"C"、"D"
        letters = row["choices"]["label"] # e.g. ["A", "B", "C", "D"]
                                          # 例如 ["A", "B", "C", "D"]
        assert answer_string in letters, f"ARC answer {answer_string} must be one of {letters}" # sanity check
        assert answer_string in letters, f"ARC 答案 {answer_string} 必须是 {letters} 之一" # 健全性检查
        # create and return the Conversation object
        # 创建并返回 Conversation 对象
        user_message = render_mc(question, letters, choices)
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": answer_string}
        ]
        conversation = {
            "messages": messages,
            "letters": letters, # useful during evaluation, so we can narrow and clamp the assistant prediction to one of the letters
                                 # 在评估期间有用，这样我们可以将助手预测缩小并限制为其中一个字母
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        # the assert here is not strictly speaking needed, but currently the way we eval, we expect this to be true
        # 这里的断言严格来说不是必需的，但目前我们评估的方式，我们期望这是真的
        # I'm going to leave the assert here to prevent footguns, but possibly in the future can remove it.
        # 我将在这里保留断言以防止陷阱，但将来可能可以删除它。
        assert assistant_response in conversation['letters'], f"ARC answer {assistant_response} is expected to be one of {conversation['letters']}"
        assert assistant_response in conversation['letters'], f"ARC 答案 {assistant_response} 应该是 {conversation['letters']} 之一"
        assistant_message = conversation['messages'][-1]['content'] # e.g. "A"
        return assistant_response == assistant_message

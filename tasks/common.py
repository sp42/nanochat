"""
Base class for all Tasks.
所有任务的基类。
A Task is basically a dataset of conversations, together with some
任务基本上是一个对话数据集，以及一些
metadata and often also evaluation criteria.
元数据，通常还有评估标准。
Example tasks: MMLU, ARC-Easy, ARC-Challenge, GSM8K, HumanEval, SmolTalk.
示例任务：MMLU、ARC-Easy、ARC-Challenge、GSM8K、HumanEval、SmolTalk。
"""

import random

class Task:
    """
    Base class of a Task. Allows for lightweight slicing of the underlying dataset.
    任务的基类。允许对底层数据集进行轻量级切片。
    """

    def __init__(self, start=0, stop=None, step=1):
        # allows a lightweight logical view over a dataset
        # 允许对数据集进行轻量级逻辑视图
        assert start >= 0, f"Start must be non-negative, got {start}"
        assert stop is None or stop >= start, f"Stop should be greater than or equal to start, got {stop} and {start}"
        assert step >= 1, f"Step must be strictly positive, got {step}"
        self.start = start
        self.stop = stop # could be None here
                         # 这里可能是 None
        self.step = step

    @property
    def eval_type(self):
        # one of 'generative' | 'categorical'
        # 'generative' | 'categorical' 之一
        raise NotImplementedError

    def num_examples(self):
        raise NotImplementedError

    def get_example(self, index):
        raise NotImplementedError

    def __len__(self):
        start = self.start
        stop = self.num_examples() if self.stop is None else self.stop
        step = self.step
        span = stop - start
        num = (span + step - 1) // step # ceil_div(span, step)
        assert num >= 0, f"Negative number of examples???: {num}" # prevent footguns
                                                                  # 防止陷阱
        return num

    def __getitem__(self, index: int):
        assert isinstance(index, int), f"Index must be an integer, got {type(index)}"
        physical_index = self.start + index * self.step
        conversation = self.get_example(physical_index)
        return conversation

    def evaluate(self, problem, completion):
        raise NotImplementedError


class TaskMixture(Task):
    """
    For SFT Training it becomes useful to train on a mixture of datasets.
    对于 SFT 训练，在数据集混合上训练变得有用。
    Fun trick: if you wish to oversample any task, just pass it in multiple times in the list.
    有趣的技巧：如果你想对任何任务进行过采样，只需在列表中多次传入它。
    """

    def __init__(self, tasks, **kwargs):
        super().__init__(**kwargs)
        # tasks is a list of Task objects
        # tasks 是 Task 对象的列表
        self.tasks = tasks
        self.lengths = [len(task) for task in self.tasks]
        self.num_conversations = sum(self.lengths)
        # Build list of all (task_idx, local_idx) pairs
        # 构建所有 (task_idx, local_idx) 对的列表
        self.index_map = []
        for task_idx, task_length in enumerate(self.lengths):
            for local_idx in range(task_length):
                self.index_map.append((task_idx, local_idx))
        # Deterministically shuffle to mix tasks throughout training
        # 确定性打乱以在整个训练过程中混合任务
        rng = random.Random(42)
        rng.shuffle(self.index_map)
        # Note: this is not the most elegant or best solution, but it's ok for now
        # 注意：这不是最优雅或最好的解决方案，但目前可以

    def num_examples(self):
        return self.num_conversations

    def get_example(self, index):
        """
        Access conversations according to a deterministic shuffle of all examples.
        根据所有示例的确定性打乱访问对话。
        This ensures tasks are mixed throughout training, regardless of dataset size.
        这确保任务在整个训练过程中混合，无论数据集大小如何。
        """
        assert 0 <= index < self.num_conversations, f"Index {index} out of range for mixture with {self.num_conversations} conversations"
        task_idx, local_idx = self.index_map[index]
        return self.tasks[task_idx][local_idx]


class TaskSequence(Task):
    """
    For SFT Training sometimes we want to sequentially train on a list of tasks.
    对于 SFT 训练，有时我们希望按顺序在任务列表上训练。
    This is useful for cases that require a training curriculum.
    这对于需要训练课程的场景很有用。
    """

    def __init__(self, tasks, **kwargs):
        super().__init__(**kwargs)
        self.tasks = tasks
        self.lengths = [len(task) for task in self.tasks]
        self.num_conversations = sum(self.lengths)

    def num_examples(self):
        return self.num_conversations

    def get_example(self, index):
        assert 0 <= index < self.num_conversations, f"Index {index} out of range for sequence with {self.num_conversations} conversations"
        for task_idx, task_length in enumerate(self.lengths):
            if index < task_length:
                return self.tasks[task_idx][index]
            index -= task_length


def render_mc(question, letters, choices):
    """
    The common multiple choice rendering format we will use.
    我们将使用的通用多项选择渲染格式。

    Note two important design decisions:
    注意两个重要的设计决策：
    1)
    Bigger models don't care as much, but smaller models prefer to have
    更大的模型不太在意，但更小的模型更喜欢
    the letter *after* the choice, which results in better binding.
    字母在选项*之后*，这会产生更好的绑定。
    2)
    There is no whitespace between the delimiter (=) and the letter.
    分隔符（=）和字母之间没有空格。
    This is actually critical because the tokenizer has different token ids
    这实际上很关键，因为分词器有不同的 token id
    for " A" vs. "A". The assistant responses will be just the letter itself,
    对于 " A" 和 "A"。助手响应将只是字母本身，
    i.e. "A", so it is important that here in the prompt it is the exact same
    即 "A"，所以重要的是在提示中它是完全相同的
    token, i.e. "A" with no whitespace before it. Again, bigger models don't care
    token，即 "A" 前面没有空格。同样，更大的模型不太在意
    about this too much, but smaller models do care about some of these details.
    这些，但更小的模型确实在意这些细节。
    """
    query = f"Multiple Choice question: {question}\n"
    query += "".join([f"- {choice}={letter}\n" for letter, choice in zip(letters, choices)])
    query += "\nRespond only with the letter of the correct answer."
    return query


if __name__ == "__main__":
    # very lightweight test of slicing
    # 非常轻量级的切片测试
    from tasks.mmlu import MMLU

    ds = MMLU(subset="auxiliary_train", split="train")
    print("Length of MMLU: ", len(ds))
    ex = ds[5]
    print("5th example: ", ex)

    ds = MMLU(subset="auxiliary_train", split="train", start=5, stop=10)
    print("Length of sliced MMLU[5:10]: ", len(ds))
    print("0th example of sliced MMLU: ", ds[0])

    print("They match: ", ex == ds[0])

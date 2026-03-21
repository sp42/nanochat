"""
Evaluate the Chat model on HumanEval dataset.
在 HumanEval 数据集上评估聊天模型。
Btw this dataset is a misnomer and has nothing to do with humans.
顺便说一句，这个数据集名称有误，与人类无关。
It is a coding benchmark.
它是一个编程基准测试。
"""

import re
from datasets import load_dataset
from nanochat.execution import execute_code
from tasks.common import Task

def extract_imports(prompt):
    """Extract import statements from the beginning of a code block."""
    """从代码块开头提取导入语句。"""
    imports = []
    for line in prompt.split('\n'):
        stripped = line.strip()
        if stripped.startswith('import ') or stripped.startswith('from '):
            imports.append(stripped)
        elif stripped and not stripped.startswith('#'):
            # Stop at first non-import, non-comment line
            # 在第一个非导入、非注释行停止
            break
    return '\n'.join(imports)

def extract_program(completion):
    """
    Extract Python code from LLM completion.
    从 LLM 补全中提取 Python 代码。

    Handles various output formats:
    处理各种输出格式：
    - Code wrapped in ```python ... ``` or ``` ... ``` blocks
    - 包裹在 ```python ... ``` 或 ``` ... ``` 块中的代码
    - Plain code without markdown blocks
    - 没有 markdown 块的普通代码
    - Extra text before/after code blocks
    - 代码块前后的额外文本

    Returns the first code block if found, otherwise returns the whole completion.
    如果找到，返回第一个代码块，否则返回整个补全。
    """
    # Try to find markdown code blocks (```python or just ```)
    # 尝试查找 markdown 代码块（```python 或仅 ```）
    # Match ```python\n...\n``` or ```\n...\n```
    # 匹配 ```python\n...\n``` 或 ```\n...\n```
    pattern = r'```(?:python)?\s*\n(.*?)\n```'
    matches = re.findall(pattern, completion, re.DOTALL)

    if matches:
        # Return the first code block found
        # 返回找到的第一个代码块
        return matches[0].strip()

    # No code blocks found, return the whole completion
    # 未找到代码块，返回整个补全
    return completion.strip()

class HumanEval(Task):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ds = load_dataset("openai/openai_humaneval", split="test").shuffle(seed=42)

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        """ Get a single problem from the dataset. """
        """ 从数据集中获取单个问题。 """
        row = self.ds[index]
        prompt = row['prompt'] # prompts in HumanEval are the beginning of the program
                               # HumanEval 中的提示是程序的开头
        solution = row['canonical_solution'] # the correct continuation of the program
                                            # 程序的正确延续
        entry_point = row['entry_point'] # the function to check
                                         # 要检查的函数
        test = row['test'] # the test cases
                           # 测试用例
        complete_solution = f"{prompt}\n{solution}"
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": complete_solution},
        ]
        conversation = {
            "messages": messages,
            "entry_point": entry_point, # needed during evaluation
                                        # 评估期间需要
            "test": test, # needed during evaluation
                          # 评估期间需要
        }
        return conversation

    def evaluate(self, conversation, completion):
        """ Given (conversation, completion), return boolean success of the completion. """
        """ 给定 (conversation, completion)，返回补全的布尔成功值。 """
        # the prompt will contain the imports and the function signature
        # 提示将包含导入和函数签名
        imports = extract_imports(conversation['messages'][0]['content'])
        # the completion will usually contain the whole function
        # 补全通常包含整个函数
        # but not always with the needed imports, so we manually append them
        # 但并不总是包含所需的导入，所以我们手动添加它们
        completion_code = extract_program(completion)
        program = (
            imports
            + "\n\n"
            + completion_code
            + "\n\n"
            + conversation['test']
            + "\n"
            + f"check({conversation['entry_point']})"
        )
        result = execute_code(program)
        success = result.success
        return success

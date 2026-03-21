"""
Task intended to make nanochat better in spelling and counting, for example:
旨在让 nanochat 在拼写和计数方面更好的任务，例如：

"How many r are in strawberry?" -> 3
"strawberry 中有多少个 r？" -> 3

An interesting part of this task is that we will get the assistant to
solve the problem using a combination of manual counting and Python.
这个任务的一个有趣部分是，我们将让助手通过手动计数和 Python 的组合来解决问题。
This is a good problem solving "instinct" to mix into the model and RL
may further refine it to trust one over the other. If we were extra fancy
这是一个很好的问题解决"直觉"可以融入模型，RL 可能会进一步优化它以信任其中一个。
(which we could/should be) we'd add small errors here and there to allow
the model also learn recoveries. We can do this in future versions.
如果我们更花哨一点（我们可以/应该这样做），我们会在各处添加小错误，让模型也学习恢复。我们可以在未来版本中这样做。

There are two tasks in this file:
这个文件中有两个任务：
1. SpellingBee: Counting the number of occurrences of a letter in a word
1. SpellingBee: 计算单词中某个字母的出现次数
2. SimpleSpelling: Simply spelling words
2. SimpleSpelling: 简单拼写单词

(1) is the goal, but (2) exists as a highly condensed version of the part
that makes (1) difficult, which is word spelling. This is non-trivial for an
(1) 是目标，但 (2) 作为使 (1) 困难的部分的高度浓缩版本存在，即单词拼写。
LLM because it has to learn how every token (a little semantic chunk/atom)
对于 LLM 来说，这是非平凡的，因为它必须学习每个 token（一个小的语义块/原子）
maps to the sequence of individual characters that make it up. Larger models
如何映射到组成它的单个字符序列。更大的模型
learn this eventually on their own, but if we want this capability to exist
最终会自己学会这个，但如果我们希望这种能力存在
in smaller models, we have to actively encourage it by over-representing it
在更小的模型中，我们必须通过在训练数据中过度表示它来积极鼓励它
in the training data. SFT is a good place to do this.
SFT 是一个做这个的好地方。

To preview a few example conversations, run:
要预览一些示例对话，运行：
python -m tasks.spellingbee
"""

import re
import random
from tasks.common import Task
from nanochat.common import download_file_with_lock

# Letters of the alphabet
# 字母表的字母
LETTERS = "abcdefghijklmnopqrstuvwxyz"
# A list of 370K English words of large variety
# 一个包含 370K 个多样化英语单词的列表
WORD_LIST_URL = "https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words_alpha.txt"
# A number bigger than 370K to separate train and test random seeds
# 一个大于 370K 的数字，用于分离训练和测试的随机种子
TEST_RANDOM_SEED_OFFSET = 10_000_000

# Identical to gsm8k's answer extraction
# 与 gsm8k 的答案提取相同
ANSWER_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
def extract_answer(completion):
    """
    Extract the numerical answer after #### marker.
    提取 #### 标记后的数字答案。
    """
    match = ANSWER_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    return None

# User message templates for data augmentation
# 用于数据增强的用户消息模板
USER_MSG_TEMPLATES = [
    "How many {letter} are in the word {word}",
    "How many {letter} are in {word}",
    "Count the number of {letter} in {word}",
    "How many times does {letter} appear in {word}",
    "What's the count of {letter} in {word}",
    "In the word {word}, how many {letter} are there",
    "How many letter {letter} are in the word {word}",
    "Count how many {letter} appear in {word}",
    "Tell me the number of {letter} in {word}",
    "How many occurrences of {letter} are in {word}",
    "Find the count of {letter} in {word}",
    "Can you count the {letter} letters in {word}",
    "What is the frequency of {letter} in {word}",
    "How many {letter}s are in {word}",
    "How many {letter}'s are in {word}",
    "Count all the {letter} in {word}",
    "How many times is {letter} in {word}",
    "Number of {letter} in {word}",
    "Total count of {letter} in {word}",
    "How many {letter} does {word} have",
    "How many {letter} does {word} contain",
    "What's the number of {letter} in {word}",
    "{word} has how many {letter}",
    "In {word}, count the {letter}",
    "How many {letter} appear in {word}",
    "Count the {letter} in {word}",
    "Give me the count of {letter} in {word}",
    "How many instances of {letter} in {word}",
    "Show me how many {letter} are in {word}",
    "Calculate the number of {letter} in {word}",
    # Spanish
    "¿Cuántas {letter} hay en {word}?",
    "¿Cuántas veces aparece {letter} en {word}?",
    "Cuenta las {letter} en {word}",
    "¿Cuántas letras {letter} tiene {word}?",
    # Chinese (Simplified)
    "{word}中有多少个{letter}",
    "{word}里有几个{letter}",
    "数一下{word}中的{letter}",
    "{word}这个词里有多少{letter}",
    # Korean
    "{word}에 {letter}가 몇 개 있나요",
    "{word}에서 {letter}의 개수는",
    "{word}에 {letter}가 몇 번 나오나요",
    "{word}라는 단어에 {letter}가 몇 개",
    # French
    "Combien de {letter} dans {word}",
    "Combien de fois {letter} apparaît dans {word}",
    "Compte les {letter} dans {word}",
    # German
    "Wie viele {letter} sind in {word}",
    "Wie oft kommt {letter} in {word} vor",
    "Zähle die {letter} in {word}",
    # Japanese
    "{word}に{letter}は何個ありますか",
    "{word}の中に{letter}がいくつ",
    "{word}に{letter}が何回出てくる",
]

class SpellingBee(Task):

    def __init__(self, size=1000, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"], "SpellingBee split must be train|test"
        assert split in ["train", "test"], "SpellingBee split 必须是 train|test"
        self.size = size
        self.split = split
        filename = WORD_LIST_URL.split("/")[-1]
        word_list_path = download_file_with_lock(WORD_LIST_URL, filename)
        with open(word_list_path, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f]
        self.words = words

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return self.size

    def get_example(self, index):
        seed = index if self.split == 'train' else TEST_RANDOM_SEED_OFFSET + index
        rng = random.Random(seed)

        # pick a random word
        # 随机选择一个单词
        word = rng.choice(self.words)
        # pick a letter from it (90%) or a random letter (10%)
        # 从中选择一个字母（90%）或随机选择一个字母（10%）
        letter = rng.choice(word) if rng.random() < 0.9 else rng.choice(LETTERS)

        # get the correct answer by simply counting
        # 通过简单计数获得正确答案
        count = word.count(letter)

        # create a user message, with a bunch of variations as data augmentation
        # 创建用户消息，使用多种变体作为数据增强
        template = rng.choice(USER_MSG_TEMPLATES)
        # 30% chance to lowercase the template (lazy people don't use shift)
        # 30% 的机会将模板小写（懒惰的人不使用 shift）
        if rng.random() < 0.3:
            template = template.lower()
        quote_options = ['', "'", '"']
        letter_quote = rng.choice(quote_options) # is the letter quoted?
                                                  # 字母是否被引用？
        word_quote = rng.choice(quote_options) # is the word quoted?
                                               # 单词是否被引用？
        letter_wrapped = f"{letter_quote}{letter}{letter_quote}"
        word_wrapped = f"{word_quote}{word}{word_quote}"
        user_msg = template.format(letter=letter_wrapped, word=word_wrapped)
        if rng.random() < 0.5: # 50% of people don't even use question marks
                               # 50% 的人甚至不使用问号
            user_msg += "?"

        # Now create the ideal assistant response - build as parts (text + tool calls)
        # 现在创建理想的助手响应 - 构建为部分（文本 + 工具调用）
        assistant_parts = []
        word_letters = ",".join(list(word))
        manual_text = f"""We are asked to find the number '{letter}' in the word '{word}'. Let me try a manual approach first.

First spell the word out:
{word}:{word_letters}

Then count the occurrences of '{letter}':
"""
        # Little simulated loop of the solution process
        # 解决过程的小模拟循环
        # TODO: This is where the fun starts, we could simulate cute little mistakes
        # TODO: 这里是有趣的地方，我们可以模拟可爱的小错误
        # and get the model to review its work and recover from them.
        # 并让模型检查其工作并从中恢复。
        # You might of course hope this could arise in RL too, but realistically you'd want to help it out a bit.
        # 你当然希望这也能在 RL 中出现，但现实上你会想帮它一把。
        running_count = 0
        for i, char in enumerate(word, 1):
            if char == letter:
                running_count += 1
                # note: there deliberately cannot be a space here between i and char
                # 注意：这里故意在 i 和 char 之间不能有空格
                # because this would create a different token! (e.g. " a" and "a" are different tokens)
                # 因为这会创建不同的 token！（例如 " a" 和 "a" 是不同的 token）
                manual_text += f"{i}:{char} hit! count={running_count}\n"
            else:
                manual_text += f"{i}:{char}\n"

        manual_text += f"\nThis gives us {running_count}."
        assistant_parts.append({"type": "text", "text": manual_text})
        # Part 2: Python verification
        # 第 2 部分：Python 验证
        assistant_parts.append({"type": "text", "text": "\n\nLet me double check this using Python:\n\n"})
        # Part 3: Python tool call
        # 第 3 部分：Python 工具调用
        python_expr = f"'{word}'.count('{letter}')"
        assistant_parts.append({"type": "python", "text": python_expr})
        # Part 4: Python output
        # 第 4 部分：Python 输出
        assistant_parts.append({"type": "python_output", "text": str(count)})
        # Part 5: Final answer
        # 第 5 部分：最终答案
        assistant_parts.append({"type": "text", "text": f"\n\nPython gives us {count}.\n\nMy final answer is:\n\n#### {count}"})

        # return the full conversation
        # 返回完整对话
        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_parts}
        ]
        conversation = {
            "messages": messages,
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        """
        Given (conversation, completion), return evaluation outcome (0 = wrong, 1 = correct)
        给定 (conversation, completion)，返回评估结果（0 = 错误，1 = 正确）
        Identical to gsm8k's evaluation.
        与 gsm8k 的评估相同。
        """
        assert isinstance(assistant_response, str), "Assuming simple string response for now"
        assert isinstance(assistant_response, str), "目前假设简单字符串响应"
        # First extract the ground truth answer from the conversation
        # 首先从对话中提取真实答案
        assistant_message = conversation['messages'][-1]
        assert assistant_message['role'] == "assistant", "Last message must be from the Assistant"
        assert assistant_message['role'] == "assistant", "最后一条消息必须来自助手"
        assert isinstance(assistant_message['content'], list), "This is expected to be a list of parts"
        assert isinstance(assistant_message['content'], list), "这应该是一个部分列表"
        # The last text part contains the final answer with ####
        # 最后一个文本部分包含带有 #### 的最终答案
        last_text_part = assistant_message['content'][-1]['text']
        # Extract both the ground truth answer and the predicted answer
        # 提取真实答案和预测答案
        ref_num = extract_answer(last_text_part)
        pred_num = extract_answer(assistant_response)
        # Compare and return the success as int
        # 比较并返回成功作为整数
        is_correct = int(pred_num == ref_num)
        return is_correct

    def reward(self, conversation, assistant_response):
        """ Use simple 0-1 reward just like gsm8k."""
        """ 使用简单的 0-1 奖励，就像 gsm8k 一样。"""
        is_correct = self.evaluate(conversation, assistant_response)
        is_correct_float = float(is_correct)
        return is_correct_float


class SimpleSpelling(Task):
    """Much simpler task designed to get the model to just practice spelling words."""
    """更简单的任务，旨在让模型练习拼写单词。"""

    def __init__(self, size=1000, split="train", **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"], "SpellingBee split must be train|test"
        assert split in ["train", "test"], "SpellingBee split 必须是 train|test"
        self.size = size
        self.split = split
        filename = WORD_LIST_URL.split("/")[-1]
        word_list_path = download_file_with_lock(WORD_LIST_URL, filename)
        with open(word_list_path, 'r', encoding='utf-8') as f:
            words = [line.strip() for line in f]
        rng = random.Random(42)
        rng.shuffle(words) # use a different word order than the SpellingBee task
                           # 使用与 SpellingBee 任务不同的单词顺序
        self.words = words

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return self.size

    def get_example(self, index):
        seed = index if self.split == 'train' else TEST_RANDOM_SEED_OFFSET + index
        rng = random.Random(seed)
        # pick a random word
        # 随机选择一个单词
        word = rng.choice(self.words)
        word_letters = ",".join(list(word))
        # return the full conversation
        # 返回完整对话
        messages = [
            {"role": "user", "content": f"Spell the word: {word}"},
            {"role": "assistant", "content": f"{word}:{word_letters}"}
        ]
        conversation = {
            "messages": messages,
        }
        return conversation


if __name__ == "__main__":

    # preview the SpellingBee task, first 10 examples
    # 预览 SpellingBee 任务，前 10 个示例
    task = SpellingBee()
    for i in range(10):
        ex = task.get_example(i)
        print("=" * 100)
        print(ex['messages'][0]['content'])
        print("-" * 100)
        # Assistant content is now a list of parts
        # 助手内容现在是部分列表
        assistant_parts = ex['messages'][1]['content']
        for part in assistant_parts:
            if part['type'] == 'text':
                print(part['text'], end='')
            elif part['type'] == 'python':
                print(f"<<{part['text']}=", end='')
            elif part['type'] == 'python_output':
                print(f"{part['text']}>>", end='')
        print()
        print("-" * 100)

    # # preview the SimpleSpelling task, first 10 examples
    # # 预览 SimpleSpelling 任务，前 10 个示例
    # task = SimpleSpelling()
    # for i in range(10):
    #     ex = task.get_example(i)
    #     print("=" * 100)
    #     print(ex['messages'][0]['content'])
    #     print("-" * 100)
    #     print(ex['messages'][1]['content'])

    # # also scrutinize the tokenization (last example only)
    # # 还要检查分词（仅最后一个示例）
    # from nanochat.tokenizer import get_tokenizer
    # tokenizer = get_tokenizer()
    # ids, mask = tokenizer.render_conversation(ex)
    # print(tokenizer.visualize_tokenization(ids, mask, with_token_id=True))

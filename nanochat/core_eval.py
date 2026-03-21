"""
Functions for evaluating the CORE metric, as described in the DCLM paper.
用于评估 CORE 指标的函数，如 DCLM 论文所述。
https://arxiv.org/abs/2406.11794

TODOs:
待办事项：
- All tasks ~match except for squad. We get 31% reference is 37%. Figure out why.
- 所有任务大致匹配，除了 squad。我们得到 31%，参考值是 37%。需要弄清楚原因。
"""
import random

from jinja2 import Template
import torch
import torch.distributed as dist

# -----------------------------------------------------------------------------
# Prompt rendering utilities
# 提示渲染工具
# -----------------------------------------------------------------------------

def render_prompts_mc(item, continuation_delimiter, fewshot_examples=None):
    """Render complete prompts for a multiple choice question"""
    """为多项选择题渲染完整提示"""
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.query }}{{ continuation_delimiter }}{{ example.choices[example.gold] }}

{% endfor -%}
{{ item.query }}{{ continuation_delimiter }}{{ choice }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    prompts = [template.render(choice=choice, **context) for choice in item['choices']]
    return prompts


def render_prompts_schema(item, continuation_delimiter, fewshot_examples=None):
    """Render complete prompts for a schema question"""
    """为 schema 问题渲染完整提示"""
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context_options[example.gold] }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ context }}{{ continuation_delimiter }}{{ item.continuation }}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    prompts = [template.render(context=context_option, **context)
               for context_option in item['context_options']]
    return prompts


def render_prompts_lm(item, continuation_delimiter, fewshot_examples=None):
    """
    Render complete prompt for a language modeling task.
    为语言建模任务渲染完整提示。
    Notice that we manually trim the context in the template,
    注意我们在模板中手动修剪上下文，
    which in some datasets seems to have trailing whitespace (which we don't want).
    在某些数据集中似乎有尾随空格（我们不想要）。
    """
    template_str = """
{%- for example in fewshot_examples -%}
{{ example.context | trim }}{{ continuation_delimiter }}{{ example.continuation }}

{% endfor -%}
{{ item.context | trim }}{{ continuation_delimiter }}{% if include_continuation %}{{ item.continuation }}{% endif %}""".strip()
    template = Template(template_str)
    fewshot_examples = fewshot_examples or []
    context = {
        'fewshot_examples': fewshot_examples,
        'continuation_delimiter': continuation_delimiter,
        'item': item
    }
    # Return two prompts: without and with the continuation
    # 返回两个提示：不带和带 continuation
    prompt_without = template.render(include_continuation=False, **context)
    prompt_with = template.render(include_continuation=True, **context)
    # Due to the way the data seems to be stored, I think I need to strip in the case of LM here.
    # 由于数据存储的方式，我认为在 LM 情况下需要在这里去除空格。
    # Otherwise we may get trailing whitespaces in prompt_without (which get absorbed into the next
    # 否则我们可能会在 prompt_without 中得到尾随空格（被吸收到下一个
    # token in prompt_with), meaning we don't get a nice and clean prefix in the token space
    # prompt_with 的 token 中），意味着我们在 token 空间中没有得到干净的前缀
    # to detect the final continuation. Tokenizers...
    # 来检测最终的 continuation。分词器...
    prompt_without = prompt_without.strip()
    return [prompt_without, prompt_with]


def find_common_length(token_sequences, direction='left'):
    """
    Find the length of the common prefix or suffix across token sequences
    找到 token 序列中公共前缀或后缀的长度
    - direction: 'left' for prefix, 'right' for suffix
    - direction: 'left' 表示前缀，'right' 表示后缀
    """
    min_len = min(len(seq) for seq in token_sequences)
    indices = {
        'left': range(min_len),
        'right': range(-1, -min_len-1, -1)
    }[direction]
    # Find the first position where the token sequences differ
    # 找到 token 序列不同的第一个位置
    for i, idx in enumerate(indices):
        token = token_sequences[0][idx]
        if not all(seq[idx] == token for seq in token_sequences):
            return i
    return min_len


def stack_sequences(tokens, pad_token_id):
    """Stack up a list of token sequences, pad to longest on the right"""
    """堆叠 token 序列列表，在右侧填充到最长"""
    bsz, seq_len = len(tokens), max(len(x) for x in tokens)
    input_ids = torch.full((bsz, seq_len), pad_token_id, dtype=torch.long)
    for i, x in enumerate(tokens):
        input_ids[i, :len(x)] = torch.tensor(x, dtype=torch.long)
    return input_ids


def batch_sequences_mc(tokenizer, prompts):
    # In multiple choice, contexts are the same but the continuation is different (common prefix)
    # 在多项选择中，上下文相同但 continuation 不同（公共前缀）
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    # figure out the start and end of each continuation
    # 找出每个 continuation 的开始和结束
    answer_start_idx = find_common_length(tokens, direction='left')
    start_indices = [answer_start_idx] * len(prompts)
    end_indices = [len(x) for x in tokens]
    return tokens, start_indices, end_indices


def batch_sequences_schema(tokenizer, prompts):
    # In schema tasks, contexts vary but continuation is the same (common suffix)
    # 在 schema 任务中，上下文不同但 continuation 相同（公共后缀）
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    # figure out the start and end of each context
    # 找出每个上下文的开始和结束
    suffix_length = find_common_length(tokens, direction='right')
    end_indices = [len(x) for x in tokens]
    start_indices = [ei - suffix_length for ei in end_indices]
    return tokens, start_indices, end_indices


def batch_sequences_lm(tokenizer, prompts):
    # In LM tasks, we have two prompts: without and with continuation
    # 在 LM 任务中，我们有两个提示：不带和带 continuation
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    tokens_without, tokens_with = tokens
    start_idx, end_idx = len(tokens_without), len(tokens_with)
    assert start_idx < end_idx, "prompt without is supposed to be a prefix of prompt with"
    assert tokens_without == tokens_with[:start_idx], "prompt without is supposed to be a prefix of prompt with"
    # we only need the with continuation prompt in the LM task, i.e. batch size of 1
    # 在 LM 任务中我们只需要带 continuation 的提示，即批大小为 1
    return [tokens_with], [start_idx], [end_idx]


@torch.no_grad()
def forward_model(model, input_ids):
    """
    Take BxT tensor of token ids, return BxT tensor of losses and argmax predictions.
    接收 BxT 的 token id 张量，返回 BxT 的损失和 argmax 预测张量。
    The last column of losses is set to nan because we don't have autoregressive targets there.
    损失的最后一列设为 nan，因为那里没有自回归目标。
    """
    batch_size, seq_len = input_ids.size()
    outputs = model(input_ids)
    # Roll the tensor to the left by one position to get the (autoregressive) target ids
    # 将张量向左滚动一个位置以获得（自回归）目标 id
    target_ids = torch.roll(input_ids, shifts=-1, dims=1)
    # Calculate cross entropy at all positions
    # 计算所有位置的交叉熵
    losses = torch.nn.functional.cross_entropy(
        outputs.view(batch_size * seq_len, -1),
        target_ids.view(batch_size * seq_len),
        reduction='none'
    ).view(batch_size, seq_len)
    # Set the last column to be nan because there is no autoregressive loss there
    # 将最后一列设为 nan，因为那里没有自回归损失
    losses[:, -1] = float('nan')
    # Get the argmax predictions at each position
    # 获取每个位置的 argmax 预测
    predictions = outputs.argmax(dim=-1)
    return losses, predictions


@torch.no_grad()
def evaluate_example(idx, model, tokenizer, data, device, task_meta):
    """Evaluate a single example, return True if correct, False otherwise"""
    """评估单个示例，正确返回 True，否则返回 False"""
    item = data[idx]
    task_type = task_meta['task_type']
    num_fewshot = task_meta['num_fewshot']
    continuation_delimiter = task_meta['continuation_delimiter']

    # Sample few-shot examples (excluding current item)
    # 采样 few-shot 示例（排除当前项）
    fewshot_examples = []
    if num_fewshot > 0:
        rng = random.Random(1234 + idx)
        available_indices = [i for i in range(len(data)) if i != idx]
        fewshot_indices = rng.sample(available_indices, num_fewshot)
        fewshot_examples = [data[i] for i in fewshot_indices]

    # Render prompts and batch sequences based on task type
    # 根据任务类型渲染提示并批处理序列
    if task_type == 'multiple_choice':
        prompts = render_prompts_mc(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_mc(tokenizer, prompts)
    elif task_type == 'schema':
        prompts = render_prompts_schema(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_schema(tokenizer, prompts)
    elif task_type == 'language_modeling':
        prompts = render_prompts_lm(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_lm(tokenizer, prompts)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    # Some models can't forward sequences beyond a certain length (e.g. GPT-2)
    # 某些模型无法前向超过一定长度的序列（例如 GPT-2）
    # In these cases, we have to truncate sequences to max length and adjust the indices
    # 在这些情况下，我们必须将序列截断到最大长度并调整索引
    if hasattr(model, 'max_seq_len') and model.max_seq_len is not None:
        max_tokens = model.max_seq_len
        new_tokens, new_start_idxs, new_end_idxs = [], [], []
        for t, s, e in zip(tokens, start_idxs, end_idxs):
            if len(t) > max_tokens:
                num_to_crop = len(t) - max_tokens
                new_tokens.append(t[-max_tokens:]) # take the last max_tokens tokens
                                                    # 取最后 max_tokens 个 token
                new_start_idxs.append(s - num_to_crop) # shift the indices down
                                                       # 向下移动索引
                new_end_idxs.append(e - num_to_crop)
                assert s - num_to_crop >= 0, "this should never happen right?"
                assert e - num_to_crop >= 0, "this should never happen right?"
            else:
                new_tokens.append(t) # keep unchanged
                                     # 保持不变
                new_start_idxs.append(s)
                new_end_idxs.append(e)
        tokens, start_idxs, end_idxs = new_tokens, new_start_idxs, new_end_idxs

    # Stack up all the sequences into a batch
    # 将所有序列堆叠成一个批次
    pad_token_id = tokenizer.get_bos_token_id() # use BOS as pad token is ok
                                                 # 使用 BOS 作为 pad token 是可以的
    input_ids = stack_sequences(tokens, pad_token_id)
    input_ids = input_ids.to(device)

    # Forward the model, get the autoregressive loss and argmax prediction at each token
    # 前向模型，获取每个 token 的自回归损失和 argmax 预测
    losses, predictions = forward_model(model, input_ids)

    # See if the losses/predictions come out correctly
    # 检查损失/预测是否正确输出
    if task_type == 'language_modeling':
        # language modeling task is currently always batch size 1
        # 语言建模任务目前始终是批大小 1
        si = start_idxs[0]
        ei = end_idxs[0]
        # predictions[i] predict input_ids[i+1] autoregressively
        # predictions[i] 自回归地预测 input_ids[i+1]
        predicted_tokens = predictions[0, si-1:ei-1]
        actual_tokens = input_ids[0, si:ei]
        is_correct = torch.all(predicted_tokens == actual_tokens).item()
    elif task_type in ['multiple_choice', 'schema']:
        # For MC/schema: find the option with lowest average loss
        # 对于 MC/schema：找到平均损失最低的选项
        mean_losses = [losses[i, si-1:ei-1].mean().item()
                        for i, (si, ei) in enumerate(zip(start_idxs, end_idxs))]
        pred_idx = mean_losses.index(min(mean_losses))
        is_correct = pred_idx == item['gold']
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    return is_correct


def evaluate_task(model, tokenizer, data, device, task_meta):
    """
    This function is responsible for evaluating one task across many examples.
    此函数负责评估跨多个示例的一个任务。
    It also handles dispatch to all processes if the script is run with torchrun.
    如果脚本使用 torchrun 运行，它还处理分发到所有进程。
    """
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    correct = torch.zeros(len(data), dtype=torch.float32, device=device)
    # stride the examples to each rank
    # 将示例步进分配到每个 rank
    for idx in range(rank, len(data), world_size):
        is_correct = evaluate_example(idx, model, tokenizer, data, device, task_meta)
        correct[idx] = float(is_correct)
    # sync results across all the processes if running distributed
    # 如果分布式运行，同步所有进程的结果
    if world_size > 1:
        dist.barrier()
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    # compute the mean
    # 计算平均值
    mean_correct = correct.mean().item()
    return mean_correct

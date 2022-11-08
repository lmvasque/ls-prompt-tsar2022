from pathlib import Path

import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence


def get_splits(df):
    df_train = df.sample(frac=0.9)
    # df_test = df.drop(df_train.index).sample(frac=0.5)
    df_val = df.drop(df_train.index)

    return df_train, df_val


def get_n_tokens(context, n_tokens, place=""):
    split = context.split(" ")
    result = context
    if len(split) > n_tokens:
        if "last" in place:
            result = " ".join(split[-n_tokens:])
        elif "first" in place:
            result = " ".join(split[:n_tokens])
    return result


def get_contexts(token, context):
    if token not in context:
        print(f"The token '{token}' does not exists in '{context}'. ")
    token_index = context.index(token)
    left_context = context[:token_index]
    right_context = context[(token_index + len(token)):]

    return left_context, right_context


def get_variable_columns(line):
    cols = line.split("\t", 2)
    if len(cols) == 3:
        text, token, gold_subs = cols
    elif len(cols) == 2:
        text, token = cols
        gold_subs = token  # just a stupid trick to make the data consistent, we don't actually use it in the testing stage
    else:
        raise ValueError(f"Wrong format: {line}")
    return text, token, gold_subs


def read_data(files, encoding):
    output = []
    for file in files:
        with open(file, encoding=encoding) as f:
            lines = [s.strip() for s in f.readlines()]
            for line in lines:
                text, token, gold_subs = get_variable_columns(line)
                text = text.encode(encoding).decode('utf-8')
                output.append([text, token, gold_subs.split("\t"), Path(file).name])

    df = pd.DataFrame(data=output, columns=["text", "token", "gold_subs", "source"])

    return df


def build_prompt(context, token, context_window, prompt1, prompt2, template, tokenizer):
    left_context, right_context = get_contexts(token, context)
    prompts = get_en_prompts(token, left_context, right_context, context_window, prompt1,
                             prompt2, tokenizer.mask_token, "")

    return prompts[template]


def get_en_prompts(token, left_context, right_context, context_window, prompt1, prompt2, mask_token, prompt_name):
    prompts = {
        f"all_context{prompt_name}": f"{left_context}{token} (a {prompt1} {prompt2} for {token} is {mask_token}){right_context}",
        f"k_context{prompt_name}": f"{get_n_tokens(left_context, context_window, 'last')}{token} (a {prompt1} {prompt2} for {token} "
                                   f"is {mask_token}) "
                                   f"{get_n_tokens(right_context, context_window, 'first')}",
        f"no_context{prompt_name}": f"An {prompt1} {prompt2} for {token} is {mask_token} .",
        f"triplets{prompt_name}": f"{left_context}{token} (a {prompt1} {prompt2} for {token} are x, y and [MASK]){right_context}",
    }

    return prompts


def build_labels(context, token, gold_subs, context_window, prompt1, prompt2, template, top_k):
    left_context, right_context = get_contexts(token, context)
    prompts = get_en_prompts(token, left_context, right_context, context_window, prompt1, prompt2,
                             gold_subs[top_k], "_labels")

    return prompts[template + '_labels']


def collate_data(batch):
    '''We have to collate the data ourselves since we have some customised features like lables, texts, etc.'''
    input_ids = [torch.LongTensor(i['input_ids']) for i in batch]
    labels = [torch.LongTensor(i['labels']) for i in batch]
    attention_mask = [torch.LongTensor(i['attention_mask']) for i in batch]
    texts = [i["text"] for i in batch]
    tokens = [i["token"] for i in batch]
    gold_subs = [i["gold_subs"] for i in batch]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=1)
    labels = pad_sequence(labels, batch_first=True, padding_value=1)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    assert input_ids.size()[1] == labels.size()[1], "Check collate_data function"

    return {'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'texts': texts,
            'tokens': tokens,
            'gold_subs': gold_subs,
            }

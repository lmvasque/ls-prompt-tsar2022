from pathlib import Path

import pandas as pd

import utils


def read_data(files, encoding):
    output = []
    for file in files:
        with open(file, encoding=encoding) as f:
            lines = [s.strip() for s in f.readlines()]

            if "tsar2022_pt_test_none" not in file:
                lines = lines[1:]
                
            for line in lines:
                # text, token, gold_subs = line.split("\t", 2)
                text, token, gold_subs = utils.get_variable_columns(line)
                gold_subs = gold_subs.split("\t")
                output.append([text, token, gold_subs, Path(file).name])

    df = pd.DataFrame(data=output, columns=["text", "token", "gold_subs", "source"])

    return df


def build_prompt(context, token, context_window, prompt1, prompt2, template, tokenizer):
    left_context, right_context = utils.get_contexts(token, context)
    prompts = get_pt_prompts(token, left_context, right_context, context_window, prompt1,
                             prompt2, tokenizer.mask_token, "")

    return prompts[template]


def get_pt_prompts(token, left_context, right_context, context_window, prompt1, prompt2, mask_token, prompt_name):
    prompts = {
        f"all_context_pt{prompt_name}": f"{left_context}{token} (Um {prompt1} {prompt2} para {token} é {mask_token}){right_context}",
        f"k_context_pt{prompt_name}": f"{utils.get_n_tokens(left_context, context_window, 'last')}{token} (Um {prompt1} {prompt2} para {token} "
                                      f"é {mask_token}) "
                                      f"{utils.get_n_tokens(right_context, context_window, 'first')}",
        f"no_context_v0_pt{prompt_name}": f"Um {prompt1} {prompt2} para {token} é {mask_token} .",
    }

    return prompts

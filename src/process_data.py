import glob
import os
from pathlib import Path

import pandas as pd
from datasets import Dataset

import config
import utils

OUTPUT_DIR = 'results'
os.makedirs(OUTPUT_DIR, exist_ok=True)
SUBS_LABELS = ["gold_subs", "cand_subs"]


class LoadPromptData:
    def __init__(self, tokenizer, params):
        self.tokenizer = tokenizer
        self.df_train = self.df_val = self.df_test = None
        self.seed = params['seed']
        self.top_train = params['top_train']
        self.ft_type = params['ft_type']

    def read_data(self, files):
        return utils.read_data(files, config.EN_ENCODING)

    def build_prompt(self, context, token, context_window, prompt1, prompt2, template):
        return utils.build_prompt(context, token, context_window, prompt1, prompt2, template, self.tokenizer)

    def build_labels(self, context, token, gold_subs, context_window, prompt1, prompt2, template, top_k):
        return utils.build_labels(context, token, gold_subs, context_window, prompt1, prompt2, template, top_k)

    def tokenize_data(self, example):
        '''To convert input prompts and labels to token ids for the LM
        '''
        tok_input = example['prompt'] + example['labels']  # this is to make sure the two sequence have the same length
        temp = self.tokenizer(tok_input, padding=True)
        # temp = self.tokenizer(example["prompt"], padding=True)
        example["attention_mask"] = temp["attention_mask"][:len(example["prompt"])]
        example["input_ids"] = temp["input_ids"][:len(example["prompt"])]
        # example["token_type_ids"] = temp["token_type_ids"]
        # temp = self.tokenizer(example["labels"], padding=True)
        example["labels"] = temp["input_ids"][len(example["prompt"]):]

        if self.ft_type == 'loss':
            sub_inps = []
            for i in example["gold_subs"]:
                sub_inps.append(i[:self.top_train])
            example["subs_train"] = sub_inps

        return example

    def get_n_tokens(self, context, n_tokens, place=""):
        split = context.split(" ")
        result = context
        if len(split) > n_tokens:
            if "last" in place:
                result = " ".join(split[-n_tokens:])
            elif "first" in place:
                result = " ".join(split[:n_tokens])
        return result

    def fix_words(self, context, token, source):
        if token not in context:
            db = config.LEXICAL_CHANGES[source][context]
            if token in db.keys():
                token = db[token]
            else:
                config_template = f"{{\"{context}\" : {{\"{token}\" : \"{token}\"}}}},"
                raise ValueError(f"The token '{token}' does not exists in '{context}' in file {source}. "
                                 f"You can include an alternative in the config file using this template:{config_template}")
        return token

    def check_data_sanity(self, prompt, label):
        if len(prompt.split(' ')) != len(label.split(' ')):
            print("Prompt and label don't match ", prompt, label)
            return False
        return True

    def get_train_data(self, data_path, prompt1, prompt2, template, context_window=-1):
        df = self.read_data(data_path)
        df = df.drop_duplicates(subset=['text', 'token'], ignore_index=True)
        df["token"] = df.apply(lambda f: self.fix_words(f.text, f.token, f.source), axis=1)
        df['prompt'] = df.apply(
            lambda f: self.build_prompt(f.text, f.token, context_window, prompt1, prompt2, template),
            axis=1)
        df['labels'] = df.apply(
            lambda f: self.build_labels(f.text, f.token, f.gold_subs, context_window, prompt1, prompt2, template, 0),
            axis=1)
        # df['check'] = df.apply(lambda f: self.check_data_sanity(f.prompt, f.labels), axis=1)
        self.df_train, self.df_val = utils.get_splits(df)
        if self.top_train > 1 and self.ft_type == 'augment':
            self.df_train = self.augment_data(self.df_train, context_window, prompt1, prompt2, template, self.top_train)
        train_data = Dataset.from_pandas(self.df_train).map(self.tokenize_data, batched=True)
        val_data = Dataset.from_pandas(self.df_val).map(self.tokenize_data, batched=True)
        # test_data = Dataset.from_pandas(self.df_test).map(self.tokenize_data, batched=True)
        return train_data, val_data

    def augment_data(self, df_train, context_window, prompt1, prompt2, template, top_train):
        '''To generate more pairs of (input, labels) from the list of substitutions
        '''
        new_row = []
        for row in df_train.iterrows():
            i = 1  # we already have top-1 in the dataframe
            while i < len(row[1]['gold_subs']) and i < top_train:
                temp = row[1].copy()
                temp['labels'] = self.build_labels(temp['text'], temp['token'],
                                                   temp['gold_subs'], context_window, prompt1, prompt2, template, i)
                new_row.append(temp)
                i += 1
        df_new = pd.DataFrame(new_row, columns=df_train.columns.values)
        df_train = pd.concat([df_train, df_new])
        return df_train

    def get_test_data(self, data_path, prompt1, prompt2, template, context_window=-1):
        df = self.read_data(data_path)
        df = df.drop_duplicates(subset=['text', 'token'], ignore_index=True)
        df["token"] = df.apply(lambda f: self.fix_words(f.text, f.token, f.source), axis=1)
        df['prompt'] = df.apply(
            lambda f: self.build_prompt(f.text, f.token, context_window, prompt1, prompt2, template),
            axis=1)
        # this is just a trick to have a dummy column of 'labels' similarly to the train data, but we don't actually use it
        df['labels'] = df.apply(
            lambda f: self.build_labels(f.text, f.token, f.gold_subs, context_window, prompt1, prompt2, template, 0),
            axis=1)
        test_data = Dataset.from_pandas(df).map(self.tokenize_data, batched=True)
        return test_data


def get_metrics():
    output = [",".join(HEADER)]
    for file in glob.glob(f"{OUTPUT_DIR}/*out"):
        subset, ft_type, dataset, prompt_1, prompt_2, model, template, k, top_train, _ = Path(file).stem.split("__")
        results = [subset, ft_type, dataset, prompt_1, prompt_2, model, template, k, top_train]
        with open(file) as f1:
            lines = [s.strip() for s in f1.readlines()]
            flag = False
            for line in lines:
                if flag:
                    if "=" in line:
                        key, value = line.split(" = ")
                        results.append(value)
                else:
                    if "RESULTS" in line:
                        flag = True
            output.append(",".join(results))

    with open(f"{OUTPUT_DIR}/summary.csv", "w") as f1:
        for r in output:
            f1.write(r + "\n")

# get_metrics()

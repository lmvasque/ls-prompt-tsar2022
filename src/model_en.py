import glob
import os
import subprocess
from pathlib import Path

from transformers import pipeline, AutoTokenizer

from config import *
import utils


class LsPromptEn:

    def __init__(self, model_name, parsed_files):
        self.lang = "en"
        self.model_name = model_name
        self.model = pipeline('fill-mask', model=self.model_name, top_k=10)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.parsed_files = parsed_files
        self.encoding = EN_ENCODING

        os.makedirs(OUTPUT_DIR, exist_ok=True)

    def read_data(self, files):
        return utils.read_data(files, self.encoding)

    def build_prompt(self, context, token, n, prompt1, prompt2, template):
        return utils.build_prompt(context, token, n, prompt1, prompt2, template, self.tokenizer)

    def fix_words(self, context, token, source):
        if token not in context:
            if source in LEXICAL_CHANGES.keys() and context in LEXICAL_CHANGES[source].keys():
                db = LEXICAL_CHANGES[source][context]
                token = db[token]
            else:
                config_template = f"\"{context}\" : {{\"{token}\" : \"{token}\"}},"
                # with open("tmp.txt", "a+", encoding=self.encoding) as f1:
                #     f1.write(config_template + "\n")
                raise ValueError(f"The token '{token}' does not exists in '{context}' in file {source}. "
                                 f"You can include an alternative in the config file using this template:\n{config_template}")
        return token

    def get_candidates(self, prompt, token):
        model_output = self.model(prompt)
        cand_list = [item['token_str'] for item in model_output]
        remove_list = ["[UNK]", token]
        for r in remove_list:
            if r in cand_list:
                cand_list.remove(r)

    def save_results(self, df_exp, prompt1, prompt2, template, k):
        name = self.model_name.replace("-", "_")
        name = name.replace("/", "_")
        for label in SUBS_LABELS:
            tmp_file = f"{OUTPUT_DIR}/{label}.txt"
            df_exp[label] = df_exp.apply(lambda f: "\t".join(f[label]), axis=1)
            df_exp[["text", "token", label]].to_csv(tmp_file, sep="\t", index=False)

            final_file = f"{OUTPUT_DIR}/{prompt1}__{prompt2}__{name}__{template}__{k}_{label}.tsv".lower()

            # Removing "" quotes added by the to_csv methods
            with open(tmp_file) as fin, open(final_file, "w+") as fout:
                for line in fin:
                    line = line.replace('"', "")
                    fout.write(line)
            os.remove(tmp_file)

        return final_file

    def run(self, prompt1, prompt2, template, k=-1):

        df = self.read_data(self.parsed_files)
        df = df.drop_duplicates(subset=['text', 'token'], ignore_index=True)
        df["token"] = df.apply(lambda f: self.fix_words(f.text, f.token, f.source), axis=1)
        df["prompt"] = df.apply(lambda f: self.build_prompt(f.text, f.token, k, prompt1, prompt2, template),
                                axis=1)
        df_train, df_val, df_test = utils.get_splits(df)

        df_exp = df_val.copy()
        df_exp["cand_subs"] = df_exp.apply(lambda f: self.get_candidates(f.prompt, f.token), axis=1)

        final_file = self.save_results(df_exp, prompt1, prompt2, template, k)
        print(f"Experiment Done: {Path(final_file).stem}")


def eval_results():
    eval_files = glob.glob(f"{OUTPUT_DIR}/*tsv")
    for s in ["_cand_subs.tsv", "_gold_subs.tsv"]:
        eval_files = [file.replace(s, "") for file in eval_files]

    eval_files = list(dict.fromkeys(eval_files))

    for f in eval_files:
        f = f.strip()
        cmd = f"python3 {TSAR_EVAL} " \
              f"--gold_file \"{f}_{SUBS_LABELS[0]}.tsv\" " \
              f"--predictions_file \"{f}_{SUBS_LABELS[1]}.tsv\" " \
              f"--output_file \"{f}.out\""
        print(f"Running evaluation script: {cmd}")
        subprocess.run(cmd, shell=True)


def get_metrics():
    output = [",".join(HEADER)]
    print(OUTPUT_DIR)
    for file in glob.glob(f"{OUTPUT_DIR}/*out"):
        subset, mode, ft_mode, dataset, prompt_1, prompt_2, model, template, top_k, top_train, _ = Path(file).stem.split("__")
        top_train = top_train.split("_")[0]
        results = [subset, mode, ft_mode, dataset, prompt_1, prompt_2, model, template, top_k, top_train]
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

    # with open(f"{OUTPUT_DIR}/summary.csv", "w") as f1:
    with open("../exp_summary.csv", "w") as f1:
        for r in output:
            f1.write(r + "\n")


def main():
    ls_prompt_en = LsPromptEn("bert-large-uncased", EN_PARSED_FILES)
    ls_prompt_en.run("simple", "word", "all_context")
    # ls_prompt_en.run("simple", "word", "k_context", 5)
    # ls_prompt_en.run("simple", "word", "k_context", 10)
    # ls_prompt_en.run("simple", "word", "no_context")
    # ls_prompt_en.run("easier", "word", "all_context")
    # ls_prompt_en.run("easier", "word", "k_context", 5)
    # ls_prompt_en.run("easier", "word", "k_context", 10)
    # ls_prompt_en.run("easier", "word", "no_context")
    # ls_prompt_en.run("simple", "word", "triplets")

    eval_results()
    get_metrics()

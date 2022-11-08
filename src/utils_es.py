import glob
import json
import re
from pathlib import Path

import nltk
import pandas as pd

import utils


def read_data(files, encoding):
    output = []
    for file in files:
        with open(file, encoding=encoding) as f:
            lines = [s.strip() for s in f.readlines()]
            for line in lines:
                # text, token, gold_subs = line.split("\t", 2)
                text, token, gold_subs = utils.get_variable_columns(line)

                text = text.encode(encoding).decode('utf-8')
                token = token.encode(encoding).decode('utf-8')
                gold_subs = [s.encode(encoding).decode('utf-8') for s in gold_subs.split("\t")]

                output.append([text, token, gold_subs, Path(file).name])

    df = pd.DataFrame(data=output, columns=["text", "token", "gold_subs", "source"])

    return df


def build_prompt(context, token, context_window, prompt1, prompt2, template, tokenizer):
    left_context, right_context = utils.get_contexts(token, context)
    prompts = get_es_prompts(token, left_context, right_context, context_window, prompt1,
                             prompt2, tokenizer.mask_token, "")

    return prompts[template]


def get_es_prompts(token, left_context, right_context, context_window, prompt1, prompt2, mask_token, prompt_name):
    prompts = {
        f"all_context_es{prompt_name}": f"{left_context}{token} (Un {prompt1} {prompt2} para {token} es {mask_token}){right_context}",
        f"k_context_es{prompt_name}": f"{utils.get_n_tokens(left_context, context_window, 'last')}{token} (Un {prompt1} {prompt2} para {token} "
                                      f"es {mask_token}) "
                                      f"{utils.get_n_tokens(right_context, context_window, 'first')}",
        f"no_context_v0_es{prompt_name}": f"Un {prompt1} {prompt2} de {token} es {mask_token} .",
    }

    return prompts


def filter_non_words(path, output_file):
    files = glob.glob(path)
    pattern = re.compile("[A-Z][a-z]+")
    encoding = "utf-8"

    for file in files:
        words_unique = {}
        with open(file, encoding=encoding) as f1:
            print("Processing file: {}".format(file))
            for line in f1:
                line = line.strip()
                words = nltk.tokenize.word_tokenize(line, language="spanish")
                words = [w.lower() for w in words if re.match(pattern, w)]
                for w in words:
                    if w in words_unique.keys():
                        words_unique[w] += 1
                    else:
                        words_unique[w] = 1

        words_unique_sorted = dict(sorted(words_unique.items(), reverse=True, key=lambda item: item[1]))
        with open(output_file.format(Path(file).stem), 'w') as fout:
            out = json.dumps(words_unique_sorted, ensure_ascii=False) + '\n'
            fout.write(out)


def has_accent_vowel(candidate):
    vowels = ["á", "é", "í", "ó", "ú", "ü", "ñ"]
    for letter in vowels:
        if letter in candidate:
            return True

    return False


def merge_dictionaries():
    files = glob.glob("data/*json")
    output_file = "data/spanish_all.json"
    encoding = "utf-8"
    words_unique = {}
    exclude = ["spanish_corpora_EMEA.json", "spanish_corpora_OpenSubtitles2018.json",
               "spanish_all_full.json", "wikicorpus.json"]

    for file in files:

        name = Path(file).name
        if name in exclude:
            continue

        with open(file, encoding=encoding) as f1:

            if "spanish_all.json" not in file:
                print("Processing file: {}".format(file))

                data = json.load(f1)

                for word, count in data.items():
                    if "/" in word or re.match(r'.*\d+.*', word):
                        continue

                    if word in words_unique.keys():
                        words_unique[word] += count
                    else:
                        words_unique[word] = count

    words_unique_sorted = dict(sorted(words_unique.items(), reverse=True, key=lambda item: item[1]))
    with open(output_file.format(Path(file).stem), 'w') as fout:
        out = json.dumps(words_unique_sorted, ensure_ascii=False) + '\n'
        fout.write(out)

# filter_non_words("data/spanish-corpora/raw/*txt", "data/spanish_corpora_{}.json")

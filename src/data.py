import os
import re
from pathlib import Path

import pandas as pd

from config import *


def parse_nnseval():
    output_file = f"{DATA_DIR_EN}/parsed/" + Path(NNS_EVAL_PATH).stem.lower() + ".tsv"
    parse_rank_syn_format(NNS_EVAL_PATH, output_file)


def parse_benchls():
    output_file = f"{DATA_DIR_EN}/parsed/" + Path(BENCH_LS_PATH).stem.lower() + ".tsv"
    parse_rank_syn_format(BENCH_LS_PATH, output_file)


def parse_rank_syn_format(path, output_file):
    with open(path) as fin, open(output_file, "w") as fout:
        lines = [s.strip() for s in fin.readlines()]
        for line in lines:
            sentence, complex_word, position, substitutions = line.split("\t", 3)
            candidates = [sentence, complex_word]
            for s in substitutions.split("\t"):
                candidate = s.split(":")[1]
                candidates.append(candidate)

            fout.write("\t".join(candidates) + "\n")


def parse_lexmturk():
    output_file = f"{DATA_DIR_EN}/parsed/" + Path(LEX_MTURK_PATH).stem.lower() + ".tsv"
    with open(LEX_MTURK_PATH, encoding="ISO-8859-1") as fin, open(output_file, "w") as fout:
        lines = [s.strip() for s in fin.readlines()]
        lines = lines[1:]
        for line in lines:
            sentence, complex_word, substitutions = line.split("\t", 2)
            sentence = sentence.replace('"', '')
            candidates = [sentence, complex_word]
            candidates.extend(substitutions.split("\t"))
            fout.write("\t".join(candidates) + "\n")


## TODO: Fix the data sorting according to its level
def parse_cefr():
    output_file = f"{DATA_DIR_EN}/parsed/" + Path(CEFR_PATH).stem.lower() + ".cefr.tsv"
    with open(CEFR_PATH) as fin, open(output_file, "w") as fout:
        lines = [s.strip() for s in fin.readlines()]

        for line in lines:
            sentence, complex_word, level, substitutions = line.split("\t", 3)
            sentence = sentence.replace('"', '')
            substitutions = substitutions.split("\t")
            candidates = [sentence, complex_word]
            end = len(substitutions) - 2
            unsorted_candidates = []
            for i in range(0, end, 3):
                word = substitutions[i]
                level = substitutions[i + 1]
                correct = substitutions[i + 2]
                if "1" in correct:
                    unsorted_candidates.append((word, level))

            sorted_candidates = sorted(unsorted_candidates, key=lambda x: x[1])
            candidates.extend([x[0] for x in sorted_candidates])
            fout.write("\t".join(candidates) + "\n")


def fix_odd_lines(lines):
    lines_fixed = []
    for i in range(0, len(lines)):
        line = lines[i]
        if re.match("^\\d+", line):
            lines_fixed.append(line)
        else:
            lines_fixed[-1] = f"{lines_fixed[-1]} {line}"

    return lines_fixed


def comma_to_tab(s):
    candidates = s.split(",")
    candidates = [c.strip() for c in candidates]
    return "\t".join(candidates)


def parse_easier():
    encoding = "ISO-8859-1"
    output_file = f"{DATA_DIR_ES}/parsed/" + Path(EASIER_PATH).stem.lower()

    with open(EASIER_PATH, encoding=encoding) as fin:
        lines = [s.strip() for s in fin.readlines()]
    names = lines[0]
    lines = lines[1:]
    lines = fix_odd_lines(lines)

    with open(EASIER_PATH + ".tmp", "w", encoding=encoding) as fin2:
        for tmp in lines:
            fin2.write(tmp + "\n")

    df = pd.read_csv(EASIER_PATH + ".tmp", sep=";", encoding=encoding, engine='python')
    df.columns = names.split(";")
    df = df.drop(columns=["document_id", "word_id"])
    df["Sinónimos"] = df["Sinónimos"].apply(lambda s: comma_to_tab(s))
    df[["sentence", "word", "Sinónimos"]].to_csv(output_file + ".csv", sep="\t", encoding=encoding, index=False, header=None)
    clean_file(output_file + ".csv", encoding)
    os.remove(output_file + ".csv")
    os.rename(output_file + ".csv.tsv", output_file + ".tsv")


def to_tab_string(s):
    synonyms = eval(s)
    output = "\t".join(synonyms)
    return output


def parse_simplex():
    output_file = f"{DATA_DIR_PT}/parsed/" + Path(SIMPLEX_PATH).stem.lower() + ".csv"
    output_file = output_file.replace(" ", "_")
    df = pd.read_excel(SIMPLEX_PATH, index_col=0)
    df["sinônimos_anotados"] = df["sinônimos_anotados"].apply(lambda s: to_tab_string(s))
    df[["sentença", "palavra_dificil", "sinônimos_anotados"]].to_csv(output_file, sep="\t", index=False)

    with open(output_file) as fin, open(output_file + ".tsv", "w+") as fout:
        for line in fin:
            line = line.replace('"', "")
            line = line.replace('*', "")
            fout.write(line)


def parse_ls_chinese():
    output_file = f"{DATA_DIR_ZH}/parsed/" + Path(LS_CHINESE_PATH).stem.lower() + ".tsv"
    with open(LS_CHINESE_PATH) as fin, open(output_file, "w") as fout:
        lines = [s.strip() for s in fin.readlines()]
        for line in lines:
            sentence, complex_word, pos, num, substitutions = line.split("\t", 4)
            candidates = [sentence, complex_word]
            for s in substitutions.split(","):
                candidates.append(s)
            fout.write("\t".join(candidates) + "\n")


def clean_file(output_file, encoding):
    with open(output_file, encoding=encoding) as fin, open(output_file + ".tsv", "w+", encoding=encoding) as fout:
        for line in fin:
            line = line.replace('"', "")
            line = line.replace('*', "")
            fout.write(line)

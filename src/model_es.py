import re

from transformers import pipeline, AutoTokenizer

from config import *
from src import utils_es
from src.model_en import LsPromptEn, eval_results, get_metrics


class LsPromptEs(LsPromptEn):

    def __init__(self, model_name, parsed_files, mask):
        self.lang = "es"
        self.model_name = model_name
        self.model = pipeline('fill-mask', model=self.model_name, top_k=10)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.parsed_files = parsed_files
        self.encoding = "latin-1"
        self.mask = mask

    def read_data(self, files):
        return utils_es.read_data(files, self.encoding)

    def build_prompt(self, context, token, context_window, prompt1, prompt2, template):
        utils_es.build_prompt(context, token, context_window, prompt1, prompt2, template, self.tokenizer)

    def run(self, prompt1, prompt2, template, n=-1):
        super().run(prompt1, prompt2, template, n)

    def get_candidates(self, prompt, token):
        model_output = self.model(prompt)
        cand_list = [item['token_str'].strip() for item in model_output]
        remove_list = ["\\[UNK\\]", f"^{token}$", "^$"]
        for regex in remove_list:
            for c in cand_list:
                if re.match(regex, c) or len(c) == 1:
                    cand_list.remove(c)

        return cand_list


def main_es():
    # TODO: Try with T5

    ls_prompt_es = LsPromptEs("bertin-project/bertin-roberta-base-spanish", parsed_files=ES_PARSED_FILES, mask="<mask>")
    ls_prompt_es.run("reemplazar", "", "no_context_v5")
    ls_prompt_es.run("reemplazar", "", "no_context_v6")
    ls_prompt_es.run("sinónimo", "fácil", "no_context_v7")
    ls_prompt_es.run("simple", "", "no_context_v8")

    ls_prompt_es = LsPromptEs("xlm-roberta-large", parsed_files=ES_PARSED_FILES, mask="<mask>")
    ls_prompt_es.run("reemplazar", "", "no_context_v5")
    ls_prompt_es.run("reemplazar", "", "no_context_v6")
    ls_prompt_es.run("sinónimo", "fácil", "no_context_v7")
    ls_prompt_es.run("simple", "", "no_context_v8")

    ls_prompt_es = LsPromptEs("bert-base-multilingual-uncased", parsed_files=ES_PARSED_FILES, mask="[MASK]")
    ls_prompt_es.run("reemplazar", "", "no_context_v5")
    ls_prompt_es.run("reemplazar", "", "no_context_v6")
    ls_prompt_es.run("sinónimo", "fácil", "no_context_v7")
    ls_prompt_es.run("simple", "", "no_context_v8")

    eval_results()
    get_metrics()

# main_es()

from transformers import pipeline, AutoTokenizer

from config import *
import utils_pt
from model_en import eval_results, get_metrics
from model_es import LsPromptEs


class LsPromptPt(LsPromptEs):

    def __init__(self, model_name, parsed_files, mask):
        self.lang = "pt"
        self.model_name = model_name
        self.model = pipeline('fill-mask', model=self.model_name, top_k=10)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.parsed_files = parsed_files
        # self.encoding = "latin-1"
        self.encoding = config.PT_ENCODING
        self.mask = mask

    def read_data(self, files):
        return utils_pt.read_data(files, self.encoding)

    def build_prompt(self, context, token, context_window, prompt1, prompt2, template):
        utils_pt.build_prompt(context, token, context_window, prompt1, prompt2, template, self.tokenizer)

    def run(self, prompt1, prompt2, template, n=-1):
        super().run(prompt1, prompt2, template, n)


def main():

    # TODO: Try with T5

    # ls_prompt_pt = LsPromptPt("xlm-roberta-large", parsed_files=PT_PARSED_FILES, mask="<mask>")
    # ls_prompt_pt.run("sinônimo", "fácil", "no_context_v0")
    # ls_prompt_pt.run("sinônimo", "fácil", "no_context_v1")
    # ls_prompt_pt.run("sinônimo", "", "no_context_v2")
    # ls_prompt_pt.run("sinônimo", "", "no_context_v3")
    # ls_prompt_pt.run("sinônimo", "fácil", "all_context")
    # ls_prompt_pt.run("sinônimo", "fácil", "k_context")

    ls_prompt_pt = LsPromptPt("bert-base-multilingual-uncased", parsed_files=PT_PARSED_FILES, mask="[MASK]")
    ls_prompt_pt.run("sinônimo", "fácil", "no_context_v0")
    ls_prompt_pt.run("sinônimo", "fácil", "no_context_v1")
    ls_prompt_pt.run("sinônimo", "", "no_context_v2")
    ls_prompt_pt.run("sinônimo", "", "no_context_v3")
    ls_prompt_pt.run("sinônimo", "fácil", "all_context")
    ls_prompt_pt.run("sinônimo", "fácil", "k_context")


    eval_results()
    get_metrics()


# main()

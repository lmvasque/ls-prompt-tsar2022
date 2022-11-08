import pandas as pd

import utils_pt
import utils
from config import PT_ENCODING
from process_data import LoadPromptData


class LoadPromptDataPt(LoadPromptData):
    def __init__(self, tokenizer, params):
        super().__init__(tokenizer, params)
        self.encoding = PT_ENCODING

    def read_data(self, files):
        return utils_pt.read_data(files, self.encoding)

    def build_prompt(self, context, token, context_window, prompt1, prompt2, template):
        return utils_pt.build_prompt(context, token, context_window, prompt1, prompt2, template, self.tokenizer)

    def build_labels(self, context, token, gold_subs, context_window, prompt1, prompt2, template, top_k):
        left_context, right_context = utils.get_contexts(token, context)
        prompts = utils_pt.get_pt_prompts(token, left_context, right_context, context_window, prompt1, prompt2,
                                          gold_subs[top_k], "_labels")

        return prompts[template + '_labels']

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

import argparse
import os
import random
import re
import subprocess
import time

import numpy as np
import torch
from nltk.corpus import wordnet
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForMaskedLM
from transformers import (AutoTokenizer,
                          AdamW, get_scheduler)

import utils
from config import *
from process_data import LoadPromptData
from process_data_es import LoadPromptDataEs
from process_data_pt import LoadPromptDataPt

TSAR_EVAL = 'scripts/eval/tsar_eval.py'

data_path = {
    "benchLS": [BENCH_LS_PARSED_PATH],
    "all": EN_PARSED_FILES,
    "NNS": [NNS_EVAL_PARSED_PATH],
    "lexmturk": [LEX_MTURK_PARSED_PATH],
    "cefr": [CEFR_PARSED_PATH],
    "easier": [EASIER_PARSED_PATH],
    "simplex": [SIMPLEX_PARSED_PATH]
}

sp_dictionary = {}


def main(params, wandb=None):
    assert params['top_train'] >= 1, "The number of trained substitution must be >=1"
    os.environ["PYTHONHASHSEED"] = str(params['seed'])
    random.seed(params['seed'])
    np.random.seed(params['seed'])
    torch.manual_seed(params['seed'])

    params['device'] = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    params["data_path"] = data_path[params["train_data"]]
    '''Step 1: Prepare the data'''
    print(f"\n ===== Model: {params['pretrained_model']}  {params['template']} ===== \n")
    tokenizer = AutoTokenizer.from_pretrained(params['pretrained_model'])

    if "EN" in params['language']:
        pData = LoadPromptData(tokenizer, params)
    elif "ES" in params['language']:
        pData = LoadPromptDataEs(tokenizer, params)
    elif "PT" in params['language']:
        pData = LoadPromptDataPt(tokenizer, params)
    else:
        raise ValueError(f"Language '{params['lang']}' not supported.")

    train_data, val_data = pData.get_train_data(
        params["data_path"],
        params['prompt1'],
        params['prompt2'],
        params['template'],
        context_window=params['context_window'],
    )

    train_loader = DataLoader(train_data,
                              shuffle=True,
                              batch_size=params['batch_size'],
                              collate_fn=utils.collate_data)
    val_loader = DataLoader(val_data,
                            shuffle=False,
                            batch_size=params['batch_size'],
                            collate_fn=utils.collate_data)

    model = AutoModelForMaskedLM.from_pretrained(params['pretrained_model'])
    
    if params['train_type'] == 'zero':  # this should be the same as using pipeline
        eval(model, val_loader, tokenizer, params)
    else:
        '''Step 2: fine tune the model and eval on the validation set'''
        train(train_loader, val_loader, model, tokenizer, params, wandb)

    '''Step 3: run on the test set'''
    if params["test_data"] != "":
        params["data_path"] = [params["test_data"]]
        test_data = pData.get_test_data(
            [params["test_data"]],
            params['prompt1'],
            params['prompt2'],
            params['template'],
            context_window=params['context_window'],
        )
        test_loader = DataLoader(test_data,
                                 shuffle=False,
                                 batch_size=params['batch_size'],
                                 collate_fn=utils.collate_data)
        eval(model, test_loader, tokenizer, params, type='test')


def train(train_data, dev_data, model, tokenizer, params, wandb=None):
    t_start = time.time()
    print(f"Device:{params['device']}\n")
    model.to(params['device'])
    num_training_steps = len(train_data) * params['epoch']
    progress_bar = tqdm(range(num_training_steps))
    optimizer = AdamW(model.parameters(), lr=5e-5)

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    model.train()
    best_scores = {'MAP@3': 0.0, 'Accuracy@3@top_gold_1': 0.0, 'Potential@3': 0.0}
    best_epoch = {'MAP@3': 0, 'Accuracy@3@top_gold_1': 0, 'Potential@3': 0}
    for epoch in range(params['epoch']):
        epoch_loss = 0
        log_info = {}
        for batch in train_data:
            in_batch = {k: v.to(params['device']) for k, v in batch.items() if
                        k in ["input_ids", "attention_mask", "labels"]}
            outputs = model(**in_batch)
            loss = outputs.loss
            loss.backward()
            epoch_loss += outputs.loss.item()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

        log_info['loss'] = epoch_loss / len(train_data)
        print(f"Loss value: {log_info['loss']} at epoch {epoch}")
        scores = eval(model, dev_data, tokenizer, params)
        log_info.update(scores)
        for metric in best_scores:
            if best_scores[metric] < scores[metric]:
                best_scores[metric] = scores[metric]
                best_epoch[metric] = epoch + 1

        if wandb != None:
            wandb.log(log_info, step=epoch)

    t_end = time.time()

    model_name = f"train_{params['train_type']}_{params['ft_type']}_{params['train_data']}_{params['prompt1']}_" \
                 f"{params['prompt2']}_{params['pretrained_model']}_{params['template']}_{params['context_window']}_" \
                 f"{params['top_train']}"
    model.save_pretrained(f"model/{model_name}")
    # tokenizer.save_pretrained(f"model/{model_name}")
    print('Total training time: {}'.format(_humanized_time(t_end - t_start)))
    print('Best scores: ', best_scores)
    print('Best epoch: ', best_epoch)


def get_top_k_predictions(batch, token_logits, tokenizer, params):
    mask_token_index = torch.where(batch["input_ids"] == tokenizer.mask_token_id)[1]
    results = []
    for index in range(len(mask_token_index)):
        result = []
        mask_token_logits = token_logits[index, mask_token_index[index], :]
        top_k_tokens = torch.topk(torch.unsqueeze(mask_token_logits, 0), params['top_k'], dim=1).indices[0].tolist()
        for t in top_k_tokens:
            token = tokenizer.decode([t])
            if token not in tokenizer.all_special_tokens:
                result.append(token.strip())

        complex_token = batch['tokens'][index]
        if params["filter_candidates"]:
            result = filter_candidates(result, complex_token)
        results.append([batch['texts'][index],
                        batch['tokens'][index],
                        '\t'.join(batch['gold_subs'][index]),
                        '\t'.join(result)])

    return results


def filter_candidates(candidates, token):
    # (A) remove known bad tokens
    remove_list = ["[UNK]", token, "unknown"]
    for r in remove_list:
        if r in candidates:
            candidates.remove(r)

    # (B) remove non-words
    candidates = [c for c in candidates if re.match("[A-z']+", c) is not None]

    if "ES" in params["language"] or "PT" in params["language"]:
        # Lower case candidates
        candidates = [c.lower() for c in candidates]
        low_token = token.lower()
        if low_token in candidates:
            candidates.remove(low_token)

        # Remove duplicates
        candidates = list(dict.fromkeys(candidates))

        # Remove small pieces
        candidates = [c for c in candidates if len(c) > 2]

        # if "ES" in params["language"]:
        #     # Remove non-words using dictionary
        #     new_candidates = []
        #
        #     for c in candidates:
        #         if utils_es.has_accent_vowel(c) or c in sp_dictionary.keys():
        #             new_candidates.append(c)
        #
        #     candidates = new_candidates


    if "EN" in params["language"]:
        # (C) antonym filtering
        antonyms = []
        for synset in wordnet.synsets(token):
            for lemma in synset.lemmas():
                if lemma.antonyms():
                    for a in lemma.antonyms():
                        antonyms.append(a.name())

        for a in set(antonyms):
            if a in candidates:
                candidates.remove(a)

    # # (D) frequency reranking
    # freqs = [freq.get(c,0) for c in candidates]
    # candidates = [c for _, c in sorted(zip(freqs, candidates), reverse=True)]

    return candidates


def eval(model, eval_data, tokenizer, params, type='eval'):
    model.to(params['device'])
    model.eval()
    predictions = []

    for batch in eval_data:
        in_batch = {k: v.to(params['device']) for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
        outputs = model(**in_batch)
        predictions += get_top_k_predictions(batch, outputs.logits, tokenizer, params)

    '''Print prediction to file'''
    model_name = params['pretrained_model'].replace("/", "-")
    file_name = "results/" + '__'.join([type, params['train_type'],
                                        params['ft_type'], params['train_data'],
                                        params['prompt1'], params['prompt2'], model_name,
                                        params['template'], str(params['context_window']),
                                        str(params['top_train']),
                                        'Yes' if params['filter_candidates'] else 'No']
                                       )

    with(open(file_name + '_pred.tsv', 'w', encoding='UTF-8')) as pred:
        for index in range(len(predictions)):
            pred.write('\t'.join([predictions[index][0], predictions[index][1], predictions[index][3]]) + '\n')

    '''This is used in the training step only, to evaluate the performance given the gold labels'''
    if type != 'test':
        with(open(file_name + '_gold.tsv', 'w', encoding='UTF-8')) as gold:
            for index in range(len(predictions)):
                gold.write('\t'.join(predictions[index][:3]) + '\n')
        eval_result(file_name + '_gold.tsv', file_name + '_pred.tsv')
        scores = get_scores(file_name + '_pred.tsv.out')
        return scores


def get_scores(file):
    results = {}
    with open(file) as f1:
        lines = [s.strip() for s in f1.readlines()]
        flag = False
        for line in lines:
            if flag:
                if "=" in line:
                    key, value = line.split(" = ")
                    results[key] = float(value)
            else:
                if "RESULTS" in line:
                    flag = True
    return results


def eval_result(file_gold, file_pred):
    cmd = f"python3 {TSAR_EVAL} " \
          f"--gold_file {file_gold} " \
          f"--predictions_file {file_pred} " \
          f"--output_file \"{file_pred}.out\""
    print(f"Running evaluation script: {cmd}")
    result = subprocess.run(cmd, shell=True)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model', type=str, default="bert-base-uncased",
                        help="Name or path of the pretrained model")
    parser.add_argument('--epoch', type=int, default=5, help="Number of epochs")
    parser.add_argument('--top_train', type=int, default=3, help="Number of substitutions used in the training phase")
    parser.add_argument('--top_k', type=int, default=10,
                        help="Number of top_k to return by the model in the prediction phase")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size")
    parser.add_argument('--prompt1', type=str, default="easier", help="Prompt 1")
    parser.add_argument('--prompt2', type=str, default="word", help="Prompt 2")
    parser.add_argument('--template', type=str, default="no_context", help="The prompt template")
    parser.add_argument('--context_window', type=int, default=5, help="Context window")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--train_data', type=str, default="benchLS", help="Path to the parsed data")
    parser.add_argument('--test_data', type=str, default="data/tsar2022_en_test_none.tsv",
                        help="Path to the testing data")
    parser.add_argument('--train_type', type=str, default="finetune", help="Training type: finetune or zero")
    parser.add_argument('--ft_type', type=str, default="augment",
                        help="We will augment the data using the gold substitutions to fine-tune the model")
    parser.add_argument('--language', type=str, default="EN",
                        help="Languange: EN, ES or PT for selecting the correct parser")
    parser.add_argument('--filter_candidates', default=True, help="Filter candidates as a postprocessing step",
                        action='store_true')

    args = parser.parse_args()
    return vars(args)  # convert to a dictionary


def _humanized_time(second):
    """
        Returns a human readable time.
    """
    m, s = divmod(second, 60)
    h, m = divmod(m, 60)
    return "%dh %02dm %02ds" % (h, m, s)


if __name__ == '__main__':
    params = get_arguments()
    main(params)

# UoM&MMU@[TSAR-2022 Shared Task](https://taln.upf.edu/pages/tsar2022-st/)
## PromptLS: Prompt Learning for Lexical Simplification


Official repository for the paper "[UoM&MMU at TSAR-2022 Shared Task — PromptLS: Prompt Learning for Lexical Simplification](https://drive.google.com/file/d/10nOMKuM62khIfRea8-XHdG6jsyMXsZtP/view?usp=share_link)", a prompt-based learning approach for Lexical Simplification for the **Text Simplification, Accessibility, and Readability (TSAR-2022) Shared Task** 
by [@lmvasquezr](https://twitter.com/lmvasquezr), [@nguyenthnhung](https://twitter.com/nguyenthnhung), [@MattShardlow](https://twitter.com/MattShardlow) and [@SAnaniadou](https://twitter.com/SAnaniadou). 

If you have any question, please don't hesitate to [contact us](mailto:lvasquezcr@gmail.com?subject=[GitHub]%20Investigating%20TS%20Eval%20Question). Feel free to submit any issue/enhancement in [GitHub](https://github.com/lmvasque/ts-explore/issues). 

## Datasets 

We have trained our models using the following models:

|  Language   |    Datasets     | Instances  |                                    Download                                    |                                             Reference                                              |
|:-----------:|:---------------:|:----------:|:------------------------------------------------------------------------------:|:--------------------------------------------------------------------------------------------------:| 
|   English   |    LexMTurk     |    500     |           [source](https://cs.pomona.edu/~dkauchak/simplification/ )           |                     [(Horn et al., 2014)](https://aclanthology.org/P14-2075/)                      |
|   English   |     NNSEval     |    239     |           [source](https://zenodo.org/record/2552381#.Y2ququzP0-R)             |       [(Paetzold and Specia, 2016b)](https://ojs.aaai.org/index.php/AAAI/article/view/9885)        |
|   English   |     BenchLS     |    929     |             [source](http://ghpaetzold.github.io/data/BenchLS.zip)             |                 [(Paetzold and Specia, 2016a)](https://aclanthology.org/L16-1491/)                 |
|   English   |      CEFR       |    414     | [source](http://www-bigdata.ist.osaka-u.ac.jp/arase/pj/lex-simplification.zip) |                    [(Uchida et al., 2018)](https://aclanthology.org/L18-1514/)                     |
|   Spanish   |     EASIER      |   5,130    |    [source](https://github.com/LURMORENO/EASIER_CORPUS/blob/main/SGSS.csv )    |               [(Alarcon et al., 2021)](https://ieeexplore.ieee.org/document/9400837)               |
| Portuguese  | SIMPLEX-PB-3.0  |   1,582    |          [source](https://github.com/nathanshartmann/SIMPLEX-PB-3.0)           |  [(Hartmann and Aluisio, 2021)](https://linguamatica.com/index.php/linguamatica/article/view/323)  |

We also used the [trial and official test datasets](https://github.com/LaSTUS-TALN-UPF/TSAR-2022-Shared-Task/tree/main/datasets) provided by the Shared Task.

## Models

Our models were fine-tuned using prompt-learning for **Lexical Simplification**. 
We have released our pretrained models in HuggingFace, in the following links (current model page in bold):

| Model Name                                                           | Run # |  Language   | Setting       |
|----------------------------------------------------------------------|-------|:-----------:|---------------|
| **[prompt-ls-en-1](https://huggingface.co/lmvasque/prompt-ls-en-1)** | **1** | **English** | **fine-tune** |
| [prompt-ls-en-2](https://huggingface.co/lmvasque/prompt-ls-en-2)     | 2     |   English   | fine-tune     |
| [roberta-large](https://huggingface.co/roberta-large)                | 3     |   English   | zero-shot     |
| [prompt-ls-es-1](https://huggingface.co/lmvasque/prompt-ls-es-1)     | 1     |   Spanish   | fine-tune     |
| [prompt-ls-es-2](https://huggingface.co/lmvasque/prompt-ls-es-2)     | 2     |   Spanish   | fine-tune     |
| [prompt-ls-es-3](https://huggingface.co/lmvasque/prompt-ls-es-3)     | 3     |   Spanish   | fine-tune     |
| [prompt-ls-pt-1](https://huggingface.co/lmvasque/prompt-ls-pt-1)     | 1     | Portuguese  | fine-tune     |
| [prompt-ls-pt-2](https://huggingface.co/lmvasque/prompt-ls-pt-2)     | 2     | Portuguese  | fine-tune     |
| [prompt-ls-pt-3](https://huggingface.co/lmvasque/prompt-ls-pt-3)     | 3     | Portuguese  | fine-tune     |


For the zero-shot setting, we used the original models with no further training. Links to these models are also updated in the table above.

## Reproducibility

We have also made available [our source code](https://github.com/lmvasque/ls-prompt-tsar2022/tree/main/src) and 
scripts (see below) for fine-tuning the selected models from scratch using prompt learning.
You can also find additional scripts we used for [benchmarking the best models](https://github.com/lmvasque/ls-prompt-tsar2022/tree/main/scripts/benchmark) and for running our selected configuration for the [final submissions runs](https://github.com/lmvasque/ls-prompt-tsar2022/tree/main/scripts/submission). 

## Model fine-tuning (from scratch) & testing

### 1. To fine-tune a LM using the prompt data:

1. Create a virtual env
```bash 
conda create -n test-env
```
2. Install the required libraries from 'requirements.txt'
```bash 
pip install -r requirements.txt
```
3. Activate the virtual env
```
conda activate test-env
```
4. Prepare your inputs. Make sure the input data has the same format as [data/en/parsed](https://github.com/lmvasque/ls-prompt-tsar2022/tree/main/data/en/parsed).
5. Run the following command:
```bash
python3 train.py --pretrained_model bert-base-uncased \
    --epoch 5 \
    --top_train 1 \
    --top_k 10 \
    --batch_size 8 \
    --prompt1 "easier" \
    --prompt2 "word" \
    --template "all_context" \
    --train_data "benchLS" \
    --train_type "finetune"
```

#### Notes:
- When the "train_type" is "zero", the performance will be same as the pipeline.
- The option 'filter_candidates' is ``True`` by default.
- When the top_train > 1, the default finetune mode is 'augment'.

### 2. To test the model on unlabelled data:
 
To test the model on unlabelled data and with a pretrained model in HuggingFace, you can run the command line below, 
updating the following params: 
```bash
python3 train.py --pretrained_model bert-base-uncased \
    --epoch 5 \
    --top_train 1 \
    --top_k 10 \
    --batch_size 8 \
    --prompt1 "easier" \
    --prompt2 "word" \
    --template "all_context" \
    --train_data "benchLS" \
    --train_type "zero" \
    --test_data "data/tsar2022_en_test_none.tsv"
```

#### Notes
- With the parameter ``--train_type "zero" \`` the model can be tested in any pretrained model (mask-based) indicated in: ``--pretrained_model``.
- To run together from scratch the training and test setup, you can run using: ``--train_type "finetuning" \``. The code will call the training function first and then use the trained model to test.
- When 'test_data' is empty, it will only run the training step.

## Citation

If you use our results and scripts in your research, please cite our work: 
"[UoM&MMU at TSAR-2022 Shared Task — PromptLS: Prompt Learning for Lexical Simplification](https://drive.google.com/file/d/10nOMKuM62khIfRea8-XHdG6jsyMXsZtP/view?usp=share_link)". 

```
@inproceedings{vasquez-rodriguez-etal-2022-prompt-ls,
    title = "UoM\&MMU at TSAR-2022 Shared Task — PromptLS: Prompt Learning for Lexical Simplification",
    author = "V{\'a}squez-Rodr{\'\i}guez, Laura  and
      Nguyen, Nhung T. H. and
      Shardlow, Matthew and
      Ananiadou, Sophia",
    booktitle = "Shared Task on Text Simplification, Accessibility, and Readability (TSAR-2022), EMNLP 2022",
    month = dec,
    year = "2022",
}
```


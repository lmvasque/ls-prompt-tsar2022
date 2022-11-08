#!/bin/bash

get_command () {
  model_name=`echo $model | sed 's?/?_?g'`
  cmd="python3 src/train.py  --pretrained_model $model \
  --top_train $top_train \
  --prompt1 $prompt1 \
  --prompt2 $prompt2 \
  --template $context \
  --train_data $dataset \
  --context_window $window \
  --filter_candidates \
  --train_type $train \
  --test_data data/tsar2022_en_test_none.tsv"
  sleep 5
}

run_command() {
  get_command
  echo "$cmd"
  $cmd > logs/en_$train-$dataset-$prompt1-$prompt2-"$model_name"-$context-$top_train-$window.log
  echo ""
}

cmd=""

#################
## Submission 1 #
#################
train="finetune"
dataset="all"
prompt1="simple"
prompt2="word"
model="roberta-large"
context="k_context"
window=5
top_train=5
run_command


#################
## Submission 2 #
#################
train="finetune"
dataset="all"
prompt1="easier"
prompt2="word"
model="bert-base-multilingual-uncased"
context="k_context"
window=10
top_train=10
run_command

#################
## Submission 3 #
#################
train="zero"
dataset="all"
prompt1="easier"
prompt2="word"
model="roberta-large"
context="k_context"
window=5
top_train=1
run_command

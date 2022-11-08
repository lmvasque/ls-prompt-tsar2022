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
  --language $language \
  --train_type $train \
  --test_data data/tsar2022_es_test_none.tsv \
  --top_k $top_k"
  sleep 5
}

run_command() {
  get_command
  echo "$cmd"
  $cmd > logs/$language_$train-$dataset-$prompt1-$prompt2-"$model_name"-$context-$top_train-$window.log
  echo ""
}

#####################
## General Settings #
#####################
train="finetune"
dataset="easier"
language="ES"
top_k=20
cmd=""
window="-1"


#################
## Submission 1 #
#################
model="bertin-project/bertin-roberta-base-spanish"
prompt1="sinónimo"
prompt2="fácil"
context="all_context_es"
top_train=3
run_command


#################
## Submission 2 #
#################
model="bertin-project/bertin-roberta-base-spanish"
prompt1="palabra"
prompt2="simple"
context="all_context_es"
top_train=10
run_command

#################
## Submission 3 #
#################
model="bertin-project/bertin-roberta-base-spanish"
prompt1="sinónimo"
prompt2="fácil"
context="k_context_es"
top_train=10
window=10
run_command

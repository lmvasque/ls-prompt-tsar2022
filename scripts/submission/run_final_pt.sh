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
  --test_data data/tsar2022_pt_test_none.tsv \
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
dataset="simplex"
language="PT"
top_k=20
cmd=""
window="-1"
model="rdenadai/BR_BERTo"


#################
## Submission 1 #
#################
prompt1="palavra"
prompt2="simples"
context="no_context_v0_pt"
top_train=8
run_command


#################
## Submission 2 #
#################
prompt1="sinônimo"
prompt2="fácil"
context="no_context_v0_pt"
top_train=10
run_command

#################
## Submission 3 #
#################
prompt1="sinônimo"
prompt2="simples"
context="k_context_pt"
top_train=10
window=5
run_command

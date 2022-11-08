#!/bin/bash

for model in "xlm-roberta-large" "bert-base-multilingual-uncased" "josu/albert-pt-br" "josu/roberta-pt-br" "rdenadai/BR_BERTo" "facebook/xlm-roberta-xl"
do
    model_name=`echo $model | sed 's?/?_?g'`
    for context in "no_context_v0_pt" "k_context_pt" "all_context_pt"
    do
        for prompt1 in "sinônimo" "palavra"
        do
            for prompt2 in "fácil" "simples"
            do
                for dataset in "simplex"
                do
                    for top_train in 1 2 3 5 10
                    do
                        for train in "finetune"
                        do
                            if [ "$context" = "k_context_pt" ]; then
                                for window in 5 10
                                do
                                    echo $train-$dataset-$prompt1-"$model_name"-$context-$window.log
                                    python3 src/train.py --pretrained_model $model \
                                    --top_train $top_train \
                                    --prompt1 $prompt1 \
                                    --prompt2 $prompt2 \
                                    --template $context \
                                    --train_data $dataset \
                                    --context_window $window \
                                    --filter_candidates \
                                    --language PT \
                                    --top_k 20 \
                                    --train_type $train > logs/$train-$dataset-$prompt1-$prompt2-"$model_name"-$context-$top_train-$window.log
                                    sleep 10
                                done
                            else
                                echo $train-$dataset-$prompt1-"$model_name"-$context.log
                                python3 src/train.py --pretrained_model $model \
                                    --top_train $top_train \
                                    --prompt1 $prompt1 \
                                    --prompt2 $prompt2 \
                                    --template $context \
                                    --train_data $dataset \
                                    --context_window -1 \
                                    --filter_candidates \
                                    --language PT \
                                     --top_k 20 \
                                    --train_type $train > logs/$train-$dataset-$prompt1-$prompt2-"$model_name"-$context-$top_train-$window.log
                                    sleep 10
                            fi    
                        done
                    done
                done
            done
        done
    done
done

python3 src/main.py


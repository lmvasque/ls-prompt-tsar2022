#!/bin/bash

for model in "roberta-large" "bert-large-uncased" "xlm-roberta-large" "bert-base-multilingual-uncased" "albert-large-v2"
do
    for context in "k_context" "all_context" "no_context"
    do   
        for prompt1 in "simple" "easier"
        do  
            for prompt2 in "word" "synonym"
            do
                for dataset in "benchLS" "all"
                do  
                    for top_train in 1 2 4 6 8 10
                    do
                        for train in "finetune" "zero"
                        do 
                            if [ "$context" = "k_context" ]; then
                                for window in 5 10
                                do
                                    echo $train-$dataset-$prompt-$model-$context-$window.log
                                    python3 src/train.py --pretrained_model $model \
                                    --top_train $top_train \
                                    --prompt1 $prompt1 \
                                    --prompt2 $prompt2 \
                                    --template $context \
                                    --train_data $dataset \
                                    --context_window $window \
                                    --train_type $train > logs/$train-$dataset-$prompt1-$prompt2-$model-$context-$top_train-$window.log
                                done
                            else
                                echo $train-$dataset-$prompt-$model-$context.log
                                python3 src/train.py --pretrained_model $model \
                                    --top_train $top_train \
                                    --prompt1 $prompt1 \
                                    --prompt2 $prompt2 \
                                    --template $context \
                                    --train_data $dataset \
                                    --context_window -1 \
                                    --train_type $train > logs/$train-$dataset-$prompt1-$prompt2-$model-$context-$top_train-$window.log
                            fi    
                        done
                    done
                done
            done
        done
    done
done

python3 src/main.py


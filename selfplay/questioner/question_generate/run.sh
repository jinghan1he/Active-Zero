#!/bin/bash

# load the model name from the command line
questioner_model_path=$1
solver_model_path=$2
save_name=$3
rollout_n=$4

# set environment variable
export VLLM_DISABLE_COMPILE_CACHE=1

python -m selfplay.questioner.question_generate.generate \
        --model "$questioner_model_path" \
        --save_name "$save_name" \
        --rollout_n "$rollout_n"

sleep 5

python -m selfplay.questioner.question_generate.evaluate \
        --model "$solver_model_path" \
        --save_name "$save_name"

sleep 5

python -m selfplay.questioner.question_generate.save --max_score 0.8 --min_score 0.3 --save_name $save_name

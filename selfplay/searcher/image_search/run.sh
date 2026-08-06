#!/bin/bash

# load the model name from the command line
model_name=$1
save_name=$2

top_k=5
num_queries=6000

export VLLM_DISABLE_COMPILE_CACHE=1

python -m selfplay.searcher.image_search.query_generate \
    --model "$model_name" \
    --num_samples "$num_queries" \
    --save_name "$save_name"

python -m selfplay.searcher.image_search.image_retrieve \
    --top_k "$top_k" \
    --save_name "$save_name"

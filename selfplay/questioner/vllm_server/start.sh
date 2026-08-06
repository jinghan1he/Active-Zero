#!/bin/bash

questioner_model_path=$1

mkdir -p ${STORAGE_PATH}/logs/vllm_server/questioner
pkill -f "start_questioner_server"

export VLLM_DISABLE_COMPILE_CACHE=1

for i in {0..7}; do
    CUDA_VISIBLE_DEVICES=$i python -m selfplay.questioner.vllm_server.start_questioner_server --port 500${i} --model_path $questioner_model_path > ${STORAGE_PATH}/logs/vllm_server/questioner/${i}.log 2>&1 &
done

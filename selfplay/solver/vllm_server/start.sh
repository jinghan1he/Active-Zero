#!/bin/bash

solver_model_path=$1

mkdir -p ${STORAGE_PATH}/logs/vllm_server/solver
pkill -f "start_solver_server"

export VLLM_DISABLE_COMPILE_CACHE=1

for i in {0..7}; do
    CUDA_VISIBLE_DEVICES=$i python -m selfplay.solver.vllm_server.start_solver_server --port 600${i} --model_path $solver_model_path > ${STORAGE_PATH}/logs/vllm_server/solver/${i}.log 2>&1 &
done
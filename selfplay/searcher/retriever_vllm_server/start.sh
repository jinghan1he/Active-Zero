#!/bin/bash

mkdir -p ${STORAGE_PATH}/logs/vllm_server/retriever

for i in {0..7}; do
    CUDA_VISIBLE_DEVICES=$i python -m selfplay.searcher.retriever_vllm_server.start_retriever_server --port 700${i} > ${STORAGE_PATH}/logs/vllm_server/retriever/${i}.log 2>&1 & 
done

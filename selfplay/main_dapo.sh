#!/bin/bash

set -x

set -euo pipefail

export PYTHONUNBUFFERED=1

mkdir -p \
  "$STORAGE_PATH/models" \
  "$STORAGE_PATH/generated_question" \
  "$STORAGE_PATH/temp_results" \
  "$STORAGE_PATH/local_train_dataset" \
  "$STORAGE_PATH/searched_image"

Base_model=Qwen/Qwen2.5-VL-3B-Instruct
Model_abbr=Qwen2.5-VL-3B-Instruct
echo "Model_abbr: $Model_abbr"

# Initialize first iteration with base model
bash selfplay/searcher/searcher_train.sh \
    $Base_model \
    $Base_model \
    $Base_model \
    ${Model_abbr}_searcher_v1

bash selfplay/questioner/questioner_train.sh  \
    $Base_model \
    ${STORAGE_PATH}/models/${Model_abbr}_searcher_v1/global_step_10/actor/huggingface \
    $Base_model \
    ${Model_abbr}_questioner_v1
    
bash selfplay/solver/solver_train_dapo.sh \
    $Base_model \
    ${STORAGE_PATH}/models/${Model_abbr}_searcher_v1/global_step_10/actor/huggingface \
    ${STORAGE_PATH}/models/${Model_abbr}_questioner_v1/global_step_10/actor/huggingface \
    ${Model_abbr}_solver_v1

bash Evaluation/eval_gen_questions.sh ${STORAGE_PATH//save_}_v1 ${STORAGE_PATH}/models/${Model_abbr}_solver_v1/global_step_30/actor/huggingface

for i in {2..3}; do
    prev=$((i-1))
    
    bash selfplay/searcher/searcher_train.sh \
        ${STORAGE_PATH}/models/${Model_abbr}_searcher_v${prev}/global_step_10/actor/huggingface \
        ${STORAGE_PATH}/models/${Model_abbr}_questioner_v${prev}/global_step_10/actor/huggingface \
        ${STORAGE_PATH}/models/${Model_abbr}_solver_v${prev}/global_step_30/actor/huggingface \
        ${Model_abbr}_searcher_v${i}

    bash selfplay/questioner/questioner_train.sh \
        ${STORAGE_PATH}/models/${Model_abbr}_questioner_v${prev}/global_step_10/actor/huggingface \
        ${STORAGE_PATH}/models/${Model_abbr}_searcher_v${i}/global_step_10/actor/huggingface \
        ${STORAGE_PATH}/models/${Model_abbr}_solver_v${prev}/global_step_30/actor/huggingface \
        ${Model_abbr}_questioner_v${i}

    bash selfplay/solver/solver_train_dapo.sh \
        ${STORAGE_PATH}/models/${Model_abbr}_solver_v${prev}/global_step_30/actor/huggingface \
        ${STORAGE_PATH}/models/${Model_abbr}_searcher_v${i}/global_step_10/actor/huggingface \
        ${STORAGE_PATH}/models/${Model_abbr}_questioner_v${i}/global_step_10/actor/huggingface \
        ${Model_abbr}_solver_v${i}

    bash Evaluation/eval_gen_questions.sh ${STORAGE_PATH//save_}_v${i} ${STORAGE_PATH}/models/${Model_abbr}_solver_v${i}/global_step_30/actor/huggingface

done

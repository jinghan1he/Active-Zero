#!/bin/bash

set -x

solver_model_path=$1
searcher_model_path=$2
questioner_model_path=$3
experiment_name=$4

export VLLM_DISABLE_COMPILE_CACHE=1

if [ -d "${STORAGE_PATH}/models/${experiment_name}/global_step_30/actor/huggingface" ]; then
    echo "Solver model ${experiment_name} already exists, skipping training..."
else

    if [ -d "${STORAGE_PATH}/local_train_dataset/${experiment_name}" ] && \
    [ -f "${STORAGE_PATH}/local_train_dataset/${experiment_name}/summary.json" ]; then
        echo "Local train dataset ${experiment_name} already exists, skipping..."
    else
        echo 'start generating question'
        bash selfplay/questioner/question_generate/run.sh $questioner_model_path $solver_model_path $experiment_name 1
    fi

    echo 'start training solver'
    python3 -m verl.trainer.main \
        config=train_examples/config.yaml \
        data.train_files=${STORAGE_PATH}/local_train_dataset/${experiment_name} \
        data.val_files=chamber111/VPPO_MMK12_validation@train \
        data.format_prompt=./selfplay/solver/format_prompt.jinja \
        data.val_format_prompt=./train_examples/format_prompt/math.jinja \
        data.mini_rollout_batch_size=128 \
        worker.actor.model.model_path=$solver_model_path \
        worker.actor.clip_ratio_low=0.2 \
        worker.actor.clip_ratio_high=0.28 \
        algorithm.online_filtering=True \
        worker.reward.reward_function=./selfplay/solver/reward_func.py:compute_score \
        trainer.experiment_name=${experiment_name} \
        trainer.save_checkpoint_path=${STORAGE_PATH}/models/${experiment_name}/ \
        trainer.nnodes=1 \
        trainer.max_steps=30 \
        trainer.save_freq=10 \
        trainer.save_limit=-1

    sleep 5

    echo "merging model"
    python scripts/model_merger.py --local_dir ${STORAGE_PATH}/models/${experiment_name}/global_step_30/actor

    sleep 10

    echo "solver training finished"
fi

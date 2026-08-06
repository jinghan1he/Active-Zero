#!/bin/bash

set -x

questioner_model_path=$1
searcher_model_path=$2
solver_model_path=$3
save_path=$4
echo "save_path: $save_path"

if [ -d "${STORAGE_PATH}/models/${save_path}/global_step_10/actor/huggingface" ]; then
    echo "Questioner model ${save_path} already exists, skipping training..."
else
    echo 'start searching images'
    bash selfplay/searcher/image_search/run.sh $searcher_model_path ${save_path/questioner/solver}

    bash selfplay/solver/vllm_server/start.sh $solver_model_path
    sleep 30

    echo "start training questioner: $questioner_model_path -> $save_path"
    python3 -m verl.trainer.main \
        config=train_examples/config.yaml \
        data.train_files=${SOURCE_DATASET} \
        data.train_indices_file="${STORAGE_PATH}/searched_image/${save_path/questioner/solver}_indices.npy" \
        data.val_files=/path/to/data_cache/R1-Onevision-Val \
        data.format_prompt=./selfplay/questioner/format_prompt.jinja \
        data.mini_rollout_batch_size=128 \
        worker.actor.model.model_path=$questioner_model_path \
        worker.reward.reward_function=./selfplay/questioner/reward_func.py:compute_score \
        trainer.experiment_name=$save_path \
        trainer.save_checkpoint_path=${STORAGE_PATH}/models/$save_path \
        trainer.max_steps=10 \
        trainer.save_freq=10 \
        trainer.save_limit=-1

    sleep 5

    echo "merging model"
    python scripts/model_merger.py --local_dir ${STORAGE_PATH}/models/${save_path}/global_step_10/actor

    sleep 10

    echo "questioner training finished"
fi

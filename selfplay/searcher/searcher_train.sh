#!/bin/bash

searcher_model_path=$1
questioner_model_path=$2
solver_model_path=$3
save_path=$4
echo "save_path: $save_path"

if [ -d "${STORAGE_PATH}/models/${save_path}/global_step_10/actor/huggingface" ]; then
    echo "Searcher model ${save_path} already exists, skipping training..."
else
    bash selfplay/searcher/retriever_vllm_server/start.sh

    bash selfplay/questioner/vllm_server/start.sh $questioner_model_path
    sleep 130 # wait for questioner vllm server to start

    bash selfplay/solver/vllm_server/start.sh $solver_model_path
    sleep 30 # wait for solver vllm server to start

    # start training searcher
    echo "Start training searcher: $searcher_model_path -> $save_path"
    python3 -m verl.trainer.main \
        config=train_examples/config.yaml \
        data.train_files=/path/to/data_cache/dummy_dataset_1m \
        data.val_files=/path/to/data_cache/dummy_dataset_1k \
        data.format_prompt=./selfplay/searcher/format_prompt.jinja \
        data.filter_overlong_prompts=False \
        data.mini_rollout_batch_size=128 \
        worker.rollout.sampling_seed=-1 \
        worker.actor.model.model_path=$searcher_model_path \
        worker.reward.reward_function=./selfplay/searcher/reward_func.py:compute_score \
        trainer.experiment_name=$save_path \
        trainer.save_checkpoint_path=${STORAGE_PATH}/models/$save_path \
        trainer.max_steps=10 \
        trainer.save_freq=5 \
        trainer.save_limit=-1

    sleep 5

    echo "merging model"
    python scripts/model_merger.py --local_dir ${STORAGE_PATH}/models/${save_path}/global_step_10/actor

    sleep 10

    echo "searcher training finished"

fi
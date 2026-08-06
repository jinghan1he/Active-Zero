#!/bin/bash

set -x

experiment_name=$1
model_path=$2

export PYTHONUNBUFFERED=1
export TORCH_COMPILE_DISABLE=1

DATASETS=(
  "Evaluation/Datasets/DynaMath"
  "Evaluation/Datasets/MathVerse"
  "Evaluation/Datasets/MathVision"
  "Evaluation/Datasets/We-Math"
  "Evaluation/Datasets/LogicVista"
  "Evaluation/Datasets/RealWorldQA"
  "Evaluation/Datasets/MMMU_Pro"
  "Evaluation/Datasets/M3CoT"
  "Evaluation/Datasets/VisNumBench"
  "Evaluation/Datasets/HallusionBench"
  "Evaluation/Datasets/mm-vet"
  "zli12321/MMMU@test"
  "hiyouga/geometry3k@test"
  "chamber111/VPPO_MMK12_validation@train"
)

# ------------------------------------------------------------------
# STATIC pieces of the command line (everything that never changes)
# ------------------------------------------------------------------
BASE_CMD="python3 -m verl.trainer.main \
  config=val_examples/config.yaml \
  worker.actor.model.model_path=${model_path} \
  worker.rollout.val_override_config.n=1 \
  worker.rollout.val_override_config.temperature=0.0 \
  data.max_response_length=4096 \
  trainer.experiment_name=${experiment_name} \
  trainer.n_gpus_per_node=8 \
  trainer.val_only=true"

# ------------------------------------------------------------------
# LOOP over datasets
# ------------------------------------------------------------------
for DS in "${DATASETS[@]}"; do
  # extract the dataset name from the path
  SHORT_NAME=${DS##*/}
  SHORT_NAME=${SHORT_NAME%%@*}

  RESULTS_PATH="Evaluation/Raw-Outputs/${experiment_name}/${SHORT_NAME}.jsonl"

  if [ -f ${RESULTS_PATH} ]; then
    echo ">>> Results file already exists: ${RESULTS_PATH}"
    continue
  fi

  echo ">>> Evaluating on ${DS}"
  CMD="${BASE_CMD} \
    data.val_files=${DS} \
    trainer.response_path=${RESULTS_PATH}"

  # show the command (optional)
  echo "$CMD" | sed 's/  */ /g'
  echo "------------------------------------------------------------"

  # run it
  eval $CMD
  
  # sleep for a few seconds before next dataset (except for the last one)
  sleep 10
done


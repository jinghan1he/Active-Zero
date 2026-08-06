#!/bin/bash

set -x

experiment_name=$1
model_path=$2
format_prompt=$3

if [ -z "$format_prompt" ]; then
    format_prompt="./train_examples/format_prompt/math.jinja"
fi

export PYTHONUNBUFFERED=1
export TORCH_COMPILE_DISABLE=1

node_count=$(ray status | grep -c "node_")
num_gpus=$((node_count * 8))

DATASETS=(
  "Evaluation/Datasets/DynaMath"
  "Evaluation/Datasets/MathVerse"
  "Evaluation/Datasets/MathVerse-VD"
  "Evaluation/Datasets/MathVision"
  "Evaluation/Datasets/We-Math"
  "Evaluation/Datasets/MathVista"
  "Evaluation/Datasets/LogicVista"
  "Evaluation/Datasets/RealWorldQA"
  "Evaluation/Datasets/MMMU_Pro_10options"
  "Evaluation/Datasets/MMMU_Pro_4options"
  "Evaluation/Datasets/MMStar"
  "Evaluation/Datasets/VisNumBench"
  "Evaluation/Datasets/HallusionBench"
  "zli12321/MMMU@test"
)

# ------------------------------------------------------------------
# STATIC pieces of the command line (everything that never changes)
# ------------------------------------------------------------------
BASE_CMD="python -m Evaluation.vllm_generate \
    --model_path ${model_path} \
    --num_workers ${num_gpus} \
    --format_prompt ${format_prompt}"
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
    --dataset_path ${DS} \
    --output_path ${RESULTS_PATH}"

  # show the command (optional)
  echo "$CMD" | sed 's/  */ /g'
  echo "------------------------------------------------------------"

  # run it
  eval $CMD
  
  # sleep for a few seconds before next dataset (except for the last one)
  sleep 10
done


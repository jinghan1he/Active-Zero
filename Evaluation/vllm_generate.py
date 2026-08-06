import argparse
import json
import math
import os
from pathlib import Path
import ray
import vllm
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer

from verl.utils.dataset import RLHFDataset, collate_fn
from verl.workers.rollout.vllm_rollout_spmd import _process_multi_modal_data


def load_tokenizer_and_processor(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        processor = AutoProcessor.from_pretrained(model_path)
    except Exception:
        processor = None

    return tokenizer, processor


@ray.remote(num_gpus=1)
def generate_results(args, shard_idx, shard_total):

    tokenizer, processor = load_tokenizer_and_processor(args.model_path)

    # build dataset and shard indices
    val_dataset = RLHFDataset(
        data_path=args.dataset_path,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key="problem",
        answer_key="answer",
        image_key="images",
        max_prompt_length=4096,
        truncation="right",
        format_prompt=args.format_prompt,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )

    total_len = len(val_dataset)
    shard_size = math.ceil(total_len / shard_total)
    start = shard_idx * shard_size
    end = min(total_len, start + shard_size)
    if start >= end:
        return ""  # empty shard

    subset = Subset(val_dataset, list(range(start, end)))

    val_loader = DataLoader(
        dataset=subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False,
    )

    model = vllm.LLM(
        model=args.model_path,
        skip_tokenizer_init=False,
        dtype="bfloat16",
        seed=1,
        max_model_len=4096+4096,
        gpu_memory_utilization=0.9,
        max_num_batched_tokens=8192,
        disable_mm_preprocessor_cache=True,
    )

    sampling_kwargs = {
        "max_tokens": 4096,
        "temperature": 0.0,
        "n": 1,
        "stop_token_ids": [tokenizer.eos_token_id],
    }
    sample_params = vllm.SamplingParams(**sampling_kwargs)

    shard_path = Path(args.output_path)
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    shard_path = shard_path.with_suffix(f".shard{shard_idx}.jsonl")

    with shard_path.open("w", encoding="utf-8") as out_f:
        for batch in val_loader:

            batch_indices = batch["dataset_index"]
            batch_raw_prompt_ids = batch["raw_prompt_ids"]
            batch_multi_modal_data = batch.get("multi_modal_data", None)

            if batch_multi_modal_data is not None:
                vllm_inputs = []
                for raw_prompt_ids, multi_modal_data in zip(batch_raw_prompt_ids, batch_multi_modal_data):
                    vllm_inputs.append(
                        {
                            "prompt_token_ids": list(raw_prompt_ids),
                            "multi_modal_data": _process_multi_modal_data(
                                multi_modal_data,
                                args.min_pixels,
                                args.max_pixels,
                                args.video_fps,
                            ),
                        }
                    )
            else:
                vllm_inputs = [{"prompt_token_ids": list(raw_prompt_ids)} for raw_prompt_ids in batch_raw_prompt_ids]

            completions = model.generate(vllm_inputs, sampling_params=sample_params, use_tqdm=True)
            
            batch_prompt_texts = [tokenizer.decode(p, skip_special_tokens=True) for p in batch_raw_prompt_ids]
            for completion, prompt_text, index in zip(completions, batch_prompt_texts, batch_indices):
                for output in completion.outputs:
                    record = {
                        "dataset_index": index,
                        "prompt": prompt_text,
                        "response": output.text,
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "shard_idx": shard_idx,
        "shard_path": str(shard_path),
    }


def main():
    parser = argparse.ArgumentParser(description="vLLM inference script for evaluation")
    parser.add_argument("--output_path", type=str, default="test.jsonl", help="Output path (merged file)")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Model path")
    parser.add_argument("--dataset_path", type=str, default="Evaluation/Datasets/DynaMath", help="Dataset path or HF dataset spec")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of Ray workers")
    parser.add_argument("--ray_address", type=str, default="auto", help="Ray cluster address; use 'auto' for local")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--min_pixels", type=int, default=200704, help="Minimum number of pixels")
    parser.add_argument("--max_pixels", type=int, default=1003520, help="Maximum number of pixels")
    parser.add_argument("--format_prompt", type=str, default="./train_examples/format_prompt/math.jinja", help="Format prompt")
    parser.add_argument("--video_fps", type=float, default=2.0, help="Video FPS")
    args = parser.parse_args()

    ray.init(address=args.ray_address, ignore_reinit_error=True, namespace="vllm-eval")

    futures = [generate_results.remote(args, shard_idx, args.num_workers) for shard_idx in range(args.num_workers)]
    results = ray.get(futures)
    results = sorted(results, key=lambda x: x["shard_idx"])
    shard_files = [result["shard_path"] for result in results]

    # merge shards
    with open(args.output_path, "w", encoding="utf-8") as merged_f:
        for shard_file in shard_files:
            with open(shard_file, "r", encoding="utf-8") as sf:
                for line in sf:
                    merged_f.write(line)
    
    for shard_file in shard_files:
        os.remove(shard_file)

    print(f"Merged {len(shard_files)} shards into {args.output_path}")


if __name__ == "__main__":
    main()

import vllm
from transformers import AutoTokenizer
import argparse
import random
from tqdm import tqdm
from typing import List, Dict
from jinja2 import Template
from vllm.outputs import RequestOutput
import json
import regex as re
import os
import glob
import numpy as np
from datasets import load_dataset, load_from_disk
from PIL import Image
import math
from torch.utils.data import Dataset, DataLoader
import ray
from verl.utils.dataset import process_image

STORAGE_PATH = os.getenv("STORAGE_PATH")
SOURCE_DATASET = os.getenv("SOURCE_DATASET")


def load_source_dataset(
    data_path,
    indices_file,
    indices_override=None,
    return_indices=False,
):
    if os.path.isdir(data_path):
        dataset = load_from_disk(data_path)
    else:
        dataset = load_dataset(data_path, split="train")

    selected_indices = indices_override
    if indices_override is not None:
        dataset = dataset.select(indices_override)
    else:
        selected_indices = np.load(indices_file).tolist()
        print(f"[Questioner] Selected {len(selected_indices)} indices from {indices_file}")
        dataset = dataset.select(selected_indices)

    if return_indices:
        return dataset, selected_indices
    return dataset


class VQADataset(Dataset):
    def __init__(self, dataset, format_prompt_path):
        self.dataset = dataset
        format_prompt = open(format_prompt_path, encoding="utf-8").read()
        template = Template(format_prompt.strip())
        user_prompt = template.render().strip("<image>")
        self.prompt = (
            f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{user_prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        processed_image = process_image(sample['images'][0], args.min_pixels, args.max_pixels)
        
        return {
            "prompt": self.prompt,
            "image": processed_image,
            "image_idx": int(sample["answer"])
        }


def collate_fn(batch):
    # Prepare batch for vLLM
    batch_chats = []
    batch_indices = []
    for item in batch:
        batch_chats.append({
            "prompt": item["prompt"],
            "multi_modal_data": {"image": item["image"]}
        })
        batch_indices.append(item["image_idx"])
    return {
        "batch_chats": batch_chats,
        "batch_indices": batch_indices
    }


@ray.remote(num_gpus=1, num_cpus=2)
def generate_questions_for_indices(
    args_dict: Dict,
    shard_indices: List[int],
    shard_id: int,
):
    args = argparse.Namespace(**args_dict)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = vllm.LLM(
        model=args.model,
        tokenizer=args.model,
        seed=shard_id,
        disable_mm_preprocessor_cache=True,
        max_model_len=6144,
        dtype="bfloat16",
        max_num_batched_tokens=8192,
    )

    source_dataset = load_source_dataset(
        SOURCE_DATASET,
        args.indices_file,
        indices_override=shard_indices,
    )

    dataset = VQADataset(source_dataset, args.format_prompt)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues with images
        collate_fn=collate_fn,
    )

    sample_params = vllm.SamplingParams(
        max_tokens=2048,
        temperature=1.0,
        top_p=0.95,
        n=args.rollout_n,
        stop_token_ids=[tokenizer.eos_token_id],
    )

    num_questions = 0
    num_errors = 0
    shard_output = f"{args.output_file}.shard{shard_id}"

    with open(shard_output, "w") as f:
        for batch in dataloader:
            completions: List[RequestOutput] = model.generate(
                batch["batch_chats"],
                sampling_params=sample_params,
            )

            for completion, image_idx in zip(completions, batch["batch_indices"]):
                for output in completion.outputs:
                    response = output.text
                    # Extract question type, question, and answer
                    question_types = re.findall(r"<type>(.*?)</type>", response, re.DOTALL)
                    questions = re.findall(r"<question>(.*?)</question>", response, re.DOTALL)
                    answers = re.findall(r"<answer>(.*?)</answer>", response, re.DOTALL)

                    if questions and answers:
                        question_type = question_types[-1].strip() if question_types else "unknown"
                        question = questions[-1].strip()
                        answer = answers[-1].strip()
                        f.write(json.dumps({
                            "question_type": question_type,
                            "question": question,
                            "answer": answer,
                            "image_idx": image_idx,
                        }) + "\n")
                        num_questions += 1
                    else:
                        num_errors += 1

    print(f"[Questioner][Shard {shard_id}] Generated {num_questions} questions ({num_errors} errors) -> {shard_output}")
    return {
        "shard_id": shard_id,
        "num_questions": num_questions,
        "num_errors": num_errors,
        "output_file": shard_output,
    }


def run_distributed(args):
    ray.init(address=args.ray_address, ignore_reinit_error=True)

    print(f"[Questioner] Loading source dataset from {SOURCE_DATASET}...")
    _, selected_indices = load_source_dataset(
        SOURCE_DATASET,
        args.indices_file,
        return_indices=True,
    )

    if not selected_indices:
        print("[Questioner] No indices to process, exiting.")
        return

    shards = [
        list(map(int, shard))
        for shard in np.array_split(selected_indices, args.num_nodes)
        if len(shard) > 0
    ]

    args_dict = vars(args)
    futures = [
        generate_questions_for_indices.remote(args_dict, shard, shard_id)
        for shard_id, shard in enumerate(shards)
    ]

    results = ray.get(futures)
    results = sorted(results, key=lambda x: x["shard_id"])

    total_questions = 0
    total_errors = 0
    with open(args.output_file, "w") as out_f:
        for res in results:
            shard_file = res["output_file"]
            if not os.path.exists(shard_file):
                raise FileNotFoundError(f"Missing shard output {shard_file}")
            with open(shard_file, "r") as shard_f:
                for line in shard_f:
                    out_f.write(line)
            total_questions += res["num_questions"]
            total_errors += res["num_errors"]

    for res in results:
        try:
            os.remove(res["output_file"])
        except OSError:
            pass

    print(f"[Questioner] Generated {total_questions} questions and saved to {args.output_file} ({total_errors} errors)")


def main(args):

    args.indices_file = f"{STORAGE_PATH}/searched_image/{args.save_name}_indices.npy"
    args.output_file = f"{STORAGE_PATH}/generated_question/{args.save_name}.json"
    args.output_results_file = f"{STORAGE_PATH}/generated_question/{args.save_name}_results.json"

    if os.path.exists(args.output_file) and os.path.getsize(args.output_file) > 0:
        print(f"[Questioner] Question file {args.output_file} already exists, skipping...")
        return
    elif os.path.exists(args.output_results_file) and os.path.getsize(args.output_results_file) > 0:
        print(f"[Questioner] Results file {args.output_results_file} already exists, skipping...")
        return

    run_distributed(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Model name or path")
    parser.add_argument("--format_prompt", type=str, default="selfplay/questioner/format_prompt.jinja", help="Path to format prompt")
    parser.add_argument("--save_name", type=str, default="vqa_generated", help="Base name for output file")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for processing")
    parser.add_argument("--rollout_n", type=int, default=1, help="Number of rollouts to generate")
    parser.add_argument("--num_nodes", type=int, default=16, help="Number of Ray nodes to shard workload")
    parser.add_argument("--ray_address", type=str, default="auto", help="Ray address, e.g., 'auto' to connect to an existing cluster")
    parser.add_argument("--max_pixels", type=int, default=1003520, help="Maximum number of pixels for image processing.")
    parser.add_argument("--min_pixels", type=int, default=200704, help="Minimum number of pixels for image processing.")
    args = parser.parse_args()
    
    main(args)
    
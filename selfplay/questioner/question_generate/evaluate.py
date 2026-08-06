import json
import vllm
from transformers import AutoTokenizer
import argparse
import os
import stopit  # Use the robust, thread-safe stopit library for timeouts
from tqdm import tqdm
from jinja2 import Template
from datasets import load_from_disk, load_dataset
from mathruler.grader import extract_boxed_content, grade_answer
import ray
from PIL import Image
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader
from vllm.outputs import RequestOutput
from typing import List

from verl.utils.dataset import process_image


STORAGE_PATH = os.getenv("STORAGE_PATH")
SOURCE_DATASET = os.getenv("SOURCE_DATASET")


@stopit.threading_timeoutable(default='TIMED_OUT')
def grade_answer_with_timeout(res1, res2):
    """
    Wraps the mathruler 'grade_answer' function with a timeout.
    If the function takes too long, it returns 'TIMED_OUT' instead of hanging.
    """
    # The actual timeout value is passed as a keyword argument on each call.
    return grade_answer(res1, res2)


def load_source_dataset(data_path):
    if os.path.isdir(data_path):
        dataset = load_from_disk(data_path)
    else:
        dataset = load_dataset(data_path, split="train")
    return dataset


def load_input_data(input_file):
    data = []
    for line in open(input_file, "r"):
        try:
            data.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"[Evaluator] WARNING: Invalid JSON line: '{line}'")
            continue
    return data


class VQADataset(Dataset):
    def __init__(self, dataset, format_prompt_path, questions, image_indices, question_types):
        self.dataset = dataset
        self.format_prompt = open(format_prompt_path, encoding="utf-8").read()
        self.format_prompt = Template(self.format_prompt.strip())
        self.questions = questions
        self.image_indices = image_indices
        self.question_types = question_types

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        image_idx = int(self.image_indices[idx])
        image = self.dataset[image_idx]['images'][0]
        processed_image = process_image(image, args.min_pixels, args.max_pixels)
        
        # Create prompt in qwen2.5-VL format
        question = self.questions[idx]
        question_type = self.question_types[idx]
        user_prompt = self.format_prompt.render(content=question).strip("<image>")
        prompt = (
            f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{user_prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        
        return {
            "prompt": prompt,
            "image": processed_image,
            "image_idx": image_idx,
            "question": question,
            "question_type": question_type
        }


def collate_fn(batch):
    # Prepare batch for vLLM
    chats, indices, questions, question_types = [], [], [], []
    for item in batch:
        chats.append({
            "prompt": item["prompt"],
            "multi_modal_data": {"image": item["image"]}
        })
        indices.append(item["image_idx"])
        questions.append(item["question"])
        question_types.append(item["question_type"])
    return {
        "chats": chats,
        "indices": indices,
        "questions": questions,
        "question_types": question_types
    }


@ray.remote(num_gpus=1, num_cpus=2)
def evaluate_shard(
    args,
    shard_id,
    shard_questions,
    shard_image_indices,
    shard_question_types,
):

    print(f"[Evaluator][Shard {shard_id}] Loading data and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = vllm.LLM(
        model=args.model,
        tokenizer=args.model,
        gpu_memory_utilization=0.85,
        seed=shard_id,
        disable_mm_preprocessor_cache=True,
        max_model_len=6144,
        dtype="bfloat16",
        max_num_batched_tokens=8192,
    )
    sample_params = vllm.SamplingParams(
        max_tokens=2048,
        temperature=1.0,
        top_p=1.0,
        top_k=40,
        stop_token_ids=[tokenizer.eos_token_id],
        n=args.rollout_n,
    )

    source_dataset = load_source_dataset(SOURCE_DATASET)
    dataset = VQADataset(
        source_dataset,
        args.format_prompt,
        shard_questions,
        shard_image_indices,
        shard_question_types,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # avoid multiprocessing issues
        collate_fn=collate_fn
    )

    shard_output = f"{args.output_file}.shard{shard_id}"
    num_questions = 0
    with open(shard_output, "w") as f:
        for batch in dataloader:
            completions: List[RequestOutput] = model.generate(
                batch["chats"], 
                sampling_params=sample_params
            )
            for completion, question, image_idx, question_type in zip(completions, batch["questions"], batch["indices"], batch["question_types"]):
                # Extract the boxed content from all generated samples
                results = [extract_boxed_content(output.text) for output in completion.outputs]

                answer_counts = {}
                for result in results:

                    if not result or result == "None":
                        continue

                    matched = False
                    for existing_answer in answer_counts:
                        # OPTIMIZATION: Perform cheap string comparisons first.
                        if result == existing_answer or ('no ' in result.lower() and 'no ' in existing_answer.lower()):
                            answer_counts[existing_answer] += 1
                            matched = True
                            break
                        
                        # If cheap checks fail, use the expensive, timed grader.
                        # Check both directions (A vs B and B vs A).
                        match_1 = grade_answer_with_timeout(result, existing_answer, timeout=10)
                        if match_1 == 'TIMED_OUT':
                            print(f"[Evaluator][Shard {shard_id}] GRADER TIMEOUT on: '{result[:30]}...' vs '{existing_answer[:30]}...'")
                            continue # Skip to the next existing_answer
                        
                        if match_1:
                            answer_counts[existing_answer] += 1
                            matched = True
                            break

                        match_2 = grade_answer_with_timeout(existing_answer, result, timeout=10)
                        if match_2 == 'TIMED_OUT':
                            print(f"[Evaluator][Shard {shard_id}] GRADER TIMEOUT on: '{existing_answer[:30]}...' vs '{result[:30]}...'")
                            continue

                        if match_2:
                            answer_counts[existing_answer] += 1
                            matched = True
                            break

                    if not matched:
                        answer_counts[result] = 1

                if not answer_counts:
                    continue

                # Determine the majority answer and its score
                majority_answer = max(answer_counts, key=answer_counts.get)
                max_count = answer_counts[majority_answer]
                score = max_count / len(results)

                # Skip certain question types that are hard to grade automatically
                if "证明" in question or 'box' in question.lower() or 'text' in majority_answer.lower():
                    continue

                f.write(json.dumps({
                    "question": question,
                    "answer": majority_answer,
                    "score": score,
                    "image_idx": image_idx,
                    "question_type": question_type,
                    'results': results
                }) + "\n")
                num_questions += 1

    print(f"[Evaluator][Shard {shard_id}] Done. Valid questions: {num_questions}. Output: {shard_output}")
    return {
        "shard_id": shard_id,
        "output_file": shard_output,
        "num_questions": num_questions,
    }


def run_distributed(args):
    ray.init(address=args.ray_address, ignore_reinit_error=True)

    print(f"[Evaluator] Loading data from: {args.input_file}")
    data = load_input_data(args.input_file)
    if not data:
        print("[Evaluator] No data to process, exiting.")
        return

    questions = [item['question'] for item in data]
    image_indices = [item['image_idx'] for item in data]
    question_types = [item['question_type'] for item in data]

    shards = [
        {
            "questions": [questions[i] for i in idxs],
            "image_indices": [image_indices[i] for i in idxs],
            "question_types": [question_types[i] for i in idxs],
        }
        for idxs in np.array_split(np.arange(len(questions)), args.num_nodes)
        if len(idxs) > 0
    ]

    futures = [
        evaluate_shard.remote(
            args,
            shard_id,
            shard["questions"],
            shard["image_indices"],
            shard["question_types"],
        )
        for shard_id, shard in enumerate(shards)
    ]

    results = ray.get(futures)
    results = sorted(results, key=lambda x: x["shard_id"])

    total_questions = 0
    with open(args.output_file, "w") as out_f:
        for res in results:
            shard_file = res["output_file"]
            with open(shard_file, "r") as shard_f:
                for line in shard_f:
                    out_f.write(line)
            total_questions += res["num_questions"]

    for res in results:
        os.remove(res["output_file"])

    os.remove(args.input_file)
    print(f"[Evaluator] Processed {total_questions} questions. Saving results to: {args.output_file}")


def main(args):

    args.input_file = f"{STORAGE_PATH}/generated_question/{args.save_name}.json"
    args.output_file = f"{STORAGE_PATH}/generated_question/{args.save_name}_results.json"

    if os.path.exists(args.output_file) and os.path.getsize(args.output_file) > 0:
        print(f"[Evaluator] Results file {args.output_file} already exists, skipping...")
        return
    run_distributed(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Path to the model in Hugging Face format.")
    parser.add_argument("--rollout_n", type=int, default=9, help="Number of candidate answers to generate per question (n).")
    parser.add_argument("--save_name", type=str, required=True, help="A base name for input and output files.")
    parser.add_argument("--format_prompt", type=str, default="selfplay/solver/format_prompt.jinja", help="Path to format prompt.")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for processing.")
    parser.add_argument("--num_nodes", type=int, default=16, help="Number of Ray nodes to shard workload.")
    parser.add_argument("--ray_address", type=str, default="auto", help="Ray address, e.g., 'auto' to connect to an existing cluster.")
    parser.add_argument("--max_pixels", type=int, default=1003520, help="Maximum number of pixels for image processing.")
    parser.add_argument("--min_pixels", type=int, default=200704, help="Minimum number of pixels for image processing.")
    args = parser.parse_args()

    main(args)

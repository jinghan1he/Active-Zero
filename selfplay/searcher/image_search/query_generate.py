import vllm
import torch
from transformers import AutoTokenizer
import argparse
from tqdm import tqdm
from typing import List
from jinja2 import Template
from vllm.outputs import RequestOutput
import os
import json
import ray
import regex as re
from torch.utils.data import Dataset, DataLoader


STORAGE_PATH = os.getenv("STORAGE_PATH")


class QueryDataset(Dataset):
    def __init__(self, num_samples, format_prompt_path):
        self.num_samples = num_samples
        self.format_prompt = open(format_prompt_path, encoding="utf-8").read()
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        template = Template(self.format_prompt.strip())
        user_prompt = template.render().strip()
        
        # Create prompt in qwen2.5-VL format (text-only, no image)
        prompt = (
            f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        # if "<think>" not in user_prompt:
        #     prompt = prompt + "<think>\n</think>"
        return prompt


@ray.remote(num_gpus=1)
def generate_queries(args, shard_idx, total_shards):

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = vllm.LLM(
        model=args.model,
        tokenizer=args.model,
        seed=shard_idx,
        max_model_len=6144,
        dtype="bfloat16",
        max_num_batched_tokens=8192,
    )
    
    shard_size = args.num_samples // total_shards
    # Create dataset and dataloader
    print(f"[ImageSearch] Loading format prompt from {args.format_prompt}...")
    dataset = QueryDataset(shard_size, args.format_prompt)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=0,
        collate_fn=lambda x: x
    )
    
    sample_params = vllm.SamplingParams(
        max_tokens=2048,
        temperature=1.0,
        top_p=0.95,
        n=1,
        stop_token_ids=[tokenizer.eos_token_id],
    )
    
    # Process batches
    num_errors = 0
    num_queries = 0
    
    shard_output = f"{args.output_file}.shard{shard_idx}"
    os.makedirs(os.path.dirname(shard_output), exist_ok=True)
    with open(shard_output, "w") as f:
        for batch in dataloader:
            # Generate responses for the batch
            completions: List[RequestOutput] = model.generate(batch, sampling_params=sample_params)
            
            # Process each completion in the batch
            for completion in completions:
                response = completion.outputs[0].text
                # Extract image_type and query
                image_types = re.findall(r"<type>(.*?)</type>", response, re.DOTALL)
                queries = re.findall(r"<query>(.*?)</query>", response, re.DOTALL)

                if queries and image_types:
                    image_type = image_types[-1].strip()
                    query = queries[-1].strip()
                    f.write(json.dumps({
                        "image_type": image_type,
                        "query": query,
                    }) + "\n")
                    num_queries += 1
                elif queries:
                    query = queries[-1].strip()
                    f.write(json.dumps({
                        "image_type": "all",
                        "query": query,
                    }) + "\n")
                    num_queries += 1
                else:
                    num_errors += 1

    return {
        "shard_idx": shard_idx,
        "num_errors": num_errors,
        "num_queries": num_queries,
        "output_file": shard_output,
    }


def main(args):

    args.output_file = f"{STORAGE_PATH}/searched_image/{args.save_name}.json"
    args.indices_file = f"{STORAGE_PATH}/searched_image/{args.save_name}_indices.json"
    
    if os.path.exists(args.output_file) and os.path.getsize(args.output_file) > 0:
        print(f"[ImageSearch] Query file {args.output_file} already exists, skipping...")
        return
    elif os.path.exists(args.indices_file) and os.path.getsize(args.indices_file) > 0:
        print(f"[ImageSearch] Indices file {args.indices_file} already exists, skipping...")
        return

    ray.init(address=args.ray_address, ignore_reinit_error=True)
    futures = [generate_queries.remote(args, shard_idx, args.num_workers) for shard_idx in range(args.num_workers)]
    results = ray.get(futures)
    results = sorted(results, key=lambda x: x["shard_idx"])

    total_errors = sum(result["num_errors"] for result in results)
    total_queries = sum(result["num_queries"] for result in results)
    shard_files = [result["output_file"] for result in results]
    # Save results
    with open(args.output_file, "w") as f:
        for shard_file in shard_files:
            with open(shard_file, "r") as sf:
                for line in sf:
                    f.write(line)
    
    print(f"[ImageSearch] Generated {total_queries} queries and saved to {args.output_file} ({total_errors} errors)")

    for shard_file in shard_files:
        os.remove(shard_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct", help="Model name or path")
    parser.add_argument("--num_samples", type=int, default=16000, help="Number of samples to process")
    parser.add_argument("--format_prompt", type=str, default="selfplay/searcher/format_prompt.jinja", help="Path to format prompt")
    parser.add_argument("--save_name", type=str, default="query_generated", help="Base name for output file")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for processing")
    parser.add_argument("--ray_address", type=str, default="auto", help="Ray cluster address; use 'auto' for local")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of Ray workers")
    args = parser.parse_args()

    main(args)

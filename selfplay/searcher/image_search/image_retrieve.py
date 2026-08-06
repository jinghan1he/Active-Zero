#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Image Search
------------

聚合所有 query 文件，过滤无效 query，然后使用文到图检索器将 query 转化为检索到的图像 indices。
"""

import argparse
import os
import json
import ray
import numpy as np
import torch
import faiss
from transformers import AutoModel, AutoProcessor


STORAGE_PATH = os.getenv("STORAGE_PATH")
INDEX_PATH = os.getenv("INDEX_PATH")


def split_list(lst, n=8):
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]


@ray.remote(num_gpus=1)
def retrieve_images(args, 
    shard_id, 
    shard_queries, 
    shard_image_types,
):

    device = "cuda"
    print(f"[ImageSearch] Loading FAISS index from: {INDEX_PATH}")
    index = faiss.read_index(INDEX_PATH)
    print(f"[ImageSearch] Loading model: {args.model_name} on {device}...")
    model = AutoModel.from_pretrained(args.model_name).to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model_name)

    inputs = processor(text=shard_queries, padding="max_length", return_tensors="pt").to(device)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        text_features = text_features.cpu().numpy().astype("float32")

    scores, indices = index.search(text_features, args.top_k)
    results = [(scores[i], indices[i]) for i in range(len(shard_queries))]

    shard_output = f"{args.output_file}.shard{shard_id}"
    with open(shard_output, "w") as f:
        for query, image_type, result in zip(shard_queries, shard_image_types, results):
            f.write(json.dumps({
                "query": query,
                "image_type": image_type,
                "score": result[0].tolist(),
                "indices": result[1].tolist(),
            }) + "\n")
    
    print(f"[ImageSearch][Shard {shard_id}] Done. Saved {len(results)} search results to {shard_output}")
    return {
        "shard_id": shard_id,
        "output_file": shard_output,
        "num_results": len(results),
    }


def main(args):
    args.input_file = f"{STORAGE_PATH}/searched_image/{args.save_name}.json"
    args.output_file = f"{STORAGE_PATH}/searched_image/{args.save_name}_indices.json"
    args.indices_file = f"{STORAGE_PATH}/searched_image/{args.save_name}_indices.npy"

    if os.path.exists(args.indices_file) and os.path.getsize(args.indices_file) > 0:
        print(f"[ImageSearch] Indices file {args.indices_file} already exists, skipping...")
        return
    
    ray.init(address=args.ray_address, ignore_reinit_error=True)

    all_queries = [json.loads(line) for line in open(args.input_file)]
    print(f"[ImageSearch] Loaded {len(all_queries)} queries from query file: {args.input_file}")
    
    query_texts = [query["query"] for query in all_queries]
    image_types = [query["image_type"] for query in all_queries]
    
    shards = [
        {
            "query_texts": [query_texts[i] for i in idxs],
            "image_types": [image_types[i] for i in idxs],
        }
        for idxs in np.array_split(np.arange(len(query_texts)), args.num_workers)
        if len(idxs) > 0
    ]

    futures = [
        retrieve_images.remote(
            args, 
            shard_id, 
            shard["query_texts"], 
            shard["image_types"],
        ) for shard_id, shard in enumerate(shards)
    ]
    results = ray.get(futures)
    results = sorted(results, key=lambda x: x["shard_id"])
    
    total_results = 0
    all_indices = []    
    with open(args.output_file, "w") as f:
        for res in results:
            shard_file = res["output_file"]
            with open(shard_file, "r") as shard_f:
                for line in shard_f:
                    item = json.loads(line)
                    all_indices.extend(item["indices"])
                    f.write(line)
            total_results += res["num_results"]
    
    all_indices = list(set(all_indices))
    np.save(args.indices_file, np.array(all_indices))

    for res in results:
        os.remove(res["output_file"])
    
    os.remove(args.input_file)
    print(f"[ImageSearch] Processed {total_results} search results. Saved {len(all_indices)} unique indices to {args.indices_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_name", type=str, required=True, help="Base name for retrieved images (e.g., 'retrieved_images')")
    parser.add_argument("--top_k", type=int, default=1, help="Number of top results to retrieve for each query")
    parser.add_argument("--model_name", type=str, default="google/siglip2-so400m-patch16-naflex", help="Text encoder model name for retrieval.")
    parser.add_argument("--ray_address", type=str, default="auto", help="Ray cluster address; use 'auto' for local")
    parser.add_argument("--num_workers", type=int, default=16, help="Number of Ray workers")
    args = parser.parse_args()
    
    main(args)


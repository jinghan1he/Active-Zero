import torch
from transformers import AutoModel, AutoProcessor
from datasets import load_from_disk
from PIL import Image
import faiss
import numpy as np
from tqdm import tqdm
import ray


model_name = "google/siglip2-so400m-patch16-naflex"
data_path = "/path/to/data_cache/the_cauldron/data"
index_path = "tool/retriever/the_cauldron.index"
batch_size = 64

if not ray.is_initialized():
    ray.init(ignore_reinit_error=True)


@ray.remote(num_gpus=1)
def process_shard(shard_idx, num_shards, model_name, data_path, batch_size):
    """在单块数据上提取特征；每个 Ray 任务绑定 1 张 GPU。"""
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to("cuda").eval()

    dataset = load_from_disk(data_path).shard(
        num_shards=num_shards, index=shard_idx, contiguous=True
    )

    all_embeddings = []
    for i in tqdm(range(0, len(dataset), batch_size), disable=True):
        batch = dataset[i : i + batch_size]
        images = [Image.open(img_path[0]).convert("RGB") for img_path in batch["images"]]
        inputs = processor(images=images, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            feats = model.get_image_features(**inputs)
            feats = feats / feats.norm(p=2, dim=-1, keepdim=True)
            all_embeddings.append(feats.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def extract_features_distributed(
    model_name: str,
    data_path: str,
    batch_size: int = 64,
) -> np.ndarray:
    num_workers = int(ray.cluster_resources().get("GPU", 1))
    futures = [
        process_shard.remote(i, num_workers, model_name, data_path, batch_size)
        for i in range(num_workers)
    ]
    shard_embeddings = ray.get(futures)
    return np.concatenate(shard_embeddings, axis=0)


if __name__ == "__main__":
    print(f"Using model: {model_name}")
    print(f"Loading dataset shards from: {data_path}")

    print("Extracting features with Ray...")
    embeddings = extract_features_distributed(
        model_name=model_name, data_path=data_path, batch_size=batch_size
    )

    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype("float32"))

    faiss.write_index(index, index_path)
    print(f"Index built and saved to {index_path}")
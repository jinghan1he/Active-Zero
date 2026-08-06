#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Searcher Server
---------------

仿照 `start_searcher_server.py` 的结构，实现一个独立的检索服务：

1. 启动时加载 SigLIP 文本编码模型和 FAISS 索引（只加载一次，常驻内存）。
2. 提供 HTTP 接口 `/batch_retrieve`，通过查询参数 `name` 接收一个本地 JSON 任务文件路径。
3. 任务文件格式（由 `image_search/query_generate.py` 写入）：

   {
       "query_texts": ["q1", "q2", ...],
       "top_k": 10
   }

4. 服务器读取任务文件，进行批量检索，将结果写入 `<name>_results.json`，并返回简单 JSON 应答。

结果文件格式与 `image_search/image_search.py` 中的读取逻辑对应：

   [
       {
           "scores": [...],   # list[float]
           "indices": [...]   # list[int]
       },
       ...
   ]
"""

from flask import Flask, request, jsonify
import argparse
import random
import os
import json
import time
from typing import List, Tuple, Dict
from collections import defaultdict
import torch
import numpy as np
import faiss
from transformers import AutoModel, AutoProcessor
from sklearn.cluster import AgglomerativeClustering
from collections import Counter

INDEX_PATH = os.getenv("INDEX_PATH")

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=7000)
parser.add_argument(
    "--model_name",
    type=str,
    default="google/siglip2-so400m-patch16-naflex",
    help="Text encoder model name for retrieval.",
)
args = parser.parse_args()


class T2IRetriever:

    def __init__(self, index_path: str = None, model_name: str = "google/siglip2-so400m-patch16-naflex") -> None:
        print(f"[Retriever {args.port}] Loading FAISS index from: {index_path}")
        self.index = faiss.read_index(index_path)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Retriever {args.port}] Loading model: {model_name} on {self.device}...")
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)
        print(f"[Retriever {args.port}] Initialization complete.")

    def _compute_image_embedding_distance_matrix(self, image_indices: np.ndarray) -> np.ndarray:
        """
        从 FAISS 索引中获取图像嵌入向量，并计算距离矩阵。
        
        Args:
            image_indices: 图像索引数组
            
        Returns:
            距离矩阵 (n_samples, n_samples)，使用余弦距离
        """
        unique_indices = np.unique(image_indices)
        
        # 从 FAISS 索引中获取图像嵌入向量
        # 使用 reconstruct_batch 批量获取向量（如果支持），否则逐个获取
        # 批量获取唯一索引的向量
        unique_embeddings = self.index.reconstruct_batch(unique_indices.astype(np.int64))
        # 创建索引映射
        index_map = {int(idx): i for i, idx in enumerate(unique_indices)}
        # 根据原始顺序重新排列
        selected_embeddings = unique_embeddings[[index_map[int(idx)] for idx in image_indices]]
        
        # 归一化嵌入向量（确保是单位向量）
        norms = np.linalg.norm(selected_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # 避免除零
        selected_embeddings = selected_embeddings / norms
        
        # 计算余弦相似度矩阵
        similarity_matrix = np.dot(selected_embeddings, selected_embeddings.T)
        
        # 转换为距离矩阵（1 - 相似度）
        dist_matrix = 1 - similarity_matrix
        
        # 确保对称性和对角线为0
        dist_matrix = (dist_matrix + dist_matrix.T) / 2
        np.fill_diagonal(dist_matrix, 0)
        
        return np.clip(dist_matrix, 0, 1)

    def _compute_diversity_penalty(self, image_indices: np.ndarray, distance_threshold: float = 0.5, linkage: str = "average") -> np.ndarray:
        """
        基于图像嵌入计算多样性惩罚，模仿 reward_func.py 中的 cluster_share_per_problem 逻辑。
        
        Args:
            image_indices: 检索到的图像索引数组
            distance_threshold: 聚类距离阈值
            linkage: 聚类链接方式
            
        Returns:
            每个图像在其所属聚类中的共享比例（作为惩罚）
        """
        if len(image_indices) == 0:
            return np.array([])

        if len(image_indices) == 1:
            return np.array([0.0])
        
        start_time = time.time()

        # 计算距离矩阵
        dist_matrix = self._compute_image_embedding_distance_matrix(image_indices)
        
        # 使用层次聚类
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,
            metric="precomputed",
            linkage=linkage
        )
        labels = clustering.fit_predict(dist_matrix)
        
        # 计算每个聚类的大小和比例
        total = len(image_indices)
        cluster_size = Counter(labels)
        cluster_ratio = {lab: sz / total for lab, sz in cluster_size.items()}
        
        # 返回每个图像在其所属聚类中的共享比例
        proportions = np.array([cluster_ratio[lab] for lab in labels])

        print(f"end clustering, time: {time.time() - start_time}, num samples: {len(image_indices)}")
        
        return proportions

    def batch_retrieve(self, query_texts: List[str], type_to_indices: Dict[str, List[int]]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch retrieve, return the indices of the top-k results for each query and diversity penalty.
        
        Returns:
            Tuple of (indices, penalty_image)
        """
        # Batch encode
        inputs = self.processor(text=query_texts, padding="max_length", return_tensors="pt").to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            text_features = text_features.cpu().numpy().astype("float32")

        # FAISS retrieve
        scores, indices = self.index.search(text_features, 5)

        results = np.array([indices[i][random.randint(0, 4)] for i in range(len(query_texts))])
        
        # Compute diversity penalty by type
        penalties_by_type = {}
        for qtype, indices in type_to_indices.items():
            results_this_type = results[indices]
            print(f"start clustering for type: {qtype}, num samples: {len(results_this_type)}")
            penalties = self._compute_diversity_penalty(results_this_type, distance_threshold=0.1)
            for ii, orig_idx in enumerate(indices):
                penalties_by_type[orig_idx] = penalties[ii]

        # 将多样性惩罚按原始顺序排列
        penalty_image = []
        for i in range(len(query_texts)):
            penalty_image.append(penalties_by_type[i])

        return results, penalty_image


# Singleton retriever
_retriever: T2IRetriever = T2IRetriever(
    index_path=INDEX_PATH,
    model_name=args.model_name,
)


app = Flask(__name__)


@app.route("/batch_retrieve", methods=["GET"])
def batch_retrieve_endpoint():
    """
    读取 `name` 指定的 JSON 任务文件，进行批量检索并将结果写回 `_results.json`。
    """
    name = request.args.get("name")

    print(f"[Retriever {args.port}] Received batch_retrieve request for task file: {name}")

    if not os.path.exists(name):
        return jsonify({"error": f"Task file not found: {name}"}), 404

    # Read task file
    with open(name, "r") as f:
        queries = json.load(f)

    if not isinstance(queries, list):
        return jsonify({"error": "Field 'queries' must be a list."}), 400

    print(f"[Retriever {args.port}] Processing {len(queries)} queries")

    query_texts = []
    type_to_indices = defaultdict(list)
    for i, query in enumerate(queries):
        query_texts.append(query["query"])
        type_to_indices[query["type"]].append(i)

    # Call retrieve
    retriever_results, penalty_image = _retriever.batch_retrieve(query_texts, type_to_indices)

    out_path = name.replace(".json", ".npy")
    penalty_path = name.replace(".json", "_penalty.npy")
    
    # Save retrieval results and diversity penalty
    np.save(out_path, retriever_results)
    np.save(penalty_path, penalty_image)
    
    os.remove(name)

    print(f"[Retriever {args.port}] Finished {name}, results saved to {out_path}, penalty saved to {penalty_path}")
    return jsonify({"message": f"Processed {name}, results saved to {out_path}, penalty saved to {penalty_path}."})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=args.port, threaded=True)

# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''
This reward function implements text-to-image retrieval reward.
It extracts query_text from model output, searches in image database,
and computes reward based on whether correct image is in top-k results.
'''

import os
import re
import torch
import numpy as np
import faiss
import json
import time
import random
import requests
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any
from transformers import AutoModel, AutoProcessor
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer

STORAGE_PATH = os.getenv("STORAGE_PATH")
VLLM_HOST = os.getenv("VLLM_HOST", "127.0.0.1")
os.environ["NO_PROXY"] = "0.0.0.0,127.0.0.1"


def extract_query_text(predict: str) -> Optional[str]:
    """
    Extract query text from model prediction.
    The model is required to enclose output with <query></query> tags.
    
    Returns:
        The extracted query text, or None if no valid query is found.
    """
    # Primary: extract from <query> tag (required format)
    match = re.search(r"<query>(.*?)</query>", predict, re.DOTALL)
    if match:
        query = match.group(1).strip()
        if query:  # Ensure non-empty query
            return query
    
    # No valid <query> tag found
    return ""


def extract_query_type(predict: str) -> Optional[str]:
    match = re.search(r"<type>(.*?)</type>", predict, re.DOTALL)
    if match:
        query_type = match.group(1).strip()
        if query_type:  # Ensure non-empty query type
            return query_type
    return ""


def format_reward(response: str) -> float:
    pattern = re.compile(
        r"<type>.*?</type>\s*"
        r"<query>.*?</query>", 
        re.DOTALL
    )
    
    # Use search or fullmatch depending on how strict you want to be 
    # about leading/trailing whitespace or text.
    format_match = re.search(pattern, response)
    
    return 1.0 if format_match else 0.0


def is_effectively_empty(sentences):
    if not sentences:
        return True

    token_pattern = re.compile(r"(?u)\b\w\w+\b")
    
    for s in sentences:
        if s and token_pattern.search(str(s)):
            return False
            
    return True


def _vectorized_bleu_distance_matrix(sentences, n_gram=4):
    """
    使用矩阵运算模拟 BLEU 距离计算
    """
    n_samples = len(sentences)
    # 1. 预计算每个句子的分词长度 (用于 Brevity Penalty)
    tokenized_sentences = [s.split() for s in sentences]
    lengths = np.array([len(s) for s in tokenized_sentences])
    
    # 存储每个 n-gram 阶数的精确度矩阵
    log_precisions = np.zeros((n_samples, n_samples))
    
    for n in range(1, n_gram + 1):
        # 2. 提取 n-gram 频次矩阵 (稀疏矩阵)
        # binary=True 表示只看是否存在，简化计算；若追求极致精确可设为 False 并手动算 min
        cv = CountVectorizer(ngram_range=(n, n), analyzer='word', tokenizer=lambda x: x.split(), lowercase=False, token_pattern=None)
        ngram_matrix = cv.fit_transform(sentences)
        
        # 3. 矩阵点积：计算句子 i 和句子 j 之间共同拥有的 n-gram 数量
        # intersection[i, j] 表示第 i 个句子（候选）在第 j 个句子（参考）中找到的匹配数
        intersection = (ngram_matrix @ ngram_matrix.T).toarray()
        
        # 4. 计算 Precision: 匹配数 / 候选句子的 n-gram 总数
        # 候选句子 i 的 n-gram 总数即为 ngram_matrix 每一行的和
        row_sums = ngram_matrix.sum(axis=1).A1
        row_sums[row_sums == 0] = 1  # 防止除以 0
        
        # 得到 p_n 矩阵
        p_n = intersection / row_sums[:, np.newaxis]
        
        # 5. 平滑处理 (类似 SmoothingFunction().method1)
        p_n = np.maximum(p_n, 1e-9)
        log_precisions += (1.0 / n_gram) * np.log(p_n)

    # 6. 计算最终分数
    scores = np.exp(log_precisions)
    
    # 7. 计算简短惩罚 Brevity Penalty (BP)
    # BP = exp(1 - r/c) if c < r else 1
    # 这里 r 是参考句(j)长度，c 是候选句(i)长度
    r = lengths[np.newaxis, :] # 参考句长度矩阵 (1, n)
    c = lengths[:, np.newaxis] # 候选句长度矩阵 (n, 1)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        bp = np.exp(1 - r / c)
    bp[c >= r] = 1
    
    bleu_matrix = scores * bp
    
    # 8. 转换成距离并强制对称 (由于原有逻辑是 dist[i,j]=dist[j,i])
    dist_mat = 1 - bleu_matrix
    dist_mat = (dist_mat + dist_mat.T) / 2
    np.fill_diagonal(dist_mat, 0) # 对角线距离为 0
    
    return np.clip(dist_mat, 0, 1)


def cluster_share_per_problem(
        problems,
        distance_threshold: float = 0.5,
        linkage: str = "average"):
    if not problems:
        return []
    if len(problems) == 1:
        return [0.0]
    if is_effectively_empty(problems):
        return [0.0] * len(problems)
    start_time = time.time()
    # dist_mat = _bleu_distance_matrix(problems)
    dist_mat = _vectorized_bleu_distance_matrix(problems)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="precomputed",
        linkage=linkage
    )
    labels = clustering.fit_predict(dist_mat)
    print(f'end clustering, time: {time.time() - start_time}, num samples: {len(problems)}')
    total = len(problems)
    cluster_size = Counter(labels)
    cluster_ratio = {lab: sz / total for lab, sz in cluster_size.items()}

    proportions = [cluster_ratio[lab] for lab in labels]
    return proportions


def generate_temp_filename(prefix="temp", suffix=".json"):
    timestamp = int(time.time() * 1000) 
    rand_part = random.randint(0, 99999)
    return f"{STORAGE_PATH}/temp_results/{prefix}_{timestamp}_{rand_part}{suffix}"

def split_list(lst, n=4):
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]

def fetch_retriever(index, filename):
    """Send HTTP request to retriever service with filename."""
    response = requests.get(f"http://{VLLM_HOST}:{7000+index}/batch_retrieve?name={filename}")
    return True

def fetch_questioner(index, filename):
    """Send HTTP request to questioner service with filename."""
    response = requests.get(f"http://{VLLM_HOST}:{5000+index}/hello?name={filename}")
    return True

def fetch_solver(index, filename):
    """Send HTTP request to solver service with filename."""
    response = requests.get(f"http://{VLLM_HOST}:{6000+index}/hello?name={filename}")
    return True

def batch_retrieve_via_file(query_texts, num_workers=8):
    query_batches = split_list(query_texts, num_workers)
    temp_filenames = [generate_temp_filename(prefix=f"temp_{i}", suffix=".json") for i in range(num_workers)]
    for i, batch in enumerate(query_batches):
        with open(temp_filenames[i], 'w') as f:
            json.dump(batch, f, indent=4)
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(fetch_retriever, i, temp_filenames[i]) for i in range(num_workers)]
        
        for future in as_completed(futures):
            print(f"[RewardFunc] Retriever request completed: {future.result()}")

    image_filenames = [temp_filenames[i].replace('.json', '.npy') for i in range(num_workers)]
    return image_filenames


def compute_score(reward_inputs: list[dict[str, Any]], format_weight: float = 0.1, num_workers: int = 8) -> list[dict[str, float]]:
    """
    Compute reward scores based on the complete pipeline:
    1. Extract queries -> retrieve images
    2. Send image indices to questioner -> generate questions
    3. Send questions to solver -> get answers
    4. Compute reward based on solver results and clustering
    
    Args:
        reward_inputs: List of reward inputs containing response and ground truth
        format_weight: Weight for format reward
        num_workers: Number of parallel workers for server requests
    Returns:
        List of score dictionaries with 'overall', 'format', 'accuracy' keys
    """
    # Step 1: Extract query texts from predictions
    query_texts, valid_indices, valid_queries, format_scores = [], [], [], []
    type_to_indices = defaultdict(list)
    for i, reward_input in enumerate(reward_inputs):
        predict = reward_input["response"]
        predict = re.sub(r"\s*(<|>|/)\s*", r"\1", predict)
        query = extract_query_text(predict)
        query_type = extract_query_type(predict)
        query_texts.append(query)
        type_to_indices[query_type].append(i)
        format_scores.append(format_reward(predict))
        if query:
            valid_indices.append(i)
            valid_queries.append({"query": query, "type": query_type})

    penalties_by_type = {}
    for qtype, indices in type_to_indices.items():
        texts_this_type = [query_texts[i] for i in indices]
        print(f'start clustering for type: {qtype}, num samples: {len(texts_this_type)}')
        penalties = cluster_share_per_problem(texts_this_type, distance_threshold=0.5)
        # Map back from local group indices to original indices
        for ii, orig_idx in enumerate(indices):
            penalties_by_type[orig_idx] = penalties[ii]

    # 3. Assemble the penalty_text array in original order
    penalty_text = []
    for i in range(len(query_texts)):
        penalty_text.append(penalties_by_type[i])

    if not valid_queries:
        # No valid queries, return zero scores
        return [{
            "overall": 0.0,
            "format": 0.0,
            "accuray": 0.0,
        } for _ in reward_inputs]
    
    # Step 2: Retrieve images via retrieve_server
    image_filenames = batch_retrieve_via_file(valid_queries, num_workers=num_workers)

    penalty_filenames = [image_filenames[i].replace('.npy', '_penalty.npy') for i in range(num_workers)]
    penalty_image = []
    for i in range(num_workers):
        penalty_image.extend(np.load(penalty_filenames[i]).tolist())
        os.remove(penalty_filenames[i])
    
    # Send requests to questioner_server in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(fetch_questioner, i, image_filenames[i]) for i in range(num_workers)]

        for future in as_completed(futures):
            print(f"[RewardFunc] Questioner request completed: {future.result()}")
    
    question_filenames = [image_filenames[i].replace('.npy', '.json') for i in range(num_workers)]
    
    # Send requests to solver_server in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(fetch_solver, i, question_filenames[i]) for i in range(num_workers)]
        
        for future in as_completed(futures):
            print(f"[RewardFunc] Solver request completed: {future.result()}")
    
    result_filenames = [question_filenames[i].replace('.json', '_results.json') for i in range(num_workers)]

    # Collect solver results
    results = []
    for i in range(num_workers):
        with open(result_filenames[i], 'r') as f:
            results.extend(json.load(f))
        os.remove(result_filenames[i])
    
    assert len(penalty_image) == len(results) == len(valid_indices), f"penalty: {len(penalty_image)}, results: {len(results)}, valid_indices: {len(valid_indices)}"
    final_results = []
    for i in range(len(query_texts)):
        if i in valid_indices:
            valid_pos = valid_indices.index(i)
            results[valid_pos]['penalty'] = penalty_image[valid_pos]
            final_results.append(results[valid_pos])
        else:
            final_results.append({
                'question': 'None',
                'answer':   'None',
                'score':    -1,
                'penalty':  0,
                'results':  []}
            )
    
    # Step 5: Compute reward based on solver results and clustering
    assert len(final_results) == len(query_texts) == len(format_scores), f"final_results: {len(final_results)}, query_texts: {len(query_texts)}, format_scores: {len(format_scores)}"
    scores = []
    for i in range(len(final_results)):
        if query_texts[i]:
            score = (min(final_results[i]["score"],1-final_results[i]["score"]))
        else:
            score = -1
        final_score = (1 - format_weight) * score + format_weight * format_scores[i] - final_results[i]['penalty'] - penalty_text[i]
        scores.append({"overall": final_score, "format": format_scores[i], "diversity_penalty": final_results[i]['penalty'] + penalty_text[i]})
    
    return scores

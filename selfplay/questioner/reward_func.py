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
This reward function is for regular [CoT] -> [Answer] GRPO finetuning
'''
import re, os, json
import time
import random
import requests
import numpy as np
from typing import Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import CountVectorizer


STORAGE_PATH = os.getenv("STORAGE_PATH")
VLLM_HOST = os.getenv("VLLM_HOST", "127.0.0.1")
os.environ["NO_PROXY"] = "0.0.0.0,127.0.0.1"


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


def _bleu_distance_matrix(sentences):
    n = len(sentences)
    dist = np.zeros((n, n))
    smoother = SmoothingFunction().method1
    tokenized_sentences = [s.split() for s in sentences]
    for i in range(n):
        for j in range(i, n):
            if i == j:
                score = 1.0
            else:
                ref = tokenized_sentences[j]
                hyp = tokenized_sentences[i]
                score = sentence_bleu(ref, hyp, smoothing_function=smoother)
            dist[i, j] = dist[j, i] = 1 - score
    return dist


def cluster_share_per_problem(
        problems,
        distance_threshold: float = 0.5,
        linkage: str = "average"):
    if not problems:
        return []
    print('start clustering')
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
    print(f'end clustering, time: {time.time() - start_time}')
    total = len(problems)
    cluster_size = Counter(labels)
    cluster_ratio = {lab: sz / total for lab, sz in cluster_size.items()}

    proportions = [cluster_ratio[lab] for lab in labels]
    return proportions


def format_reward(response: str) -> float:
    """
    Rewards responses that strictly follow the generated question format:
    <think>...</think>
    <type>...</type>
    <question>...</question>
    <answer>...</answer>
    """
    # The pattern checks for the sequence of the four required tags.
    # It uses re.DOTALL to allow for newlines between and within tags.
    pattern = re.compile(
        r"<think>.*?</think>\s*"
        r"<type>.*?</type>\s*"
        r"<question>.*?</question>\s*"
        r"<answer>.*?</answer>", 
        re.DOTALL
    )
    
    # Use search or fullmatch depending on how strict you want to be 
    # about leading/trailing whitespace or text.
    format_match = re.search(pattern, response)
    
    return 1.0 if format_match else 0.0


def match(generation):
    pattern = r"<type>(.*?)</type>.*?<question>(.*?)</question>.*?<answer>(.*?)</answer>"
    match_obj = re.search(pattern, generation, re.DOTALL)

    if match_obj:
        return {
            "question": match_obj.group(2).strip(),
            "answer": match_obj.group(3).strip(),
            "question_type": match_obj.group(1).strip()
        }
    return None


def generate_temp_filename(prefix="temp", suffix=".json"):
    timestamp = int(time.time() * 1000) 
    rand_part = random.randint(0, 99999)
    return f"{STORAGE_PATH}/temp_results/{prefix}_{timestamp}_{rand_part}{suffix}"


def split_list(lst, n=8):
    k, m = divmod(len(lst), n)
    return [lst[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n)]


def fetch(index,i):
    response = requests.get(f"http://{VLLM_HOST}:{6000+index}/hello?name={i}")
    return True


def generate_results(data, num_workers=8):
    datas = split_list(data, num_workers)
    random_names = [generate_temp_filename(prefix=f"temp_{i}", suffix=".json") for i in range(num_workers)]
    for i in range(num_workers):
        with open(random_names[i],'w') as f:
            json.dump(datas[i],f,indent=4)

    final_results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(fetch, i,random_names[i]) for i in range(num_workers)]

        for future in as_completed(futures):
            print(f"[RewardFunc] Solver request completed: {future.result()}")

    for i in range(num_workers):
        with open(random_names[i].replace('.json','_results.json'),'r') as f:
            final_results.extend(json.load(f))
    for i in range(num_workers):
        os.remove(random_names[i].replace('.json','_results.json'))
    return final_results


def compute_score(reward_inputs: list[dict[str, Any]], format_weight: float = 0.1, save_path: str = "") -> list[dict[str, float]]:
    results = []
    format_scores = []
    for reward_input in reward_inputs:
        predict = reward_input["response"]
        image_idx = reward_input["ground_truth"]
        predict = re.sub(r"\s*(<|>|/)\s*", r"\1", predict)  # handle qwen2.5vl-32b format
        dirty_results = match(predict)
        if dirty_results == None:
            item = {"question": "", "answer": "", "question_type": ""}
        else:
            item = dirty_results
        item["image_idx"] = image_idx
        results.append(item)
        format_scores.append(format_reward(predict))
    final_results = generate_results(results, num_workers=8)
    penalty = cluster_share_per_problem([result['question'] for result in final_results], distance_threshold=0.5)
    assert len(penalty) == len(final_results)
    scores = []
    for i, (final_result, format_score) in enumerate(zip(final_results, format_scores)):
        if final_result['question']:
            score = min(final_result["score"],1-final_result["score"])
        else:
            score = -1
        final_score = (1 - format_weight) * score + format_weight * format_score - penalty[i]
        scores.append({"overall": final_score, "format": format_score, "diversity_penalty": penalty[i]})

    if save_path:
        mode = "w" if not os.path.exists(save_path) else "a"
        with open(save_path, mode) as f:
            for final_result in final_results:
                f.write(json.dumps(final_result) + "\n")
                f.flush()
    return scores

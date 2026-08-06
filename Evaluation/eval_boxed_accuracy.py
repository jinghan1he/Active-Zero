#!/usr/bin/env python3
"""
Evaluation script: Extract content within \\boxed{} from model output responses, and use LLM Judge to compare with ground truth answers to calculate accuracy
"""

import os
import json
import re
from typing import Dict, List, Tuple
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import argparse
from datasets import load_dataset, load_from_disk
from dotenv import load_dotenv
from tqdm import tqdm
from mathruler.grader import extract_boxed_content, grade_answer

load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='Qwen-VL-7B-solver_v3', help='Specify the model name to load')
parser.add_argument('--use_llm_judge', action='store_true', default=False, help='Specify whether to use LLM judge')
args = parser.parse_args()

JUDGE_MODEL = "deepseek-chat"

def extract_answer(text: str) -> str:
    if not text:
        return ""

    answer = extract_boxed_content(text)
    
    if answer != "None":
        return answer

    if "</think>" in text:
        return text.split("</think>")[1].strip()[:100]
    
    # If no boxed format found, return the first 100 characters of the response (stripped)
    # This handles cases where the answer is directly output (e.g., "B", "C", etc.)
    return text.strip()[:100]


@lru_cache(maxsize=10000)
def judge_answer_with_llm(predicted_answer: str, ground_truth_answer: str, question: str = "", predicted_response: str = "") -> bool:

    client = OpenAI(
            api_key=os.environ.get('DS_API_KEY'),
            base_url="https://api.deepseek.com"
        )
    
    user_content = f"""Given the question {question}, please judge whether the following two answers express the same meaning. Please only answer "correct" or "incorrect". Correct answer: {ground_truth_answer}. Answer to be judged: {predicted_answer}. Judgment result (only answer "correct" or "incorrect"). You don't need to reason, just answer "correct" or "incorrect". Don't say anything else. Only answer "correct" or "incorrect" directly without thinking.""" 
    response = client.chat.completions.create( 
        model=JUDGE_MODEL, 
        messages=[ 
            {"role": "system", "content": "You are an answer evaluation assistant. Your task is to judge whether two answers are substantially equivalent. When evaluating, you should ignore superficial differences such as format, spaces, punctuation, case, etc., and focus on whether they are consistent in core content, logical meaning and information expression. The judgment criteria should be lenient and inclusive, as long as the expressed meaning is basically the same, it is considered equivalent."}, 
            {"role": "user", "content": user_content} 
            ], 
            temperature=0.6, 
        )
    
    result = response.choices[0].message.content.strip().lower()
    if result not in ['correct', 'incorrect']:
        result = extract_answer(result).strip().lower()

    if result == 'correct':
        return True
    elif result == 'incorrect':
        return False
    else:
        print(f"Invalid result: {result}")
        return False


def load_ground_truth(truth_dataset: str) -> Dict[int, str]:
    """
    使用 datasets 库加载指定数据集名称（或本地数据集路径），并提取标准答案。
    默认从 test 切分中读取，样本顺序与索引保持一致，便于通过 dataset_index 对齐。
    """
    if os.path.isdir(truth_dataset):
        ds = load_from_disk(truth_dataset)
    else:
        truth_dataset, split_name = truth_dataset.split("@")
        ds = load_dataset(truth_dataset, split=split_name)
    
    ground_truth = {idx: str(row["answer"]).strip() for idx, row in enumerate(ds)}

    return ground_truth

def load_judge_results(judge_results_file: str) -> Dict[int, bool]:
    correct = 0
    errors = []
    samples = []
    with open(judge_results_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            samples.append(data)
            if data['judge_correct']:
                correct += 1
            else:
                errors.append(data)
    return correct, len(samples), errors, samples

def evaluate_dataset(predictions_file: str, truth_dataset: str, use_llm_judge: bool = True, max_workers: int = 32) -> Tuple[int, int, List[Dict], List[Dict]]:
    # 使用 datasets.load_dataset 从指定 truth_dataset 加载标准答案
    ground_truth = load_ground_truth(truth_dataset)
    
    # Read all prediction results
    samples = []
    with open(predictions_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            dataset_index = data.get('dataset_index', -1)
            if dataset_index in ground_truth:
                response = data.get('response', '')
                question = data.get('prompt', '')
                question = re.search(r'user\n\s*(.*?)\s*\nassistant', question, re.DOTALL).group(1).strip()   # question contains the prompt and the answer
                predicted = extract_answer(response)
                samples.append({
                    'dataset_index': dataset_index,
                    'predicted': predicted,
                    'true_answer': ground_truth[dataset_index],
                    'question': question,
                    'response': response
                })
    
    total = len(samples)
    correct = 0
    errors = []
    
    if not use_llm_judge:
        # Traditional method: normalize answers for comparison
        for sample in tqdm(samples, desc="Evaluating dataset"):
            if grade_answer(sample['predicted'], sample['true_answer']):
                correct += 1
            else:
                errors.append({
                    'index': sample['dataset_index'],
                    'predicted': sample['predicted'],
                    'true_answer': sample['true_answer'],
                    'question': sample['question'][:200] + '...' if len(sample['question']) > 200 else sample['question'],
                    'response': sample['response'][:200] + '...' if len(sample['response']) > 200 else sample['response']
                })
    else:
        # Use multi-threaded concurrent LLM Judge calls
        def judge_sample(sample):
            if not sample['predicted']:
                return sample, False
            is_correct = judge_answer_with_llm(sample['predicted'], sample['true_answer'], sample['question'], sample['response'])
            return sample, is_correct
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(judge_sample, sample): sample for sample in samples}
            for future in as_completed(futures):
                sample, is_correct = future.result()
                sample['judge_correct'] = is_correct
                if is_correct:
                    correct += 1
                else:
                    errors.append({
                        'index': sample['dataset_index'],
                        'predicted': sample['predicted'],
                        'true_answer': sample['true_answer'],
                        'question': sample['question'][:200] + '...' if len(sample['question']) > 200 else sample['question'],
                        'response': sample['response'][:200] + '...' if len(sample['response']) > 200 else sample['response']
                    })
    
    return correct, total, errors, samples


def main():
    datasets = {
        'DynaMath': (
            f'Evaluation/Raw-Outputs/{args.model}/DynaMath.jsonl',
            'Evaluation/Datasets/DynaMath',
        ),
        'HallusionBench': (
            f'Evaluation/Raw-Outputs/{args.model}/HallusionBench.jsonl',
            'Evaluation/Datasets/HallusionBench',
        ),
        'LogicVista': (
            f'Evaluation/Raw-Outputs/{args.model}/LogicVista.jsonl',
            'Evaluation/Datasets/LogicVista',
        ),
        'MathVerse': (
            f'Evaluation/Raw-Outputs/{args.model}/MathVerse.jsonl',
            'Evaluation/Datasets/MathVerse',
        ),
        'MathVerse-VD': (
            f'Evaluation/Raw-Outputs/{args.model}/MathVerse-VD.jsonl',
            'Evaluation/Datasets/MathVerse-VD',
        ),
        'MathVision': (
            f'Evaluation/Raw-Outputs/{args.model}/MathVision.jsonl',
            'Evaluation/Datasets/MathVision',
        ),
        'MathVista': (
            f'Evaluation/Raw-Outputs/{args.model}/MathVista.jsonl',
            'Evaluation/Datasets/MathVista',
        ),
        'MMMU_Pro': (
            f'Evaluation/Raw-Outputs/{args.model}/MMMU_Pro.jsonl',
            'Evaluation/Datasets/MMMU_Pro',
        ),
        'MMMU_Pro_10options': (
            f'Evaluation/Raw-Outputs/{args.model}/MMMU_Pro_10options.jsonl',
            'Evaluation/Datasets/MMMU_Pro_10options',
        ),
        'MMMU_Pro_4options': (
            f'Evaluation/Raw-Outputs/{args.model}/MMMU_Pro_4options.jsonl',
            'Evaluation/Datasets/MMMU_Pro_4options',
        ),
        'MMStar': (
            f'Evaluation/Raw-Outputs/{args.model}/MMStar.jsonl',
            'Evaluation/Datasets/MMStar',
        ),
        'RealWorldQA': (
            f'Evaluation/Raw-Outputs/{args.model}/RealWorldQA.jsonl',
            'Evaluation/Datasets/RealWorldQA',
        ),
        'VisNumBench': (
            f'Evaluation/Raw-Outputs/{args.model}/VisNumBench.jsonl',
            'Evaluation/Datasets/VisNumBench',   
        ),
        'We-Math': (
            f'Evaluation/Raw-Outputs/{args.model}/We-Math.jsonl',
            'Evaluation/Datasets/We-Math',
        ),
        'M3CoT': (
            f'Evaluation/Raw-Outputs/{args.model}/M3CoT.jsonl',
            'Evaluation/Datasets/M3CoT',
        ),
        'mm-vet': (
            f'Evaluation/Raw-Outputs/{args.model}/mm-vet.jsonl',
            'Evaluation/Datasets/mm-vet',
        ),
        'VPPO_MMK12_validation': (
            f'Evaluation/Raw-Outputs/{args.model}/VPPO_MMK12_validation.jsonl',
            'chamber111/VPPO_MMK12_validation@train',
        ),
        'geometry3k': (
            f'Evaluation/Raw-Outputs/{args.model}/geometry3k.jsonl',
            'hiyouga/geometry3k@test',
        ),
        'MMMU': (
            f'Evaluation/Raw-Outputs/{args.model}/MMMU.jsonl',
            'zli12321/MMMU@test',
        ),
    }
    
    # Collect statistics for all datasets
    all_results = {}
    total_correct = 0
    total_samples = 0
    
    if args.use_llm_judge:
        output_file = f'Evaluation/Results/{args.model}_llm_judge.txt'
    else:
        output_file = f'Evaluation/Results/{args.model}_exact_match.txt'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if not os.path.exists(output_file):
        open(output_file, 'w', encoding='utf-8').close()
    
    print("=" * 80)
    print("Starting evaluation of accuracy for each dataset...")
    if args.use_llm_judge:
        print(f"Using Judge Model: {JUDGE_MODEL}")
    else:
        print("Using Exact Match")
    print(f"Results will be saved to: {output_file}")
    print("=" * 80)
    print()
    
    for dataset_name, (pred_file, truth_dataset) in datasets.items():
        if dataset_name == 'VPPO_MMK12_validation' and args.use_llm_judge:
            continue
        if not os.path.exists(pred_file):
            continue
        print(f"Evaluating: {dataset_name}")
        print(f"  Prediction file: {pred_file}")
        print(f"  Ground truth dataset: {truth_dataset}")
        if args.use_llm_judge:
            output_dir = f'Evaluation/Judge-Outputs/{args.model}'
            os.makedirs(output_dir, exist_ok=True)
            output_pred_file = os.path.join(output_dir, os.path.basename(pred_file))
        try:
            if args.use_llm_judge and os.path.exists(output_pred_file):
                correct, total, errors, samples = load_judge_results(output_pred_file)
            else:
                correct, total, errors, samples = evaluate_dataset(pred_file, truth_dataset, use_llm_judge=args.use_llm_judge)
            accuracy = correct / total * 100 if total > 0 else 0
            
            all_results[dataset_name] = {
                'correct': correct,
                'total': total,
                'accuracy': accuracy,
                'errors': errors
            }
            
            total_correct += correct
            total_samples += total
            
            print(f"  ✓ Correct: {correct}/{total}")
            print(f"  ✓ Accuracy: {accuracy:.2f}%")
            
            # Append results to file immediately
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(f"{dataset_name}: {correct}/{total} = {accuracy:.2f}%\n")
            
            # Save judge results to file
            if args.use_llm_judge and not os.path.exists(output_pred_file):
                with open(output_pred_file, 'w', encoding='utf-8') as f:
                    for sample in samples:
                        f.write(json.dumps(sample, ensure_ascii=False) + '\n')
                
                print(f"  ✓ Judge results saved to: {output_pred_file}")
            
            print()
            
        except Exception as e:
            print(f"  ✗ Evaluation error: {str(e)}")
            print()
            continue
    
    # Print overall statistics
    print("=" * 80)
    print("Overall Statistics")
    print("=" * 80)
    print()
    
    for dataset_name, result in all_results.items():
        print(f"{dataset_name:20s}: {result['correct']:4d}/{result['total']:4d} = {result['accuracy']:6.2f}%")
    
    print("-" * 80)
    overall_accuracy = total_correct / total_samples * 100 if total_samples > 0 else 0
    print(f"{'Overall':20s}: {total_correct:4d}/{total_samples:4d} = {overall_accuracy:6.2f}%")
    print()
    
    # Append overall results to file
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(f"Overall: {total_correct}/{total_samples} = {overall_accuracy:.2f}%\n")
    
    print(f"All results saved to: {output_file}")
    print("=" * 80)
    print("Evaluation completed!")
    print("=" * 80)


if __name__ == '__main__':
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Refactored Version: This script employs the 'stopit' library to apply fine-grained, thread-safe
timeout control directly to the `grade_answer` function. This approach is more robust than a
global timeout and avoids the 'signal only works in main thread' error common in multi-threaded
Flask applications. The comparison logic is optimized to perform cheap checks first.

Setup Instructions:
    # 1. Install the required library (note the change from previous versions)
    pip install stopit

    # 2. Run the server
    python your_server_file_name.py --port 5000 --model_path Qwen/Qwen3-4B-Base
'''

from flask import Flask, request, jsonify
import vllm
import argparse
import json
import os
import gc
import time
from PIL import Image
from io import BytesIO
import math
from jinja2 import Template
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
from mathruler.grader import extract_boxed_content, grade_answer
import stopit  # 1. Import the thread-safe 'stopit' library

# ------------------------- Command-Line Arguments ------------------------- #
# (This section remains unchanged)
parser = argparse.ArgumentParser()
parser.add_argument('--port', type=str, default='5000')
parser.add_argument('--model_path', type=str, default='Qwen/Qwen3-4B-Base')
parser.add_argument('--gpu_mem_util', type=float, default=0.9, help='The maximum GPU memory utilization fraction for vLLM.')
parser.add_argument("--format_prompt", type=str, default="selfplay/solver/format_prompt.jinja", help="Path to format prompt.")
parser.add_argument("--max_pixels", type=int, default=1003520, help="Maximum number of pixels for image processing.")
parser.add_argument("--min_pixels", type=int, default=200704, help="Minimum number of pixels for image processing.")
parser.add_argument("--batch_size", type=int, default=128, help="Number of prompts per generation batch to limit memory usage.")
args = parser.parse_args()

format_prompt = open(args.format_prompt, encoding="utf-8").read()
format_prompt = Template(format_prompt.strip())

# ------------------------- vLLM Initialization ------------------------ #
# (This section remains unchanged)
print(f'[Solver {args.port}] Loading model...')

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
model = vllm.LLM(
    model=args.model_path,
    tokenizer=args.model_path,
    gpu_memory_utilization=args.gpu_mem_util,
    disable_mm_preprocessor_cache=True,
    max_model_len=6144,
    dtype="bfloat16",
    enable_sleep_mode=True,
    max_num_batched_tokens=8192,
    tensor_parallel_size=4,
)

sample_params = vllm.SamplingParams(
    max_tokens=2048,
    temperature=1.0,
    top_p=0.95,
    top_k=40,
    stop_token_ids=[tokenizer.eos_token_id],
    n=10, # Generate 10 candidate answers for each question
)

dataset_name = os.getenv("SOURCE_DATASET")
if os.path.isdir(dataset_name):
    dataset = load_from_disk(dataset_name)
else:
    dataset = load_dataset(dataset_name, split="train")

def process_image(image, min_pixels, max_pixels):
    if isinstance(image, str):
        image = Image.open(image)
    elif isinstance(image, dict):
        image = Image.open(BytesIO(image["bytes"]))
    elif isinstance(image, bytes):
        image = Image.open(BytesIO(image))

    image.load()  # avoid "Too many open files" errors
    if max_pixels is not None and (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if min_pixels is not None and (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != "RGB":
        image = image.convert("RGB")

    return image

model.sleep(level=1)


# ------------------------ Timeout Utility (Refactored) --------------------------- #
# 2. Use the 'stopit.threading_timeoutable' decorator for thread-safe timeouts.
#    It returns a default value on timeout instead of raising an exception.
@stopit.threading_timeoutable(default='TIMED_OUT')
def grade_answer_with_timeout(res1, res2):
    """
    This wrapper applies a timeout to each individual `grade_answer` call.
    If the function's execution exceeds the specified timeout, it will return 'TIMED_OUT'.
    The timeout duration is passed as a keyword argument during the function call.
    """
    return grade_answer(res1, res2)

# ---------------------------- Flask Application --------------------------- #
app = Flask(__name__)

@app.route('/hello', methods=['GET'])
def hello():
    '''The main processing endpoint: reads a task file, invokes vLLM, consolidates answers, and writes results.'''

    model.wake_up()
    gc.collect()

    name = request.args.get('name', 'None')
    print(f'[Solver {args.port}] Received request for task file: {name}')

    # ---------- Load Data ----------
    with open(name, 'r') as f:
        data = json.load(f)

    valid_indices = []
    valid_prompts = []
    for i, item in enumerate(data):
        q, a, t, img_idx = item['question'], item['answer'], item['question_type'], int(item['image_idx'])
        if q and a and t and img_idx:
            user_prompt = format_prompt.render(content=q).strip("<image>")
            prompt = (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{user_prompt} The final answer should only contain the numeric value or the minimal short answer without any units or punctuation, and MUST appear inside \\boxed{{}}.<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            valid_prompts.append((img_idx, prompt))
            valid_indices.append(i)

    print(f'[Solver {args.port}] {len(valid_prompts)} valid prompts prepared, generating in batches...')

    # ---------- vLLM Batched Generation ----------
    responses = []
    for batch_start in range(0, len(valid_prompts), args.batch_size):
        batch = valid_prompts[batch_start:batch_start + args.batch_size]
        batch_chats = []
        for img_idx, prompt in batch:
            img = process_image(dataset[img_idx]['images'][0], args.min_pixels, args.max_pixels)
            batch_chats.append({"prompt": prompt, "multi_modal_data": {"image": img}})

        batch_responses = model.generate(batch_chats, sampling_params=sample_params, use_tqdm=True)
        responses.extend(batch_responses)
        del batch_chats, batch_responses
        gc.collect()
        print(f'[Solver {args.port}] Batch {batch_start // args.batch_size + 1}/{(len(valid_prompts) + args.batch_size - 1) // args.batch_size} done.')

    del valid_prompts
    print(f'[Solver {args.port}] Generation completed.')

    # ---------- Results Post-Processing (Core Refactoring & Optimization Here) ----------
    def process_single(question, golden_answer, response):
        '''Consolidates and grades vLLM outputs for a single question, returning a result dictionary.'''
        results = [extract_boxed_content(out.text) for out in response.outputs]
        # print(f"[process_single] Processing question: '{question[:70]}...'")

        answer_counts = {}
        for res in results:
            if not res or res == "None": continue # Skip empty results
            matched = False
            
            for exist_ans in list(answer_counts.keys()):
                # 3. OPTIMIZATION: Perform cheap comparisons first to avoid expensive calls.
                if res == exist_ans or ('no ' in res.lower() and 'no ' in exist_ans.lower()):
                    answer_counts[exist_ans] += 1
                    matched = True
                    break # Match found, break from the inner loop over exist_ans
                
                # 4. If cheap checks fail, proceed to the expensive, timed grade_answer calls.
                try:
                    is_match = False
                    # First direction: res vs exist_ans
                    match_result_1 = grade_answer_with_timeout(res, exist_ans, timeout=10)
                    if match_result_1 == 'TIMED_OUT':
                        print(f"      [grader] TIMEOUT comparing '{res[:30]}...' with '{exist_ans[:30]}...'.")
                    elif match_result_1:
                        is_match = True

                    # Second direction (only if first failed): exist_ans vs res
                    if not is_match:
                        match_result_2 = grade_answer_with_timeout(exist_ans, res, timeout=10)
                        if match_result_2 == 'TIMED_OUT':
                             # Log timeout for the second direction as well
                            print(f"      [grader] TIMEOUT comparing '{exist_ans[:30]}...' with '{res[:30]}...'. Skipping pair.")
                        elif match_result_2:
                            is_match = True
                    
                    if is_match:
                        answer_counts[exist_ans] += 1
                        matched = True
                        break # Match found, break from the inner loop

                except Exception as e:
                    # Catch any other potential errors from the grader function itself.
                    print(f"      [grader] ERROR comparing '{res[:30]}...' with '{exist_ans[:30]}...': {e}. Skipping.")
                    continue # Continue to the next comparison in the inner loop
            
            if not matched:
                answer_counts[res] = 1

        if not answer_counts:
            majority_ans, max_count = '', 0
        else:
            majority_ans = max(answer_counts, key=answer_counts.get)
            max_count = answer_counts[majority_ans]

        score = max_count / len(results) if results else 0.0

        return {
            'question': question,
            'answer':   majority_ans,
            'score':    score,
            'results':  results
        }

    results_all = []
    response_idx = 0
    for i, item in enumerate(data):
        q, a, img_idx = item['question'], item['answer'], int(item['image_idx'])
        if i not in valid_indices:
            results_all.append({'question': q, 'answer': a, 'score': -1, 'results': [], 'image_idx': img_idx})
        else:
            response = responses[response_idx]
            response_idx += 1
            new_item = process_single(q, a, response)
            new_item['image_idx'] = img_idx
            results_all.append(new_item)
    print(f'[Solver {args.port}] All results have been processed.')

    out_path = name.replace('.json', '_results.json')
    with open(out_path, 'w') as f:
        json.dump(results_all, f, indent=4)

    os.remove(name)

    print(f'[Solver {args.port}] Processed {name}, results saved to {out_path}. Resuming idle worker.')
    del responses, results_all, data
    gc.collect()
    model.sleep(level=1)
    time.sleep(1)
    return jsonify({'message': f'Processed {name}, results saved to {out_path}.'})

# ------------------------- Main Application Entrypoint --------------------------- #
# (This section remains unchanged)
if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=int(args.port), threaded=True)
    finally:
        print(f'[Solver {args.port}] Application shutdown complete.')
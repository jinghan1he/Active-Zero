#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Questioner Server
-----------------

接收包含 image_indices 的任务文件，使用 vLLM 根据图像生成问题。
参考 start_solver_server.py 的结构。
'''

from flask import Flask, request, jsonify
import vllm
import argparse
import json
import os
import threading
import time
import torch
import numpy as np
from jinja2 import Template
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
from io import BytesIO
from PIL import Image
import math
import regex as re
from verl.utils.dataset import process_image

# ------------------------- Command-Line Arguments ------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument('--port', type=str, default='5000')
parser.add_argument('--model_path', type=str, default='Qwen/Qwen2.5-VL-7B-Instruct')
parser.add_argument('--gpu_mem_util', type=float, default=0.9, help='The maximum GPU memory utilization fraction for vLLM.')
parser.add_argument("--format_prompt", type=str, default="selfplay/questioner/format_prompt.jinja", help="Path to format prompt.")
parser.add_argument("--max_pixels", type=int, default=1003520, help="Maximum number of pixels for image processing.")
parser.add_argument("--min_pixels", type=int, default=200704, help="Minimum number of pixels for image processing.")
args = parser.parse_args()

format_prompt = open(args.format_prompt, encoding="utf-8").read()
format_prompt = Template(format_prompt.strip())
user_prompt = format_prompt.render().strip("<image>")
prompt = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{user_prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

# ------------------------- vLLM Initialization ------------------------ #
print(f'[Questioner {args.port}] Loading model...')

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = vllm.LLM(
    model=args.model_path,
    tokenizer=args.model_path,
    gpu_memory_utilization=args.gpu_mem_util,
    disable_mm_preprocessor_cache=True,
    max_model_len=6144,
    dtype="bfloat16",
    enable_sleep_mode=True,
    max_num_batched_tokens=8192,
)

sample_params = vllm.SamplingParams(
    max_tokens=2048,
    temperature=1.0,
    top_p=0.95,
    n=1,
    stop_token_ids=[tokenizer.eos_token_id],
)

# Load dataset
dataset_name = os.getenv("SOURCE_DATASET")
if os.path.isdir(dataset_name):
    dataset = load_from_disk(dataset_name)
else:
    dataset = load_dataset(dataset_name, split="train")

print(f'[Questioner {args.port}] Initialization complete.')


# ---------------------- GPU Idle Utilization Thread ---------------------- #
# stop_event = threading.Event()
# pause_event = threading.Event()

# def gpu_idle_worker():
#     '''
#     This worker occupies the GPU with a continuous matrix multiplication loop when idle,
#     preventing potential performance drops from GPU power state changes.
#     '''
#     print('[QuestionerServer] GPU idle worker started.')
#     running = True
#     while not stop_event.is_set():
#         if pause_event.is_set():
#             if running:
#                 print('[QuestionerServer] Paused.')
#                 running = False
#             time.sleep(0.1)
#             continue
#         else:
#             if not running:
#                 print('[QuestionerServer] Resumed.')
#                 running = True
#         try:
#             a = torch.rand((2000, 2000), dtype=torch.float32, device='cuda')
#             b = torch.rand((2000, 2000), dtype=torch.float32, device='cuda')
#             torch.matmul(a, b)
#             torch.cuda.synchronize()
#         except RuntimeError as e:
#             print(f'[QuestionerServer] Caught a RuntimeError: {e}. Sleeping for 1s...')
#             time.sleep(1)
#     print('[QuestionerServer] GPU idle worker stopped.')

# idle_thread = threading.Thread(target=gpu_idle_worker, daemon=True)
# idle_thread.start()
model.sleep(level=1)

# ---------------------------- Flask Application --------------------------- #
app = Flask(__name__)

@app.route('/hello', methods=['GET'])
def hello():
    '''The main processing endpoint: reads a task file with image indices, invokes vLLM to generate questions, and writes results.'''

    # --- Pause the GPU idle worker to free up resources ---
    # pause_event.set()
    # torch.cuda.synchronize()
    model.wake_up()

    name = request.args.get('name')
    print(f'[Questioner {args.port}] Received request for task file: {name}')

    # ---------- Load Data ----------
    image_indices = np.load(name, allow_pickle=True).tolist()

    # 加载图像
    images = []
    for img_idx in image_indices:
        img = dataset[int(img_idx)]['images'][0]
        processed_image = process_image(img, args.min_pixels, args.max_pixels)
        images.append(processed_image)

    # 准备 prompts
    valid_chats = []
    for img in images:
        valid_chats.append({
            "prompt": prompt,
            "multi_modal_data": {"image": img}
        })

    print(f'[Questioner {args.port}] Valid chat prompts have been prepared for {len(valid_chats)} images.')

    # ---------- vLLM Generation ----------
    responses = model.generate(valid_chats, sampling_params=sample_params, use_tqdm=True)
    print(f'[Questioner {args.port}] Generation completed.')

    # ---------- Results Post-Processing ----------
    results_all = []
    for response, img_idx in zip(responses, image_indices):
        response_text = response.outputs[0].text
        try:
            # Extract question type, question, and answer
            question_types = re.findall(r"<type>(.*?)</type>", response_text, re.DOTALL)
            questions = re.findall(r"<question>(.*?)</question>", response_text, re.DOTALL)
            answers = re.findall(r"<answer>(.*?)</answer>", response_text, re.DOTALL)

            if questions and answers:
                question_type = question_types[-1].strip() if question_types else "unknown"
                question = questions[-1].strip()
                answer = answers[-1].strip()
                results_all.append({
                    "question_type": question_type,
                    "question": question,
                    "answer": answer,
                    "image_idx": img_idx,
                })
            else:
                results_all.append({
                    "question_type": "unknown",
                    "question": response_text,
                    "answer": "",
                    "image_idx": img_idx,
                })
        except Exception as e:
            print(f'[Questioner {args.port}] Error processing response for image_idx {img_idx}: {e}')
            results_all.append({
                "question_type": "error",
                "question": response_text,
                "answer": "",
                "image_idx": img_idx,
            })

    print(f'[Questioner {args.port}] All results have been processed.')

    out_path = name.replace('.npy', '.json')
    with open(out_path, 'w') as f:
        json.dump(results_all, f, indent=4)
    
    os.remove(name)

    # --- Resume the GPU idle worker ---
    # pause_event.clear()
    print(f'[Questioner {args.port}] Processed {name}, results saved to {out_path}. Resuming idle worker.')
    model.sleep(level=1)
    time.sleep(1)
    return jsonify({'message': f'Processed {name}, results saved to {out_path}.'})

# ------------------------- Main Application Entrypoint --------------------------- #
if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=int(args.port), threaded=True)
    finally:
        # stop_event.set()
        # idle_thread.join()
        print(f'[Questioner {args.port}] Application shutdown complete.')


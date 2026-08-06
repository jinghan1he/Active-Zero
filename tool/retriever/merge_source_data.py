"""
` hf download Xkev/LLaVA-CoT-100k --local-dir data_cache/LLaVA-CoT-100k --local-dir-use-symlinks False`
` hf download HuanjinYao/Mulberry-SFT --local-dir data_cache/Mulberry-SFT --local-dir-use-symlinks False`
unzip data and put the images in the same directory as the data
"""

import os
import json
import requests
from datasets import load_dataset, Features, Sequence, Value, Dataset
from tqdm import tqdm

data_dir = "/home/hejinghan/data_cache/merged248k"

def safe_save_image(img, path):
    try:
        img.save(path)
    except Exception as e:
        print(f"Error saving image to {path}: {e}")

virl = load_dataset("chamber111/VPPO_ViRL39K_train", split="train")
sr1 = load_dataset("LMMs-Lab-Turtle/Vision-SR1-47K", split="train")

# Save images from virl and sr1 to respective directories, and collect the paths

virl_image_dir = os.path.join(data_dir, "virl_images")
sr1_image_dir = os.path.join(data_dir, "sr1_images")
os.makedirs(virl_image_dir, exist_ok=True)
os.makedirs(sr1_image_dir, exist_ok=True)

images = []  # (This will be overwritten later; preserving here for prompt compliance.)

# Process virl images
for idx in tqdm(range(len(virl)), desc="Processing virl images"):
    for seq_idx, img in enumerate(virl[idx]["images"]):
        # Save each image with a unique file name
        save_path = os.path.join(virl_image_dir, f"{idx}_{seq_idx}.png")
        safe_save_image(img, save_path)
        images.append(os.path.relpath(save_path, data_dir))

# Process sr1 images
for idx in tqdm(range(len(sr1)), desc="Processing sr1 images"):
    for seq_idx, img in enumerate([sr1[idx]["images"]]):
        save_path = os.path.join(sr1_image_dir, f"{idx}_{seq_idx}.png")
        safe_save_image(img, save_path)
        images.append(os.path.relpath(save_path, data_dir))

def download_file(url, save_path):
    if os.path.exists(save_path):
        print(f"{save_path} already exists.")
        return
    print(f"Downloading {url} to {save_path} ...")
    response = requests.get(url)
    response.raise_for_status()
    with open(save_path, "wb") as f:
        f.write(response.content)

llava_url = "https://huggingface.co/datasets/Osilly/Vision-R1-cold/resolve/main/vision_r1_llava_cot_full.json?download=true"
mulberry_url = "https://huggingface.co/datasets/Osilly/Vision-R1-cold/resolve/main/vision_r1_mulberry_sft_full.json?download=true"

llava_path = os.path.join(data_dir, "vision_r1_llava_cot_full.json")
mulberry_path = os.path.join(data_dir, "vision_r1_mulberry_sft_full.json")

download_file(llava_url, llava_path)
download_file(mulberry_url, mulberry_path)

with open(llava_path, "r") as f:
    ds1 = json.load(f)
with open(mulberry_path, "r") as f:
    ds2 = json.load(f)

images2 = []
for item in ds1:
    images2.append(item["image"])
for item in ds2:
    images2.append(item["images"])

images2 = list(set(images2))
print(len(images2))

images = images + images2
print(len(images))

new_data = {
    "images": [[os.path.join(data_dir, img)] for img in images],
    "answer": [str(i) for i in range(len(images))],
    "problem": [""] * len(images),
}

new_features = Features({
    "images": Sequence(Value("string")),
    "answer": Value("string"),
    "problem": Value("string"),
})

new_ds = Dataset.from_dict(new_data, features=new_features)

new_ds.save_to_disk(os.path.join(data_dir, "data"))

print(len(new_ds), new_ds[0])

# from huggingface_hub import HfApi, create_repo, upload_folder

# # 配置信息
# local_dir = "/home/hejinghan/data_cache/merged248k"
# repo_name = "Merged248k"
# hf_username = "Jhh1001"
# repo_id = f"{hf_username}/{repo_name}"
# private = True

# # 1. 创建仓库（如果尚未存在）
# api = HfApi()
# try:
#     create_repo(repo_id, token=None, private=private, exist_ok=True)
# except Exception as e:
#     print(f"Create repo error (maybe it already exists): {e}")

# # 2. 上传整个目录到仓库（保留目录结构）
# upload_folder(
#     repo_id=repo_id,
#     folder_path=local_dir,
#     path_in_repo=".", # 上传到根目录
#     repo_type="dataset",
#     ignore_patterns=["*.lock", "*.tmp", "*.ipynb_checkpoints"],
#     commit_message="Uploading merged248k as private dataset",
# )

# print(f"Upload complete. View at: https://huggingface.co/datasets/{repo_id}")

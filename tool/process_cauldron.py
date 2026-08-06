import glob
import os

from datasets import Dataset, get_dataset_config_names, load_dataset
from tqdm import tqdm


DATASET_NAME = "HuggingFaceM4/the_cauldron"
BASE_DATA_DIR = "/home/hejinghan/data_cache/the_cauldron"


def process_the_cauldron():
    configs = get_dataset_config_names(DATASET_NAME)

    for config in tqdm(configs, desc="处理子集"):
        if config in ["clevr_math", "okvqa"]:
            continue

        subset_dir = os.path.join(BASE_DATA_DIR, config)
        os.makedirs(subset_dir, exist_ok=True)

        try:
            ds_stream = load_dataset(DATASET_NAME, config, split="train", streaming=True)
        except Exception as e:
            print(f"跳过子集 {config}，原因: {e}")
            continue

        existing_count = len(glob.glob(os.path.join(subset_dir, "*.jpg")))
        saved_count = 0
        skipped_count = 0

        for row_idx, row in enumerate(tqdm(ds_stream, desc=config, leave=False)):
            images = row.get("images") or []
            if len(images) != 1:
                skipped_count += 1
                continue

            img_path = os.path.join(subset_dir, f"{row_idx}.jpg")
            if os.path.exists(img_path):
                continue

            try:
                image = images[0]
                if image.mode != "RGB":
                    image = image.convert("RGB")
                image.save(img_path, format="JPEG", quality=95)
                saved_count += 1
            except Exception as e:
                print(f"跳过 {config} 的第 {row_idx} 条图片，原因: {e}")

        print(
            f"{config}: 已有 {existing_count} 张，新增 {saved_count} 张，"
            f"跳过 {skipped_count} 条非单图样本"
        )


def construct_dataset():
    data = []
    subsets = glob.glob(os.path.join(BASE_DATA_DIR, "*"))
    subsets = sorted(subsets)
    total_rows = 0
    for subset in subsets:
        images = glob.glob(os.path.join(subset, "*.jpg"))
        images = sorted(images, key=lambda x: int(x.split("/")[-1].split(".")[0]))
        for image in images:
            data.append({
                "images": [image],
                "problem": "",
                "answer": str(total_rows)
            })
            total_rows += 1    
    dataset = Dataset.from_list(data)
    dataset.save_to_disk(os.path.join(BASE_DATA_DIR, "data1"))
    print(f"Total rows: {total_rows}")
    print(dataset[0])


if __name__ == "__main__":
    # process_the_cauldron()
    construct_dataset()

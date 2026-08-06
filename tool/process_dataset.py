from jinja2 import Template
import re
import base64
from io import BytesIO
from PIL import Image as PILImage

from datasets import load_dataset, load_from_disk, Features, Sequence, Image, Value, Dataset


TARGET_FEATURES = Features({
        "problem": Value("string"),
        "images": Sequence(Image(decode=True)), 
        "answer": Value("string"),
    })

with open("tool/mc.jinja", encoding="utf-8") as f:
    MC_FORMAT_PROMPT = f.read()
    MC_FORMAT_PROMPT = Template(MC_FORMAT_PROMPT)

def mathverse():
    dataset = load_dataset("AI4Math/MathVerse", "testmini", split="testmini")

    def transform_example(example):
        image = example.get("image", None)
        images = [image] if image is not None else []
        
        return {
            "problem": "<image> " + example.get("question", ""),
            "images": images,
            "answer": example.get("answer", ""),
        }
    
    dataset = dataset.map(
        transform_example, 
        remove_columns=dataset.column_names,
        features=TARGET_FEATURES
    )

    dataset.save_to_disk("Evaluation/Datasets/MathVerse/")
    
    return dataset


def mathverse_v():
    dataset = load_dataset("AI4Math/MathVerse", "testmini", split="testmini")
    
    # dataset = dataset.filter(lambda example: example.get("problem_version", "") in ["Vision Dominant", "Vision Intensive", "Vision Only"])
    dataset = dataset.filter(lambda example: example.get("problem_version", "") in ["Vision Dominant", "Vision Intensive"])

    def transform_example(example):
        image = example.get("image", None)
        images = [image] if image is not None else []
        question = example.get("question", "")
        if not question:
            question = "Please read the question and choices from the image, and answer the correct choice based on the image."
        
        return {
            "problem": "<image> " + question,
            "images": images,
            "answer": example.get("answer", ""),
        }
    
    dataset = dataset.map(
        transform_example, 
        remove_columns=dataset.column_names,
        features=TARGET_FEATURES
    )

    # dataset.save_to_disk("Evaluation/Datasets/MathVerse-V/")
    dataset.save_to_disk("Evaluation/Datasets/MathVerse-VD/")
    
    return dataset


def mathvision():
    dataset = load_dataset("MathLLMs/MathVision", split="test")

    def transform_example(example):
        image = example.get("decoded_image", None)
        images = [image] if image is not None else []

        problem = example.get("question", "")
        problem = "<image> " + problem.replace("<image1>", "").strip()
        options = example.get("options", [])
        problem = MC_FORMAT_PROMPT.render(content=problem, options=options)
        
        return {
            "problem": problem,
            "images": images,
            "answer": example.get("answer", ""),
        }
    
    dataset = dataset.map(
        transform_example, 
        remove_columns=dataset.column_names,
        features=TARGET_FEATURES
    )

    dataset.save_to_disk("Evaluation/Datasets/MathVision/")
    return dataset


def wemath():
    dataset = load_dataset("We-Math/We-Math", split="testmini")

    def transform_example(example):
        image = example.get("image_path", None)
        images = [image] if image is not None else []
        
        return {
            "problem": "<image> " + example.get("question", "") + " " + example.get("option", ""),
            "images": images,
            "answer": example.get("answer", ""),
        }
    
    dataset = dataset.map(
        transform_example, 
        remove_columns=dataset.column_names,
        features=TARGET_FEATURES
    )

    dataset.save_to_disk("Evaluation/Datasets/We-Math/")
    return dataset


def dynamath():
    dataset = load_dataset("DynaMath/DynaMath_Sample", split="all")

    def transform_example(example):
        image = example.get("decoded_image", None)
        images = [image] if image is not None else []
        
        return {
            "problem": "<image> " + example.get("question", ""),
            "images": images,
            "answer": example.get("ground_truth", ""),
        }
    
    dataset = dataset.map(
        transform_example, 
        remove_columns=dataset.column_names,
        features=TARGET_FEATURES
    )

    dataset.save_to_disk("Evaluation/Datasets/DynaMath/")
    return dataset


def mathvista():
    dataset = load_dataset("AI4Math/MathVista", split="testmini")

    def transform_example(example):
        image = example.get("decoded_image", None)
        images = [image] if image is not None else []
        question = "<image> " + example.get("question", "")
        answer = example.get("answer", "")

        options = example.get("choices", [])
        if options:
            question = MC_FORMAT_PROMPT.render(content=question, options=options)
            answer = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[options.index(answer)]
        
        return {
            "problem": question,
            "images": images,
            "answer": answer,
        }
    
    dataset = dataset.map(
        transform_example, 
        remove_columns=dataset.column_names,
        features=TARGET_FEATURES
    )

    dataset.save_to_disk("Evaluation/Datasets/MathVista/")
    return dataset


def mmstar():

    dataset = load_dataset("Lin-Chen/MMStar", split="val")

    def transform_example(example):
        image = example.get("image", None)
        images = [image] if image is not None else []
        
        return {
            "problem": "<image> " + example.get("question", ""),
            "images": images,
            "answer": example.get("answer", ""),
        }

    dataset = dataset.map(
        transform_example, 
        remove_columns=dataset.column_names,
        features=TARGET_FEATURES
    )

    dataset.save_to_disk("Evaluation/Datasets/MMStar/")
    return dataset


def logicvista():
    dataset = load_dataset("lscpku/LogicVista", split="test")

    def transform_example(example):
        image = example.get("image", None)
        images = [image] if image is not None else []
        
        return {
            "problem": "<image> " + example.get("question", ""),
            "images": images,
            "answer": example.get("answer", ""),
        }
    
    dataset = dataset.map(
        transform_example, 
        remove_columns=dataset.column_names,
        features=TARGET_FEATURES
    )

    dataset.save_to_disk("Evaluation/Datasets/LogicVista/")
    return dataset


def realworldqa():
    dataset = load_dataset("lmms-lab/RealWorldQA", split="test")

    def transform_example(example):
        image = example.get("image", None)
        images = [image] if image is not None else []
        
        return {
            "problem": "<image> " + example.get("question", "").strip("Please answer directly with only the letter of the correct option and nothing else."),
            "images": images,
            "answer": example.get("answer", ""),
        }

    dataset = dataset.map(
        transform_example, 
        remove_columns=dataset.column_names,
        features=TARGET_FEATURES
    )

    dataset.save_to_disk("Evaluation/Datasets/RealWorldQA/")
    return dataset


def hallusionbench():
    dataset = load_dataset("zli12321/hallusionbench", split="test")

    def transform_example(example):
        image = example.get("images", None)
        images = [image] if image is not None else []
        
        return {
            "problem": "<image> " + example.get("problem", "").replace("<image>", ""),
            "images": images,
            "answer": example.get("answer", ""),
        }
    
    dataset = dataset.map(
        transform_example, 
        remove_columns=dataset.column_names,
        features=TARGET_FEATURES
    )

    dataset.save_to_disk("Evaluation/Datasets/HallusionBench/")
    return dataset


def visnumbench():
    dataset = load_dataset("zli12321/visnumbench", split="test")

    def transform_example(example):
        image = example.get("images", None)
        images = [image] if image is not None else []
        
        return {
            "problem": "<image> " + example.get("problem", "").replace("<image>", ""),
            "images": images,
            "answer": example.get("answer", ""),
        }
    
    dataset = dataset.map(
        transform_example, 
        remove_columns=dataset.column_names,
        features=TARGET_FEATURES
    )

    dataset.save_to_disk("Evaluation/Datasets/VisNumBench/")
    return dataset


def mmmu_pro():
    dataset = load_dataset("MMMU/MMMU_Pro", "vision", split="test")

    def transform_example(example):
        image = example.get("image", None)
        images = [image] if image is not None else []
        
        return {
            "problem": "<image> Please read the question and choices from the image, and answer the correct choice based on the image.",
            "images": images,
            "answer": example.get("answer", ""),
        }
    
    dataset = dataset.map(
        transform_example, 
        remove_columns=dataset.column_names,
        features=TARGET_FEATURES
    )

    dataset.save_to_disk("Evaluation/Datasets/MMMU_Pro/")
    return dataset


def mmmu_pro_standard(n_options=10):
    dataset = load_dataset("MMMU/MMMU_Pro", f"standard ({n_options} options)", split="test")

    # 过滤出image_1列为空的row
    dataset = dataset.filter(lambda x: x.get("image_2") is None and x.get("question", "").count("<image") <= 1)

    def transform_example(example):
        images = []
        for i in range(1, 8):
            image = example.get(f"image_{i}", None)
            if image:
                images.append(image)

        question = re.sub(r"<image \d+>", "<image>", example.get("question"))
        if "<image>" not in question:
            question = "<image> " + question.strip()
        options = eval(example.get("options", "[]"))
        question = MC_FORMAT_PROMPT.render(content=question, options=options)
        
        return {
            "problem": question,
            "images": images,
            "answer": example.get("answer", ""),
        }
    
    dataset = dataset.map(
        transform_example, 
        remove_columns=dataset.column_names,
        features=TARGET_FEATURES
    )

    dataset.save_to_disk(f"Evaluation/Datasets/MMMU_Pro_{n_options}options/")
    return dataset


def m3cot():
    dataset = load_dataset("LightChen2333/M3CoT", split="test")

    def transform_example(example):
        image = example.get("image", None)
        images = [image] if image is not None else []

        problem = example.get("question", "")
        problem = "<image> " + problem.replace("<image1>", "").strip()
        options = example.get("choices", [])
        problem = MC_FORMAT_PROMPT.render(content=problem, options=options)
        
        return {
            "problem": problem,
            "images": images,
            "answer": example.get("answer", ""),
        }

    dataset = dataset.map(
        transform_example, 
        remove_columns=dataset.column_names,
        features=TARGET_FEATURES
    )

    dataset.save_to_disk("Evaluation/Datasets/M3CoT/")
    return dataset


def mmvet():
    dataset = load_dataset("zli12321/mm-vet", split="test")

    def transform_example(example):
        image = example.get("images", None)
        images = [image] if image is not None else []
        
        return {
            "problem": example.get("problem", ""),
            "images": images,
            "answer": example.get("answer", ""),
        }

    dataset = dataset.map(
        transform_example, 
        remove_columns=dataset.column_names,
        features=TARGET_FEATURES
    )

    dataset.save_to_disk("Evaluation/Datasets/mm-vet/")
    return dataset


def r1_onevision():
    dataset = load_from_disk("/home/hejinghan/data_cache/R1-Onevision-Deduplicated")

    def base64_to_pil(base64_str):
        """Convert base64 string to PIL Image"""
        img_data = base64.b64decode(base64_str)
        img = PILImage.open(BytesIO(img_data))
        img = img.convert("RGB")
        return img

    def transform_example(example):
        image_str = example["image"]
        
        # Convert base64 strings to PIL Images
        pil_img = base64_to_pil(image_str)
        images = [pil_img]

        return {
            "problem": example["problem"],
            "images": images,
            "answer": example["answer"],
        }
    
    dataset = dataset.map(
        transform_example,
        remove_columns=dataset.column_names,
        features=TARGET_FEATURES
    )

    dataset.save_to_disk("/home/hejinghan/data_cache/R1-Onevision/")
    return dataset


def dummy_dataset_for_searcher(num_rows, save_path):

    # 2. 构造数据字典
    # 使用列表乘法可以非常快速地在内存中生成重复的 dummy 内容
    data = {
        "problem": ["dummy problem"] * num_rows,
        "answer": ["dummy answer"] * num_rows
    }

    # 3. 创建 Dataset 对象
    print(f"正在创建包含 {num_rows} 条数据的数据集...")
    dataset = Dataset.from_dict(data)

    # 4. 存储到磁盘
    print(f"正在保存到磁盘: {save_path}...")
    dataset.save_to_disk(save_path)

    print("保存完成！")

    # 5. (可选) 验证加载
    loaded_ds = load_from_disk(save_path)
    print(f"验证数据集大小: {len(loaded_ds)}")
    print(f"第一条数据内容: {loaded_ds[0]}")


if __name__ == "__main__":

    # r1_onevision()
    # dummy_dataset_for_searcher(num_rows=1000, save_path="dummy_dataset_1k")
    # dummy_dataset_for_searcher(num_rows=1000000, save_path="dummy_dataset_1m")
    mathverse()
    mathvision()
    mathvista()
    wemath()
    dynamath()
    logicvista()
    visnumbench()
    realworldqa()
    hallusionbench()
    m3cot()
    mmvet()
    mathverse_v()
    mmmu_pro()
    mmmu_pro_standard(4)
    mmmu_pro_standard(10)
    mmstar()
    
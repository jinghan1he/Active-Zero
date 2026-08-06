import pandas as pd
from datasets import Dataset, load_dataset
import base64
import io
from io import BytesIO
from PIL import Image as PILImage
from datasets import Features, Sequence, Value, Image

# dataset_name = "chamber111/VPPO_ViRL39K_train"
# output_dir = "/home/hejinghan/data_cache/ViRL39K"
# target_repo = "Jhh1001/ViRL39K" 

dataset_name = "LMMs-Lab-Turtle/Vision-SR1-47K"
output_dir = "/home/hejinghan/data_cache/Vision-SR1-47K"
target_repo = "Jhh1001/Vision-SR1-47K" 

ds = load_dataset(dataset_name, split="train")

# convert PIL image to base64 
def pil_to_base64(image, format="PNG"):
    # 1. 创建一个内存缓冲区
    buffer = BytesIO()
    
    # 2. 将 PIL 图像保存到缓冲区，并指定格式
    image.save(buffer, format=format)
    
    # 3. 获取缓冲区的二进制字节内容
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    return img_str

def base64_to_pil(image_base64):
    return PILImage.open(io.BytesIO(base64.b64decode(image_base64))).convert("RGB")

ds = ds.map(lambda x: {"image": pil_to_base64(x["images"])})

print("Deduplicating dataset...")
df = ds.to_pandas()

unique_images = df['image'].unique()

print("Constructing new dataset structure...")
new_data = {
    "images": [[base64_to_pil(image)] for image in unique_images],
    "problem": [""] * len(unique_images),
    "answer": [str(i) for i in range(len(unique_images))]  # index as answer
}
new_features = Features({
    "images": Sequence(Image(decode=True)),
    "problem": Value("string"),
    "answer": Value("string"),
})
new_ds = Dataset.from_dict(new_data, features=new_features)



new_ds.save_to_disk(output_dir)

# print("Uploading to Hugging Face...")
# new_ds.push_to_hub(target_repo)

print("Done!")
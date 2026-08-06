import torch
from transformers import AutoModel, AutoProcessor
from datasets import load_from_disk
import faiss
from PIL import Image
from imgcat import imgcat


class T2I_Searcher:

    def __init__(self, source_dataset) -> None:
        self.index = faiss.read_index(f"/home/hejinghan/projects/SelfPlay-VL/tool/retriever/{source_dataset}.index")

        model_name = "google/siglip2-so400m-patch16-naflex"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"正在加载模型: {model_name}...")
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)

        dataset_path = f"/home/hejinghan/data_cache/{source_dataset}/data"
        self.dataset = load_from_disk(dataset_path) # 根据实际情况选择 split

    def search_in_dataset(self, query_text, top_k=5):
        
        # 编码查询文本
        inputs = self.processor(text=[query_text], padding="max_length", return_tensors="pt").to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
            text_features = text_features.cpu().numpy().astype('float32')

        # 检索
        scores, indices = self.index.search(text_features, top_k)
        
        
        # 展示结果
        print(f"\n查询内容: '{query_text}'")
        samples = []
        for i in range(top_k):
            idx = int(indices[0][i])
            score = scores[0][i]
            
            # 直接从原数据集中按索引提取数据
            sample = self.dataset[idx]
            samples.append(sample)
            print(f"Rank {i+1}: 得分 {score:.4f}, 数据ID: {idx}")
            img = Image.open(sample['images'][0]).convert("RGB").resize((256, 256))
            imgcat(img)
            # sample['image'].show() # 如果在 Jupyter 中可以取消注释直接看图
            
        return samples
    
    def show_image(self, img):
        img = Image.open(img).convert("RGB").resize((256, 256))
        return img

# 测试
if __name__ == "__main__":
    searcher = T2I_Searcher("the_cauldron")
    while True:
        query = input("query: ")
        samples = searcher.search_in_dataset(query, 10)

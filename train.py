import json
import os
from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_PATH = "trained_model"

def train_model(actions_file="actions.json"):
    """Huấn luyện SentenceTransformer với dữ liệu mẫu và lưu mô hình."""
    if not os.path.exists(actions_file):
        print(f"Không tìm thấy {actions_file}. Đảm bảo rằng file tồn tại.")
        return
    
    # Load dữ liệu từ file JSON
    with open(actions_file, "r", encoding="utf-8") as f:
        actions = json.load(f)
    
    # Load mô hình Sentence-BERT
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # Mã hóa sẵn các câu mẫu
    encoded_templates = {key: model.encode(samples) for key, samples in actions.items()}

    # Lưu mô hình và embeddings
    os.makedirs(MODEL_PATH, exist_ok=True)
    model.save(MODEL_PATH)

    # Chuyển embeddings sang list trước khi lưu
    np.save(os.path.join(MODEL_PATH, "encoded_templates.npy"), {k: v.tolist() for k, v in encoded_templates.items()})
    
    print("Huấn luyện hoàn tất. Mô hình đã được lưu.")

if __name__ == "__main__":
    train_model()

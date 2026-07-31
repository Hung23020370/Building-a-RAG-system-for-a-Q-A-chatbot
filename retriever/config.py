"""Cấu hình chung cho package retriever.
Mọi giá trị nhạy cảm hoặc phụ thuộc môi trường (đường dẫn, API key nếu sau này
tích hợp thêm dịch vụ ngoài) đều đọc qua biến môi trường / file .env,
KHÔNG hard-code trực tiếp trong code để tránh lộ khi push lên GitHub.
"""
import os
from dotenv import load_dotenv

load_dotenv()  # tự động tìm file .env ở thư mục gốc project

# --- Đường dẫn dữ liệu ---
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./vector_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "uet_legal_docs")

# --- Model ---
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")

# --- Tham số BM25 ---
BM25_K1 = float(os.getenv("BM25_K1", "1.5"))
BM25_B = float(os.getenv("BM25_B", "0.75"))

# --- Tham số retrieval / fusion ---
DEFAULT_POOL_SIZE = int(os.getenv("DEFAULT_POOL_SIZE", "50"))
DEFAULT_FINAL_TOP_K = int(os.getenv("DEFAULT_FINAL_TOP_K", "5"))
DEFAULT_REWRITE_WEIGHT = float(os.getenv("DEFAULT_REWRITE_WEIGHT", "0.8"))
WRRF_K = int(os.getenv("WRRF_K", "60"))

# --- Tham số sinh câu trả lời ---
GENERATION_SEED = int(os.getenv("GENERATION_SEED", "42"))
GEN_TEMPERATURE = float(os.getenv("GEN_TEMPERATURE", "0.2"))
GEN_TOP_P = float(os.getenv("GEN_TOP_P", "0.85"))
GEN_TOP_K = int(os.getenv("GEN_TOP_K", "40"))
GEN_REPETITION_PENALTY = float(os.getenv("GEN_REPETITION_PENALTY", "1.15"))
GEN_MAX_NEW_TOKENS_ANSWER = int(os.getenv("GEN_MAX_NEW_TOKENS_ANSWER", "512"))
GEN_MAX_NEW_TOKENS_REWRITE = int(os.getenv("GEN_MAX_NEW_TOKENS_REWRITE", "64"))

# --- Placeholder cho các API key ngoài (nếu sau này tích hợp Google / LlamaIndex Cloud) ---
# KHÔNG in giá trị này ra log/console dưới bất kỳ hình thức nào.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

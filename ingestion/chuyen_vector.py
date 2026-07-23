import os
import json
from sentence_transformers import SentenceTransformer
import chromadb
from tqdm import tqdm # Để hiển thị thanh tiến trình cho đẹp

# ==========================================
# CẤU HÌNH THƯ MỤC VÀ MÔ HÌNH
# ==========================================
INPUT_JSON = "./ingestion/uet_rag_chunks.json" # File kết quả từ bước trước
DB_PATH = "./vector_db" # Thư mục lưu database cục bộ
COLLECTION_NAME = "uet_legal_docs"
MODEL_NAME = "BAAI/bge-m3"
BATCH_SIZE = 32 # Xử lý theo cụm để tránh tràn RAM/VRAM

# ==========================================
# 1. KHỞI TẠO MÔ HÌNH VÀ DATABASE
# ==========================================
print(f"⏳ Đang tải mô hình Embedding '{MODEL_NAME}'... (Lần đầu sẽ hơi lâu để tải model)")
# Sử dụng thiết bị có sẵn (Tự động nhận CUDA nếu có GPU, không thì chạy CPU)
model = SentenceTransformer(MODEL_NAME)

print(f"🗄️ Đang khởi tạo ChromaDB tại thư mục: {DB_PATH}")
chroma_client = chromadb.PersistentClient(path=DB_PATH)

# Tạo collection (Nếu đã có thì lấy ra dùng tiếp)
# bge-m3 mặc định sinh ra vector có số chiều (dimension) là 1024
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"} # Khuyến nghị dùng Cosine Similarity cho BGE models
)

# ==========================================
# 2. ĐỌC DỮ LIỆU TỪ JSON
# ==========================================
if not os.path.exists(INPUT_JSON):
    raise FileNotFoundError(f"❌ Không tìm thấy file dữ liệu: {INPUT_JSON}")

with open(INPUT_JSON, 'r', encoding='utf-8') as f:
    chunks_data = json.load(f)

print(f"📦 Đã tải {len(chunks_data)} chunks từ {INPUT_JSON}")

# ==========================================
# 3. THỰC THI EMBEDDING VÀ LƯU TRỮ (BATCHING)
# ==========================================
print("🚀 Bắt đầu quá trình Embedding và lưu vào Vector DB...")

# Tách dữ liệu thành các batch nhỏ để xử lý
for i in tqdm(range(0, len(chunks_data), BATCH_SIZE), desc="Đang xử lý"):
    batch = chunks_data[i : i + BATCH_SIZE]
    
    # Bóc tách các thành phần
    texts = [item["text"] for item in batch]
    ids = [item["chunk_id"] for item in batch]
    
    # ChromaDB yêu cầu metadata là dạng dict với các value là chuỗi, số hoặc boolean
    metadatas = [
        {
            "source": item["source"],
            "article_id": item["article_id"],
        } 
        for item in batch
    ]
    
    # Sinh vectors
    # bge-m3 xử lý rất tốt, normalize_embeddings=True là bắt buộc để dùng cosine similarity
    embeddings = model.encode(texts, normalize_embeddings=True).tolist()
    
    # Đẩy vào ChromaDB
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=texts
    )

# ==========================================
# 4. KIỂM TRA KẾT QUẢ
# ==========================================
print("\n" + "="*80)
print("🎉 HOÀN TẤT GIAI ĐOẠN 3!")
print(f"✅ Đã lưu thành công {collection.count()} vectors vào collection '{COLLECTION_NAME}'.")
print(f"📂 Dữ liệu nhị phân được lưu an toàn tại: {DB_PATH}")
print("="*80)
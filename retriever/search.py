"""Khởi tạo ChromaDB (BGE-M3) + BM25 index, và các hàm tìm kiếm theo từng luồng."""
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
from pyvi import ViTokenizer
from rank_bm25 import BM25Okapi

from . import config


class SearchIndex:
    """Đóng gói ChromaDB collection + BM25 index, khởi tạo 1 lần và tái sử dụng
    xuyên suốt vòng đời app (tránh build lại BM25 mỗi lần gọi)."""

    def __init__(self):
        self.client = chromadb.PersistentClient(path=config.VECTOR_DB_PATH)
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.EMBEDDING_MODEL_NAME,
            normalize_embeddings=True,  # bắt buộc để công thức L2->cosine đúng
        )
        self.collection = self.client.get_collection(name=config.COLLECTION_NAME)

        self._check_normalization()

        all_docs = self.collection.get()
        self.chunks_goc = all_docs["documents"]
        self.chunk_ids = all_docs["ids"]
        self.id_to_text = dict(zip(self.chunk_ids, self.chunks_goc))

        tokenized_corpus = [ViTokenizer.tokenize(doc).split(" ") for doc in self.chunks_goc]
        self.bm25 = BM25Okapi(tokenized_corpus, k1=config.BM25_K1, b=config.BM25_B)

    def _check_normalization(self):
        """In cảnh báo nếu vector trong collection có vẻ chưa được normalize —
        vì công thức chuyển L2 -> cosine chỉ đúng khi vector là unit vector."""
        sample = self.collection.get(limit=1, include=["embeddings"])
        if len(sample["embeddings"]) > 0:
            norm = np.linalg.norm(sample["embeddings"][0])
            if abs(norm - 1.0) > 0.01:
                print(f"⚠️ [SearchIndex] Vector mẫu trong collection có norm={norm:.4f} "
                      f"(khác 1.0) — cần re-index với normalize_embeddings=True để "
                      f"công thức cosine chính xác.")

    def bm25_search(self, query: str, pool_size: int):
        tokenized_query = ViTokenizer.tokenize(query).split(" ")
        scores_all = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(
            range(len(scores_all)), key=lambda i: scores_all[i], reverse=True
        )[:pool_size]
        ids = [self.chunk_ids[i] for i in top_indices]
        scores = [scores_all[i] for i in top_indices]
        return ids, scores

    def vector_search(self, query: str, pool_size: int):
        query_vector = self.emb_fn([query])[0]
        results = self.collection.query(query_embeddings=[query_vector], n_results=pool_size)
        ids = results["ids"][0]
        cosines = [1 - (dist / 2) for dist in results["distances"][0]]
        return ids, cosines

    def get_text(self, doc_id: str) -> str:
        return self.id_to_text.get(doc_id, "")
from . import config
from .search import SearchIndex
from .llm import QwenLLM
from .fusion import wrrf_fusion_multi

class RAGPipeline:
    def __init__(self):
        print("⏳ Đang khởi tạo SearchIndex (ChromaDB + BM25)...")
        self.index = SearchIndex()
        print("⏳ Đang khởi tạo QwenLLM...")
        self.llm = QwenLLM()
        print("✅ RAGPipeline đã sẵn sàng.")

    def retrieve(self, query: str, pool_size: int = None, final_top_k: int = None) -> dict:
        """Chạy retrieval + fusion (Chỉ giữ 2 luồng: BM25 và Vector)."""
        pool_size = pool_size or config.DEFAULT_POOL_SIZE
        final_top_k = final_top_k or config.DEFAULT_FINAL_TOP_K

        bm25_orig_ids, bm25_orig_scores = self.index.bm25_search(query, pool_size)
        vec_orig_ids, vec_orig_scores = self.index.vector_search(query, pool_size)

        streams = [
            (bm25_orig_ids, bm25_orig_scores),
            (vec_orig_ids, vec_orig_scores),
        ]
        weights = [1.0, 1.0]

        fused = wrrf_fusion_multi(streams, k=config.WRRF_K, weights=weights)
        top_ids = [doc_id for doc_id, _ in fused[:final_top_k]]
        top_scores = [score for _, score in fused[:final_top_k]]
        top_docs = [self.index.get_text(doc_id) for doc_id in top_ids]

        return {
            "top_ids": top_ids,
            "top_scores": top_scores,
            "top_docs": top_docs,
        }

    def answer(self, query: str) -> dict:
        """Chạy toàn bộ pipeline: retrieval + fusion + CoT sinh câu trả lời."""
        retrieval = self.retrieve(query)
        answer_text = self.llm.generate_answer(query, retrieval["top_docs"])
        return {
            "answer": answer_text,
            "top_ids": retrieval["top_ids"],
            "top_scores": retrieval["top_scores"],
            "top_docs": retrieval["top_docs"],
        }
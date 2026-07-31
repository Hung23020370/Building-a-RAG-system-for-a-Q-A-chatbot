"""Pipeline cấp cao: gộp SearchIndex (BM25 + BGE-M3) + fusion (WRRF) + QwenLLM
thành 1 điểm truy cập duy nhất (RAGPipeline) để app.py sử dụng."""
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

    def retrieve(self, query: str, use_rewrite: bool = True,
                 pool_size: int = None, final_top_k: int = None,
                 rewrite_weight: float = None) -> dict:
        """Chạy retrieval + fusion.
        use_rewrite=False -> 2 luồng (BM25 gốc + BGE-M3 gốc)
        use_rewrite=True  -> 4 luồng (BM25 gốc + BM25 rewrite + BGE-M3 gốc + BGE-M3 rewrite)
        Cùng 1 câu rewrite được dùng cho cả BM25 và BGE-M3.
        """
        pool_size = pool_size or config.DEFAULT_POOL_SIZE
        final_top_k = final_top_k or config.DEFAULT_FINAL_TOP_K
        rewrite_weight = rewrite_weight if rewrite_weight is not None else config.DEFAULT_REWRITE_WEIGHT

        bm25_orig_ids, bm25_orig_scores = self.index.bm25_search(query, pool_size)
        vec_orig_ids, vec_orig_scores = self.index.vector_search(query, pool_size)

        streams = [
            (bm25_orig_ids, bm25_orig_scores),
            (vec_orig_ids, vec_orig_scores),
        ]
        weights = [1.0, 1.0]
        rewritten_query = None

        if use_rewrite:
            rewritten_query = self.llm.rewrite_query(query)

            bm25_rw_ids, bm25_rw_scores = self.index.bm25_search(rewritten_query, pool_size)
            streams.append((bm25_rw_ids, bm25_rw_scores))
            weights.append(rewrite_weight)

            vec_rw_ids, vec_rw_scores = self.index.vector_search(rewritten_query, pool_size)
            streams.append((vec_rw_ids, vec_rw_scores))
            weights.append(rewrite_weight)

        fused = wrrf_fusion_multi(streams, k=config.WRRF_K, weights=weights)
        top_ids = [doc_id for doc_id, _ in fused[:final_top_k]]
        top_scores = [score for _, score in fused[:final_top_k]]
        top_docs = [self.index.get_text(doc_id) for doc_id in top_ids]

        return {
            "rewritten_query": rewritten_query,
            "top_ids": top_ids,
            "top_scores": top_scores,
            "top_docs": top_docs,
        }

    def answer(self, query: str, use_rewrite: bool = True) -> dict:
        """Chạy toàn bộ pipeline: retrieval + fusion + sinh câu trả lời.
        Trả về dict gồm câu trả lời và thông tin retrieval (để hiển thị debug trên UI)."""
        retrieval = self.retrieve(query, use_rewrite=use_rewrite)
        answer_text = self.llm.generate_answer(query, retrieval["top_docs"])
        return {
            "answer": answer_text,
            "rewritten_query": retrieval["rewritten_query"],
            "top_ids": retrieval["top_ids"],
            "top_scores": retrieval["top_scores"],
            "top_docs": retrieval["top_docs"],
        }
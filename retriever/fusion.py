"""Các hàm kết hợp (fusion) kết quả từ nhiều luồng retrieval khác nhau
(BM25 gốc, BM25 rewrite, BGE-M3...) thành 1 bảng xếp hạng duy nhất."""


def min_max_normalize(scores):
    """Chuẩn hóa 1 mảng điểm số về khoảng [0, 1]."""
    if not scores:
        return []
    min_s, max_s = min(scores), max(scores)
    if min_s == max_s:
        return [1.0] * len(scores)
    return [(s - min_s) / (max_s - min_s) for s in scores]


def wrrf_fusion_multi(streams, k=60, weights=None):
    """Kết hợp N luồng retrieval bằng WRRF (Weighted Reciprocal Rank Fusion
    với điểm tin cậy chuẩn hóa min-max).

    Args:
        streams: list các tuple (ids, scores) — mỗi tuple là kết quả của 1 luồng.
        k: hằng số RRF (mặc định 60, theo chuẩn RRF gốc).
        weights: trọng số cho từng luồng, mặc định bằng nhau (1.0).

    Returns:
        list các tuple (doc_id, fused_score) đã sắp xếp giảm dần theo điểm.
    """
    if weights is None:
        weights = [1.0] * len(streams)
    assert len(weights) == len(streams), "Số trọng số phải khớp số luồng"

    hybrid_scores = {}
    for (ids, scores), w in zip(streams, weights):
        c_scores = min_max_normalize(scores)
        for rank, (doc_id, conf) in enumerate(zip(ids, c_scores), start=1):
            hybrid_scores[doc_id] = hybrid_scores.get(doc_id, 0.0) + w * (conf / (k + rank))

    return sorted(hybrid_scores.items(), key=lambda item: item[1], reverse=True)
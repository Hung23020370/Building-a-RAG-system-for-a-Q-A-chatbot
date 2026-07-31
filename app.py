"""Giao diện Chatbot chính — tra cứu quy chế/thủ tục hành chính - đào tạo.
Chạy bằng lệnh: streamlit run app.py
"""
import time
import streamlit as st

from retriever import RAGPipeline

st.set_page_config(
    page_title="Trợ lý tra cứu quy chế UET",
    page_icon="🎓",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_pipeline() -> RAGPipeline:
    """Load model + index 1 lần duy nhất, giữ lại giữa các lần rerun của Streamlit."""
    return RAGPipeline()


def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Cấu hình")
        use_rewrite = st.toggle(
            "Bật Query Rewriting",
            value=True,
            help="Viết lại câu hỏi theo thuật ngữ hành chính chuẩn trước khi tìm kiếm BM25.",
        )
        show_sources = st.toggle(
            "Hiển thị tài liệu tham khảo",
            value=True,
            help="Hiện các đoạn văn bản đã được truy xuất để tạo câu trả lời.",
        )
        st.divider()
        if st.button("🗑️ Xóa lịch sử hội thoại", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        st.divider()
        st.caption(
            "Trợ lý chỉ trả lời dựa trên văn bản quy chế/thủ tục hành chính - đào tạo "
            "của trường. Nếu câu hỏi ngoài phạm vi, trợ lý sẽ nêu rõ không đủ dữ liệu."
        )
        return use_rewrite, show_sources


def render_sources(top_ids, top_scores, top_docs):
    with st.expander(f"📚 {len(top_ids)} tài liệu tham khảo"):
        for rank, (doc_id, score, text) in enumerate(zip(top_ids, top_scores, top_docs), 1):
            st.markdown(f"**Top {rank}** · điểm: `{score:.5f}` · mã: `{doc_id}`")
            st.text(text)
            if rank < len(top_ids):
                st.markdown("---")


def main():
    st.title("🎓 Trợ lý tra cứu quy chế & thủ tục hành chính")
    st.caption("Hỏi về học phí, ký túc xá, học vụ, tốt nghiệp, thủ tục hành chính...")

    use_rewrite, show_sources = render_sidebar()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.spinner("Đang khởi động mô hình lần đầu (có thể mất vài phút)..."):
        pipeline = load_pipeline()

    # Hiển thị lại lịch sử hội thoại
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources") and show_sources:
                render_sources(*msg["sources"])

    query = st.chat_input("Nhập câu hỏi của bạn...")
    if not query:
        return

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm và tổng hợp câu trả lời..."):
            t0 = time.time()
            result = pipeline.answer(query, use_rewrite=use_rewrite)
            elapsed = time.time() - t0

        st.markdown(result["answer"])
        st.caption(f"⏱️ {elapsed:.1f}s"
                   + (f" · Câu rewrite: _{result['rewritten_query']}_" if result["rewritten_query"] else ""))

        sources = (result["top_ids"], result["top_scores"], result["top_docs"])
        if show_sources:
            render_sources(*sources)

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": sources,
    })


if __name__ == "__main__":
    main()

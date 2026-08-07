import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import config

# =====================================================================
# HẰNG SỐ (Xử lý lỗi Magic String)
# =====================================================================
FALLBACK_MSG = "Không có thông tin"
REJECTION_MSG = "Tôi không có đủ dữ liệu để trả lời đầy đủ câu hỏi này."

# =====================================================================
# 1. HỆ THỐNG PROMPT TỪ Ý TƯỞNG COT
# =====================================================================

EXTRACTION_SYSTEM_PROMPT = f"""Bạn là chuyên gia trích xuất thông tin.
Nhiệm vụ: Đọc NỘI DUNG và tìm TẤT CẢ các thông tin liên quan đến CÂU HỎI.

QUY TẮC BẮT BUỘC: Bạn phải xuất ra ĐÚNG 2 thẻ <nhap> và <trich_xuat> theo định dạng dưới đây. Tuyệt đối không chép lại lời hướng dẫn của tôi.

ĐỊNH DẠNG ĐẦU RA YÊU CẦU:
<nhap>
[TỰ BẠN gạch đầu dòng liệt kê tất cả các địa điểm, trường hợp, con số, hoặc điều kiện mà bạn nhìn thấy trong NỘI DUNG NHƯNG KHÔNG ĐƯỢC TRÙNG NHAU. Không được bỏ sót.]
</nhap>

<trich_xuat>
[Chép lại Y NGUYÊN các câu/đoạn văn bản trong phần NỘI DUNG có chứa những ý mà bạn vừa liệt kê ở trên. Không tự bịa thêm chữ. Các câu văn được liệt kê phải là câu văn hoàn chỉnh trong TÀI LIỆU, kết thúc bằng dấu chấm]
</trich_xuat>

Nếu NỘI DUNG không có thông tin, hãy ghi đúng 1 câu: "{FALLBACK_MSG}".
"""

ANSWER_SYSTEM_PROMPT = f"""Bạn là trợ lý ảo của trường Đại học Công Nghệ (UET - VNU).
NHIỆM VỤ: Trả lời câu hỏi CHỈ dựa trên NỘI DUNG được cung cấp, gộp TẤT CẢ các ý trong NỘI DUNG có liên quan đến câu hỏi vào một câu trả lời duy nhất. Không được bỏ sót ý nào liên quan.
Nếu câu hỏi đã đủ rõ nhưng NỘI DUNG không chứa thông tin giải quyết, hoặc nằm ngoài phạm vi, hoặc không chắc chắn câu trả lời: CHỈ trả lời đúng 1 câu duy nhất:
"{REJECTION_MSG}"
Xuất ra ĐÚNG 1 thẻ theo định dạng:
<tra_loi>
(câu trả lời hoàn chỉnh)
</tra_loi>
"""

REVIEW_SYSTEM_PROMPT = """Bạn đang rà soát lại một câu trả lời để đảm bảo không bỏ sót ý liên quan.

Cho: CÂU HỎI, CÂU TRẢ LỜI HIỆN TẠI, và DANH SÁCH CÂU CÓ THỂ BỊ THIẾU (trích từ nguồn gốc).

Với từng câu trong danh sách, xét xem nó có thực sự liên quan đến CÂU HỎI hay không.
- Nếu có ít nhất 1 câu liên quan mà CÂU TRẢ LỜI HIỆN TẠI chưa đề cập, hãy viết lại toàn bộ câu trả lời (giữ nguyên nội dung cũ, bổ sung thêm ý còn thiếu).
- Nếu không có câu nào trong danh sách thực sự liên quan đến câu hỏi, giữ nguyên câu trả lời cũ.

Xuất ra ĐÚNG 1 thẻ:
<tra_loi>
(câu trả lời cuối cùng, đầy đủ)
</tra_loi>
"""

# =====================================================================
# 2. CÁC HÀM XỬ LÝ TEXT & COVERAGE
# =====================================================================

def _build_chinese_bad_words_ids(tokenizer):
    bad_ids = []
    for token_id in range(len(tokenizer)):
        token_str = tokenizer.decode([token_id])
        if re.search(r"[\u4e00-\u9fff]", token_str):
            bad_ids.append([token_id])
    return bad_ids

def clean_output(text: str) -> str:
    match = re.search(r"[\u4e00-\u9fff]", text)
    if match:
        text = text[: match.start()].rstrip()
    return text

def _extract_tag(text: str, tag: str):
    match = re.search(rf"<{tag}>\s*(.*?)(?:</{tag}>|$)", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None

def _extract_key_entities(sentence: str) -> set:
    """Đã Fix Regex: Hỗ trợ bắt cả Acronym (từ viết hoa toàn bộ/chứa số như UET, GPA, BGE-M3) 
    và các cụm danh từ riêng viết hoa chữ cái đầu."""
    # Bắt từ viết hoa toàn bộ (vd: UET, BGE-M3) hoặc chuỗi tên riêng nhiều từ
    proper_nouns = re.findall(
        r'\b[A-ZĐ][A-ZĐ0-9\-]+\b|'
        r'(?:[A-ZĐ][a-zàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹ]*\s?){2,}',
        sentence,
    )
    numbers = re.findall(r'\d+', sentence)
    entities = {p.strip() for p in proper_nouns if len(p.strip()) > 2} | set(numbers)
    return entities

def check_coverage(trich_xuat: str, tra_loi: str) -> list:
    """Đã Fix Regex Tách câu: Không tách ở dấu chấm của số thập phân, tiền tệ hay ngày tháng."""
    # Look-behind và Look-ahead: Tách nếu dấu '.' không bị kẹp giữa 2 chữ số
    sentences = [s.strip() for s in re.split(r'(?<!\d)\.(?!\d)', trich_xuat) if s.strip()]
    tra_loi_lower = tra_loi.lower()
    missing = []
    for s in sentences:
        entities = _extract_key_entities(s)
        if not entities:
            continue
        if not any(e.lower() in tra_loi_lower for e in entities):
            missing.append(s)
    return missing

# =====================================================================
# 3. CLASS MODEL & PIPELINE COT
# =====================================================================

class QwenLLM:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.LLM_MODEL_NAME,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()

        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.eos_ids = list(set([self.tokenizer.eos_token_id, im_end_id]))
        self.chinese_bad_words_ids = _build_chinese_bad_words_ids(self.tokenizer)

    def _run(self, system_prompt: str, user_content: str, max_new_tokens: int) -> str:
        """Đã Fix: Sử dụng các biến tham số từ config.py thay vì hard-code"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        text_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text_prompt], return_tensors="pt").to(self.model.device)
        torch.manual_seed(config.GENERATION_SEED)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=config.GEN_TEMPERATURE,          # Lấy từ config[cite: 6]
                top_p=config.GEN_TOP_P,                      # Lấy từ config[cite: 6]
                top_k=config.GEN_TOP_K,                      # Lấy từ config[cite: 6]
                repetition_penalty=config.GEN_REPETITION_PENALTY, # Lấy từ config[cite: 6]
                do_sample=True,
                eos_token_id=self.eos_ids,
                bad_words_ids=self.chinese_bad_words_ids,
            )

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        raw_output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return clean_output(raw_output)

    def _run_with_tag(self, system_prompt: str, user_content: str, tag: str, max_new_tokens: int, max_retries: int = 1) -> str:
        raw = ""
        for attempt in range(max_retries + 1):
            raw = self._run(system_prompt, user_content, max_new_tokens=max_new_tokens)
            content = _extract_tag(raw, tag)
            if content is not None:
                return content
        return FALLBACK_MSG

    def generate_answer(self, query: str, retrieved_docs: list) -> str:
        """Chạy luồng CoT 3 bước: Trích xuất -> Tổng hợp -> Rà soát chéo"""
        ordered_docs = list(reversed(retrieved_docs))
        context_text = "\n\n---\n\n".join(ordered_docs)
        
        # BƯỚC 1: TRÍCH XUẤT
        user_content_ext = f"NỘI DUNG:\n{context_text}\n\nCÂU HỎI:\n{query}"
        extracted_text = self._run_with_tag(
            EXTRACTION_SYSTEM_PROMPT, user_content_ext, tag="trich_xuat", max_new_tokens=1024
        )

        if not extracted_text or FALLBACK_MSG.lower() in extracted_text.lower():
            return REJECTION_MSG

        # BƯỚC 2: TỔNG HỢP CÂU TRẢ LỜI
        user_content_ans = f"NỘI DUNG:\n{extracted_text}\n\nCÂU HỎI:\n{query}"
        final_answer = self._run_with_tag(
            ANSWER_SYSTEM_PROMPT, user_content_ans, tag="tra_loi", max_new_tokens=config.GEN_MAX_NEW_TOKENS_ANSWER
        )

        # Đã Fix: Bắt trọn vẹn cả trường hợp model bị lỗi thẻ/rớt mạng VÀ trường hợp model tự trả lời từ chối
        if final_answer == FALLBACK_MSG or "không có đủ dữ liệu" in final_answer.lower():
            return REJECTION_MSG

        # BƯỚC 3: CHECK COVERAGE & RÀ SOÁT
        missing_sentences = check_coverage(extracted_text, final_answer)

        if missing_sentences:
            missing_block = "\n".join(f"- {s}" for s in missing_sentences)
            user_content_review = (
                f"CÂU HỎI:\n{query}\n\n"
                f"CÂU TRẢ LỜI HIỆN TẠI:\n{final_answer}\n\n"
                f"DANH SÁCH CÂU CÓ THỂ BỊ THIẾU:\n{missing_block}"
            )
            raw_review = self._run(REVIEW_SYSTEM_PROMPT, user_content_review, max_new_tokens=config.GEN_MAX_NEW_TOKENS_ANSWER)
            reviewed_answer = _extract_tag(raw_review, "tra_loi")
            if reviewed_answer:
                final_answer = reviewed_answer

        return final_answer
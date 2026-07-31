"""Load Qwen2.5-3B-Instruct và các hàm liên quan đến sinh văn bản:
rewrite câu hỏi, sinh câu trả lời RAG, chặn token tiếng Trung, hậu xử lý."""
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pyvi import ViTokenizer

from . import config

SYSTEM_PROMPT = """Bạn là trợ lý ảo chuyên cung cấp thông tin về tuyển sinh, đào tạo và đời sống sinh viên, có nhiệm vụ
trả lời câu hỏi của người dùng một cách chính xác, ngắn gọn và chỉ sử dụng nội dung được cung cấp.
Trong quá trình phản hồi, bạn bắt buộc phải tuân thủ các quy tắc: trả lời bằng tiếng Việt, tuyệt đối
không sử dụng ký tự tiếng Trung hoặc các ngôn ngữ khác, không được nhắc tên tài liệu và không bịa đặt,
suy diễn các thông tin không có trong dữ liệu gốc. Bạn được phép tổng hợp và diễn giải lại nội dung để câu trả lời
trở nên mạch lạc, tuy nhiên, nếu câu hỏi nằm ngoài phạm vi hoặc dữ liệu không đủ chi tiết, bạn phải phản hồi
bằng cấu trúc: "Tôi không có đủ dữ liệu để trả lời đầy đủ câu hỏi này."."""

REWRITE_SYSTEM_PROMPT = """Bạn là bộ xử lý câu hỏi cho hệ thống tra cứu văn bản hành chính - đào tạo của trường đại học.
Nhiệm vụ của bạn là viết lại câu hỏi của người dùng thành 1 câu ngắn gọn, dùng ĐÚNG thuật ngữ hành chính -
học vụ chuẩn mực (ví dụ: "học phí" -> "mức thu học phí", "ktx" -> "ký túc xá", "bao nhiêu tiền" -> "mức thu"...).
QUY TẮC BẮT BUỘC:
- Chỉ chuẩn hóa từ ngữ, KHÔNG thêm bất kỳ thông tin, giả định hay chi tiết nào không có trong câu hỏi gốc.
- KHÔNG trả lời câu hỏi, KHÔNG giải thích, chỉ trả về DUY NHẤT câu đã viết lại.
- Giữ nguyên phạm vi và ý định của câu hỏi gốc.
- Tự so sánh câu trả lời được sinh ra với câu trả lời gốc, nếu câu trả lời không giống nội dung với câu gốc, hãy phản hồi lại bằng cấu trúc: "Tôi không thể viết lại câu hỏi trên".
Ví dụ:
- "Xin giấy xác nhận sinh viên ở đâu?" -> "Thủ tục xin cấp giấy xác nhận sinh viên"
  (KHÔNG suy diễn thành giấy xác nhận cư trú/tạm trú, KHÔNG đổi chủ thể của câu hỏi)"""


def _build_chinese_bad_words_ids(tokenizer):
    """Quét toàn bộ vocab, trả về id của các token chứa ký tự Hán,
    để truyền vào bad_words_ids khi generate — chặn cứng việc model
    'trôi' sang tiếng Trung khi bị repetition_penalty dồn ép."""
    bad_ids = []
    for token_id in range(len(tokenizer)):
        token_str = tokenizer.decode([token_id])
        if re.search(r"[\u4e00-\u9fff]", token_str):
            bad_ids.append([token_id])
    return bad_ids


def clean_output(text: str) -> str:
    """Lưới an toàn cuối: cắt bỏ mọi thứ từ ký tự Hán đầu tiên trở đi."""
    match = re.search(r"[\u4e00-\u9fff]", text)
    if match:
        text = text[: match.start()].rstrip()
    return text


def is_rewrite_safe(original: str, rewritten: str, min_overlap: float = 0.3) -> bool:
    """Kiểm tra câu rewrite có bị lệch nghĩa quá xa câu gốc không
    (dựa trên overlap từ vựng — bắt các trường hợp model tự suy diễn sai chủ thể)."""
    orig_words = set(ViTokenizer.tokenize(original.lower()).split())
    rw_words = set(ViTokenizer.tokenize(rewritten.lower()).split())
    if not orig_words:
        return True
    overlap_ratio = len(orig_words & rw_words) / len(orig_words)
    return overlap_ratio >= min_overlap


class QwenLLM:
    """Đóng gói model + tokenizer Qwen2.5-3B-Instruct, khởi tạo 1 lần và tái sử dụng."""

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.LLM_MODEL_NAME,
            torch_dtype=torch.bfloat16,  # đổi sang torch.float16 nếu chạy trên GPU T4
            device_map="auto",
        )
        self.model.eval()

        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.eos_ids = list(set([self.tokenizer.eos_token_id, im_end_id]))

        self.chinese_bad_words_ids = _build_chinese_bad_words_ids(self.tokenizer)

    def _run(self, system_prompt: str, user_content: str, max_new_tokens: int,
              do_sample: bool) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        text_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text_prompt], return_tensors="pt").to(self.model.device)

        torch.manual_seed(config.GENERATION_SEED)  # giữ tái lập dù đã bật sampling

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            repetition_penalty=config.GEN_REPETITION_PENALTY,
            eos_token_id=self.eos_ids,
            bad_words_ids=self.chinese_bad_words_ids,
        )
        if do_sample:
            gen_kwargs.update(
                do_sample=True,
                temperature=config.GEN_TEMPERATURE,
                top_p=config.GEN_TOP_P,
                top_k=config.GEN_TOP_K,
            )
        else:
            gen_kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)

        with torch.no_grad():
            generated_ids = self.model.generate(**model_inputs, **gen_kwargs)

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        raw_output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return clean_output(raw_output)

    def rewrite_query(self, query: str) -> str:
        """Viết lại câu hỏi theo thuật ngữ hành chính chuẩn (dùng cho luồng BM25).
        Dùng greedy vì đây là tác vụ chuẩn hóa, cần ổn định."""
        rewritten = self._run(
            REWRITE_SYSTEM_PROMPT, query,
            max_new_tokens=config.GEN_MAX_NEW_TOKENS_REWRITE, do_sample=False,
        )
        if not rewritten or len(rewritten) < 2:
            return query
        if not is_rewrite_safe(query, rewritten):
            return query
        return rewritten

    def generate_answer(self, query: str, retrieved_docs: list) -> str:
        """Sinh câu trả lời RAG. Đảo ngược thứ tự context: chunk liên quan nhất
        (fusion score cao nhất) đặt cuối, gần khối CÂU HỎI nhất — giảm hiệu ứng
        'lost in the middle'. Dùng sampling nhẹ để tránh vòng lặp dao động của greedy."""
        ordered_docs = list(reversed(retrieved_docs))
        context_text = "\n\n---\n\n".join(ordered_docs)
        user_content = f"NỘI DUNG:\n{context_text}\n\nCÂU HỎI:\n{query}"
        return self._run(
            SYSTEM_PROMPT, user_content,
            max_new_tokens=config.GEN_MAX_NEW_TOKENS_ANSWER, do_sample=True,
        )
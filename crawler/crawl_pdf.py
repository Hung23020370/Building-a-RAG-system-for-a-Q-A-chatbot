import os
import io
import re
import time
import nest_asyncio
import google.generativeai as genai
from pdf2image import convert_from_path
from llama_cloud import AsyncLlamaCloud
from dotenv import load_dotenv

nest_asyncio.apply()
load_dotenv()
# ==========================================
# 1. CẤU HÌNH API
# ==========================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")       
genai.configure(api_key=GOOGLE_API_KEY)
gemini_model = genai.GenerativeModel(model_name="gemini-3.5-flash")

LLAMA_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY") 
SAVE_DIR = "./nguon_pdf"

MAX_PAGES_PER_BATCH = 6   # số trang bảng liên tiếp tối đa gộp vào 1 lần gọi Gemini
DPI = 200

# ==========================================
# 2. PROMPT THẦN CHÚ CHO GEMINI
# ==========================================
NARRATIVE_PROMPT = """
Bạn là chuyên gia xử lý dữ liệu. Hãy đọc các hình ảnh trang tài liệu đính kèm (có thể là 1 hoặc nhiều trang liên tiếp thuộc cùng một bảng).

Yêu cầu bắt buộc:
1. TUYỆT ĐỐI KHÔNG xuất ra bảng Markdown hay HTML.
2. Nếu có bảng biểu, hãy "Văn bản hóa" (Table-to-Text) thành các gạch đầu dòng trần thuật.
3. Rã tất cả các ô bị gộp, đảm bảo mỗi câu đều có đủ ngữ cảnh (Đối tượng + Ý nghĩa cột + Số liệu).
4. Diễn đạt lại tự nhiên bằng lời văn của bạn, không cần giữ nguyên cấu trúc câu gốc — chỉ cần giữ đúng 100% số liệu.
5. Nếu có văn bản bình thường (không phải bảng), hãy giữ nguyên nội dung.
6. KHÔNG in số trang, header, footer, hoặc ký hiệu phân trang nào.
7. Nếu các trang gửi cùng lúc là phần tiếp nối của cùng 1 bảng, hãy tự nối mạch cột chính xác xuyên suốt các trang đó.

QUAN TRỌNG - ĐỊNH DẠNG OUTPUT:
8. TUYỆT ĐỐI KHÔNG viết câu mở đầu/giới thiệu kiểu "Dưới đây là...", "Sau đây là nội dung...", "Tôi đã chuyển đổi...".
9. TUYỆT ĐỐI KHÔNG viết câu kết luận/tổng kết ở cuối như "Trên đây là toàn bộ nội dung...".
10. Trả lời BẮT ĐẦU NGAY bằng nội dung thực tế, không có bất kỳ câu meta nào bao quanh.
"""

# ==========================================
# 3. CÁC HÀM TIỆN ÍCH
# ==========================================

def pil_to_gemini_part(img):
    img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return {"mime_type": "image/png", "data": buf.getvalue()}


def group_consecutive_table_pages(has_table_list, max_batch=MAX_PAGES_PER_BATCH):
    """
    Input: list bool, has_table_list[i] = True nếu trang i (0-indexed) có bảng.
    Output: list các tuple (start_idx, end_idx) - các trang bảng liên tiếp, đã gộp thành cụm,
            và tự chia nhỏ nếu cụm dài quá max_batch.
    """
    groups = []
    start = None
    for i, is_table in enumerate(has_table_list):
        if is_table and start is None:
            start = i
        elif not is_table and start is not None:
            groups.append((start, i - 1))
            start = None
    if start is not None:
        groups.append((start, len(has_table_list) - 1))

    final_groups = []
    for start, end in groups:
        length = end - start + 1
        if length <= max_batch:
            final_groups.append((start, end))
        else:
            s = start
            while s <= end:
                e = min(s + max_batch - 1, end)
                final_groups.append((s, e))
                s = e + 1
    return final_groups


def call_gemini_safe(image_parts, max_retries=3):
    """Gọi Gemini, tự chờ & thử lại khi gặp lỗi 429 (hết quota)."""
    for attempt in range(max_retries):
        try:
            response = gemini_model.generate_content(
                [NARRATIVE_PROMPT] + image_parts,
                generation_config=genai.types.GenerationConfig(temperature=0.6)
            )
            candidate = response.candidates[0] if response.candidates else None
            if candidate and candidate.finish_reason == 4:
                raise ValueError("RECITATION_BLOCKED")
            return response.text.strip() if response.text else None
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "quota" in err_str.lower():
                match = re.search(r"retry in ([\d.]+)s", err_str)
                wait_time = float(match.group(1)) + 2 if match else 60
                print(f"      ⏳ Hết quota, chờ {wait_time:.0f}s rồi thử lại (lần {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
                continue
            elif "RECITATION" in err_str and len(image_parts) > 1:
                print(f"      ⚠️ Bị chặn RECITATION, chia đôi batch ({len(image_parts)} ảnh) thử lại...")
                mid = len(image_parts) // 2
                part1 = call_gemini_safe(image_parts[:mid], max_retries - 1)
                time.sleep(1)
                part2 = call_gemini_safe(image_parts[mid:], max_retries - 1)
                if part1 and part2:
                    return part1 + "\n\n" + part2
                return part1 or part2
            else:
                print(f"      ❌ Lỗi Gemini không xác định: {e}")
                return None
    print(f"      ❌ Vẫn lỗi sau {max_retries} lần thử.")
    return None


# ==========================================
# 4. HÀM XỬ LÝ 1 FILE PDF
# ==========================================
async def process_single_pdf(client, pdf_path, output_md_path):
    print(f"\n🚀 Đang xử lý: {os.path.basename(pdf_path)}")

    try:
        file_obj = await client.files.create(file=pdf_path, purpose="parse")
        result = await client.parsing.parse(
            file_id=file_obj.id,
            tier="agentic",
            version="latest",
            expand=["markdown"],
        )

        md_pages = result.markdown.pages
        total_pages = len(md_pages)

        if total_pages == 0:
            print(f"  ❌ LlamaParse trả về rỗng — bỏ qua, giữ nguyên PDF gốc.")
            return

        print(f"  📄 Tổng số trang: {total_pages}")

        # Bước 1: quét trước toàn bộ, xác định trang nào có bảng
        has_table_list = ["<table" in md_pages[i].markdown.lower() for i in range(total_pages)]
        table_groups = group_consecutive_table_pages(has_table_list)
        print(f"    🔍 Phát hiện {len(table_groups)} cụm bảng: {table_groups}")

        # Bước 2: build nội dung theo đúng thứ tự trang gốc
        final_parts = []
        i = 0
        while i < total_pages:
            group = next((g for g in table_groups if g[0] == i), None)

            if group:
                start, end = group
                page_range = f"{start + 1}-{end + 1}"
                print(f"    ⚠️ Cụm bảng trang {page_range}! Gửi {end - start + 1} ảnh cho Gemini...")

                try:
                    images = convert_from_path(pdf_path, dpi=DPI, first_page=start + 1, last_page=end + 1)
                    image_parts = [pil_to_gemini_part(img) for img in images]
                    text = call_gemini_safe(image_parts)

                    if text:
                        final_parts.append(text)
                        print(f"      ✔️ Gemini xử lý xong cụm trang {page_range}.")
                    else:
                        print(f"      ⚠️ Gemini không trả kết quả → fallback markdown gốc.")
                        for p in range(start, end + 1):
                            final_parts.append(md_pages[p].markdown)
                except Exception as gem_err:
                    print(f"      ⚠️ Lỗi xử lý cụm trang {page_range}: {gem_err} → fallback markdown gốc")
                    for p in range(start, end + 1):
                        final_parts.append(md_pages[p].markdown)

                time.sleep(1)
                i = end + 1
            else:
                final_parts.append(md_pages[i].markdown)
                print(f"    ✔️ Trang {i + 1} sạch sẽ (LlamaParse lấy).")
                i += 1

        if not final_parts:
            print(f"  ❌ Không thu được nội dung — không ghi file, giữ nguyên PDF gốc.")
            return

        final_text = "\n\n".join(final_parts)
        with open(output_md_path, "w", encoding="utf-8") as f:
            f.write(final_text)

        print(f"  ✅ Hoàn tất lưu file: {os.path.basename(output_md_path)}")

    except Exception as e:
        print(f"  ❌ Lỗi khi xử lý {os.path.basename(pdf_path)}: {e}")
        print(f"  ⚠️ PDF gốc được GIỮ NGUYÊN để xử lý lại sau.")


# ==========================================
# 5. CHẠY CHO TOÀN BỘ THƯ MỤC
# ==========================================
async def main():
    client = AsyncLlamaCloud(api_key=LLAMA_API_KEY)

    pdf_files = [f for f in os.listdir(SAVE_DIR) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"⚠️ Không tìm thấy file PDF nào trong {SAVE_DIR}")
        return

    print(f"🔥 Tìm thấy {len(pdf_files)} file PDF. Bắt đầu xử lý...")

    for filename in pdf_files:
        pdf_path = os.path.join(SAVE_DIR, filename)
        md_path = os.path.join(SAVE_DIR, filename.replace(".pdf", ".md"))
        await process_single_pdf(client, pdf_path, md_path)

    print("\n🎉 HOÀN TẤT TOÀN BỘ QUÁ TRÌNH!")

await main()
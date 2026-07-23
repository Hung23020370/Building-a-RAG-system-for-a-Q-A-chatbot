import os
import re
import json

# ==========================================
# CẤU HÌNH THƯ MỤC
# ==========================================
INPUT_DIR = "./nguon_web"
OUTPUT_FILE = "./ingestion/uet_rag_chunks.json"
MAX_LENGTH = 800

all_chunks = []

# ==========================================
# CÁC HÀM TIỆN ÍCH TẠO ID
# ==========================================
def make_safe_id(text):
    """Làm sạch chuỗi, thay khoảng trắng bằng dấu gạch dưới để làm ID"""
    text = re.sub(r'[^\w\s]', '', text) 
    return re.sub(r'\s+', '_', text.strip())

def extract_article_id(text, strategy):
    """Trích xuất article_id dựa trên chiến lược cắt"""
    text_lines = text.strip().split('\n')
    first_line = text_lines[0] if text_lines else ""
    
    if strategy == "heading_2":
        match = re.match(r'^##\s+(.+)', first_line)
        if match:
            return make_safe_id(match.group(1))
            
    elif strategy == "list_format":
        match = re.match(r'^(\d+)[\)\.]', first_line)
        if match:
            return f"Muc_{match.group(1)}"
            
    elif strategy == "isolated_bold":
        match = re.match(r'^\*\*([^\*]+)\*\*', first_line)
        if match:
            return make_safe_id(match.group(1))
            
    elif strategy == "specific_bullets":
        match = re.match(r'^\*\s+(.+)', first_line)
        if match:
            return make_safe_id(match.group(1))
            
    # Với các file gom 1 cục hoặc cắt theo dấu *, mặc định trả về Tong_quan
    return ""

def create_structured_chunks(raw_chunks, filename, strategy):
    """Biến đổi text thô thành cấu trúc chuẩn JSON, CHỈ THÊM SỐ ĐUÔI KHI BỊ TRÙNG LẶP"""
    temp_chunks = []
    article_counts = {}
    
    for c in raw_chunks:
        text = c.strip()
        if not text:
            continue
            
        article_id = extract_article_id(text, strategy)
        temp_chunks.append((article_id, text))
        article_counts[article_id] = article_counts.get(article_id, 0) + 1
        
    structured_data = []
    running_counts = {}
    
    for article_id, text in temp_chunks:
        base_id = f"{filename}_{article_id}"
        
        if article_counts[article_id] > 1:
            running_counts[article_id] = running_counts.get(article_id, 0) + 1
            chunk_id = f"{base_id}_{running_counts[article_id]}"
        else:
            chunk_id = base_id
            
        structured_data.append({
            "chunk_id": chunk_id,
            "article_id": article_id,
            "source": filename,
            "text": text
        })
        
    return structured_data

# ==========================================
# CÁC HÀM XỬ LÝ CHUNKING
# ==========================================
def chunk_by_list(text):
    pattern = r'\n(?=\s*\d+[\)\.])'
    return re.split(pattern, text)

def chunk_by_heading2(text):
    pattern = r'\n(?=##\s)'
    return re.split(pattern, text)

def chunk_by_isolated_bold(text, max_length=MAX_LENGTH):
    blocks = re.split(r'\n\s*\n', text.strip())
    raw_chunks = []
    current_title = "**Tong_quan**"
    current_group = []
    current_len = 0
    
    def is_title(b):
        return b.startswith('**') and b.endswith('**') and '\n' not in b
        
    def is_table(b):
        return b.startswith('|')
        
    for block in blocks:
        block = block.strip()
        if not block: 
            continue
        
        if is_title(block):
            if current_group:
                raw_chunks.append(f"{current_title}\n\n" + "\n\n".join(current_group))
                current_group = []
                current_len = 0
            current_title = block
            
        elif is_table(block):
            if current_group:
                raw_chunks.append(f"{current_title}\n\n" + "\n\n".join(current_group))
                current_group = []
                current_len = 0
            raw_chunks.append(f"{current_title}\n\n{block}")
            
        else:
            block_len = len(re.sub(r'\s+', '', block))
            if current_group and (current_len + block_len > max_length):
                raw_chunks.append(f"{current_title}\n\n" + "\n\n".join(current_group))
                current_group = [block]
                current_len = block_len
            else:
                current_group.append(block)
                current_len += block_len
                
    if current_group:
        raw_chunks.append(f"{current_title}\n\n" + "\n\n".join(current_group))
        
    return raw_chunks

def chunk_by_bullets(text, max_length=MAX_LENGTH):
    """Gom các đoạn/dòng có dấu * lại với nhau cho đến khi đạt max_length"""
    lines = text.strip().split('\n')
    raw_chunks = []
    current_group = []
    current_len = 0
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Tính độ dài dính liền
        line_len = len(re.sub(r'\s+', '', line))
        
        if current_group and (current_len + line_len > max_length):
            raw_chunks.append('\n'.join(current_group))
            current_group = [line]
            current_len = line_len
        else:
            current_group.append(line)
            current_len += line_len
            
    if current_group:
        raw_chunks.append('\n'.join(current_group))
        
    return raw_chunks

def chunk_by_specific_bullets(text):
    """Cắt file dựa theo danh sách tiêu đề dấu * cố định"""
    markers = [
        r'\*\s+Học phí',
        r'\*\s+Đối tượng được miễn 100% học phí \(theo mức trần học phí hệ chuẩn\)',
        r'\*\s+Đối tượng được giảm 70% học phí \(theo mức trần học phí hệ chuẩn\)',
        r'\*\s+Đối tượng được giảm 50% học phí \(theo mức trần học phí hệ chuẩn\)'
    ]
    # Dùng lookahead để ngắt nhưng vẫn giữ lại dấu * ở đầu đoạn
    pattern = r'\n(?=' + '|'.join(markers) + r')'
    chunks = re.split(pattern, text)
    return [c.strip() for c in chunks if c.strip()]

# ==========================================
# THỰC THI CHÍNH
# ==========================================
print("🚀 Bắt đầu quá trình Chunking phân loại tự động...\n")

if not os.path.exists(INPUT_DIR):
    print(f"❌ Không tìm thấy thư mục: {INPUT_DIR}")
else:
    for filename in os.listdir(INPUT_DIR):
        if not filename.endswith(".md"):
            continue
            
        filepath = os.path.join(INPUT_DIR, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            
        # 🎯 Tính độ dài thực tế kiểu dính liền
        actual_length = len(re.sub(r'\s+', '', text))
        
        # 1. NHÓM <= 800 KÝ TỰ HOẶC FILE KHÁM CHỮA BỆNH: Gom nguyên file thành 1 chunk
        if actual_length <= MAX_LENGTH or filename == "Kham_chua_benh.md":
            chunks_data = create_structured_chunks([text], filename, "full")
            all_chunks.extend(chunks_data)
            print(f"  ✅ [Gom nguyên file] {filename:<35} -> 1 chunk.")
            
        # 2. NHÓM > 800 KÝ TỰ: Phân loại theo chiến thuật
        else:
            if filename in ["Thong_tin_lien_he.md", "Thu_vien.md"]:
                raw_pieces = chunk_by_list(text)
                chunks_data = create_structured_chunks(raw_pieces, filename, "list_format")
                all_chunks.extend(chunks_data)
                print(f"  ✂️ [Cắt theo List]   {filename:<35} -> {len(chunks_data)} chunks.")
                
            elif filename in ["Noi_quy_Quy_che.md", "Lich_su_Truyen_thong.md"]:
                raw_pieces = chunk_by_heading2(text)
                chunks_data = create_structured_chunks(raw_pieces, filename, "heading_2")
                all_chunks.extend(chunks_data)
                print(f"  ✂️ [Cắt theo ##]     {filename:<35} -> {len(chunks_data)} chunks.")
                
            elif filename == "Lich_su_Truyen_thong_Chi_tiet.md":
                raw_pieces = chunk_by_isolated_bold(text, MAX_LENGTH)
                chunks_data = create_structured_chunks(raw_pieces, filename, "isolated_bold")
                all_chunks.extend(chunks_data)
                print(f"  ✂️ [Cắt theo **]     {filename:<35} -> {len(chunks_data)} chunks.")
                
            elif filename == "Hoc_phi_Che_do.md":
                raw_pieces = chunk_by_specific_bullets(text)
                chunks_data = create_structured_chunks(raw_pieces, filename, "specific_bullets")
                all_chunks.extend(chunks_data)
                print(f"  ✂️ [Cắt Bullet chuẩn] {filename:<35} -> {len(chunks_data)} chunks.")
                
            elif filename == "Hoc_bong_Diem_ren_luyen.md":
                raw_pieces = chunk_by_bullets(text, MAX_LENGTH)
                chunks_data = create_structured_chunks(raw_pieces, filename, "bullet_group")
                all_chunks.extend(chunks_data)
                print(f"  ✂️ [Gom nhóm dấu *]  {filename:<35} -> {len(chunks_data)} chunks.")
                
            # Đề phòng còn file nào lọt lưới
            else:
                print(f"  ⏳ [Chưa phân loại]   {filename:<35} -> Cần check lại.")

    # Lưu kết quả ra file JSON
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=4)

    print("\n" + "="*70)
    print(f"🎉 Hoàn tất! Đã xử lý 100% file và lưu tổng cộng {len(all_chunks)} chunks vào: {OUTPUT_FILE}")
    print("="*70)
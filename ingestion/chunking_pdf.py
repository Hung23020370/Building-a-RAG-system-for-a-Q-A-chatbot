import os
import re
import json

# ==========================================
# CẤU HÌNH THƯ MỤC & FILE
# ==========================================
INPUT_DIR = "./nguon_pdf" 
OUTPUT_FILE = "./ingestion/uet_rag_chunks.json" 
MAX_LENGTH = 800

# ==========================================
# CÁC HÀM XỬ LÝ CHO FILE ĐẶC BIỆT (Khối *)
# ==========================================
def create_pdf_structured_chunks(raw_chunks, filename):
    """Tạo cấu trúc JSON tối giản: chunk_id là tenfile_1, article_id trống"""
    structured_data = []
    for i, text in enumerate(raw_chunks, 1):
        structured_data.append({
            "chunk_id": f"{filename}_{i}",
            "article_id": "",
            "source": filename,
            "text": text.strip()
        })
    return structured_data

def chunk_by_asterisk_blocks(text, max_length=MAX_LENGTH):
    """Cắt văn bản tại vị trí xuống dòng và bắt đầu bằng dấu *."""
    blocks = re.split(r'\n(?=\*)', text.strip())
    raw_chunks = []
    current_group = []
    current_len = 0
    
    for block in blocks:
        block = block.strip()
        if not block: continue
        block_len = len(re.sub(r'\s+', '', block))
        
        if current_group and (current_len + block_len > max_length):
            raw_chunks.append('\n\n'.join(current_group))
            current_group = [block]
            current_len = block_len
        else:
            current_group.append(block)
            current_len += block_len
            
    if current_group:
        raw_chunks.append('\n\n'.join(current_group))
    return raw_chunks

# ==========================================
# CÁC HÀM XỬ LÝ CHO FILE LUẬT (Phân cấp)
# ==========================================
def make_safe_id(text):
    text = re.sub(r'[^\w\s]', '', text) 
    return re.sub(r'\s+', '_', text.strip())

def extract_article_id(text, strategy):
    text_lines = text.strip().split('\n')
    first_line = text_lines[0].strip() if text_lines else ""
    
    if strategy == "heading_2":
        match = re.match(r'^##\s+(.+)', first_line)
        if match: return make_safe_id(match.group(1))
    elif strategy == "list_format":
        match = re.match(r'^(\d+)[\)\.]', first_line)
        if match: return f"Muc_{match.group(1)}"
    elif strategy == "isolated_bold":
        match = re.match(r'^\*\*([^\*]+)\*\*', first_line)
        if match: return make_safe_id(match.group(1))
    elif strategy == "legal_hierarchy":
        match = re.search(r'^[\#\*]*\s*Điều\s+(\d+[a-zA-Z]*)', first_line, re.IGNORECASE)
        if match: return f"Dieu_{match.group(1)}"
            
    return None 

def create_structured_chunks(raw_chunks, filename, strategy):
    """Hàm tạo JSON có kế thừa và chống trùng lặp hậu tố _1"""
    grouped_chunks = []
    current_id = "Tong_quan"
    current_cluster = []
    
    for text in raw_chunks:
        text = text.strip()
        if not text: continue
        extracted_id = extract_article_id(text, strategy)
        if extracted_id is not None:
            if current_cluster:
                grouped_chunks.append((current_id, current_cluster))
            current_id = extracted_id
            current_cluster = [text]
        else:
            current_cluster.append(text)
            
    if current_cluster:
        grouped_chunks.append((current_id, current_cluster))
        
    structured_data = []
    cluster_counts = {}
    seen_clusters = {}
    
    for cid, _ in grouped_chunks:
        cluster_counts[cid] = cluster_counts.get(cid, 0) + 1
        
    for article_id, cluster in grouped_chunks:
        seen_clusters[article_id] = seen_clusters.get(article_id, 0) + 1
        prefix = f"_{seen_clusters[article_id]}" if cluster_counts[article_id] > 1 and seen_clusters[article_id] > 1 else ""
        base_id = f"{filename}_{article_id}{prefix}"
        
        if len(cluster) == 1:
            structured_data.append({
                "chunk_id": base_id, "article_id": article_id, "source": filename, "text": cluster[0]
            })
        else:
            for i, text in enumerate(cluster, 1):
                structured_data.append({
                    "chunk_id": f"{base_id}_{i}", "article_id": article_id, "source": filename, "text": text
                })
    return structured_data

def chunk_legal_hierarchy(text, max_length=MAX_LENGTH):
    """Cắt theo 3 cấp: Điều -> Khoản -> Điểm"""
    raw_chunks = []
    blocks_dieu = re.split(r'\n(?=^\s*[\#\*]*\s*Điều\s+\d+)', text.strip(), flags=re.MULTILINE)
    
    for block_dieu in blocks_dieu:
        block_dieu = block_dieu.strip()
        if not block_dieu: continue
        len_dieu = len(re.sub(r'\s+', '', block_dieu))
        
        if len_dieu <= max_length:
            raw_chunks.append(block_dieu)
        else:
            blocks_khoan = re.split(r'\n(?=^\s*[\#\*]*\s*\d+[\.\)])', block_dieu, flags=re.MULTILINE)
            current_group = []
            current_len = 0
            
            for block_khoan in blocks_khoan:
                block_khoan = block_khoan.strip()
                if not block_khoan: continue
                len_khoan = len(re.sub(r'\s+', '', block_khoan))
                
                if len_khoan <= max_length:
                    if current_group and (current_len + len_khoan > max_length):
                        raw_chunks.append('\n\n'.join(current_group))
                        current_group = [block_khoan]
                        current_len = len_khoan
                    else:
                        current_group.append(block_khoan)
                        current_len += len_khoan
                else:
                    if current_group:
                        raw_chunks.append('\n\n'.join(current_group))
                        current_group = []
                        current_len = 0
                        
                    blocks_diem = re.split(r'\n(?=^\s*[\#\*]*\s*[a-z]\))', block_khoan, flags=re.MULTILINE)
                    sub_group = []
                    sub_len = 0
                    
                    for block_diem in blocks_diem:
                        block_diem = block_diem.strip()
                        if not block_diem: continue
                        len_diem = len(re.sub(r'\s+', '', block_diem))
                        
                        if sub_group and (sub_len + len_diem > max_length):
                            raw_chunks.append('\n\n'.join(sub_group))
                            sub_group = [block_diem]
                            sub_len = len_diem
                        else:
                            sub_group.append(block_diem)
                            sub_len += len_diem
                            
                    if sub_group:
                        raw_chunks.append('\n\n'.join(sub_group))
                        
            if current_group:
                raw_chunks.append('\n\n'.join(current_group))
    return raw_chunks

# ==========================================
# THỰC THI CHÍNH
# ==========================================
print("🚀 Bắt đầu quá trình Chunking Giai đoạn 2 (Xử lý hợp nhất)...\n")

new_chunks = []
SPECIAL_FILES = [
    "Hoc_phi_Che_do_Signed_Dinh_muc_hoc_phi_nam_hoc_25-26-2080_dhcn.md",
    "Ke_hoach_hoc_tap_2025_2026.md"
]

if not os.path.exists(INPUT_DIR):
    print(f"❌ Không tìm thấy thư mục: {INPUT_DIR}")
else:
    for filename in os.listdir(INPUT_DIR):
        if not filename.endswith(".md"):
            continue
            
        filepath = os.path.join(INPUT_DIR, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read().strip()

        # =========================================================
        # 🗑️ TIỀN XỬ LÝ: XÓA MỤC LỤC Ở CUỐI FILE QUY CHẾ
        # =========================================================
        if filename == "Noi_quy_Quy_che_Final_QC-dH-_2014_Ban-hanh-25-12-2014.md":
            # Cắt đôi văn bản ngay tại vị trí có chữ "MỤC LỤC"
            parts = re.split(r'\n\s*#+\s*MỤC LỤC', text, flags=re.IGNORECASE)
            
            # Ghi đè lại biến text, CHỈ GIỮ LẠI phần nội dung bên trên (parts[0])
            text = parts[0].strip()

        # ---------------------------------------------------------
        # NHÁNH 1: XỬ LÝ CÁC FILE ĐẶC BIỆT
        # ---------------------------------------------------------
        if filename in SPECIAL_FILES:
            raw_pieces = chunk_by_asterisk_blocks(text, MAX_LENGTH)
            chunks_data = create_pdf_structured_chunks(raw_pieces, filename)
            new_chunks.extend(chunks_data)
            print(f"  ✂️ [Cắt theo khối *] {filename[:45]}... -> {len(chunks_data)} chunks.")
            
        # ---------------------------------------------------------
        # NHÁNH 2: XỬ LÝ FILE LUẬT & PHỤ LỤC
        # ---------------------------------------------------------
        else:
            parts = re.split(r'\n(?=[\#\*]*\s*PHỤ LỤC)', text, flags=re.IGNORECASE, maxsplit=1)
            main_text = parts[0]
            appendix_text = parts[1] if len(parts) > 1 else ""
            
            # Xử lý phần chính (Luật)
            raw_pieces_main = chunk_legal_hierarchy(main_text, MAX_LENGTH)
            chunks_data_main = create_structured_chunks(raw_pieces_main, filename, "legal_hierarchy")
            new_chunks.extend(chunks_data_main)
            count_main = len(chunks_data_main)
            
            # Xử lý phần phụ lục (Cắt theo *)
            count_appx = 0
            if appendix_text:
                raw_pieces_appx = re.split(r'\n(?=^\s*\*\s+)', appendix_text.strip(), flags=re.MULTILINE)
                current_appx_group = []
                current_appx_len = 0
                appx_chunk_index = 1
                
                for piece in raw_pieces_appx:
                    piece = piece.strip()
                    if not piece: continue
                    piece_len = len(re.sub(r'\s+', '', piece))
                    
                    if current_appx_group and (current_appx_len + piece_len > MAX_LENGTH):
                        new_chunks.append({
                            "chunk_id": f"{filename}_Phu_luc_{appx_chunk_index}",
                            "article_id": "Phu_luc",
                            "source": filename,
                            "text": '\n\n'.join(current_appx_group)
                        })
                        appx_chunk_index += 1
                        current_appx_group = [piece]
                        current_appx_len = piece_len
                        count_appx += 1
                    else:
                        current_appx_group.append(piece)
                        current_appx_len += piece_len
                        
                if current_appx_group:
                    new_chunks.append({
                        "chunk_id": f"{filename}_Phu_luc_{appx_chunk_index}",
                        "article_id": "Phu_luc",
                        "source": filename,
                        "text": '\n\n'.join(current_appx_group)
                    })
                    count_appx += 1

            print(f"  ⚖️ [Cắt theo Luật]   {filename[:45]}... -> {count_main} chunks (Chính) + {count_appx} chunks (Phụ lục).")

    # ==========================================
    # LƯU VÀ NỐI VÀO FILE JSON CŨ (NẾU CÓ)
    # ==========================================
    if os.path.exists(OUTPUT_FILE):
        print(f"\n📥 Tìm thấy file cũ {OUTPUT_FILE}. Đang nạp dữ liệu cũ để nối thêm...")
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            try:
                existing_data = json.load(f)
            except json.JSONDecodeError:
                existing_data = [] 
        
        existing_data.extend(new_chunks)
        final_data_to_save = existing_data
    else:
        final_data_to_save = new_chunks

    # Ghi lại toàn bộ dữ liệu ra file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_data_to_save, f, ensure_ascii=False, indent=4)

    print("\n" + "="*90)
    print(f"🎉 Hoàn tất! Đã xử lý {len(new_chunks)} chunks mới.")
    print(f"📦 Tổng số lượng chunks trong file '{OUTPUT_FILE}' hiện tại là: {len(final_data_to_save)} chunks.")
    print("="*90)
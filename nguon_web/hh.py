import os
import re

# ==========================================
# CẤU HÌNH ĐẦU VÀO
# ==========================================
INPUT_DIR = "./"
MAX_LENGTH = 800

def count_valid_files(directory, max_len):
    if not os.path.exists(directory):
        print(f"❌ Không tìm thấy thư mục: {directory}")
        return

    valid_count = 0
    total_count = 0
    
    print(f"🚀 Bắt đầu quét các file có độ dài thực <= {max_len} (bỏ qua khoảng trắng)...\n")

    for filename in os.listdir(directory):
        if not filename.endswith(".md"):
            continue
            
        total_count += 1
        filepath = os.path.join(directory, filename)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # 🎯 BÍ QUYẾT Ở ĐÂY: Xóa toàn bộ khoảng trắng, tab, xuống dòng
        text_no_spaces = re.sub(r'\s+', '', text)
        
        # Đếm độ dài "thực" của file
        actual_length = len(text_no_spaces)
        
        if actual_length <= max_len:
            valid_count += 1
            print(f"  ✅ [Hợp lệ] {filename} (Độ dài thực: {actual_length} ký tự)")
        else:
            print(f"  ❌ [Vượt ngưỡng] {filename} (Độ dài thực: {actual_length} ký tự)")
            
    # ==========================================
    # BẢNG TỔNG KẾT
    # ==========================================
    print("\n" + "="*50)
    print(f"📊 TỔNG KẾT:")
    print(f" - Tổng số file Markdown đã quét: {total_count}")
    print(f" - Số file có độ dài (không khoảng trắng) <= {max_len}: {valid_count}")
    print("="*50)

count_valid_files(INPUT_DIR, MAX_LENGTH)
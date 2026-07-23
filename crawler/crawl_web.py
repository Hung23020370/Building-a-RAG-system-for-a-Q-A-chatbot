import os
import requests
import shutil
import re
import urllib.parse
from bs4 import BeautifulSoup
from markdownify import markdownify as md

# ==========================================
# CÀI ĐẶT THƯ MỤC
# ==========================================
SAVE_DIR = "./nguon_web"
os.makedirs(SAVE_DIR, exist_ok=True)
BASE_URL = "https://handbook.uet.vnu.edu.vn"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

# 🛠️ HÀM MỚI: DỌN DẸP LỖI XUỐNG DÒNG CỦA MARKDOWN
def clean_markdown_format(raw_md):
    # 1. Ghép các dòng bị ngắt có khoảng trắng thụt lề (không phải là thẻ list *, #, hoặc số)
    # Ví dụ: "\n      gồm người có công" -> " gồm người có công"
    cleaned = re.sub(r'\n[ \t]+(?![ \t]*(\*|\#|\-|\d+\.))', ' ', raw_md)
    
    # 2. Ghép các dòng ngắt đơn (1 \n) nằm giữa câu văn
    cleaned = re.sub(r'(?<!\n)\n(?!\n|[ \t]*(\*|\#|\-|\d+\.))', ' ', cleaned)
    
    # 3. Dọn dẹp khoảng trắng dư thừa do nối chuỗi và giới hạn khoảng trống tối đa 2 dòng
    cleaned = re.sub(r'[ \t]{2,}', ' ', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    
    return cleaned.strip()

# ==========================================
# 1. XỬ LÝ LUỒNG 1: SHALLOW CRAWL (NHÓM 1)
# ==========================================
flow_1_links = {
    "Kham_chua_benh": "/Khám chữa bệnh/",
    "Ky_tuc_xa": "/Ký túc xá/",
    "Thu_vien": "/Thư viện/",
    "Thong_tin_lien_he": "/Thông tin liên hệ/",
    "Hoat_dong_Doan_Hoi": "/Hoạt động Đoàn-Hội/",
    "Thuc_tap_Thuc_te": "/Thực tập thực tế/",
    "Trao_doi_sinh_vien": "/Trao đổi sinh viên/",
    "Hoc_bong_Diem_ren_luyen": "/Học bổng/",
    "Thu_tuc_hanh_chinh": "/Thủ tục hành chính một cửa/",
    "Cac_cong_thong_tin_huu_ich": "/các cổng thông tin hữu ích/"
}

print("🚀 Bắt đầu cào dữ liệu Luồng 1 (Nhóm 1)...")
for file_name, path in flow_1_links.items():
    url = f"{BASE_URL}{path}"
    
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        response.encoding = 'utf-8'
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        main_content = soup.find('div', class_='main-content')
        if not main_content:
            main_content = soup.find('main') or soup.find('body')
        
        if main_content:
            text_content = ""
            
            # 🎯 XỬ LÝ ĐẶC BIỆT LẤY LINK: Dành riêng cho trang Các cổng thông tin
            if file_name == "[Cac_cong_thong_tin_huu_ich]_Noi_dung":
                cleaned_paragraphs = []
                for a_tag in main_content.find_all('a'):
                    text = a_tag.get_text(strip=True)
                    href = a_tag.get('href')
                    if text and href:
                        clean_text = re.sub(r'\s+', ' ', text)
                        cleaned_paragraphs.append(f"{clean_text}: {href}")
                text_content = '\n'.join(cleaned_paragraphs)
                        
            # 🎯 XỬ LÝ BÌNH THƯỜNG BẰNG MARKDOWNIFY
            else:
                raw_markdown = md(
                    str(main_content),
                    heading_style="ATX",
                    bullets="*", 
                    strip=['script', 'style', 'img'] 
                )
                # Dùng hàm dọn dẹp mới ở đây
                text_content = clean_markdown_format(raw_markdown)
            
            # Lưu file Markdown
            file_path = os.path.join(SAVE_DIR, f"{file_name}.md")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
            
            print(f"  ✅ Đã tải và lưu cấu trúc Markdown: {file_name}.md")
        else:
            print(f"  ⚠️ Không tìm thấy nội dung cho: {file_name}")
            
    except Exception as e:
        print(f"  ❌ Lỗi khi cào {url}: {e}")

# ==========================================
# 2. XỬ LÝ LUỒNG 2: EXTERNAL LINKS (NHÓM 3)
# ==========================================
print("\n🚀 Bắt đầu ghi dữ liệu Luồng 2 (Nhóm 3)...")

flow_2_links = {
    "Đặt lịch đối thoại - Đặt câu hỏi": "https://ctsv.uet.vnu.edu.vn:3300/",
    "Tuyển dụng - Việc làm": "https://vieclam.uet.vnu.edu.vn/",
    "Thông tin nhà trọ": "https://ctsv.uet.vnu.edu.vn:3100/"
}

file_path_nhom3 = os.path.join(SAVE_DIR, "Lien_he_ngoai.md")
try:
    with open(file_path_nhom3, 'w', encoding='utf-8') as f:
        for name, link in flow_2_links.items():
            f.write(f"{name} truy cập web: {link}\n")
    print("  ✅ Đã tạo file: Lien_he_ngoai.md")
except Exception as e:
    print(f"  ❌ Lỗi ghi file Nhóm 3: {e}")

# ==========================================
# 3. LUỒNG 4.1: DEEP CRAWL (BỎ TẢI PDF)
# ==========================================
print("\n🚀 XỬ LÝ LUỒNG 4.1: CÀO TEXT TỔNG QUAN...")
deep_links = {
    "Hoc_phi_Che_do": "/Học phí - Chế độ chính sách/",
    "Noi_quy_Quy_che": "/Nội quy - quy chế/"
}

for prefix, path in deep_links.items():
    url = f"{BASE_URL}{path}"
    try:
        response = requests.get(url, headers=headers, timeout=15)
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')
        
        main_content = soup.find('div', class_='main-content')
        if not main_content:
            main_content = soup.find('main') or soup.find('body')
            
        if main_content:
            raw_markdown = md(
                str(main_content),
                heading_style="ATX",
                bullets="*",
                strip=['script', 'style', 'img']
            )
            
            # Dùng hàm dọn dẹp mới ở đây
            text_content = clean_markdown_format(raw_markdown)
            
            with open(os.path.join(SAVE_DIR, f"{prefix}.md"), 'w', encoding='utf-8') as f:
                f.write(text_content)
            print(f"  ✅ Đã lưu text cấu trúc Markdown: {prefix}_Tong_quan.md")
        else:
            print(f"  ⚠️ Không tìm thấy nội dung cho: {prefix}")
            
    except Exception as e:
        print(f"  ❌ Lỗi khi xử lý {url}: {e}")

# ==========================================
# 4. LUỒNG 4.2: LỊCH SỬ - TRUYỀN THỐNG
# ==========================================
print("\n🚀 XỬ LÝ LUỒNG 4.2: LỊCH SỬ - TRUYỀN THỐNG...")
try:
    path_encoded = urllib.parse.quote("lịch sử - truyền thống")
    url_goc = f"{BASE_URL}/{path_encoded}/"
    
    response_goc = requests.get(url_goc, headers=headers, timeout=15)
    response_goc.encoding = 'utf-8'
    soup_goc = BeautifulSoup(response_goc.text, 'html.parser')
    
    md_tong_quan = []
    deep_link = None 
    
    elements = soup_goc.find_all(['div'], class_=['title', 'component'])
    
    for el in elements:
        if 'title' in el.get('class', []):
            md_tong_quan.append(f"\n## {el.get_text(strip=True)}\n")
            
        elif 'component' in el.get('class', []):
            for li in el.find_all('li'):
                md_tong_quan.append(f"- {li.get_text(strip=True)}")
                
            for col in el.find_all('div', class_='col-content'):
                if not col.find('ul'): 
                    a_tag = col.find('a')
                    if a_tag and 'lich-su' in a_tag.get('href', ''):
                        deep_link = a_tag['href']
                        md_tong_quan.append(f"👉 [Xem Chi tiết Lịch sử hình thành tại link: {deep_link}]")
                    else:
                        text = col.get_text(strip=True)
                        if text:
                            md_tong_quan.append(text)

    final_tong_quan = '\n'.join(md_tong_quan)
    final_tong_quan = clean_markdown_format(final_tong_quan)
    
    file_tong_quan = os.path.join(SAVE_DIR, "Lich_su_Truyen_thong.md")
    with open(file_tong_quan, 'w', encoding='utf-8') as f:
        f.write(final_tong_quan)
    print("  ✅ Đã lưu: Lich_su_Truyen_thong.md")
    
    if deep_link:
        print(f"  🔗 Kích hoạt cào sâu trang chi tiết: {deep_link}")
        try:
            resp_sau = requests.get(deep_link, headers=headers, timeout=15)
            resp_sau.encoding = 'utf-8'
            soup_sau = BeautifulSoup(resp_sau.text, 'html.parser')
            main_content_sau = soup_sau.find(class_='entry-content')
            
            if main_content_sau:
                raw_markdown = md(
                    str(main_content_sau), 
                    heading_style="ATX",
                    bullets="*",
                    strip=['script', 'style', 'img'] 
                )
                
                # Dùng hàm dọn dẹp mới ở đây
                cleaned_markdown = clean_markdown_format(raw_markdown)
                
                file_chi_tiet = os.path.join(SAVE_DIR, "Lich_su_Truyen_thong_Chi_tiet.md")
                with open(file_chi_tiet, 'w', encoding='utf-8') as f:
                    f.write(cleaned_markdown)
                print("  ✅ Đã bóc tách hoàn hảo Bảng & Chú thích vào: Lich_su_Truyen_thong_Chi_tiet.md")
            else:
                print("  ⚠️ Không tìm thấy class 'entry-content' tại trang chi tiết.")
                
        except Exception as e_sau:
            print(f"  ❌ Lỗi khi cào sâu: {e_sau}")

except Exception as e:
    print(f"  ❌ Lỗi khi cào Luồng 4.2: {e}")
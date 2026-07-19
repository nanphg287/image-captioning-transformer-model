import os
import csv
import ssl
import zipfile
import urllib.request
from pathlib import Path

# Bypass SSL verification for macOS certificate issues
ssl._create_default_https_context = ssl._create_unverified_context

def download_file(url: str, dest_path: Path):
    print(f"Đang tải {url}...")
    urllib.request.urlretrieve(url, dest_path)
    print(f"-> Đã tải xong và lưu tại {dest_path}")

def main():
    project_root = Path(__file__).resolve().parent
    raw_dir = project_root / "datasets" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    img_zip = raw_dir / "Flickr8k_Dataset.zip"
    text_zip = raw_dir / "Flickr8k_text.zip"

    # 1. Tải dữ liệu ảnh và text từ GitHub Release
    img_url = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip"
    text_url = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_text.zip"

    if not img_zip.exists():
        download_file(img_url, img_zip)
    else:
        print("Tập tin Flickr8k_Dataset.zip đã tồn tại. Bỏ qua tải xuống.")

    if not text_zip.exists():
        download_file(text_url, text_zip)
    else:
        print("Tập tin Flickr8k_text.zip đã tồn tại. Bỏ qua tải xuống.")

    # 2. Giải nén ảnh
    images_dir = raw_dir / "Images"
    if not images_dir.exists():
        print("Đang giải nén ảnh Flickr8k_Dataset.zip...")
        with zipfile.ZipFile(img_zip, 'r') as zip_ref:
            # Brownlee's zip contains a folder called "Flicker8k_Dataset" or similar
            # Let's extract all to a temporary folder then move/rename it
            temp_extract = raw_dir / "temp_img"
            zip_ref.extractall(temp_extract)
            
            # Find the actual images folder inside temp_extract
            # Usually it's either in the root or inside a subfolder
            candidates = list(temp_extract.glob("**/Flicker8k_Dataset"))
            if not candidates:
                candidates = list(temp_extract.glob("**/Flickr8k_Dataset"))
            
            if candidates:
                # Rename the found folder to raw/Images
                candidates[0].rename(images_dir)
                print(f"-> Giải nén thành công ảnh vào: {images_dir}")
            else:
                # If no subfolder, maybe images are direct
                temp_extract.rename(images_dir)
                print(f"-> Giải nén trực tiếp vào: {images_dir}")
                
            # Dọn dẹp thư mục tạm nếu còn
            if temp_extract.exists():
                import shutil
                shutil.rmtree(temp_extract, ignore_errors=True)
    else:
        print("Thư mục Images đã tồn tại. Bỏ qua giải nén ảnh.")

    # 3. Giải nén text
    token_txt_path = raw_dir / "Flickr8k.token.txt"
    if not token_txt_path.exists():
        print("Đang giải nén text Flickr8k_text.zip...")
        with zipfile.ZipFile(text_zip, 'r') as zip_ref:
            zip_ref.extractall(raw_dir)
        print("-> Giải nén text thành công.")
    else:
        print("Tập tin Flickr8k.token.txt đã tồn tại. Bỏ qua giải nén text.")

    # 4. Chuyển đổi Flickr8k.token.txt sang captions.txt (CSV format)
    captions_csv_path = raw_dir / "captions.txt"
    if not captions_csv_path.exists():
        print("Đang chuyển đổi Flickr8k.token.txt sang captions.txt (định dạng CSV)...")
        with open(token_txt_path, "r", encoding="utf-8") as f_in:
            lines = f_in.readlines()

        with open(captions_csv_path, "w", encoding="utf-8", newline="") as f_out:
            writer = csv.writer(f_out)
            writer.writerow(["image", "caption"]) # Viết header cột
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                # Định dạng trong file: image_name.jpg#index\tcaption
                parts = line.split("\t")
                if len(parts) == 2:
                    image_part, caption = parts
                    image_name = image_part.split("#")[0].strip()
                    writer.writerow([image_name, caption.strip()])
        print(f"-> Chuyển đổi thành công! File lưu tại: {captions_csv_path}")
    else:
        print("File captions.txt đã tồn tại. Bỏ qua chuyển đổi.")

    # 5. Dọn dẹp zip để tiết kiệm dung lượng
    print("Dọn dẹp các tập tin zip tạm...")
    if img_zip.exists():
        os.remove(img_zip)
    if text_zip.exists():
        os.remove(text_zip)
    print("=== HOÀN THÀNH CHUẨN BỊ BỘ DỮ LIỆU FLICKR8K GỐC! ===")

if __name__ == "__main__":
    main()

import os
import json
from PIL import Image
from pathlib import Path
from data.tokenizer import CaptionTokenizer
from data.vocabulary import Vocabulary

def main():
    print("=== TẠO DỮ LIỆU GIẢ LẬP ĐỂ CHẠY THỬ PROJECT ===")
    
    # 1. Đường dẫn các thư mục
    project_root = Path(__file__).resolve().parent
    datasets_dir = project_root / "datasets"
    inference_dir = datasets_dir / "inference"
    processed_dir = datasets_dir / "processed"
    resize_dir = processed_dir / "resize_image"
    
    # Tạo các thư mục nếu chưa tồn tại
    resize_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Danh sách ảnh inference có sẵn
    images = [
        "1299459550_1fd5594fa2.jpg",
        "2662890367_382eaf83bd.jpg",
        "3627679667_0e3de9fc90.jpg"
    ]
    
    # Kiểm tra xem có ảnh trong datasets/inference không
    for img_name in images:
        src = inference_dir / img_name
        if not src.exists():
            print(f"Lỗi: Không tìm thấy ảnh {src}. Hãy chắc chắn bạn chạy đúng thư mục gốc.")
            return

    # 3. Tạo resize_image
    print("Đang chuẩn hóa kích thước ảnh (resize sang 224x224)...")
    for img_name in images:
        src = inference_dir / img_name
        dest = resize_dir / img_name
        with Image.open(src) as img:
            img = img.resize((224, 224), Image.Resampling.BILINEAR)
            img.save(dest)
        print(f"-> Đã lưu ảnh resize tại: {dest}")

    # 4. Tạo dữ liệu caption mẫu
    captions_mapping = {
        "1299459550_1fd5594fa2.jpg": {
            "resize_image_path": (resize_dir / "1299459550_1fd5594fa2.jpg").as_posix(),
            "captions": [
                "A black dog is running on the grass .",
                "A dog runs across the green field ."
            ]
        },
        "2662890367_382eaf83bd.jpg": {
            "resize_image_path": (resize_dir / "2662890367_382eaf83bd.jpg").as_posix(),
            "captions": [
                "A white dog sits on the grass pool .",
                "A dog sits beside a swimming pool ."
            ]
        },
        "3627679667_0e3de9fc90.jpg": {
            "resize_image_path": (resize_dir / "3627679667_0e3de9fc90.jpg").as_posix(),
            "captions": [
                "A young girl is swimming in a pool .",
                "A girl swims in the blue water pool ."
            ]
        }
    }

    # 5. Lưu train.json, validation.json, test.json
    print("Đang tạo các file dữ liệu train.json, validation.json, test.json...")
    with open(processed_dir / "train.json", "w", encoding="utf-8") as f:
        json.dump(captions_mapping, f, ensure_ascii=False, indent=2)
        
    with open(processed_dir / "validation.json", "w", encoding="utf-8") as f:
        json.dump(captions_mapping, f, ensure_ascii=False, indent=2)
        
    with open(processed_dir / "test.json", "w", encoding="utf-8") as f:
        json.dump(captions_mapping, f, ensure_ascii=False, indent=2)
        
    print("-> Đã tạo xong các file JSON trong datasets/processed/")

    # 6. Tạo vocabulary.json từ các câu caption này
    print("Đang tự động dựng bộ từ điển vocabulary.json...")
    tokenizer = CaptionTokenizer()
    vocab = Vocabulary(tokenizer=tokenizer)
    vocab.build_from_json(json_path=processed_dir / "train.json", min_frequency=1) # Đặt tần suất min = 1 để lấy tất cả từ mẫu
    vocab.save(processed_dir / "vocabulary.json")
    print(f"-> Đã tạo vocabulary mới với {len(vocab)} từ.")
    print("-> Đã lưu tại: ", processed_dir / "vocabulary.json")
    print("\n=== HOÀN THÀNH CHUẨN BỊ DỮ LIỆU! GIỜ BẠN CÓ THỂ CHẠY THỬ PROJECT. ===")

if __name__ == "__main__":
    main()

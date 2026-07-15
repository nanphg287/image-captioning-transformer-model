from __future__ import annotations

import re

#định nghĩa lớp CaptionTokenizer để chuyển đổi một caption thành một danh sách các từ con (tokens) dạng list[str]
class CaptionTokenizer:
    """
     Ví dụ: "A girl sits in a pool ." => ["a", "girl", "sits", "in", "a", "pool"]
    """
    # Biểu thức chính quy (Regex) dùng để nhận diện và tách các từ 
    TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđĐ]+(?:'[a-z0-9]+)?")

    def tokenize(self, caption: str) -> list[str]:
         # 1. Kiểm tra kiểu dữ liệu đầu vào, nếu không phải là chuỗi thì báo lỗi
        if not isinstance(caption, str):
            raise TypeError(f"caption phải là str, nhận được {type(caption).__name__}.")
        # 2. Chuẩn hóa chuỗi: Chuyển về chữ thường (lower) và loại bỏ khoảng trắng dư ở 2 đầu (strip)
        normalized_caption = caption.lower().strip()
        
        # 3. Tìm tất cả các tokens khớp với regex pattern và trả về danh sách các chuỗi
        return self.TOKEN_PATTERN.findall(normalized_caption)

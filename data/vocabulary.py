from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from data.tokenizer import CaptionTokenizer


class Vocabulary:
    PAD_TOKEN = "<PAD>" #<PAD> (Padding - ID: 0): Dùng để chèn thêm vào các câu ngắn hơn trong một batch để đưa tất cả các câu về cùng một độ dài tối đa (max_length). Điều này giúp GPU xử lý song song hiệu quả.
    BOS_TOKEN = "<BOS>" #<BOS> (Beginning of Sentence - ID: 1): Đánh dấu điểm bắt đầu của một câu chú thích.
    EOS_TOKEN = "<EOS>" #<EOS> (End of Sentence - ID: 2): Đánh dấu điểm kết thúc của một câu chú thích. Mô hình sẽ dừng sinh từ khi gặp token này.
    UNK_TOKEN = "<UNK>" #<UNK> (Unknown - ID: 3): Đại diện cho các từ lạ/ngoại lai không nằm trong bộ từ điển đã học.

    PAD_TOKEN_ID = 0
    BOS_TOKEN_ID = 1
    EOS_TOKEN_ID = 2
    UNK_TOKEN_ID = 3

    def __init__(self, tokenizer: CaptionTokenizer):
        self.tokenizer = tokenizer

        self.token_to_id: dict[str, int] = {
            self.PAD_TOKEN: self.PAD_TOKEN_ID,
            self.BOS_TOKEN: self.BOS_TOKEN_ID,
            self.EOS_TOKEN: self.EOS_TOKEN_ID,
            self.UNK_TOKEN: self.UNK_TOKEN_ID
        }

        self.id_to_token: dict[int, str] = {
            token_id: token
            for token, token_id
            in self.token_to_id.items()
        }

    def __len__(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_token_id(self) -> int:
        return self.PAD_TOKEN_ID

    @property
    def bos_token_id(self) -> int:
        return self.BOS_TOKEN_ID

    @property
    def eos_token_id(self) -> int:
        return self.EOS_TOKEN_ID

    @property
    def unk_token_id(self) -> int:
        return self.UNK_TOKEN_ID

    def build_from_json(self, json_path: str | Path, min_frequency: int = 2) -> None:
        """
        Xây dựng vocabulary từ tất cả caption trong tập train.
        Chỉ sử dụng train dataset để tránh data leakage.
        Hàm này quét qua toàn bộ dữ liệu chú thích dạng JSON (chỉ dùng tập Train) để tìm và đưa các từ thực tế vào từ điển:

        Tách từ và Đếm tần suất: Đọc từng câu caption, dùng tokenizer để tách từ và dùng Counter để đếm xem mỗi từ xuất hiện bao nhiêu lần trong toàn bộ tập dữ liệu.
        Lọc từ theo tần suất (min_frequency): Loại bỏ các từ quá hiếm gặp (tần suất xuất hiện nhỏ hơn min_frequency) vì chúng có thể là từ gõ sai chính tả hoặc từ ít phổ biến, giúp giảm kích thước từ điển và tránh nhiễu cho mô hình.
        Sắp xếp: Sắp xếp các từ hợp lệ giảm dần theo tần suất xuất hiện. Nếu tần suất bằng nhau thì sắp xếp theo thứ tự chữ cái. Việc này đảm bảo từ điển tạo ra luôn giống hệt nhau qua mỗi lần chạy (tính nhất quán).
        Gán ID: Mỗi từ hợp lệ mới sẽ được gán ID tăng dần tiếp theo (bắt đầu từ số 4 trở đi).
        """

        if min_frequency <= 0:
            raise ValueError("min_frequency phải lớn hơn 0.")

        json_path = Path(json_path)

        if not json_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {json_path}")

        with json_path.open(mode="r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError("Dữ liệu JSON phải là một object.")

        word_counter: Counter[str] = Counter()

        for image_name, image_data in data.items():
            if not isinstance(image_data, dict):
                raise ValueError(f"Dữ liệu của {image_name} không phải object.")

            captions = image_data.get("captions")

            if not isinstance(captions, list):
                raise ValueError(f"captions của {image_name} không phải list.")

            for caption in captions:
                tokens = self.tokenizer.tokenize(caption)

                word_counter.update(tokens)

        # Sắp xếp để Vocabulary luôn được tạo giống nhau giữa các lần chạy.
        valid_tokens = [
            token
            for token, frequency
            in word_counter.items()
            if frequency >= min_frequency
        ]

        valid_tokens.sort(
            key=lambda token: (
                -word_counter[token],
                token
            )
        )

        for token in valid_tokens:
            if token in self.token_to_id:
                continue

            token_id = len(self.token_to_id)

            self.token_to_id[token] = token_id
            self.id_to_token[token_id] = token

    #độ dài tối đa của danh sách id đầu ra mặc định là none - không giới hạn
    def encode(self, caption: str, max_length: int | None = None) -> list[int]:
        """
        Chuyển caption thành token ID.
        Ví dụ: "a girl sits" ==> [BOS, a, girl, sits, EOS]
        """
        #gọi bộ tokenizer để tách chuỗi câu thành các từ viết thường A to a
        tokens = self.tokenizer.tokenize(caption)
        #nếu giới hạn độ dài chuỗi 
        if max_length is not None:
            #độ dài tối thiểu bằng 2 -> chương trình báo lỗi 
            if max_length < 2:
                raise ValueError("max_length phải ít nhất bằng 2 để chứa BOS và EOS.")

            # Chừa hai vị trí cho BOS và EOS.
            tokens = tokens[:max_length - 2]
        #duyệt qua từng từ trong danh sách và đổi sang số ID tương ứng
        #hàm .get() kiểm tra xem từ đó có trong từ điển không, nếu có trả về id của từ đó, nếu không trả về id của từ lạ (3)
        token_ids = [
            self.token_to_id.get(token, self.UNK_TOKEN_ID)
            for token in tokens
        ]
        #đưa thêm số ID của từ đầu (BOS_TOKEN) và từ cuối (EOS_TOKEN) vào danh sách token_ids
        return [
            self.BOS_TOKEN_ID,
            *token_ids, #dấu * đề trải phằng unpack danh sách id ở trên vào trong danh sách mới 
            self.EOS_TOKEN_ID
        ]
    #skip_special_tokens mặc định là true, hàm sẽ loại bỏ các ký hiệu kỹ thuật như pad, bos, eos khỏi câu văn ban đầu ra để câu đọc tự nhiên 
    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """
        Chuyển token ID về chuỗi văn bản.
        Duyệt qua danh sách token_ids, tra cứu trong bảng id_to_token để lấy lại từ gốc. Nếu không tìm thấy, mặc định là từ lạ <UNK>.
Điểm dừng: Nếu gặp EOS_TOKEN (ID: 2), quá trình giải mã dừng lại ngay lập tức (không đọc các từ đệm phía sau).
Nếu đặt skip_special_tokens=True, hàm sẽ bỏ qua không đưa các từ điều khiển như <PAD>, <BOS>, <EOS> vào câu văn bản đầu ra.
Nối các từ lại bằng khoảng trắng (" ".join(tokens)).
        """

        tokens: list[str] = []
        #danh sách các token đặc biệt dùng để loại bỏ
        special_tokens = {
            self.PAD_TOKEN,
            self.BOS_TOKEN,
            self.EOS_TOKEN
        }

        for token_id in token_ids:
            #tra cứu ngược: truyền số id vào để lấy ra chữ tương ứng
            #duyệt qua từng id trong danh sách token_ids. 
            #tra cứu trong bảng ánh xạ self.id_to_token, nếu gặp số lạ không có trong từ điển, nó sẽ trả về từ mặc định là unk
            token = self.id_to_token.get(int(token_id), self.UNK_TOKEN)
            #nếu id tương ứng là EOS_TOKEN thì dừng không duyệt tiếp nữa, tất cả các id đứng sau eos thường là các token đêm pad sẽ bị bỏ qua
            if token == self.EOS_TOKEN:
                break
            # nếu skip_special_tokens = true và từ đang xét là một token đặc biệt, bỏ qua không thêm từ này vào kết quả
            if skip_special_tokens and token in special_tokens:
                continue
            #ngược lại, thêm từ đã dịch vào danh sách tokens 
            tokens.append(token)
        #nối tất cả các từ trong danh sách lại với nhau bằng khoảng trắng để tạo thành câu hoàn chỉnh
        return " ".join(tokens)
    # save(output_path): Lưu từ điển (bảng ánh xạ token_to_id) thành định dạng file .json.
    def save(self, output_path: str | Path) -> None:

        output_path = Path(output_path)
        #tạo thư mục chứa nếu chưa tồn tại
        output_path.parent.mkdir(parents=True, exist_ok=True)

        vocabulary_data = {
            "token_to_id": self.token_to_id
        }
        #lưu dữ liệu self.token_to_id thành một file .json 
        with output_path.open(mode="w", encoding="utf-8") as file:
            json.dump(vocabulary_data, file, ensure_ascii=False, indent=2)

    @classmethod    
    # load(vocabulary_path, tokenizer): Đọc file từ điển .json đã lưu từ trước để tái cấu trúc lại đối tượng Vocabulary khi cần suy luận (inference) hoặc tiếp tục huấn luyện mà không cần phải dựng lại từ đầu.
    #hàm dùng nạp lại bộ từ điển đã lưu ở trên để phục vụ cho việc huấn luyện tiếp theo hoặc khi chạy ứng dụng thực tế
    def load(cls, vocabulary_path: str | Path, tokenizer: CaptionTokenizer) -> "Vocabulary":
        vocabulary_path = Path(vocabulary_path)

        if not vocabulary_path.exists():
            raise FileNotFoundError(f"Không tìm thấy vocabulary: {vocabulary_path}")
        #đọc file json từ điển
        with vocabulary_path.open(mode="r", encoding="utf-8") as file:
            vocabulary_data = json.load(file)
        #khởi tạo một đối tượng vocabulary trống mới 
        vocabulary = cls(tokenizer=tokenizer)
        #nạp lại bảng ánh xạ token_to_id từ file json 
        vocabulary.token_to_id = {
            str(token): int(token_id)
            for token, token_id
            in vocabulary_data["token_to_id"].items()
        }
        #tự động đảo ngược bảng token_to_id để tái tạo lại bảng id_to_token
        vocabulary.id_to_token = {
            token_id: token
            for token, token_id
            in vocabulary.token_to_id.items()
        }

        return vocabulary

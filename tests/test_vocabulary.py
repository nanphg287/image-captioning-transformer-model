import unittest
import json
import tempfile
from pathlib import Path
from data.tokenizer import CaptionTokenizer
from data.vocabulary import Vocabulary

class TestVocabulary(unittest.TestCase):
    def setUp(self):
        self.tokenizer = CaptionTokenizer()
        self.vocab = Vocabulary(self.tokenizer)
        
        # Thiết lập một từ điển giả lập đơn giản để test encode/decode
        self.vocab.token_to_id = {
            "<PAD>": 0,
            "<BOS>": 1,
            "<EOS>": 2,
            "<UNK>": 3,
            "cô": 4,
            "gái": 5,
            "ngồi": 6,
            "hồ": 7,
            "bơi": 8
        }
        self.vocab.id_to_token = {v: k for k, v in self.vocab.token_to_id.items()}

    def test_initial_tokens(self):
        """Kiểm tra các token đặc biệt khi khởi tạo"""
        vocab_new = Vocabulary(self.tokenizer)
        self.assertEqual(vocab_new.token_to_id[vocab_new.PAD_TOKEN], vocab_new.PAD_TOKEN_ID)
        self.assertEqual(vocab_new.token_to_id[vocab_new.BOS_TOKEN], vocab_new.BOS_TOKEN_ID)
        self.assertEqual(vocab_new.token_to_id[vocab_new.EOS_TOKEN], vocab_new.EOS_TOKEN_ID)
        self.assertEqual(vocab_new.token_to_id[vocab_new.UNK_TOKEN], vocab_new.UNK_TOKEN_ID)

    def test_encode(self):
        """Kiểm tra hàm encode chuyển văn bản thành IDs"""
        caption = "Cô gái ngồi hồ bơi"
        # Dự kiến: [BOS_ID, cô_ID, gái_ID, ngồi_ID, hồ_ID, bơi_ID, EOS_ID]
        expected = [1, 4, 5, 6, 7, 8, 2]
        self.assertEqual(self.vocab.encode(caption), expected)

    def test_encode_with_unk(self):
        """Kiểm tra hàm encode xử lý từ lạ (không có trong từ điển)"""
        caption = "Cô gái đá bóng"  # "đá", "bóng" không có trong từ điển
        expected = [1, 4, 5, 3, 3, 2]  # ID 3 là UNK
        self.assertEqual(self.vocab.encode(caption), expected)

    def test_encode_with_max_length(self):
        """Kiểm tra hàm encode cắt bớt từ theo max_length"""
        caption = "Cô gái ngồi hồ bơi"
        # Với max_length=5, chỉ giữ lại 5-2 = 3 từ đầu tiên ("cô gái ngồi")
        # Kết quả mong đợi: [BOS, cô, gái, ngồi, EOS] => độ dài đúng bằng 5
        expected = [1, 4, 5, 6, 2]
        self.assertEqual(self.vocab.encode(caption, max_length=5), expected)

    def test_decode_skip_special_tokens(self):
        """Kiểm tra hàm decode chuyển IDs thành chuỗi văn bản (bỏ qua token đặc biệt)"""
        token_ids = [1, 4, 5, 6, 2]
        self.assertEqual(self.vocab.decode(token_ids, skip_special_tokens=True), "cô gái ngồi")

    def test_decode_keep_special_tokens(self):
        """Kiểm tra hàm decode và giữ lại token đặc biệt"""
        token_ids = [1, 4, 5, 6, 2]
        # skip_special_tokens=False vẫn dừng lại ở <EOS> nhưng giữ lại <BOS>
        self.assertEqual(self.vocab.decode(token_ids, skip_special_tokens=False), "<BOS> cô gái ngồi")

    def test_decode_stop_at_eos(self):
        """Kiểm tra hàm decode dừng lại ngay khi gặp <EOS>"""
        token_ids = [1, 4, 5, 2, 6, 7, 0, 0]  # Có các từ sau <EOS>
        # Kết quả giải mã chỉ lấy đến trước <EOS>
        self.assertEqual(self.vocab.decode(token_ids, skip_special_tokens=True), "cô gái")

    def test_build_from_json(self):
        """Kiểm tra việc dựng từ điển từ file JSON chứa dữ liệu caption"""
        dummy_data = {
            "image1.jpg": {
                "captions": [
                    "Cô gái hồ bơi",
                    "Một cô gái ngồi ở hồ bơi"
                ]
            },
            "image2.jpg": {
                "captions": [
                    "Cô gái ngồi bơi"
                ]
            }
        }
        
        # Tạo file JSON tạm thời để test
        with tempfile.TemporaryDirectory() as tmpdir:
            json_file = Path(tmpdir) / "dummy_dataset.json"
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(dummy_data, f, ensure_ascii=False)
            
            vocab_build = Vocabulary(self.tokenizer)
            # Dựng từ điển với min_frequency=2 (từ xuất hiện ít nhất 2 lần mới được thêm)
            vocab_build.build_from_json(json_file, min_frequency=2)
            
            # Các từ xuất hiện >= 2 lần: "cô" (3 lần), "gái" (3 lần), "hồ" (2 lần), "bơi" (3 lần), "ngồi" (2 lần)
            # Các từ xuất hiện 1 lần: "một", "ở" (sẽ bị loại bỏ vì min_frequency=2)
            self.assertIn("cô", vocab_build.token_to_id)
            self.assertIn("gái", vocab_build.token_to_id)
            self.assertIn("ngồi", vocab_build.token_to_id)
            self.assertNotIn("một", vocab_build.token_to_id)
            self.assertNotIn("ở", vocab_build.token_to_id)

    def test_save_and_load(self):
        """Kiểm tra lưu từ điển thành JSON và nạp lại"""
        with tempfile.TemporaryDirectory() as tmpdir:
            vocab_file = Path(tmpdir) / "vocab.json"
            
            # Lưu từ điển hiện tại
            self.vocab.save(vocab_file)
            self.assertTrue(vocab_file.exists())
            
            # Tải lại từ điển
            loaded_vocab = Vocabulary.load(vocab_file, self.tokenizer)
            
            # So sánh độ tương đồng
            self.assertEqual(loaded_vocab.token_to_id, self.vocab.token_to_id)
            self.assertEqual(loaded_vocab.id_to_token, self.vocab.id_to_token)

if __name__ == "__main__":
    unittest.main()

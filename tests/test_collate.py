import unittest
import torch
from data.collate import ImageCaptionCollator

class TestImageCaptionCollator(unittest.TestCase):
    def setUp(self):
        # Giả lập ID của token <PAD> là 0
        self.pad_token_id = 0
        self.collator = ImageCaptionCollator(pad_token_id=self.pad_token_id)
        
        # Tạo batch dữ liệu giả lập để kiểm thử
        # Batch gồm 3 mẫu với độ dài caption_ids khác nhau: 5, 3, và 4
        self.dummy_batch = [
            {
                "filename": "image_1.jpg",
                "image": torch.randn(3, 224, 224),
                "caption": "a black cat sits",
                "caption_ids": torch.tensor([1, 10, 11, 12, 2])  # Độ dài 5 (BOS, a, black, cat, sits, EOS)
            },
            {
                "filename": "image_2.jpg",
                "image": torch.randn(3, 224, 224),
                "caption": "dog running",
                "caption_ids": torch.tensor([1, 13, 2])          # Độ dài 3 (BOS, dog, running, EOS)
            },
            {
                "filename": "image_3.jpg",
                "image": torch.randn(3, 224, 224),
                "caption": "a young girl",
                "caption_ids": torch.tensor([1, 10, 14, 2])      # Độ dài 4 (BOS, a, young, girl, EOS)
            }
        ]

    def test_collate_keys_exist(self):
        """Kiểm tra xem kết quả trả về có đủ các khóa (keys) cần thiết hay không"""
        collated = self.collator(self.dummy_batch)
        expected_keys = {
            "filenames", "images", "captions", 
            "caption_ids", "caption_lengths", "caption_padding_mask"
        }
        for key in expected_keys:
            self.assertIn(key, collated)

    def test_collate_shapes(self):
        """Kiểm tra kích thước (shape) của các Tensor đầu ra"""
        collated = self.collator(self.dummy_batch)
        
        # Batch size = 3, Số kênh ảnh = 3, kích thước ảnh = 224x224
        self.assertEqual(collated["images"].shape, (3, 3, 224, 224))
        
        # Câu dài nhất trong batch có độ dài là 5 (của mẫu thứ nhất)
        # Ma trận caption_ids phải có size: [Batch_Size, Max_Length] -> [3, 5]
        self.assertEqual(collated["caption_ids"].shape, (3, 5))
        self.assertEqual(collated["caption_padding_mask"].shape, (3, 5))
        self.assertEqual(collated["caption_lengths"].shape, (3,))

    def test_collate_padding_values(self):
        """Kiểm tra xem các câu ngắn hơn có được đệm đúng giá trị pad_token_id (0) hay không"""
        collated = self.collator(self.dummy_batch)
        caption_ids = collated["caption_ids"]
        
        # Mẫu 1 (độ dài 5): [1, 10, 11, 12, 2] -> Không cần đệm
        torch.testing.assert_close(caption_ids[0], torch.tensor([1, 10, 11, 12, 2]))
        
        # Mẫu 2 (độ dài 3): [1, 13, 2] -> Đệm thêm 2 số 0 ở cuối -> [1, 13, 2, 0, 0]
        torch.testing.assert_close(caption_ids[1], torch.tensor([1, 13, 2, 0, 0]))
        
        # Mẫu 3 (độ dài 4): [1, 10, 14, 2] -> Đệm thêm 1 số 0 ở cuối -> [1, 10, 14, 2, 0]
        torch.testing.assert_close(caption_ids[2], torch.tensor([1, 10, 14, 2, 0]))

    def test_collate_padding_mask(self):
        """Kiểm tra xem mask padding được tạo ra có chính xác không"""
        collated = self.collator(self.dummy_batch)
        mask = collated["caption_padding_mask"]
        
        # Vị trí đệm (bằng pad_token_id=0) phải là True, vị trí khác phải là False
        # Mẫu 1: [False, False, False, False, False]
        self.assertFalse(mask[0].any())
        
        # Mẫu 2: [False, False, False, True, True] (hai vị trí cuối là đệm)
        self.assertEqual(mask[1].tolist(), [False, False, False, True, True])
        
        # Mẫu 3: [False, False, False, False, True] (vị trí cuối cùng là đệm)
        self.assertEqual(mask[2].tolist(), [False, False, False, False, True])

    def test_collate_lengths(self):
        """Kiểm tra xem độ dài thực tế trước khi đệm của các câu có chính xác không"""
        collated = self.collator(self.dummy_batch)
        lengths = collated["caption_lengths"]
        
        # Độ dài thực tế mong muốn: [5, 3, 4]
        self.assertEqual(lengths.tolist(), [5, 3, 4])

    def test_collate_metadata(self):
        """Kiểm tra xem tên file và câu mô tả gốc dạng text có được bảo toàn không"""
        collated = self.collator(self.dummy_batch)
        
        expected_filenames = ["image_1.jpg", "image_2.jpg", "image_3.jpg"]
        expected_captions = ["a black cat sits", "dog running", "a young girl"]
        
        self.assertEqual(collated["filenames"], expected_filenames)
        self.assertEqual(collated["captions"], expected_captions)

if __name__ == "__main__":
    unittest.main()

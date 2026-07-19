from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence


class ImageCaptionCollator:
    """
    Gom nhiều sample thành một batch và padding caption.

    PAD position trong caption_padding_mask có giá trị True.
    Token bình thường có giá trị False.
    """

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch: list[dict[str, object]]) -> dict[str, object]:
        #1. gom nhóm ảnh và xếp chồng lên nhau
        images = torch.stack([ #torch.stack là hàm gộp các tensor lại với nhau thành 1 tensor lớn hơn
            item["image"]
            for item in batch #batch đầu vào là 1 list chứa các dictionary mỗi dict là một cặp image và caption_ids của 1 ảnh
        ])

        caption_sequences = [
            item["caption_ids"]
            for item in batch
        ]
        #2. padding các câu caption ngắn
        #pad_sequence tự động tìm câu dài nhất trong batch để điền số 0 vào cuối các câu ngắn hơn để tất cả các câu có độ dài bằng 8
        caption_ids = pad_sequence(caption_sequences, batch_first=True, padding_value=self.pad_token_id) #batch_first=True nghĩa là batch sẽ là chiều đầu tiên [Batch_Size, Max_Sequence_Length]
        #3. tính toán độ dài thật của các câu (trước khi padding)
        caption_lengths = torch.tensor(
            [
                sequence.size(0)
                for sequence in caption_sequences
            ],
            dtype=torch.long
        )
        #4. tạo mask cho padding
        caption_padding_mask = (caption_ids == self.pad_token_id)

        return {
            "filenames": [
                item["filename"]
                for item in batch
            ],
            "images": images,
            "captions": [
                item["caption"]
                for item in batch
            ],
            "caption_ids": caption_ids,
            "caption_lengths": caption_lengths,
            "caption_padding_mask": (
                caption_padding_mask
            )
        }

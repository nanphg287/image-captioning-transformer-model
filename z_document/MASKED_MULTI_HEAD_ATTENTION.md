# Cơ Chế Masked Multi-Head Self-Attention và Causal Masking trong Transformer Decoder

Tài liệu này giải thích chi tiết về cơ chế **Masked Multi-Head Self-Attention**, **Causal Masking (Mặt nạ nguyên nhân)** và **Padding Masking (Mặt nạ đệm)** trong khối Text Decoder của mô hình Image Captioning Transformer.

---

## 📌 1. Tổng Quan

Trong khối **Decoder** của kiến trúc Transformer, lớp **Masked Multi-Head Self-Attention** đóng vai trò xử lý chuỗi văn bản (Caption). Khác với Self-Attention thông thường ở khối Encoder (nơi tất cả các token có thể tự do "nhìn" lẫn nhau), Self-Attention ở Decoder bị giới hạn bởi 2 loại mặt nạ (Masks):

1. **Causal Mask (Mặt nạ nguyên nhân / Tam giác trên):** Ngăn không cho từ tại vị trí hiện tại "nhìn lén" các từ đứng phía sau nó trong tương lai.
2. **Padding Mask (Mặt nạ đệm):** Ngăn không cho mô hình chú ý đến các thẻ đệm `<PAD>` (vốn không chứa thông tin ngữ nghĩa).

```
                            Input Text Embeddings
                                     │
                                     ▼
                            [ Pre-LayerNorm ]
                                     │
                                     ▼
                          [ QKV Projections ]
                                     │
                        Query, Key, Value [B, H, L, D]
                                     │
                                     ▼
                      Attention Scores (Q × Kᵀ / √d)
                                [B, H, L, L]
                                     │
                                     ▼
        ┌────────────────────────────────────────────────────────┐
        │                 APPLY COMBINED MASK                    │
        │   Combined Mask = Causal Mask  |  Key Padding Mask    │
        │                                                        │
        │   masked_fill(combined_mask == True, value = -inf)     │
        └────────────────────────────────────────────────────────┘
                                     │
                                     ▼
                            [ Softmax (dim=-1) ]
                  (Các vị trí bị che sẽ có trọng số = 0)
                                     │
                                     ▼
                        Dropout × Value Matrix (V)
                                     │
                                     ▼
                          [ Output Projection ]
                                     │
                                     ▼
                           [ Residual Connection ]
```

---

## 🎯 2. Tại Sao Cần Masked Attention?

### 2.1. Bảo vệ tính chất Tự Hồi Quy (Autoregressive Property)
Khi thực hiện suy luận (Inference), mô hình sinh câu theo từng từ một từ trái qua phải:
- Tại thời điểm $t = 1$, mô hình nhận `<BOS>` và dự đoán từ thứ 1 (ví dụ: `"a"`).
- Tại thời điểm $t = 2$, mô hình nhận `["<BOS>", "a"]` và dự đoán từ thứ 2 (ví dụ: `"dog"`).

Trong quá trình huấn luyện (Training), chúng ta sử dụng kỹ thuật **Teacher Forcing**: truyền toàn bộ câu hoàn chỉnh vào Decoder cùng lúc để tính toán song song cho tất cả các vị trí. Nếu không che các từ ở tương lai, từ tại vị trí $t=1$ sẽ "nhìn thấy" ngay từ $t=2$, khiến mô hình **học vẹt (gian lận)** và mất hoàn toàn khả năng tự suy luận khi chạy thực tế.

### 2.2. Triệt tiêu nhiễu từ các thẻ `<PAD>`
Trong một lô dữ liệu (Batch), các câu có độ dài ngắn dài khác nhau. Các câu ngắn hơn được thêm thẻ `<PAD>` vào cuối để cân bằng kích thước ma trận. `Padding Mask` giúp mô hình hoàn toàn bỏ qua các thẻ `<PAD>` này, tránh làm sai lệch trọng số chú ý.

---

## 🔍 3. Chi Tiết Các Loại Mask

### 3.1. Causal Mask (Mặt nạ tam giác trên)
Causal Mask là một ma trận vuông boolean kích thước $L \times L$ (với $L$ là độ dài chuỗi). Các ô có giá trị `True` đại diện cho các vị trí **bị che (không được phép nhìn)**.

Ví dụ với câu có độ dài $L = 4$ (tương ứng 4 vị trí $t_0, t_1, t_2, t_3$):

$$\text{Causal Mask} = \begin{bmatrix} 
\text{False} & \text{True} & \text{True} & \text{True} \\
\text{False} & \text{False} & \text{True} & \text{True} \\
\text{False} & \text{False} & \text{False} & \text{True} \\
\text{False} & \text{False} & \text{False} & \text{False}
\end{bmatrix}$$

- **Hàng 0 ($t_0$):** Chỉ được nhìn chính nó ($t_0$), che $t_1, t_2, t_3$.
- **Hàng 1 ($t_1$):** Được nhìn $t_0, t_1$, che $t_2, t_3$.
- **Hàng 2 ($t_2$):** Được nhìn $t_0, t_1, t_2$, che $t_3$.
- **Hàng 3 ($t_3$):** Được nhìn tất cả $t_0, t_1, t_2, t_3$.

Trong PyTorch, Causal Mask được tạo cực kỳ nhanh bằng hàm `torch.triu`:
```python
@staticmethod
def create_causal_mask(sequence_length: int, device: torch.device) -> torch.Tensor:
    return torch.triu(
        torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=device),
        diagonal=1
    )
```

### 3.2. Key Padding Mask
Padding Mask là ma trận boolean kích thước $[B, L]$ trong đó:
- `False`: Token hợp lệ (được chú ý).
- `True`: Token `<PAD>` (bị che).

Khi kết hợp với Causal Mask, nó được mở rộng chiều lên $[B, 1, 1, L]$ để có thể phát sóng (broadcasting) với ma trận Attention Scores $[B, H, L, L]$.

### 3.3. Phép kết hợp Mask (Combined Mask)
Hai mặt nạ được kết hợp bằng phép toán logic `OR` (`|`):

$$\text{Combined Mask} = \text{Causal Mask} \lor \text{Key Padding Mask}$$

Nếu một vị trí bị che bởi **bất kỳ** mặt nạ nào, giá trị ô đó sẽ thành `True`.

---

## 🧮 4. Toán Học Phép `masked_fill` và Softmax

Giả sử ma trận Attention Scores chưa mask $A = \frac{Q K^T}{\sqrt{d_k}}$ tại một vị trí có dạng:

$$A = \begin{bmatrix} 4.2 & 1.5 & 3.8 \\ 2.1 & 5.0 & 0.9 \end{bmatrix}$$

Khi áp dụng `masked_fill` với `combined_mask`, tất cả các vị trí `True` sẽ bị thay thế bằng $-\infty$ (hoặc `torch.finfo(dtype).min`):

$$A_{\text{masked}} = \begin{bmatrix} 4.2 & -\infty & -\infty \\ 2.1 & 5.0 & -\infty \end{bmatrix}$$

Khi đưa qua hàm **Softmax** theo chiều ngang (`dim=-1`):

$$\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_{j} e^{x_j}}$$

Vì $e^{-\infty} = 0$, kết quả trọng số chú ý tại các vị trí bị che sẽ **bằng 0 tuyệt đối**:

$$\text{Attention Weights} = \begin{bmatrix} 1.0 & 0.0 & 0.0 \\ 0.04 & 0.96 & 0.0 \end{bmatrix}$$

Mô hình hoàn toàn bị triệt tiêu sự chú ý tới các từ trong tương lai và các thẻ `<PAD>`!

---

## 💻 5. Phân Tích Mã Nguồn PyTorch

Mã nguồn được cài đặt tại [models/masked_multi_head_self_attention.py](../models/masked_multi_head_self_attention.py).

```python
class MaskedMultiHeadSelfAttention(nn.Module):
    def forward(
        self, 
        text_embeddings: torch.Tensor, 
        padding_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        batch_size, sequence_length, _ = text_embeddings.shape

        # 1. Pre-LayerNorm
        normalized_text = self.input_norm(text_embeddings)

        # 2. Tạo Q, K, V
        query, key, value = self.qkv_projection(normalized_text)

        # 3. Tính QKᵀ / sqrt(head_dim) -> [B, H, L, L]
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale

        # 4. Tạo Causal Mask [1, 1, L, L]
        causal_mask = self.create_causal_mask(sequence_length, text_embeddings.device)
        combined_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        # 5. Kết hợp Padding Mask [B, 1, 1, L]
        if padding_mask is not None:
            key_padding_mask = padding_mask[:, None, None, :]
            combined_mask = combined_mask | key_padding_mask

        # 6. Che các vị trí bị cấm bằng -inf
        attention_scores = attention_scores.masked_fill(
            combined_mask,
            torch.finfo(attention_scores.dtype).min
        )

        # 7. Softmax tạo trọng số chú ý
        attention_weights = torch.softmax(attention_scores, dim=-1)
        dropped_weights = self.attention_dropout(attention_weights)

        # 8. Nhân với V và chiếu đầu ra (Output Projection)
        attention_output = torch.matmul(dropped_weights, value)
        _, projected_output = self.output_projection(attention_output)

        # 9. Residual Connection
        output = text_embeddings + projected_output

        return output, projected_output, attention_weights
```

---

## 💡 6. Tóm Tắt Khác Biệt Giữa Self-Attention trong Encoder vs Decoder

| Tiêu chí | Encoder Self-Attention | Decoder Masked Self-Attention |
| :--- | :--- | :--- |
| **Mục đích** | Hiểu toàn bộ ngữ cảnh hình ảnh/văn bản | Sinh văn bản tự hồi quy từng từ một |
| **Causal Mask** | ❌ Không sử dụng (được nhìn 2 chiều) | ✅ **Bắt buộc dùng** (chỉ nhìn về quá khứ) |
| **Padding Mask** | ✅ Có dùng (nếu có chuỗi đầu vào) | ✅ Có dùng (che thẻ `<PAD>`) |
| **Ma trận chú ý** | Đầy đủ $L \times L$ | Dạng tam giác dưới (Lower Triangular) |

---

## 🏁 Kết Luận

Cơ chế **Masked Multi-Head Self-Attention** cùng sự kết hợp của **Causal Mask** và **Padding Mask** là thành phần sống còn giúp khối Text Decoder trong Transformer hoạt động chính xác. Nó vừa cho phép huấn luyện song song cực nhanh (Teacher Forcing) vừa đảm bảo mô hình giữ đúng nguyên tắc tự hồi quy khi sinh ra câu caption từng từ một ở thực tế.

# Text Transformer Decoder trong mô hình Image Captioning

Tài liệu này mô tả đúng luồng xử lý đang được cài đặt trong package `models`, chủ yếu ở các file:

- `models/_2_text_embedding.py`
- `models/_3_0_text_transformer_decoder.py`
- `models/_3_1_masked_multi_head_self_attention.py`
- `models/_3_2_cross_attention.py`
- `models/_3_3_text_decoder_feed_forward.py`
- `models/_4_0_language_modeling_head.py`

Ký hiệu được sử dụng:

| Ký hiệu | Ý nghĩa |
|---|---|
| `B` | batch size |
| `L` | số token trong caption đầu vào (`text_length`) |
| `P` | số image patch (`num_patches`) |
| `D` | kích thước mô hình (`d_model`) |
| `H` | số attention head (`num_heads`) |
| `Dh` | kích thước mỗi head, `Dh = D / H` |
| `Dff` | kích thước lớp ẩn của Feed-Forward (`d_ff`) |
| `V` | kích thước vocabulary |

Luồng tổng quát:

```text
decoder_input_ids [B, L]
        │
        ▼
TextEmbedding (token embedding + positional embedding)
        │ [B, L, D]
        ▼
┌──────────────── TextDecoderBlock × num_layers ────────────────┐
│ Masked Multi-Head Self-Attention → Cross-Attention → FFN      │
└────────────────────────────────────────────────────────────────┘
        │ [B, L, D]
        ▼
Final LayerNorm
        │ decoder_features [B, L, D]
        ▼
Language Modeling Head
        │
        ▼
logits [B, L, V]
```

## 1. Đầu vào của Text Transformer Decoder

### 1.1. `text_embeddings`

`TextTransformerDecoder.forward()` không nhận trực tiếp token ID mà nhận:

```text
text_embeddings: [B, L, D]
```

Tensor này được `TextEmbedding` tạo từ `decoder_input_ids: [B, L]` theo công thức:

```text
text_embeddings = Dropout(
    TokenEmbedding(decoder_input_ids) × sqrt(D)
    + PositionEmbedding(0, 1, ..., L - 1)
)
```

- **Token embedding** biến mỗi token ID thành một vector `D` chiều chứa biểu diễn ngữ nghĩa có thể học được.
- Việc nhân với `sqrt(D)` giữ độ lớn của token embedding ở mức phù hợp khi cộng với positional embedding.
- **Learned positional embedding** cung cấp thông tin thứ tự. Attention tự thân không biết token nào đứng trước hoặc sau nếu không có tín hiệu vị trí.
- Trong huấn luyện, `decoder_input_ids` là caption đã bỏ token cuối (`caption_ids[:, :-1]`). Target là caption đã bỏ token đầu. Đây là cơ chế teacher forcing để token tại vị trí `t` dự đoán token kế tiếp.

### 1.2. `visual_features`

```text
visual_features: [B, P, D]
```

Đây là đặc trưng ảnh do `VisionTransformerEncoder` sinh ra. Mỗi trong số `P` vector biểu diễn một image patch đã mang thông tin ngữ cảnh từ encoder. `D` phải bằng `d_model` của decoder để các phép chiếu trong cross-attention hoạt động.

### 1.3. `padding_mask`

```text
padding_mask: [B, L] hoặc None
True  = vị trí PAD cần bị che
False = token hợp lệ
```

Mask này được dùng trong masked self-attention để token văn bản không lấy thông tin từ các **key** là PAD. Nó không phải causal mask; causal mask được decoder tự tạo.

Lưu ý về implementation hiện tại: padding mask chỉ che chiều key, không che riêng các vị trí PAD khi chúng đóng vai trò query. Điều này không làm token hợp lệ chú ý đến PAD, và loss cũng bỏ qua PAD bằng `ignore_index`, nhưng output tại chính các vị trí PAD vẫn có thể khác 0.

## 2. Masked Multi-Head Self-Attention

Mục tiêu của bước này là để mỗi token tổng hợp ngữ cảnh từ phần caption đã xuất hiện, nhưng tuyệt đối không nhìn thấy các token tương lai. Điều kiện này khiến quá trình huấn luyện song song vẫn nhất quán với lúc inference sinh từng token theo kiểu autoregressive.

### Bước 1 — Pre-LayerNorm

```text
Xn = LayerNorm(X)
```

LayerNorm chuẩn hóa đặc trưng của từng token trước attention, giúp độ lớn activation ổn định và gradient đi qua mạng nhiều block dễ hơn. Đây là kiến trúc **Pre-LN**. Nhánh residual vẫn giữ `X` chưa chuẩn hóa để bảo toàn đường truyền thông tin gốc.

### Bước 2 — Tạo Query, Key và Value rồi chia head

```text
Q = Xn Wq, K = Xn Wk, V = Xn Wv
[B, L, D] → [B, H, L, Dh]
```

Trong self-attention, `Q`, `K`, `V` đều đến từ cùng chuỗi văn bản:

- `Q` mô tả thông tin mà token hiện tại đang tìm kiếm.
- `K` mô tả nội dung mà mỗi token có thể được truy vấn.
- `V` là thông tin thực tế sẽ được tổng hợp.

Chia thành nhiều head cho phép các không gian con học những quan hệ khác nhau, chẳng hạn quan hệ cú pháp, chủ thể–động từ hoặc các cụm từ mô tả thuộc tính.

### Bước 3 — Tính attention score có scale

```text
S = QKᵀ / sqrt(Dh)
[B, H, L, Dh] @ [B, H, Dh, L] → [B, H, L, L]
```

Tích vô hướng lớn nghĩa là query và key tương thích. Chia cho `sqrt(Dh)` để score không tăng quá lớn khi số chiều tăng; nếu không, softmax dễ bão hòa, phân phối attention trở nên quá nhọn và gradient nhỏ.

### Bước 4 — Tạo causal mask

Với `L = 4`, mask là:

```text
False  True  True  True
False False  True  True
False False False  True
False False False False
```

`True` nghĩa là bị che. Hàng `i` là query tại vị trí `i`, cột `j` là key tại vị trí `j`. Vì mọi ô `j > i` bị che, token chỉ được chú ý đến chính nó và quá khứ. Nếu không có mask này, khi train token có thể đọc đáp án nằm ở vị trí tương lai, gây rò rỉ thông tin.

### Bước 5 — Kết hợp padding mask

Causal mask `[L, L]` được mở rộng thành `[1, 1, L, L]`. Padding mask `[B, L]` được đổi thành `[B, 1, 1, L]`, rồi kết hợp bằng phép OR:

```text
combined_mask = causal_mask OR key_padding_mask
```

Nhờ broadcasting, cùng causal mask áp dụng cho mọi batch và head, còn padding mask vẫn khác nhau theo từng mẫu. Các score bị che được gán giá trị hữu hạn nhỏ nhất của dtype, nên xác suất của chúng sau softmax xấp xỉ 0.

### Bước 6 — Softmax và dropout

```text
A = softmax(S_masked, dim=-1)
```

Softmax chạy trên chiều key cuối cùng, biến score của mỗi query thành một phân phối xác suất trên các token được phép nhìn thấy. `attention_dropout` ngẫu nhiên bỏ một phần liên kết trong lúc train để giảm overfitting. Tensor được trả về dưới tên `attention_weights` là giá trị **trước dropout**, thuận tiện cho việc quan sát attention.

### Bước 7 — Tổng hợp Value

```text
O = Dropout(A) V
[B, H, L, L] @ [B, H, L, Dh] → [B, H, L, Dh]
```

Mỗi token nhận tổng có trọng số của các value trong quá khứ. Vì vậy biểu diễn của token không còn độc lập mà đã chứa ngữ cảnh caption trước đó.

### Bước 8 — Ghép head và output projection

```text
[B, H, L, Dh] → [B, L, H × Dh] = [B, L, D]
projected_output = Dropout(O_merged Wo)
```

Các head phải được ghép lại để quay về không gian `D` chiều. Ma trận `Wo` không chỉ đổi shape; nó học cách phối hợp thông tin từ tất cả head thành một biểu diễn chung.

### Bước 9 — Residual connection

```text
self_attention_residual = X + projected_output
```

Residual giữ lại biểu diễn đầu vào và chỉ yêu cầu attention học phần thông tin bổ sung. Nó cũng tạo đường truyền gradient trực tiếp, giúp stack nhiều decoder block ổn định hơn.

## 3. Cross-Attention giữa văn bản và ảnh

Masked self-attention trả lời câu hỏi “caption đã nói gì?”. Cross-attention bổ sung câu hỏi “ở ảnh có thông tin nào liên quan đến token đang xử lý?”. Đây là cầu nối để decoder sinh caption dựa trên ảnh thay vì chỉ hoạt động như một language model.

### Bước 1 — Chuẩn hóa text state

```text
Tn = LayerNorm(T)
```

Implementation chỉ Pre-LayerNorm nhánh text trước khi tạo query. `visual_features` từ encoder không được LayerNorm thêm bên trong `CrossAttention`.

### Bước 2 — Query từ văn bản

```text
Q = Tn Wq
[B, L, D] → [B, H, L, Dh]
```

Query đại diện cho nhu cầu thông tin của từng vị trí caption. Chẳng hạn sau ngữ cảnh “a dog is”, query kế tiếp có thể tìm vùng ảnh chứa hành động của con chó.

### Bước 3 — Key và Value từ ảnh

```text
K = VisualFeatures Wk
V = VisualFeatures Wv
[B, P, D] → [B, H, P, Dh]
```

Key giúp decoder đánh giá patch nào phù hợp với query; Value mang nội dung thị giác của patch đó. Tách nguồn theo cách `Q` từ text, còn `K/V` từ ảnh chính là lý do cơ chế này được gọi là **cross-attention**.

### Bước 4 — Đo độ tương thích text–patch

```text
S_cross = QKᵀ / sqrt(Dh)
[B, H, L, Dh] @ [B, H, Dh, P] → [B, H, L, P]
```

Mỗi phần tử biểu diễn mức phù hợp giữa một vị trí text và một image patch trên một head. Phép scale có cùng mục đích ổn định softmax như trong self-attention.

### Bước 5 — Softmax trên các image patch

```text
A_cross = softmax(S_cross, dim=-1)
```

Softmax chạy trên `P`, vì mỗi query văn bản cần phân bổ sự chú ý lên các vùng ảnh. Cross-attention không dùng causal mask: toàn bộ ảnh đã có sẵn trước khi caption được sinh, nên token có thể quan sát tất cả patch. Implementation hiện tại cũng không nhận visual padding mask vì ảnh được chia thành số patch cố định.

### Bước 6 — Lấy tổng có trọng số của visual Value

```text
O_cross = Dropout(A_cross) V
[B, H, L, P] @ [B, H, P, Dh] → [B, H, L, Dh]
```

Kết quả tại mỗi vị trí text là biểu diễn thị giác được chọn lọc theo nội dung caption. Các head có thể chuyên chú ý đến các loại tín hiệu khác nhau như vật thể, màu sắc, vị trí hoặc hành động.

### Bước 7 và 8 — Ghép head, chiếu đầu ra và residual

```text
projected_cross = Dropout(ConcatHeads(O_cross) Wo)
cross_attention_residual = T + projected_cross
```

Output projection trộn thông tin từ các head và đưa tensor về `[B, L, D]`. Residual cộng kết quả thị giác vào text state đã đi qua self-attention, nhờ đó biểu diễn cuối đồng thời giữ ngữ cảnh ngôn ngữ và bằng chứng từ ảnh.

Tensor `cross_attention_weights: [B, H, L, P]` cho biết mỗi token đang chú ý đến patch nào và có thể được dùng để phân tích hoặc trực quan hóa mô hình.

## 4. Feed-Forward Network: tại sao cần và cách hoạt động

Attention chủ yếu **trộn thông tin giữa các vị trí** bằng tổng có trọng số. Nếu chỉ có attention, khả năng biến đổi phi tuyến trên đặc trưng của từng token còn hạn chế. Feed-Forward Network (FFN) đảm nhiệm việc **xử lý đặc trưng tại từng vị trí** sau khi thông tin ngôn ngữ và thị giác đã được tập hợp.

Trong code:

```text
Xn = LayerNorm(X)
F  = Dropout(Linear_Dff→D(
         Dropout(GELU(Linear_D→Dff(Xn)))
     ))
decoder_output = X + F
```

Shape thay đổi như sau:

```text
[B, L, D] → [B, L, Dff] → [B, L, D]
```

- Lớp `Linear(D, Dff)` mở rộng không gian đặc trưng, mặc định từ `512` lên `2048`, tạo thêm dung lượng để học các tổ hợp đặc trưng.
- `GELU` đưa phi tuyến vào mô hình. Không có activation, hai lớp Linear liên tiếp về bản chất có thể rút gọn thành một phép biến đổi tuyến tính duy nhất.
- Lớp `Linear(Dff, D)` nén kết quả về `D` để có thể cộng residual và truyền sang block kế tiếp.
- Dropout giúp regularization.
- Pre-LayerNorm và residual có vai trò ổn định huấn luyện giống hai attention sublayer.

Cùng một FFN được áp dụng độc lập cho mọi vị trí token: nó không trộn vị trí `i` với vị trí `j`. Việc trao đổi thông tin giữa các vị trí đã do attention đảm nhiệm. Tuy cùng dùng một bộ trọng số, mỗi token tạo output khác nhau vì input của nó chứa ngữ cảnh khác nhau.

## 5. Đầu ra của Decoder

### 5.1. Đầu ra của một `TextDecoderBlock`

Mỗi block trả về:

```text
decoder_output:          [B, L, D]
self_attention_weights: [B, H, L, L]
cross_attention_weights:[B, H, L, P]
```

`decoder_output` trở thành `text_states` đầu vào cho block kế tiếp. Các attention weight phục vụ quan sát và phân tích; chúng không được truyền làm state sang block sau.

### 5.2. Đầu ra của `TextTransformerDecoder`

Sau `num_layers` block, code áp dụng thêm:

```text
decoder_features = final_norm(decoder_states)
```

Kết quả có shape:

```text
decoder_features: [B, L, D]
```

Final LayerNorm là cần thiết trong stack Pre-LN để chuẩn hóa state cuối trước khi đưa sang head dự đoán.

Theo mặc định, decoder chỉ trả `decoder_features`. Nếu gọi với `return_attention_weights=True`, kết quả là:

```python
(
    decoder_features,
    all_self_attention_weights,   # list dài num_layers
    all_cross_attention_weights,  # list dài num_layers
)
```

Mỗi phần tử trong hai list lần lượt có shape `[B, H, L, L]` và `[B, H, L, P]`.

### 5.3. Từ decoder feature đến token dự đoán

`decoder_features` chưa phải xác suất và cũng chưa phải token ID. `LanguageModelingHead` thực hiện một phép chiếu tuyến tính:

```text
logits = decoder_features W_vocab + b_vocab
[B, L, D] → [B, L, V]
```

Vector `logits[b, t, :]` chứa điểm số cho toàn bộ vocabulary tại vị trí `t`. Khi train, logits được so sánh với token kế tiếp bằng cross-entropy. Khi inference, hệ thống lấy phân phối ở vị trí cuối để chọn token tiếp theo, nối token đó vào chuỗi rồi chạy decoder lại.

Không nên áp dụng softmax trong `LanguageModelingHead` khi dùng `CrossEntropyLoss`, vì loss này đã kết hợp `log_softmax` bên trong để ổn định số học. Softmax chỉ cần khi muốn diễn giải logits thành xác suất trong quá trình sinh hoặc phân tích.

## Tóm tắt vai trò của ba sublayer

| Sublayer | Nguồn Q | Nguồn K/V | Trộn thông tin theo | Vai trò chính |
|---|---|---|---|---|
| Masked Self-Attention | Text | Text | Các token hiện tại và quá khứ | Hiểu ngữ cảnh caption mà không nhìn tương lai |
| Cross-Attention | Text | Image patches | Các vùng của ảnh | Đưa bằng chứng thị giác vào từng vị trí caption |
| Feed-Forward | Từng text state | — | Các chiều đặc trưng tại cùng một vị trí | Biến đổi phi tuyến và tăng năng lực biểu diễn |

Toàn bộ decoder vì thế luân phiên giữa ba việc: hiểu phần câu đã sinh, tìm thông tin liên quan trong ảnh, rồi biến đổi sâu biểu diễn kết hợp để dự đoán token tiếp theo.

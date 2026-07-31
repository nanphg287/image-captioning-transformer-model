# Multi-Head Attention (MHSA) trong mô hình Transformer

## MHSA là gì ?

* Cơ chế Multi-Head Attention là một cơ chế thực hiện nhiều hoạt động Self Attention song song, mỗi hoạt động có tập hợp
  các phép chiếu Q, K và V riêng, sau đó kết hợp đầu ra của chúng thành một biểu diễn duy nhất phong phú
  hơn $$Multi-Head Attention = Multi + Head + Attention$$
    * Từ `Head` có nghĩa là một sự chú ý độc lập
    * Từ `Multi` có nghĩa là ta vận hành nhiều `Head` song song
    * Từ `Attention` có nghĩa là mỗi `Head` quyết định mức độ tập trung vào từng `Head` khác </br></br>
* Vì vậy, thay vì nhìn vào câu bằng một cặp mắt, mô hình nhìn vào câu bằng nhiều cặp mắt cùng một lúc, và mỗi cặp mắt
  tập trung vào một khía cạnh khác nhau của toàn bộ Token
* Thử lấy một ví dụ tượng tự:
    * Giả sử ta đang đọc một câu và muốn hiểu đầy đủ ý nghĩa của nó
    * Một người đọc câu đó và tập trung vào người thực hiện hành động
    * Một người khác đọc cùng câu đó và tập trung vào nơi hành động diễn ra
    * Một người khác nữa tập trung vào lý do tại sao hành động đó xảy ra
    * Cuối cùng, ta thu thập ý kiến từ mỗi người và kết hợp chúng lại để có được sự hiểu biết cuối cùng về câu đó
* Cơ chế Multi-Head Attention được giới thiệu trong bài báo nổi
  tiếng "[Attention Is All You Need](https://arxiv.org/abs/1706.03762)" năm 2017

## Tóm tắt nhanh về Self Attention

(Có thể xem ở file **[SELF_ATTENTION.md](./SELF_ATTENTION.md)**)

## Tại sao cần khả năng Chú ý đa chiều?

* Hãy xem xét câu sau: `"I Love AI"`
* Đây là 1 câu ngắn nhưng ẩn chứa nhiều mối quan hệ khác nhau:
    * `I` là chủ ngữ, `Love` là hành động, `AI` là đối tượng mà chủ thể yêu thích
    * `I` và `Love` hình thành mqh chủ ngữ-động từ
    * `Love` và `AI` hình thành mqh động từ-tân ngữ
    * `I` và `AI` được kết nối thông qua ý nghĩa của `Love`
* Vậy câu hỏi đặt ra là: liệu một trung tâm chú ý duy nhất có thể học được tất cả các mối quan hệ này cùng một lúc hay
  không?
* Câu trả lời là **không**. Một bộ não tự chú ý chỉ có thể học một loại quan hệ tại một thời điểm. Nếu một bộ não đang
  bận
  học mối liên hệ chủ ngữ-động từ, nó không thể đồng thời học mối liên hệ động từ-tân ngữ với cùng một sự tập trung
* Vì vậy nên, khả năng tập trung đa chiều (Multi-Head Attention) xuất hiện
* Với nhiều bộ xử lý, mỗi bộ xử lý có thể tập trung vào một loại mối quan hệ khác nhau. Một bộ xử lý có thể học mối liên
  hệ chủ ngữ-động từ. Một bộ xử lý khác có thể học mối liên hệ động từ-tân ngữ. Một bộ xử lý khác có thể học mối liên hệ
  tầm xa giữa chủ ngữ và tân ngữ. Và cứ thế tiếp tục
* Bằng cách này, mô hình sẽ hiểu cùng context một cách toàn diện hơn

## Nguyên lý hoạt động từng bước của cơ chế MHSA

* Giả sử chúng ta có một chuỗi đầu vào với `d_model = 512` và chúng ta muốn sử dụng `h = 8` heads.
* Ở đây, `d_model` là kích thước của vectơ đại diện cho mỗi token. Nói một cách đơn giản, mỗi token trong câu được
  chuyển
  đổi thành một danh sách các số, `d_model = 512` có nghĩa là mỗi token được biểu diễn bằng 512 số. Số này càng lớn, mỗi
  token càng có thể mang nhiều thông tin hơn
* **Bước 1**: Từ Input là một vector embedding của Token, ta chia nó ra thành 8 phần như đã quy định 8 heads, như vậy
  mỗi head sẽ làm việc với một khía cạnh nhỏ hơn $$d_k = \frac{d_{model}}{h} = \frac{512}{8} = 64$$
* **Bước 2**: Với mỗi đầu vào, ta sử dụng các Phép chiếu tuyến tính (Các ma trận trọng số ấy) đã học được của từng Head
  để tạo ra các vector $Q,K,V$:
```math
\begin{aligned} Q_i=X \cdot W_Q^i \\ K_i=X \cdot W_K^i \\ V_i=X \cdot W_V^i \end{aligned} 
  ```
Ở đây: `i` tương ứng với head thứ `i` mà ta chia ra ban đầu

  ```
                  Input token embedding
                     (d_model = 512)
                           |
       +-------------------+-------------------+
       |                   |                   |
   own W_Q, W_K, W_V   own W_Q, W_K, W_V   own W_Q, W_K, W_V
       |                   |        ...        |
       ↓                   ↓                   ↓
    Q1  K1  V1          Q2  K2  V2          Q8  K8  V8
    (each 64)           (each 64)           (each 64)
      Head 1              Head 2              Head 8
  ```
Ở đây, ta có thể thấy rằng cùng một embedding đầu vào có kích thước 512 được đưa vào mọi head, nhưng mỗi head sử
dụng ma trận trọng số riêng để tạo ra các ma trận trọng số `Q`, `K`, `V` nhỏ hơn của riêng nó với kích thước
64</br></br>
* **Bước 3**: Mỗi `head` giờ đây thực hiện Chức năng Self-Attention một cách độc lập bằng cách sử dụng các ma trận $Q_i,
  K_i, V_i$ riêng của nó: $$head_i = Attention(Q_i, K_i, V_i) = softmax(\frac{Q_i \cdot K_i^T}{\sqrt{d_k}}) \cdot V_i$$
    * Cả 8 head đều thực hiện việc này song song</br></br>
* **Bước 4**:  Chúng ta nối kết quả đầu ra của tất cả các `head` lại với nhau: $$concat(head_1, ..., head_8)$$
    * Sau khi nối lại, ta được một vectơ có kích thước $h \times d_k = 8 \times 64 = 512$, giống như `d_model`. Vì vậy,
      kích thước của vectơ đầu ra khớp với kích thước của vectơ đầu vào</br></br>
* **Bước 5**: Ta truyền đầu ra đã được nối qua thêm một lớp tuyến tính được huấn luyện để trộn thông tin từ tất cả các
  đầu ra: $$MultiHead(Q, K, V) = Concat(head_1, ..., head_8) \cdot W_O$$
    * Ở đây, $W_O$ là ma trận trọng số của lớp tuyến tính cuối cùng, được huấn luyện để kết hợp thông tin từ tất cả các
      head lại với nhau
    * Đây là kết quả cuối cùng (Output) của cơ chế Multi-Head Attention</br></br>
* Sơ đồ tổng quát:

```
                    Input Embeddings (512)
                              |
        +---------+-----------+-----------+----------+
        |         |                       |          |
        ↓         ↓          ...          ↓          ↓
      Head 1    Head 2                  Head N-1   Head N
        (each head creates its own Q, K, V of size 64)
        |         |                       |          |
        ↓         ↓                       ↓          ↓
     Output 1  Output 2      ...       Output N-1 Output N
       (64)      (64)                    (64)       (64)
        |         |                       |          |
        +---------+-----------+-----------+----------+
                              |
                              ↓
                     Concatenate -> (512)
                              |
                              ↓
                Linear Layer W_O (512 -> 512)
                              |
                              ↓
                     Final Output (512)
```

## Trường hợp sử dụng cơ chế Multi-Head Attention

* Cơ chế Multi-Head Attention được sử dụng ở ba vị trí bên trong mô hình Transformer:
    * **Encoder Self-Attention**: Mỗi token đầu vào sẽ chú ý đến mọi token đầu vào khác. Đây là cốt lõi của bộ mã hóa
    * **Decoder Masked Multi-Head Self Attention**: Mỗi token đầu ra chỉ chú ý đến chính nó và các token trước đó. Các
      token tiếp theo được che giấu bằng một mặt nạ (mask) ,bởi vì bộ giải mã tạo ra từng token một và không được phép
      nhìn vào các token tiếp theo
    * **Multi-Head Cross Attention**: Bộ giải mã chú ý đến đầu ra của bộ mã hóa. Ở đây, tín hiệu `Q` đến từ bộ giải mã,
      và tín hiệu `K`, `V` đến từ bộ mã hóa. Đây là cầu nối giữa bộ mã hóa và bộ giải mã</br></br>
* Như vậy, Multi-Head Attention là khối cấu trúc cốt lõi của toàn bộ mô hình Transformer

## Ưu điểm của Multi-Head Attention

1. **Biểu diễn phong phú hơn**: Mỗi `head` học một mô hình khác nhau, do đó kết quả cuối cùng nắm bắt được nhiều mối
   quan hệ cùng một lúc.
2. **Tính toán song song**: Tất cả các đầu đọc/ghi hoạt động song song, do đó tốc độ rất nhanh trên các GPU hiện đại.
3. **Tổng chi phí như nhau**: Vì mỗi `head` sử dụng một chiều nhỏ hơn ( d_model / h), nên tổng quá trình tính toán tương
   tự như một cơ chế chú ý lớn với chiều đầy đủ.
4. **Khả năng học tập tốt hơn**: Mỗi `head` tự học hỏi từ dữ liệu, do đó mô hình có thể nắm bắt cả các mô hình
   ngắn hạn và dài hạn cùng một lúc.
5. **Tính linh hoạt**: Các `head` token khác nhau có thể chuyên về các lĩnh vực khác nhau - cú pháp, ý nghĩa, vị trí,
   liên kết tầm xa, v.v.

* Mô hình Transformer ban đầu sử dụng 8 attention head. Các mô hình LLM hiện đại sử dụng số lượng attention head cao hơn
  nhiều. Nhiều mô hình LLM ngày nay sử dụng 32, 64, hoặc thậm chí nhiều hơn nữa attention head trong mỗi lớp, bởi vì
  nhiều attention head hơn có thể nắm bắt được nhiều mối quan hệ hơn trong dữ liệu</br></br>
* Việc có nhiều head cũng khiến quá trình suy luận tiêu tốn nhiều bộ nhớ hơn, vì vậy nhiều mô hình LLM hiện đại sử dụng
  một biến thể gọi là `Grouped Query Attention` , giúp duy trì chất lượng gần với Multi-Head Attention trong khi sử dụng
  ít bộ nhớ hơn nhiều
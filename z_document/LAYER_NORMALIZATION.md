# Layer Normalization trước khi đưa input vào Self Attention

* **Nhiệm vụ**: Giữ phân phối giá trị của vector đặc trưng ổn định khi dữ liệu đi qua nhiều lớp Attention và
  Feed-Forward,
  từ đó giúp mô hình train ổn định hơn và Gradient truyền tốt hơn

### Vấn đề xảy ra nếu không có LayerNorm

* Giả sử một token trong Transformer được biểu diễn bởi vector: $x = [x_1, x_2, \ldots, x_d]$
* Với `d_model = 4`: $x = [2,4,6,8]$
* Sau khi đi qua nhiều phép biến đổi như:
    * $Q = XW_Q, K = XW_K, V = XW_V$
    * $Attention(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$
    * $FFN(x) = \text{ReLU}(xW_1 + b_1)W_2 + b_2$
* Giá trị của vector đặc trưng có thể ngày càng lớn hoặc có phân bố rất khác
  nhau: $$[2,4,6,8]→[15,−23,48,7]→[125,−80,310,−41]$$
* Nếu Transformer có hàng chục hoặc hàng trăm layer, sự thay đổi này có thể khiến:
    * Gradient trở nên rất lớn hoặc rất nhỏ → dẫn đến **vanishing gradient** hoặc **exploding gradient**
    * Việc tối ưu bằng **Adam/AdamW** khó hơn.
    * Một số feature có độ lớn quá lớn, lấn át feature khác
    * Mạng sâu khó train
* **LayerNorm** được đặt vào để đưa vector về một thang đo ổn định hơn

### LayerNorm làm gì?

* Với một token $x = [x_1,x_2, \ldots, x_d]$ LayerNorm thực hiện 3 bước chính
    * Bước 1: Tính giá trị trung bình của toàn bộ input, đây gọi là Mean: $$\mu = \frac{1}{d}\sum_{i=1}^{d} x_i$$
        * Ví dụ : $x=[2,4,6,8]$ → $\mu = \frac{2+4+6+8}{4} = 5$</br></br>
    * Bước 2: Tính phương sai (variance) input: $$\sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i - \mu)^2$$
        * Ví dụ: $\sigma^2 = \frac{(2-5)^2 + (4-5)^2 + (6-5)^2 + (8-5)^2}{4} = 5$
        * Suy ra Std - Standard Deviation - Độ lệch chuẩn
          là: $\sigma = \sqrt{\sigma^2} = \sqrt{5} \approx 2.236$</br></br>
    * Bước 3: Chuẩn hóa (Normalization) input: $$\hat{x_i} = \frac{x_i - \mu}{\sigma}$$
        * Ví
          dụ: $\hat{x} = [\frac{2-5}{2.236}, \frac{4-5}{2.236}, \frac{6-5}{2.236}, \frac{8-5}{2.236}] \approx [-1.34, -0.45, 0.45, 1.34]$
        * Sau khi chuẩn hóa, Mean của vector mới là $0$ và Std là $1$.
* Như vậy sau khi chuẩn hóa, dù vector đầu vào có lớn cỡ nào, ví dụ như $[200,400,600,800]$ thì sau **LayerNorm**, phân
  bố của các giá trị vẫn gần giống với ban đầu $[-1.34, -0.45, 0.45, 1.34]$

### LayerNorm không chỉ đơn giản normalize

* Nếu chỉ luôn ép: $mean =0, std =1$ thì có thể làm mất khả năng biểu diễn của Neural Network
* Vì vậy nên **LayerNorm** bổ sung thêm 2 tham số học được: $\gamma$ và $\beta$
* Công thức đầy đủ: $$\hat{x_i} = \gamma \frac{x_i - \mu}{\sigma} + \beta$$
* Các giá trị này được Optimizer như AdamW cập nhật trong quá trình training

### LayerNorm trong mô hình Transformer

* Giả sử output của Transformer có shape $[B, N, D]$ ví dụ $[B, 196, 1024]$
* Khi thực hiện Normalization, LayerNorm sẽ normalize theo chiều cuối cùng $D$ (feature dimension) cho mỗi token
    * Ví dụ: Token $x = [x_1, x_2, \ldots, x_{1024}]$
    * LayerNorm tính riêng mean(token1), std(token1)

### LayerNorm được sử dụng như thế nào trong dự án Image Captioning này ?

* Dự án này sử dụng `Vision Transformer Encoder` theo kiến trúc `Pre-LayerNorm`. Nó sẽ chuẩn hóa các vector input
  embedding của từng Patch trước khi đưa vào các lớp Transformer tiếp theo. </br></br>

* Luồng xử lý Encoder sẽ có flow như
  sau: $$X \rightarrow LayerNorm \rightarrow Multi-Head Self-Attention \rightarrow Residual Add \rightarrow LayerNorm \rightarrow Feed Forward \rightarrow Residual Add$$

#### Đầu vào của Encoder:

* Là 196 Patch ảnh `16x16` được chia từ một bức ảnh `224x224`
* Mỗi Patch được biểu diễn bởi một vector embedding có kích thước `1024` (d_model = 1024)

#### Trước Self Attention, LayerNorm nhận Vector $X = [1024]$

* Pre-LayerNorm thực hiện $$\hat{X} = \text{LayerNorm}(X)$$
* LayerNorm được áp dụng độc lập cho từng patch, theo chiều `d_model = 1024`
* Sau đó mỗi Patch (Token) được Normalize độc lập với nhau theo công thức ở Phần 2
* Ví dụ nhỏ để thấy được cách hoạt động, ta chọn `d_model = 4`:
  * Có 3 Patch như sau: 
```math
  X = \begin{bmatrix} 10 & 20 & 30 & 40 \\ 1 & 2 & 3 & 4 \\ 100 & 200 & 300 & 400 \end{bmatrix} 
  ```
  * Mặc dù có độ lớn khác nhau nhưng cả 3 patch đều có pattern tương tự:
    ```text
    feature1 < feature2 < feature3 < feature4
    ```
  * Khi LayerNorm cho từng Patch, ta có kết quả:
```math
    LayerNorm(X) = \begin{bmatrix} -1.34 & 0.45 & 0.45 & 1.34 \\ -1.34 & 0.45 & 0.45 & 1.34 \\ -1.34 & 0.45 & 0.45 & 1.34 \end{bmatrix}
  ```
* Điều quan trọng không phải là tất cả patch sẽ giống nhau như ví dụ đặc biệt này, mà là LayerNorm làm cho
  Self-Attention không bị quá phụ thuộc vào độ lớn tuyệt đối của vector, mà dễ tập trung hơn vào cấu trúc tương đối giữa
  các feature.
* Sau khi chuẩn hóa Input thì các Input đã được Norm này sẽ được dùng để tính toán các Vector `Q,K,V`

#### Tại sao phải normalize trước khi tính Q và K?

* Self Attention tính score như sau: $Score_{ij} = \frac{Q_iK_j^T}{\sqrt{d_k}}$
* Giả sử vector của Patch có giá trị rất lớn: $Q_1 = [100,200], K_2=[300,400]$
    * Tích vô hướng sẽ là $Q_1 \cdot K_2^T = 100 \times 300 + 200 \times 400 = 30000 + 80000 = 110000$
    * Trong khi 1 Patch khác : $Q_3 = [1,2], K_4=[3,4]$ thì $Q_3 \cdot K_4^T = 11$ chênh lệch rất nhiều so với `110000`
    * Nếu sự khác biệt này chỉ đến từ việc activation của một patch có scale lớn, chứ không phải patch đó thực sự quan
      trọng hơn, Attention sẽ bị méo $$Softmax([110000, 11, \ldots]) \approx [1,0, 0, \ldots]$$ dẫn đến việc Attention
      quá tự tin vào 1 Patch
* Nhờ Pre-LayerNorm mà Attention score phản ánh tốt hơn quan hệ học được giữa các feature, thay vì đơn thuần bị thống
  trị bởi độ lớn activation
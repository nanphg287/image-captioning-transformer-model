# Cross Attention trong Decoder của mô hình Transformer

## Cross Attention là gì ?

* Cơ chế (Cross Attention) là một cơ chế trong đó một chuỗi xem xét một chuỗi khác, sử dụng các Truy vấn riêng của nó để
  đối chiếu với các Khóa và Giá trị của chuỗi kia
* Nói một cách đơn giản, Cross Attention = Cross + Attention:
    * Từ "`Cross`" có nghĩa là thông tin chuyển giao từ chuỗi này sang chuỗi khác
    * Từ "`Attention`" có nghĩa là mô hình quyết định mức độ tập trung vào từng phần của chuỗi kia
* Như vậy, một chuỗi đặt ra các câu hỏi, và một chuỗi khác cung cấp câu trả lời

## Tại sao ta cần Cross Attention ?

* Giả sử chúng ta muốn dịch câu tiếng Anh `How are you` sang tiếng Nga. Câu tiếng Nga tương ứng là `Как у тебя дела`
* Câu hỏi đặt ra là: khi mô hình tạo ra từ tiếng Nga `Как у`, làm sao nó biết được `Как у` nghĩa của từ đó How?
* Mô hình phải xem xét câu tiếng Anh gốc để tìm ra điều này. Nó phải xem xét đầu vào ("How are you") trong khi tạo ra
  từng từ của đầu ra ("Как у тебя дела")
* Như vậy, hiện tại ta có hai chuỗi:
    * Chuỗi từ nhập vào: "How are you?" (3 từ)
    * Chuỗi đầu ra: "Как у тебя дела" (4 từ)
* Lưu ý rằng hai chuỗi câu thậm chí không có cùng số từ. Câu tiếng Anh có ba từ, nhưng bản dịch tiếng Nga có 4 từ. Điều
  này xảy ra thường xuyên trong cuộc sống thực và cơ chế Cross Attention xử lý nó một cách tự nhiên
* Chuỗi đầu ra phải liên tục xem xét chuỗi đầu vào để tạo ra bản dịch chính xác. Từ tiếng Nga được tạo ra phải chú ý đến
  từ tiếng Anh có liên quan nhất
* Đây chính xác là những gì mà cơ chế Cross Attention thực hiện:
    * Chuỗi đầu ra chú ý đến chuỗi đầu vào
    * Hai chuỗi khác nhau, nhưng chuỗi này lại hướng sự chú ý đến chuỗi kia
    * Vì sự chú ý chuyển từ chuỗi này sang chuỗi khác, nên chúng ta gọi nó là Cross Attention - Chú ý chéo</br></br>

* Giả sử ta là một sinh viên đang viết bài kiểm tra. Ta có những suy nghĩ riêng trong đầu (đây là một chuỗi Token).
  Ta cũng có một cuốn sách tham khảo đang mở trên bàn (đây là một chuỗi Token khác)
    * Trong khi viết từng dòng của câu trả lời, ta xem xét những suy nghĩ của riêng mình (Truy vấn - Q) và kiểm tra sách
      tham khảo (Từ khóa - K và Giá trị - V) để chọn ra thông tin cần thiết
    * Đó chính xác là những gì mà Cross Attention thực hiện trong một mô hình. Một chuỗi đang đọc từ một chuỗi khác

## Query, Key, Value trong Cross Attention

* Đối với mỗi từ mà bộ giải mã tạo ra, cơ chế Cross Attention cần trả lời hai câu hỏi:
    * Những từ nào trong chuỗi đầu vào là quan trọng đối với tôi?
    * Tôi nên tập trung vào từng khía cạnh đến mức độ nào?
* Để trả lời những câu hỏi này, ta tạo ra ba vectơ:
    * **Truy vấn (Q)**: điều mà decoder đang tìm kiếm ở bước hiện tại
    * **Khóa (K)**: ý nghĩa của từng token đầu vào
    * **Giá trị (V)**: thông tin thực tế mà mỗi token đầu vào mang theo</br></br>
* Từ "Cross" trong cụm từ "Cross Attention" xuất phát từ một sự thật rất quan trọng:
    ```text
    "Truy vấn Q xuất phát từ một chuỗi, còn Khóa K và Giá trị V xuất phát từ một chuỗi khác"
    ```

* Đây chính là điều tạo nên sự "chú ý chéo". Hai chuỗi sự kiện khác nhau, nhưng chuỗi này đang chú ý đến chuỗi kia
* Nói một cách đơn giản, nếu đầu vào của ta là `How are you` và đầu ra là `Как у тебя дела`, thì:
    * Truy vấn Q đến từ phía Decoder, nơi tạo ra "Как у тебя дела" (chuỗi đầu ra output)
    * Khóa K và Giá trị V được lấy từ "How are you" (chuỗi đầu vào - input)
* Đây là điểm khác biệt chính giữa Self-Attention và Cross-Attention. Trong Self Attention, cả ba (Q, K, V) đều đến từ
  cùng một chuỗi. Trong Cross Attention, Q đến từ một chuỗi, trong khi K và V đến từ một chuỗi khác

## So sánh Self Attention và Cross Attention

| Tiêu chí                        | Self-Attention                                                | Cross-Attention                                                               |
|---------------------------------|---------------------------------------------------------------|-------------------------------------------------------------------------------|
| **Nguồn Query (Q)**             | Cùng một chuỗi đầu vào                                        | Chuỗi đầu ra / hidden state của Decoder                                       |
| **Nguồn Key (K)**               | Cùng một chuỗi đầu vào                                        | Chuỗi đầu vào / output của Encoder                                            |
| **Nguồn Value (V)**             | Cùng một chuỗi đầu vào                                        | Chuỗi đầu vào / output của Encoder                                            |
| **Masking**                     | Có thể sử dụng mask; trong Decoder thường dùng Causal Mask    | Thường không sử dụng Causal Mask                                              |
| **Hình dạng ma trận Attention** | Hình vuông `(N × N)`                                          | Hình chữ nhật `(độ dài output × độ dài input)`                                |
| **Mục đích**                    | Học và hiểu mối quan hệ giữa các phần tử trong cùng một chuỗi | Kết nối và tổng hợp thông tin giữa hai chuỗi hoặc hai nguồn dữ liệu khác nhau |
| **Ứng dụng phổ biến**           | Encoder Self-Attention, Decoder Masked Self-Attention         | Encoder-Decoder Attention, mô hình đa phương thức                             |

* Decoder của Transformer sử dụng cả hai loại cơ chế Attention:
    * Nó sử dụng cơ chế Self Attention trước để xem xét các từ mà nó đã tạo ra
    * Sau đó, nó sử dụng cơ chế Cross Attention để xem xét đầu ra của Decoder
    * Vì vậy, Cross Attention không thay thế cho Self Attetion. Chúng hoạt động cùng nhau bên trong Decoder</br></br>
* Cross Attention không bị che khuất (masking), còn trong Self Attention của Decoder một từ không được phép nhìn vào các
  từ tiếp theo , bởi vì mô hình chưa tạo ra chúng
* Nhưng trong Cross Attention, toàn bộ câu đầu vào đã được biết trước. Vì vậy, mọi từ đầu ra đều được phép nhìn vào mọi
  từ đầu vào mà không có bất kỳ hạn chế nào
* Khóa K và Giá trị V chỉ được tính toán một lần. Encoder đọc câu đầu vào một lần duy nhất và tạo ra Khóa K và Giá trị
  V. Sau đó, Decoder sử dụng lại cùng một Khóa K và Giá trị V cho mọi từ đầu ra mà nó tạo ra. Điều này làm cho cơ chế
  Cross Attention rất hiệu quả

## Cách thức hoạt động Step-by-Step của Cross Attention

* Ví dụ kinh điển về việc sử dụng phương pháp Cross Attention là mô hình Transformer gốc từ bài báo "Attention is all
  you
  need". Mô hình Transformer gồm hai phần:
    * **Encoder**: đọc chuỗi đầu vào và tạo ra một biểu diễn chi tiết của chuỗi đó
    * **Decoder**: tạo ra chuỗi đầu ra, từng token một
* Bên trong Decoder, thực chất có hai bước chú ý:
    * Đầu tiên, Decoder sử dụng Self Attention để xem xét các từ mà nó đã tạo ra
    * Sau đó, nó sử dụng Cross Attention để xem xét đầu ra của Encoder
    * Vậy nên có thể thấy, Cross Attention đóng vai trò là cầu nối giữa bộ mã hóa và bộ giải mã

### Step 1

* Encoder đọc chuỗi đầu vào và tạo ra một tập hợp các vectơ. Từ các vectơ này, ta tạo ra Khóa (K) và Giá trị (V)
    * $K = output_{encoder} \times W_K$
    * $V = output_{encoder} \times W_V$
* Encoder chỉ thực hiện việc này một lần. Sau đó, cùng một Khóa K và Giá trị V được sử dụng lại cho mọi token mà decoder
  tạo ra
* Đây chính là điều làm cho cơ chế Cross Attention trở nên hiệu quả

### Step 2

* Decoder lấy các token đã được tạo ra ở B1 và tạo ra các trạng thái ẩn masked của riêng nó. Từ các trạng thái ẩn này,
  ta tạo ra Truy vấn (Q)
    * $Q = masking_{decoder} \times W_Q$
* Ở đây, $W_Q, W_K, W_V$ là các ma trận trọng số mà mô hình học được trong quá trình huấn luyện

### Step 3

* Tính tích vô hướng của $Q$ với ma trận chuyển vị của $K$ tức là $K^T$. Đây là phép nhân ma trận cho ra Attention
  Score. Ta sử dụng ma trận chuyển vị để các hình dạng được căn chỉnh chính xác cho phép nhân
    * $Score = Q \cdot K^T$
* Score cho ta biết mức độ tương tác giữa mỗi token giải mã và mỗi token mã hóa. Điểm số càng cao nghĩa là sự khớp nối
  càng mạnh giữa truy vấn Q của token giải mã và khóa K của token mã hóa

### Step 4

* Điều chỉnh score bằng cách chia cho $\sqrt{d_k}$. Ở đây, $d_k$ là chiều không gian của vectơ Khóa - Key
    * $ScaledScore = \frac{Q \cdot K^T}{ \sqrt{d_k}}$
* Việc điều chỉnh tỷ lệ này được thực hiện để giữ cho các số nằm trong một phạm vi ổn định, tránh việc hàm softmax tạo
  ra các giá trị cực đoan, xem file [SELF_ATTENTION.md](./SELF_ATTENTION.md) (Phần: Toán học đằng sau hệ
  số $\sqrt{d_k}$)

### Step 5

* Áp dụng hàm softmax lên các điểm số đã được hiệu chỉnh. Điều này chuyển đổi các điểm số thành xác suất. Tổng của mỗi
  hàng trong ma trận lúc này bằng 1
    * $AttentionWeight = softmax(\frac{Q \cdot K^T}{\sqrt{d_k}})$
* Các trọng số này cho biết, đối với mỗi token giải mã, nó nên dành bao nhiêu sự chú ý cho mỗi token mã hóa

### Step 6

* Nhân các trọng số chú ý với ma trận Giá trị V. Điều này cho ra kết quả đầu ra cuối cùng
    * $Output = AttentionWeight \cdot V$
* Kết quả đầu ra là một vectơ mới cho mỗi vị trí của Decoder, được bổ sung thêm thông tin liên quan từ Encoder. Decoder
  sử dụng vectơ này để quyết định từ tiếp theo mà nó nên tạo ra </br></br>

Vậy công thức đầy đủ của Cross Attention là: $$Attention(Q, K, V) = softmax(\frac{Q \cdot K^T}{\sqrt{d_k}}) \cdot V$$

Công thức hoàn toàn giống với Self Attention. Sự khác biệt không nằm ở toán học. Sự khác biệt nằm ở nguồn gốc của Q, K
và V:

* Trong Cross Attention, Q đến từ Decoder, trong khi K và V đến từ Encoder
* Ta cũng đã thấy những điểm khác biệt khác trước đó: Cross Attention không bị che khuất masking, và ma trận chú ý của
  nó là hình chữ nhật, bởi vì hai chuỗi có thể có độ dài khác nhau

Luồng tương tác đầy đủ cho cơ chế Cross Attention:

```
       Encoder Output             Decoder Hidden State
      (Input Sequence)              (Output Sequence)
              |                             |
        +-----+-----+                       ↓
        ↓           ↓                       Q
        V           K                       |
        |           |                       |
        |           +-----------+-----------+
        |                       |
        |                       ↓
        |                    Q . K^T
        |                       |
        |                       ↓
        |              Divide by sqrt(d_k)
        |                       |
        |                       ↓
        |                    Softmax
        |                       |
        |                       ↓
        |               Attention Weights
        |                       |
        +-----------+-----------+
                    |
                    ↓
                Multiply
                    |
                    ↓
                 Output

```

* Ở đây, ta thấy Q đến từ bộ giải mã, trong khi K và V đến từ bộ mã hóa. Sau đó, Q và K được sử dụng để tính toán trọng
  số chú ý, và V được sử dụng để tính toán đầu ra cuối cùng

## Ví dụ đơn giản từng bước

* Giả sử chúng ta muốn dịch câu tiếng Anh "How are you" sang tiếng Tây Ban Nha: "Cómo estás"
* Decoder tạo ra các từ tiếng Tây Ban Nha từng từ một. Nó đã tạo ra từ đầu tiên là `Cómo`. Hãy xem nó dự đoán từ tiếp
  theo như thế nào. Nó vẫn chưa biết từ đó

#### Bước 1: Encoder tạo ra một Khóa (K) và một Giá trị (V) cho mỗi token tiếng Anh

* K cho `How`, K cho `are`, K cho `you`
* V cho `How`, V cho `are`, V cho `you`

Các Khóa K và Giá trị V này chỉ được tạo một lần. Decoder sẽ sử dụng lại chúng ở mọi bước

#### Bước 2: Decoder xem xét những gì nó đã tạo ra cho đến hiện tại (Cómo) và tạo ra một Truy vấn (Q) cho từ tiếp theo

#### Bước 3: Lấy truy vấn này và tính tích vô hướng với các vectơ khóa của cả ba từ tiếng Anh: `How`, `are`, và `you`

Điều này cho ra ba điểm số </br></br>

#### Bước 4: Chia các điểm số cho $\sqrt{d_k}$

#### Bước 5: Ap dụng hàm softmax để lấy trọng số chú ý

* Giả sử kết quả là:
    * Trọng số của `How` = $0.050$
    * Trọng số của `are` = $0.500$
    * Trọng số của `you` = $0.450$
* Như vậy, Decoder đang dành 5% sự chú ý cho `How`, 50% cho `are`, và 45% cho `you`. Nó đang tập trung vào `are` và
  `you`

#### Bước 6: Kết hợp các vectơ Giá trị V bằng cách sử dụng các trọng số này

* Điều này cho chúng ta một vectơ đầu ra:
    * $Output = 0.050 \times V_{(How)} + 0.500 \times V_{(are)} + 0.450 \times V_{(you)}$
* Vectơ đầu ra này chứa thông tin tiếng Anh quan trọng nhất cho từ tiếp theo

#### Bước 7: Cuối cùng, bộ giải mã sử dụng vectơ đầu ra này để dự đoán từ tiếp theo

* Nó chấm điểm cho mỗi từ trong vốn từ vựng tiếng Tây Ban Nha của mình và chọn từ có khả năng xuất hiện cao nhất
* Giả sử xác suất là:
    * `estas` = $0.70$
    * `gracias` = $0.20$
    * `Hola` = $0.10$
* Từ `estás` có xác suất cao nhất. Vì vậy, Decoder dự đoán `estás` là từ tiếp theo. Điều này hợp lý, bởi vì Decoder đang
  tập trung vào `are` và `you`, từ tiếng Tây Ban Nha duy nhất này `estás` mang ý nghĩa của cả hai từ đó
* Sau đó, Decoder sẽ cộng thêm `estás` vào kết quả đầu ra và lặp lại toàn bộ quá trình để dự đoán từ tiếp theo, cho đến
  khi câu hoàn chỉnh
* Ma trận chú ý hoàn chỉnh sẽ trông như sau:

```
                          English (Encoder)
                           How       are       you
                       +---------+---------+---------+
 Spanish     Cómo      |  0.800  |  0.100  |  0.100  |
 (Decoder)             +---------+---------+---------+
             estás     |  0.050  |  0.500  |  0.450  |
                       +---------+---------+---------+
```

* Ở đây, tổng của mỗi hàng bằng 1 do hàm softmax
* Mỗi hàng là một bước của quá trình xử lý đầu ra, và nó cho biết bước đó đã dành bao nhiêu sự chú ý cho mỗi từ tiếng
  Anh.
* Ma trận này không phải là ma trận vuông. Nó có 2 hàng (mỗi hàng tương ứng với một từ tiếng Tây Ban Nha) và 3 cột (mỗi
  cột tương ứng với một từ tiếng Anh), bởi vì hai chuỗi có độ dài khác nhau. Đây là dấu hiệu quan trọng của Cross
  Attention
* Trong Self Attention, ma trận luôn là ma trận vuông, bởi vì một chuỗi tự chú ý đến chính nó

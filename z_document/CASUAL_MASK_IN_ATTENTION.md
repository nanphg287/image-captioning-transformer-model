# Casual Mask trong Attention

* Như chúng ta đã biết, lớp Attention là thành phần quan trọng nhất của kiến trúc Transformer và là khối xây dựng then
  chốt đằng sau các mô hình ngôn ngữ lớn (LLM).
* Bên trong lớp chú ý, chúng ta sử dụng `Casual Masking` . Điều này đảm bảo rằng một token chỉ có thể chú ý đến chính nó
  và các token trong quá khứ, chứ không bao giờ chú ý đến các token trong tương lai
* `Casual Masking` là một cơ chế sử dụng `Mask Matrix` để chặn các vị trí trong tương lai trong ma trận điểm chú ý, đảm
  bảo mỗi token chỉ chú ý đến chính nó và các token trong quá khứ, chứ không bao giờ chú ý đến các token trong tương lai
* Hãy xem một ví dụ: một câu `"I love learning AI"`
* Vị trí các Token như sau:

```
[ I | love | learning | AI ]
  1     2        3       4
```

### Nếu không có sự che dấu Casual Masking

* Trong khi dự đoán "love" , mô hình cũng có thể chú trọng đến "learning" và "AI"
* Trong khi dự đoán về "learning" , người ta đã có thể nhìn thấy trước "AI"
* Đây không phải là điều ta mong muốn
* Ta không nên cho phép mô hình nhìn thấy tương lai, bởi vì làm như vậy sẽ dẫn đến rò rỉ thông tin
* Nếu các token trong tương lai hiển thị, mô hình có thể học các lối tắt bằng cách sử dụng các từ mà nó được cho là phải
  dự đoán làm đầu vào
* Điều này dẫn đến hiệu suất huấn luyện tốt một cách phi thực tế nhưng lại phá vỡ giả định cơ bản của mô hình ngôn ngữ:
  dự đoán token tiếp theo chỉ bằng cách sử dụng ngữ cảnh trong quá khứ
* Trong quá trình suy luận, các token tương lai không tồn tại
* Nếu mô hình dựa vào thông tin tương lai trong quá trình huấn luyện, nó sẽ thất bại khi được yêu cầu tạo văn bản từng
  bước một
* Casual Masking đảm bảo rằng các điều kiện huấn luyện phù hợp với các điều kiện suy luận, buộc mô hình phải học các mẫu
  ngôn ngữ thực tế thay vì ghi nhớ câu trả lời
* Vì vậy, nếu chúng ta cho phép mô hình nhìn thấy tương lai, việc huấn luyện sẽ dễ dàng hơn, nhưng quá trình tạo ra kết
  quả sẽ không chính xác

### Nếu áp dụng Casual Masking

* Trong khi dự đoán "love" , mô hình chú trọng đến "I" .
* Trong khi dự đoán về "learning" , nó chú trọng đến "I" và "love"
* Trong khi dự đoán về "AI" , nó chú trọng đến "I" , "love" và "learning"
* Ở đây, chúng ta không cho phép mô hình nhìn thấy tương lai, vì vậy nó buộc phải hoạt động chính xác như trong quá
  trình suy luận

### Cách triển khai Casual Masking

* Trong lớp Attention, ta tính toán ma trận `Attention Score`
* `Attention Score` là một con số cho biết mức độ liên quan của một token với một token khác
    * Mỗi token tự so sánh mình với tất cả các token khác
    * Điểm chú ý đo lường mức độ tập trung vào từng token
    * Điểm càng cao nghĩa là càng nhận được nhiều sự chú ý
    * Điểm thấp hơn đồng nghĩa với việc nhận được ít sự chú ý hơn</br></br>
* Kết quả là, mỗi token sẽ tính điểm với mọi token khác, tạo thành một ma trận 2D trong đó các hàng đại diện cho các
  token truy vấn và các cột đại diện cho các token khóa
* Chi tiết xem ở file [SELF_ATTENTION.md](./SELF_ATTENTION.md) (Phần `Toán học đằng sau Attention - Q,K,V`)
* Vì câu `"I love learning AI"` chứa 4 từ, nên điểm chú ý tạo thành ma trận `4 × 4`
* Mỗi hàng tương ứng với một mã truy vấn, và mỗi cột tương ứng với một giá trị token
* Mỗi giá trị trong ma trận này cho biết mức độ mà một token cần chú ý đến token khác trước khi chuẩn hóa (Softmax)
* Ví dụ về ma trận điểm chú ý (trước khi Masking):
    ```
                I    love   learning   AI
              --------------------------------
    I         | 2.0 | 1.0 |   3.0    | 0.5 |
    love      | 1.5 | 2.5 |   0.5    | 1.0 |
    learning  | 0.2 | 1.0 |   2.0    | 1.5 |
    AI        | 0.1 | 0.3 |   0.7    | 2.5 |
    ```

* Sau khi áp dụng Softmax để chuẩn hóa Attention Score:
    ```
                 I    love   learning    AI
              --------------------------------
    I         | 0.23 | 0.09 |  0.63   | 0.05 |
    love      | 0.21 | 0.58 |  0.08   | 0.13 |
    learning  | 0.08 | 0.17 |  0.47   | 0.28 |
    AI        | 0.07 | 0.08 |  0.12   | 0.73 |
    ```

* Ở giai đoạn này, mọi token đều có thể tương tác với mọi token khác, kể cả những token trong tương lai
    * Khi dự đoán "love" , mô hình không chỉ chú ý đến các token trong quá khứ mà còn cả các token trong tương lai :
      nó gán trọng số chú ý là `0,08` cho "learning" và `0,13` cho "AI" , điều này cho thấy sự rò rỉ thông tin
    * Trong khi dự đoán "learning" , mô hình đã chú ý đến từ khóa tương lai "AI" với trọng số chú ý là `0,28` , mặc dù "
      AI" không nên xuất hiện ở giai đoạn này
* Do vấn đề này, việc che giấu là cần thiết, và để làm điều đó, ta sử dụng ma trận `Casual Masking` để chặn các
  token trong tương lai khỏi việc được chú ý

### Ma trận Casual Masking

* Tạo một ma trận mặt nạ có cùng kích thước. Ma trận này xác định những vị trí nào được cho phép (1) và những vị trí
  nào bị chặn (0)
    ```
                I   love  learning   AI
              --------------------------
    I         | 1 |  0  |    0      | 0 |
    love      | 1 |  1  |    0      | 0 |
    learning  | 1 |  1  |    1      | 0 |
    AI        | 1 |  1  |    1      | 1 |
    ```
* Đây là ma trận masking và ta sẽ áp dụng nó vào ma trận điểm số để chặn các token trong tương lai
    * `0`: Sự chú ý sẽ bị chặn (Tương lai)
    * `1`: Được phép chú ý
* Sau khi áp dụng Masking, các vị trí bị chặn sẽ được đặt thành $- \infty$
* `Casual Masking` ngăn chặn việc truy cập vào các token trong tương lai bằng cách gán giá trị $- \infty$ cho các vị trí
  đó
* Ví dụ về ma trận Attention Score (trước khi che khuất):

```
            I     love   learning    AI
          --------------------------------
I         | 2.0 | 1.0 |    3.0    | 0.5 |
love      | 1.5 | 2.5 |    0.5    | 1.0 |
learning  | 0.2 | 1.0 |    2.0    | 1.5 |
AI        | 0.1 | 0.3 |    0.7    | 2.5 |

```

* Ma trận Attention Score đã được che giấu (sau khi che giấu bằng ma trận Casual Masking tạo ở trên)

```
            I     love   learning    AI
          --------------------------------
I         | 2.0 | -∞  |   -∞    | -∞  |
love      | 1.5 | 2.5 |   -∞    | -∞  |
learning  | 0.2 | 1.0 |   2.0   | -∞  |
AI        | 0.1 | 0.3 |   0.7   | 2.5 |
```

* Khi áp dụng Casual Masking, các vị trí được đánh dấu 0 trong ma trận masking sẽ bị chặn bằng cách đặt attention score
  thành −∞ , trong khi các vị trí được đánh dấu 1 vẫn giữ nguyên giá trị ban đầu trong ma trận attention score
* Vì ta phải áp dụng softmax, và do cách softmax hoạt động về mặt toán học , −∞ sẽ trở thành chính xác bằng 0 sau khi áp
  dụng softmax
* Nếu chúng ta sử dụng 0 thay thế, softmax vẫn sẽ gán một xác suất khác 0, cho phép mô hình chú ý một phần đến các token
  trong tương lai và làm mất đi mục đích của việc che giấu hệ quả
* Sau Softmax: Trọng số chú ý thu được:

```
            I     love   learning    AI
          --------------------------------
I         | 1.00 | 0.00 |  0.00 | 0.00 |
love      | 0.27 | 0.73 |  0.00 | 0.00 |
learning  | 0.11 | 0.24 |  0.65 | 0.00 |
AI        | 0.07 | 0.08 |  0.12 | 0.73 |
```

* Hiện tại, các token tương lai không nhận được bất kỳ sự quan tâm nào
    * Khi dự đoán từ "love" , mô hình chú ý đến từ "I" với trọng số chú ý là 0,27 và đến chính nó ( "love" ) với trọng
      số là 0,73 . Nó không chú ý đến các từ trong tương lai "learning" và "AI" , vì cả hai đều nhận được 0,00 điểm chú
      ý
    * Trong khi dự đoán "learning" , mô hình chú ý đến "I" ( 0,11 ) và "love" ( 0,24 ), và chính nó ("learning")
      với trọng số 0,65 . Nó không chú ý đến từ khóa tương lai "AI" , từ khóa này có trọng số chú ý là 0,00
    * Trong khi dự đoán "AI" , mô hình chú ý đến "I" ( 0,07 ), "love" ( 0,08 ) và "learning" ( 0,12 ), và chính nó
      ("AI") với 0,73
* Đây là cách mà Casual Masking ngăn cản mô hình nhận biết các token trong tương lai
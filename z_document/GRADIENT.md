# Gradient trong học máy

## Gradient

* Gradient là đại lượng thể hiện:
    * Một hàm số thay đổi nhanh đến mức nào
    * Hàm số tăng nhanh nhất theo hướng nào
* Trường hợp với hàm 1 biến $y=f(x)$, gradient chính là đạo hàm $f'(x)=\frac{df}{dx}$
* Ví dụ: $y=x^2$
    * Gradient: $f'(x)=2x$
    * Tại $x=3$, gradient là $6$
    * Điều này có nghĩa là tại $x=3$, nếu $x$ tăng một lượng rất nhỏ thì giá trị $f(x)$ có xu hướng tăng với tốc độ
      khoảng 6 lần lượng thay đổi của $x$.
        * $f′(x)>0$: hàm đang tăng.
        * $f ′(x)<0$: hàm đang giảm.
        * $f ′(x)=0$: điểm bằng phẳng, có thể là cực tiểu, cực đại hoặc điểm yên ngựa

## Gradient trong hàm nhiều biến

* Trong học máy, hàm thường phụ thuộc vào rất nhiều biến: $L(w_1,w_2,w_3,...,w_n)$
* Gradient là một vector gồm đạo hàm riêng theo nhiều biến:
```math
\nabla L = \left(\frac{\partial L}{\partial w_1}, \frac{\partial L}{\partial w_2}, \ldots, \frac{\partial L}{\partial w_n}\right) 
```
* Ví dụ: $f(x,y) = x^2 + 3y^2$
* Gradient:
```math
\nabla f = \left(\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}\right) = \begin{bmatrix}2x \\ 6y\end{bmatrix} 
```
* Tại $(x,y) = (2,1)$ : 
```math
\nabla f(2,1) = \begin{bmatrix} 4 \\ 6 \end{bmatrix}
```
* Vector này cho biết:
    * Hàm tăng nhanh nhất theo hướng $(4,6)$
    * Thay đổi x ảnh hưởng đến kết quả với mức độ 4.
    * Thay đổi y ảnh hưởng đến kết quả với mức độ 6.
    * Ngược lại hàm giảm nhanh nhất theo hướng $-\nabla f = (-4,-6)$

## Vai trò của Gradient trong học máy

* Gradient được sử dụng để tìm cách điều chỉnh các tham số của mô hình nhằm làm cho dự đoán ngày càng chính xác
* Quá trình huấn luyện thường có
  dạng: $$Input \rightarrow Model \rightarrow Prediction \rightarrow Loss \rightarrow Gradient \rightarrow Update$$
* **Loss** là con số đo mức độ sai của mô hình.
    * Loss lớn: mô hình dự đoán sai nhiều.
    * Loss nhỏ: mô hình dự đoán gần đúng.
    * Mục tiêu huấn luyện: tìm bộ tham số làm Loss nhỏ nhất.

* Giả sử mô hình có hàm Loss là $L$, trọng số của mô hình là $w$ suy ra hàm Loss của mô hình là $L(w)$
* Gradient của hàm Loss là : $$\nabla L = \frac{\partial L}{\partial w}$$
* Gradient này sẽ cho biết khi thay đổi $w$, Loss sẽ tăng hay giảm và thay đổi mạnh đến mức nào

## Gradient Descent

* Việc tìm **global minimum** của hàm mất mát trong Machine Learning thường rất khó hoặc bất khả thi.
* Thay vào đó, mô hình thường tìm một **local minimum** đủ tốt và coi đó là nghiệm gần đúng.
* Về lý thuyết, local minimum thường nằm tại điểm có đạo hàm bằng `0`, nhưng việc giải trực tiếp phương trình này rất
  khó với hàm phức tạp, dữ liệu nhiều chiều hoặc lượng dữ liệu lớn.
* **Gradient Descent (GD)** giải quyết vấn đề bằng cách bắt đầu từ một điểm ban đầu, sau đó lặp lại việc cập nhật tham
  số theo hướng làm giảm hàm mất mát cho đến khi gradient gần `0`.

### Gradient Descent của hàm 1 biến

* Với hàm chỉ có 1 biến $y=f(x)$, Gradient Descent là thuật toán giúp ta tìm giá trị $x$ sao cho $f(x)$ đạt cực tiểu,
  bằng cách liên tục di chuyển $x$ theo hướng làm hàm giảm nhanh nhất
* Ví dụ đơn giản: $y = x^2$
    * Ta biết bằng toán học rằng hàm này đạt minimum tại: $x=0$ với $y=0$
    * Nhưng hãy giả sử máy tính không biết trước $x=0$ và phải tự tìm
* **Đạo hàm cho biết hướng đi:**
    * Gradient: $f'(x) = 2x$
    * Nếu $f'(x)>0$ đồ thị đang dốc lên khi đi sang phải → muốn đi xuống thì phải đi sang trái
    * Nếu $f'(x)<0$ đồ thị đang dốc xuống khi đi sang phải → muốn đi xuống thì phải đi sang phải
    * Nếu $f'(x)=0$ mặt phẳng tại điểm đó nằm ngang → có thể đã tới cực trị
    * Gradient Descent sử dụng đúng thông tin này

#### Công thức Gradient Descent

* Với hàm một biến: $$x_{new} = x_{old} - \eta f'(x_{old})$$
* Trong đó:
    * $x_{old}$ : vị trí hiện tại
    * $f'(x_{old})$ : Độ dốc ở vị trí hiện tại (Gradient)
    * $\eta$ : Learning rate (tốc độ học) - một hằng số dương nhỏ, quyết định xem mỗi bước đi dài bao nhiêu
    * $x_{new}$ : Vị trí sau khi cập nhật
* Dấu trừ cực kỳ quan trọng: Bởi vì ta muốn đi ngược chiều đạo hàm để làm giá trị giảm xuống $$-\eta f'(x_{old})$$
* Descent nghĩa là đi ngược - vì có dấu trừ đó

### Gradient Descent cho hàm nhiều biến

* Gradient Descent cho hàm nhiều biến thực chất là sự mở rộng trực tiếp của trường hợp một biến
* Giả sử ta cần tìm global minimum cho hàm $f(\theta)$ trong đó $'theta$ là một vector, thường được dùng để ký hiệu tập
  hợp các tham số của một mô hình cần tối ưu
* Đạo hàm của hàm số đó tại một điểm $\theta$ bất kỳ được ký hiệu là $\nabla f(\theta)$
* Tương tự như hàm 1 biến, thuật toán **GD** cho hàm nhiều biến cũng bắt đầu bằng một điểm dự đoán $\theta_0$, sau đó, ở
  vòng lặp thứ $t$, quy tắc cập nhật là: $$\theta_{t+1} = \theta_t - \eta \nabla_{\theta} f(\theta_t)$$
* Vẫn là quy tắc cần nhớ: **luôn luôn đi ngược hướng với đạo hàm**
* Ví dụ: $f(x,y)=x^2 + y^2$
    * Chọn điểm bắt đầu $(x_0,y_0)=(4,3)$
    * Learning rate: $\eta = 0.1$
    * Gradient: $\nabla f(x,y) = [2x, 2y]$
    * Bước 0:
        * $\theta_0 = (4,3)$
        * Giá trị hàm: $f(4,3) = 16 + 9 = 25$
        * Gradient: $\nabla f(4,3) = [8,6] $
    * Bước 1:
        * Cập nhật $x$ : $x_1 = 4 - 0,1 \times 8 = 4 - 0.8 = 3.2$
        * Cập nhật $y$ : $y_1 = 3 - 0,1 \times 6 = 3 - 0.6 = 2.4$
        * Vậy $\theta_1 = (3.2, 2.4)$
        * Giá trị hàm mới: $f(3.2,2.4) = 3.2^2 + 2.4^2 = 16$
        * Loss đã giảm từ 25 -> 16
    * Tiếp tục lặp lại như trên ta có
        * Bước 2: Giá trị hàm mới $f(2.56,1.92) = 10.24$
        * Bước 3: Giá trị hàm mới $f(2.048,1.536) = 6.4$
        * Bước n: Giá trị hàm mới $f(x_n, y_n) = 0$

* Tổng quát cho hàm nhiều biến: $$f(x_1,...,x_n)$$
* Gradient: 
```math
\nabla f = 
\begin{bmatrix}
    \frac{\partial f}{\partial x_1} \\ 
    \frac{\partial f}{\partial x_2} \\ 
    ... \\ 
    \frac{\partial f}{\partial x_n}
\end{bmatrix}
```
* Gradient Descent: $$\theta_{t+1} = \theta_t - \eta \nabla_{\theta} f(\theta_t)$$
* Gradient Descent theo từng biến: 
```math
\begin{aligned} 
    x_1 \leftarrow x_1 - \eta \frac{\partial f}{\partial x_1} \\ 
    x_2 \leftarrow x_2 - \eta \frac{\partial f}{\partial x_2} \\ 
    ... \\ 
    x_n \leftarrow x_n - \eta \frac{\partial f}{\partial x_n} 
\end{aligned}
```
* Không có khác biệt bản chất giữa 2 biến và 1 triệu biến, chỉ có số lượng đạo hàm riêng nhiều hơn

### Gradient Descent cho hàm nhiều biến chính xác là những gì diễn ra trong Neural Network
* Một Neural Network có rất rất nhiều tham số $W_1,W_2,...,W_n$
* Loss là một hàm cực lớn $L(W_1,W_2,...,W_n)$
* Ta muốn tìm `min L`
* Backpropagation tính các đạo hàm : $\frac{\partial L}{\partial W_1}, \frac{\partial L}{\partial W_2}, \ldots, \frac{\partial L}{\partial W_n}$
* Tất cả ghép lại tạo thành gradient: $\nabla L$
* Sau đó Optimizer cập nhật: $W_i \leftarrow W_i - \eta \frac{\partial L}{\partial W_i}$
* Tất cả hàng triệu weight đều được cập nhật theo nguyên lý này
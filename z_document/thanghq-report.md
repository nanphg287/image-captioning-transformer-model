# Giải thích code by ThangHQ

Flow từ đầu đến cuối của chương trình này như sau

## Data preparation

Chỉ chạy 1 lần duy nhất hàm `prepare_training_data`, hàm này làm 3 việc

1. Gọi `prepare_dataset`: Tạo file `.json` lưu mapping giữa image_name và các captions của nó
2. Chia dataset thành 3 tập: Train + Test + Val dùng hàm `split_dataset`
3. Resize lại tất cả ảnh tập Train dùng hàm `resize_image`: Kích thước ảnh config được trong code. Hiện tại resize ảnh thành vuông 224\*224. Thực hiện resize bằng cách centre crop (giữ lại phần trung tâm kích thước 224*224, phần còn lại cắt hết)

Sample output bước 1:

```
Tổng số ảnh: 3
Tổng số caption: 15
Caption ít nhất mỗi ảnh: 5
Caption nhiều nhất mỗi ảnh: 5
Caption trung bình mỗi ảnh: 5.00
Project root: /home/gideon/image-captioning-transformer-model
Image directory: /home/gideon/image-captioning-transformer-model/datasets/raw/Images
Caption file: /home/gideon/image-captioning-transformer-model/datasets/raw/captions.txt
Output file: /home/gideon/image-captioning-transformer-model/datasets/processed/image_to_captions.json
Đã lưu mapping tại: /home/gideon/image-captioning-transformer-model/datasets/processed/image_to_captions.json
```

Sample output bước 2:

```
===== DATASET SPLIT STATISTICS =====
Tổng số ảnh: 3
Tổng số caption: 15

Train: 2 ảnh (66.67%), 10 caption
Validation: 0 ảnh (0.00%), 0 caption
Test: 1 ảnh (33.33%), 5 caption

Train file: /home/gideon/image-captioning-transformer-model/datasets/processed/train.json
Validation file: /home/gideon/image-captioning-transformer-model/datasets/processed/validation.json
Test file: /home/gideon/image-captioning-transformer-model/datasets/processed/test.json
```

Sample output bước 3:

```
------------------------------------------------------------
[1/2] Thành công: 47871819_db55ac4699.jpg
[2/2] Thành công: 42637987_866635edf6.jpg

===== KẾT QUẢ =====
Tổng số ảnh: 2
Resize thành công: 2
Resize thất bại: 0
Đã cập nhật JSON: /home/gideon/image-captioning-transformer-model/datasets/processed/train.json
```

QUESTION

1. Tại sao lại chỉ resize ảnh train? Nếu vậy ảnh khi inference đưa vào kích thước khác thì sao model đoán được tốt? Đây là vô tình hay cố ý?
2. Việc center crop sẽ gây mất dữ liệu ở góc. Đây là vô tình hay cố ý, hay là sai sót nhưng buộc phải chấp nhận?


Sample output đoạn PyTorch detect CUDA

```
Device: cuda:0
GPU: NVIDIA GeForce 930MX
```

## Transpose, chuyển về phân phối chuẩn

IMAGE_TRANSFORM thực hiện 2 bước sau

1. Convert về CHW
2. Normalize ảnh: Bản thân `ToTensor()` đã chuyển [0, 255] về [0, 1] rồi. Sau đó normalize để scale các con số không bị lệch nhiều --> Giúp training stable hơn.
3. Vector chứa `std` và `mean` đã được tính sẵn dựa trên toàn bộ ảnh trong dataset để đưa về phân phối chuẩn.

QUESTION

1. Tìm hiểu hàm class `ImageCaptionDataset` được dùng ở đâu?
2. `HWC` và `CHW` là gì? Trong code có hàm `ToTensor()` để chuyển từ HWC --> CHW, tại sao phải làm như thế?
3. `transforms.Normalize(mean=[...], std=[...])` hàm này trong code đang hard-code `mean` và `std` value. Tại sao chọn bộ mean và std này?

HWC and CHW refer to the physical order in which multidimensional image data (tensors) is stored in a computer's linear memory.

- **HWC (Height, Width, Channels)**: Also known as *channel-last* or interleaved format.
- **CHW (Channels, Height, Width)**: Also known as *channel-first* or planar format.

Images are essentially 3D matrices containing a height, a width, and color channels (like Red, Green, and Blue). However, RAM and GPU memory are strictly 1-dimensional. To store a 3D matrix, the system must "flatten" it into a 1D sequence of bytes. The data format dictates the logic of how this flattening occurs.

- **HWC (Interleaved)**: The memory stores all channel values for a single pixel before moving to the next pixel. If you read the linear memory sequentially, you see: R1, G1, B1, then R2, G2, B2, and so on.
- **CHW (Planar)**: The memory stores an entire channel (a 2D plane of pixels) before moving to the next channel. Reading sequentially, you see all Red values (R1, R2, R3...), followed by all Green values (G1, G2, G3...), and finally all Blue values.

[Tensor Physical Layouts on Memory](https://leimao.github.io/blog/Tensor-Physical-Layout-on-Memory/)

*(Note: In deep learning, you will often see this written as NHWC and NCHW, where N stands for the Batch size—the number of images processed at once).*

**Tại sao lại chọn HWC/CHW?** Phụ thuộc vào target hardware là CPU/GPU mà sử dụng, để tận dụng processing speed, cache locality, and hardware utilization during computation tối ưu theo từng loại hardware.

1. HWC dùng cho CPUs --> Libs hay dùng loại này là OpenCV, Matplotlib, PIL/Pillow, and TensorFlow/Keras (which initially optimized for legacy architectures before GPUs dominated).
2. CHW dùng cho GPUs --> PyTorch, ONNX, and NVIDIA's cuDNN library (built specifically from the ground up for GPU acceleration).

Ở trong PyTorch thì nó cung cấp sẵn hàm để convert HWC --> CHW rồi, ví dụ

```python
from torchvision import transforms
from PIL import Image

image_hwc = Image.open("dataset/sample.jpg")

# Automatically converts PIL Image (HWC) to a PyTorch Tensor (CHW)
transform = transforms.ToTensor()
tensor_chw = transform(image_hwc)
```

Do mình đang dùng thư viện Pillow (PIL) để load ảnh, nên phải dùng `transforms.ToTensor()` để convert về CHW.

## Patches

Project này sử dụng kiến trúc Vision Transform (ViT), thực chất xuất phát từ kiến trúc Transformer ban đầu được dùng cho text.

Paper của ViT là *An Image is Worth 16x16 Words*, với idea đó là treating an image like a sequence of "visual words", then feeding that sequence into a Transformer encoder. Instead of using convolution layers as the main feature extractor like CNNs, ViT splits the image into patches and lets self-attention learn relationships between all patches globally.

Đặc điểm của text là có thể tokenize by word, ví dụ "I", "Have", "a", "pen", etc. 

Nhưng với ảnh nếu một bức ảnh 224\*224 mà coi mỗi pixel == token thì KHÔNG thể đủ infrastructure để tính toán được --> Họ chia bức ảnh thành nhiều mảnh gọi là *patches*, với kích thước (trong project này) là 16*16. Lúc này patch == token trong kiến trúc ViT.

Config của patch size trong `Config.py`

**Linear Projection** là convert image patch --> fixed-size vector (format mà Transformer can process). Sau quá trình này, output được gọi là **patch embedding**.

Trong project này ảnh mình xử lý là `224*224`, với patch size = `16*16` --> Ra được 196 patches. Nói cách khác input vector của một ảnh đưa vào Linear Projection là [196, 768], tổng quát nếu thêm Batch size B vào nữa thì là [B, 196, 768]. Output vector của Linear Projection **không** quy định size, mà phải dựa vào *architecture*. Linear Projection thực hiện công thức rất đơn giản

$$
\text{patch embedding} = xW + b
$$

Nên output shape tổng quát là [B, 196, D] với $D$ là output size của Linear Projection. $D$ còn được gọi là `embedding_dim`, `hidden_size`, `model_dim`, `transformer dimension`.

Rất nhiều ViT implementation sử dụng $D = 768$

Vậy quy luật thật sự ở đây là gì? --> output size is chosen by architecture, not forced by patch size. Nên hoàn toàn có thể chọn 512, 768, 1024, etc. Luật duy nhất là phải match dimension và Encoder expect (vì output của Linear Projection sẽ feed vào Encoder, cụ thể là khớp với Encoder Hidden Size).

Các yếu tố ảnh hưởng đến $D$

1. Model capacity: Thông thường $D$ càng lớn thì patch token can store richer information. (vì chứa nhiều số trong đó hơn mà)
2. Transformer Encoder Hidden Size: The output of Linear Projection becomes the input to the Transformer Encoder.

```
Patch Projection output: [B, 196, 768]
Transformer expects:     [B, sequence_length, 768]
```

3. Số lượng attention heads: In Multi-Head Self-Attention, $D$ is split across attention heads. Tức là `head_dim = D / num_heads`, thông thường người ta sẽ chọn $D$ chia hết được cho `num_heads`, và số chiều của head `head_dims` thông thường là 64.

QUESTION

1. Trong công thức Linear Projection, nếu input là 768, mà output != 768 thì làm sao nhân được ma trận nhỉ?
2. Đọc file config thấy $D = 256$, số heads = 4 --> Số chiều của head = 64 (configuration khá lạ)
3. Tại sao cần Linear Projection? Vì suy cho thực chất thứ nó làm là đổi số chiều + tính toán đơn giản? --> Gợi ý: Tác dụng learning $W$ và $b$, output vector mới đại diện cho toàn bộ 768 values, chứ không đơn lẻ intensity value của 1 pixel, etc. 

## Add [CLS] Token

Sau khi Patch Embedding xong thì ra được shape $[B, 196, D]$, riêng với image classification mình có một bước đặc biệt: Add thêm special *learnable* token gọi là `[CLS] Token`.

Fact: CLS Token là một trick để giúp model create one global representation for classification. CLS Token này tại thời điểm Transformer ra đời chưa có, nó ra đời vào lúc BERT ra đời (NLP Transformer model), sau này ViT mượn ý tưởng này về.

Tại sao gọi CLS Token là "learnable"? Vì thực tế token này không phải patch thật (vì không có ảnh ứng với patch này). Nó là một vector có shape y hệt các patch embedding khác, trong quá trình training, CLS Token vector này sẽ được update liên tục, cuối cùng nó sẽ là "summary receiver" for the whole image. Chính bản thân CLS Token là một parameters được learn trong quá trình training, nó ngang hàng với $W$ và $b$.

ViT sẽ *prepend* token vào patch sequence. Vì vậy sau bước này số patch embedding tăng từ 196 --> 197 trở thành $[B, 197, D]$. Nôm na nó sẽ như này `[CLS], patch_1, patch_2, patch_3, ...`. Tại thời điểm ban đầu PyTorch random initialize CLS Token, sau này back-propagation sẽ update lại.

```
Patch embeddings: [B, 196, 768]
CLS token:        [B, 1, 768]
After concat:     [B, 197, 768]
```

## Add Positional embedding

Tại thời điểm này ta có `[CLS], patch_1, patch_2, patch_3, ..., patch_196`. Mặc dù mỗi patch đã chứa dữ liệu, nhưng không có dữ liệu về vị trí của patch trong bức ảnh (trên dưới trái phải, ...).

--> ViT add thêm Positional embedding (tất nhiên same shape)

```
Token embeddings:      [B, 197, D]
Positional embeddings: [1, 197, D]
Result:                [B, 197, D]
```

Bởi vì trong Computer Vision thì: Mũi trên Miệng vs Miệng trên Mũi nó rất khác nhau --> Vị trí của một patch trong bức ảnh rất quan trọng.

A positional embedding is a learnable vector assigned to each position in the sequence.

Ví dụ cụ thể: For ViT with [CLS]:

```
Number of tokens = 197
Embedding dimension = 768
```

So positional embedding has shape: [1, 197, 768], tức là

```
position_embedding_for_CLS      = vector of 768 numbers
position_embedding_for_patch_1  = vector of 768 numbers
position_embedding_for_patch_2  = vector of 768 numbers
...
position_embedding_for_patch_196 = vector of 768 numbers
```

Positional Embedding KHÔNG làm thay đổi shape của output, lí do vì nó là cộng matrix (khác với CLS là concat).

Fact: Có nhiều biến thể, kiểu cộng Positional Embedding với Patch Embedding chỉ là một cách phổ biến thôi. Còn một cách concat nữa cũng có trong thực tế, nhưng hiếm.

Does [CLS] also get positional embedding? 

## Transformer Encoder Block

[Transformers for Vision](https://d2l.ai/chapter_attention-mechanisms-and-transformers/vision-transformer.html)

Sau khi Positional Embeddings thì ảnh được feed qua hàng loạt Transformer Encoder Block. Thông thường một Encoder Block sẽ chứa các thành phần sau

```
1. Layer Normalization
2. Multi-Head Self-Attention
3. Residual Connection
4. Layer Normalization
5. MLP / Feed-Forward Network
6. Residual Connection
```

Biểu diễn đơn giản hơn thì nó như này

```
x = x + MultiHeadSelfAttention(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

Mặc dù một Encoder Block chứa rất nhiều layer, nhưng thực tế nó chỉ có 2 nhiệm vụ chính

1. **Self-attention**: patches talk to each other
2. **MLP/Feed-forward**: each patch thinks/processes its own updated information.

Còn lại Layer Normalization và Residual Connection chỉ có tác dụng stablize training và preserve information thôi.

| Component | Simple meaning |
| :--- | :--- |
| LayerNorm | Clean/stabilize token values |
| Multi-Head Self-Attention | Let tokens communicate |
| Residual Connection | Keep old information while adding new information |
| MLP | Further process each token individually |

### LayerNorm

[Transformer Architecture](https://waylandz.com/llm-transformer-book/)

LayerNorm means **normalize each token vector**. Nó chính là re-scale 768 values trong vector về một khoảng giá trị nào đó. Bởi vì có thể trong vector đôi khi gặp những value rất bé/rất lớn --> Gây ra neuron network mất tính ổn định.

Phương pháp normalize có nhiều, ví dụ: Min-max scaling, z-score normalization, etc. 

Trong PyTorch thường dùng `nn.LayerNorm` thì nó dùng z-score normalization.

### Multi-Head Self-Attention

Đây là phần CỰC KỲ QUAN TRỌNG.

**Problem**: Mình có 196 patches

```
Patch 1 = information about top-left image area
Patch 2 = information about next image area
Patch 50 = information about middle image area
...
```

Mỗi patch ứng với một mảnh nhỏ của bức ảnh. Nhưng để hiểu được cả bức ảnh cần hiểu được *relationship* giữa các patch.

Ví dụ ảnh chó: ear patch + eye patch + nose patch + body patch = dog

--> Self attention: Cho phép mỗi patch ask question "Which other patches are important to me?".

Ta có 197 patch tokens (1 CLS + 196 patch tokens) --> Với mỗi token ta có 3 vector $Q$ $K$ và $V$.

| Vector | Human meaning |
| :--- | :--- |
| Query | What am I looking for? |
| Key | What do I contain? |
| Value | What information can I provide? |

--> Với 1 ảnh có 197 token --> có 197 query vectors + 197 key vectors + 197 value vectors.

$${\displaystyle \text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V}$$

Trong đó $QK^T$ (tử số) được gọi là *attention score*.

Sau đó áp dụng softmax vào để tính *attention_weights* = $\text{softmax}(\frac{QK^T}{\sqrt{d_k}})$

Pre-LN Transformer block

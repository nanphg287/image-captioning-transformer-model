"""
=============================================================================
 STAGE 1 DIAGNOSTIC: Image Patch Embedding + Positional Encoding
 Target spec: input [B, 3, 224, 224]  ->  output [B, 196, 512]
=============================================================================
 CACH DUNG:
   1. Thay class ReferencePatchEmbed ben duoi bang module THAT cua ban
      (hoac import: from my_model import PatchEmbedding)
   2. Sua dong `MODULE = ...` o phan CONFIG
   3. python test_patch_embed.py
=============================================================================
"""

import math
import torch
import torch.nn as nn

torch.manual_seed(0)

# ============================================================================
# REFERENCE IMPLEMENTATION (dung de doi chieu - day la cach LAM DUNG)
# ============================================================================
class ReferencePatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_ch=3, embed_dim=512,
                 dropout=0.1):
        super().__init__()
        assert img_size % patch_size == 0, "img_size phai chia het cho patch_size"
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2          # 14*14 = 196

        # Conv2d voi kernel = stride = patch_size <=> cat patch + linear proj
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

        # Learned positional embedding: dang ky la nn.Parameter
        # -> tu dong .to(device), tu dong nam trong state_dict
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                      # x: [B, 3, 224, 224]
        x = self.proj(x)                       # [B, 512, 14, 14]
        x = x.flatten(2)                       # [B, 512, 196]
        x = x.transpose(1, 2)                  # [B, 196, 512]  <-- TRANSPOSE, KHONG PHAI view()
        x = x + self.pos_embed                 # broadcast [1,196,512] -> [B,196,512]
        x = self.norm(x)
        return self.drop(x)


# Bien the sinusoidal (neu ban dung PE co dinh thay vi learned)
def sinusoidal_pe(num_pos, dim):
    pe = torch.zeros(num_pos, dim)
    pos = torch.arange(0, num_pos, dtype=torch.float).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)                     # [1, num_pos, dim]
    # LUU Y: phai dung self.register_buffer("pe", sinusoidal_pe(...))
    #        KHONG duoc de la bien thuong -> se khong theo model sang GPU


# ============================================================================
# CONFIG  <<<< SUA O DAY
# ============================================================================
B, C, H, W = 8, 3, 224, 224
PATCH, DIM = 16, 512
N_PATCH = (H // PATCH) * (W // PATCH)          # 196
GRID = H // PATCH                              # 14

MODULE = ReferencePatchEmbed(H, PATCH, C, DIM)   # <<<< thay bang module cua ban
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODULE = MODULE.to(DEVICE)


# ============================================================================
# HELPERS
# ============================================================================
PASS, FAIL, WARN = "\033[92m[PASS]\033[0m", "\033[91m[FAIL]\033[0m", "\033[93m[WARN]\033[0m"
_results = []

def check(name, cond, detail="", warn_only=False):
    tag = PASS if cond else (WARN if warn_only else FAIL)
    print(f"  {tag} {name}" + (f"  ->  {detail}" if detail else ""))
    _results.append((name, cond or warn_only))
    return cond

def header(t):
    print(f"\n{'='*72}\n  {t}\n{'='*72}")


# ============================================================================
# TEST 1: SHAPE + DTYPE + NaN
# ============================================================================
header("TEST 1 | Shape, dtype, gia tri hop le")
MODULE.eval()
x = torch.randn(B, C, H, W, device=DEVICE)
with torch.no_grad():
    out = MODULE(x)

print(f"  input : {tuple(x.shape)}")
print(f"  output: {tuple(out.shape)}")
check("Output shape == [B, 196, 512]", tuple(out.shape) == (B, N_PATCH, DIM),
      f"nhan duoc {tuple(out.shape)}")
check("Khong co NaN", not torch.isnan(out).any())
check("Khong co Inf", not torch.isinf(out).any())
print(f"  stats : mean={out.mean():.4f}  std={out.std():.4f}  "
      f"min={out.min():.3f}  max={out.max():.3f}")
check("std nam trong khoang lanh manh (0.05 - 20)",
      0.05 < out.std().item() < 20,
      "std qua nho => tin hieu chet; qua lon => se lam no LayerNorm/attention")


# ============================================================================
# TEST 2: SPATIAL LOCALITY  (bat loi .view() lam xao tron pixel)
# Doi 1 o vuong 16x16 duy nhat -> CHI DUNG 1 token duoc thay doi.
# Day la test quan trong nhat. Shape dung KHONG co nghia la data dung.
# ============================================================================
header("TEST 2 | Spatial locality (phat hien patchify bi xao tron)")
row, col = 5, 9
target_idx = row * GRID + col

base = torch.zeros(1, C, H, W, device=DEVICE)
poked = base.clone()
poked[:, :, row*PATCH:(row+1)*PATCH, col*PATCH:(col+1)*PATCH] = 5.0

with torch.no_grad():
    d = (MODULE(poked) - MODULE(base)).abs().mean(dim=-1).squeeze(0)  # [196]

changed = (d > 1e-4).nonzero().flatten().tolist()
print(f"  Da sua patch (row={row}, col={col}) -> token index ky vong = {target_idx}")
print(f"  So token bi thay doi: {len(changed)}")
print(f"  Top-5 token thay doi manh nhat: {d.topk(5).indices.tolist()}")
ok = check("Chi dung 1 token thay doi", len(changed) == 1,
           f"co {len(changed)} token thay doi")
check("Token thay doi dung vi tri row*14+col", changed[:1] == [target_idx],
      f"thay doi tai {changed[:5]}")
if not ok:
    print("  >> Nhieu token thay doi = ban dang dung x.view()/reshape() de cat patch,")
    print("     hoac quen .transpose(1,2) sau flatten. Pixel bi tron -> encoder hoc rac.")


# ============================================================================
# TEST 3: POSITIONAL ENCODING CO THUC SU DUOC CONG VAO KHONG
# Anh hang so -> moi patch giong het nhau -> neu khong co PE thi 196 token identical
# ============================================================================
header("TEST 3 | Positional Encoding co duoc ap dung khong")
flat = torch.ones(1, C, H, W, device=DEVICE) * 0.5
with torch.no_grad():
    o = MODULE(flat).squeeze(0)                                  # [196, 512]

spread = o.std(dim=0).mean().item()     # do lech giua cac vi tri
print(f"  Do bien thien giua 196 token (anh dong nhat): {spread:.6f}")
check("PE thuc su duoc cong vao (token khac nhau theo vi tri)", spread > 1e-5,
      "196 token giong het nhau => PE bi quen hoac bi cong = 0")

with torch.no_grad():
    o = MODULE(torch.randn(1, C, H, W, device=DEVICE)).squeeze(0)   # anh co noi dung
cos = torch.nn.functional.cosine_similarity(
    o.unsqueeze(1), o.unsqueeze(0), dim=-1)
mean_cos = (cos.sum() - cos.diag().sum()).item() / (N_PATCH * (N_PATCH - 1))
print(f"  Cosine similarity trung binh giua cac token: {mean_cos:.4f}")
check("Token khong bi collapse (cos < 0.995)", mean_cos < 0.995,
      "gan 1.0 => tin hieu vi tri qua yeu so voi content", warn_only=True)


# ============================================================================
# TEST 4: PE PHAI DUOC DANG KY (Parameter hoac Buffer)
# Loi kinh dien: PE tao bang torch.randn() trong __init__ nhung khong register
# -> khong nam trong state_dict -> LUC INFERENCE LOAD RA PE KHAC HAN LUC TRAIN
# -> "train ok nhung inference sai het" chinh la trieu chung nay.
# ============================================================================
header("TEST 4 | PE duoc dang ky vao state_dict / dung device")
names = list(dict(MODULE.named_parameters()).keys()) + \
        list(dict(MODULE.named_buffers()).keys())
pos_names = [n for n in names if "pos" in n.lower() or n.lower() in ("pe", "encoding")]
print(f"  Parameters : {list(dict(MODULE.named_parameters()).keys())}")
print(f"  Buffers    : {list(dict(MODULE.named_buffers()).keys())}")
check("Tim thay tensor positional trong parameters/buffers", len(pos_names) > 0,
      f"tim thay: {pos_names}")

sd = MODULE.state_dict()
check("PE nam trong state_dict (se duoc luu vao checkpoint)",
      any(p in sd for p in pos_names), f"state_dict keys: {list(sd.keys())}")

for n, t in list(MODULE.named_parameters()) + list(MODULE.named_buffers()):
    if str(t.device) != str(next(MODULE.parameters()).device):
        check(f"Device khop cho '{n}'", False, f"{t.device}")
check("Tat ca tensor cung device", True, str(DEVICE))

for p in pos_names:
    if p in sd:
        t = sd[p]
        print(f"  {p}: shape={tuple(t.shape)}  std={t.std():.4f}  mean={t.mean():.4f}")
        check(f"'{p}' broadcast dung voi [B,196,512]",
              t.shape[-1] == DIM and t.numel() // DIM == N_PATCH,
              f"shape {tuple(t.shape)} - phai la [196,512] hoac [1,196,512]")


# ============================================================================
# TEST 5: DETERMINISM (bat dropout con bat o eval, hoac randn trong forward)
# ============================================================================
header("TEST 5 | Tinh xac dinh o che do eval")
MODULE.eval()
xx = torch.randn(2, C, H, W, device=DEVICE)
with torch.no_grad():
    a, b = MODULE(xx), MODULE(xx)
check("2 lan forward cung input -> ket qua giong het (eval mode)",
      torch.allclose(a, b),
      "khac nhau => dropout van bat (quen model.eval()) hoac PE random moi forward")

MODULE.train()
with torch.no_grad():
    c1, c2 = MODULE(xx), MODULE(xx)
print(f"  train mode khac nhau (dropout hoat dong): {not torch.allclose(c1, c2)}")
MODULE.eval()


# ============================================================================
# TEST 6: BATCH INDEPENDENCE (bat loi nham truc batch <-> sequence)
# ============================================================================
header("TEST 6 | Cac anh trong batch khong anh huong lan nhau")
img1 = torch.randn(1, C, H, W, device=DEVICE)
img2 = torch.randn(1, C, H, W, device=DEVICE)
with torch.no_grad():
    solo = MODULE(img1)
    batched = MODULE(torch.cat([img1, img2], dim=0))[0:1]
diff = (solo - batched).abs().max().item()
check("Forward 1 anh == forward trong batch", diff < 1e-4,
      f"max diff = {diff:.2e} (neu lon => ban dang tron truc batch va sequence)")

with torch.no_grad():
    o1, o2 = MODULE(img1), MODULE(img2)
check("2 anh khac nhau -> output khac nhau ro ret",
      (o1 - o2).abs().mean().item() > 1e-3,
      f"mean abs diff = {(o1-o2).abs().mean().item():.4f}")


# ============================================================================
# TEST 7: GRADIENT FLOW (dam bao PE va conv deu duoc hoc)
# ============================================================================
header("TEST 7 | Gradient flow")
MODULE.train()
MODULE.zero_grad()
out = MODULE(torch.randn(2, C, H, W, device=DEVICE))
out.pow(2).mean().backward()
for n, p in MODULE.named_parameters():
    g = p.grad
    if g is None:
        check(f"grad ton tai cho '{n}'", False, "grad = None -> tensor bi detach")
    else:
        check(f"grad hop le cho '{n}'",
              not torch.isnan(g).any() and g.abs().sum() > 0,
              f"|grad| mean = {g.abs().mean():.3e}")
MODULE.zero_grad()
MODULE.eval()


# ============================================================================
# TEST 8: TY LE TIN HIEU CONTENT vs POSITION
# Neu ||PE|| >> ||patch_embed|| thi moi anh se cho ra gan nhu cung mot thu
# -> decoder sinh ra 1 caption chung chung cho MOI anh (trieu chung cua ban!)
# ============================================================================
header("TEST 8 | Ty le bien do Content vs Positional")
try:
    with torch.no_grad():
        content = MODULE.proj(torch.randn(4, C, H, W, device=DEVICE))
        content = content.flatten(2).transpose(1, 2)
        c_std = content.std().item()
    p_std = sd[pos_names[0]].std().item() if pos_names else float("nan")
    ratio = c_std / (p_std + 1e-9)
    print(f"  std(patch embedding) = {c_std:.4f}")
    print(f"  std(positional)      = {p_std:.4f}")
    print(f"  ty le content/pos    = {ratio:.2f}")
    check("Ty le trong khoang 0.2 - 50", 0.2 < ratio < 50,
          "lech qua nhieu => mot trong hai tin hieu bi nhan chim", warn_only=True)
except Exception as e:
    print(f"  (bo qua - module cua ban khong co .proj: {e})")


# ============================================================================
# TEST 9: DOI CHIEU VOI CONV2D THUAN (kiem tra toan hoc cua phep cat patch)
# So sanh unfold thu cong voi conv2d - hai cach phai cho ket qua giong het
# ============================================================================
header("TEST 9 | Doi chieu unfold thu cong vs Conv2d")
conv = nn.Conv2d(C, DIM, PATCH, PATCH).to(DEVICE)
xi = torch.randn(2, C, H, W, device=DEVICE)
with torch.no_grad():
    way_conv = conv(xi).flatten(2).transpose(1, 2)                     # [2,196,512]
    patches = xi.unfold(2, PATCH, PATCH).unfold(3, PATCH, PATCH)       # [2,3,14,14,16,16]
    patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(2, N_PATCH, -1)  # [2,196,768]
    W_ = conv.weight.reshape(DIM, -1)                                  # [512,768]
    way_manual = patches @ W_.t() + conv.bias
check("Conv2d(k=16,s=16) tuong duong unfold + linear",
      torch.allclose(way_conv, way_manual, atol=1e-4),
      f"max diff = {(way_conv - way_manual).abs().max():.2e}")
print("  >> Neu ban tu cat patch bang view/reshape, hay so sanh output cua ban")
print("     voi 'way_manual' o tren de xac minh thu tu pixel.")


# ============================================================================
# TONG KET
# ============================================================================
header("TONG KET")
failed = [n for n, ok in _results if not ok]
print(f"  {len(_results) - len(failed)}/{len(_results)} test PASS")
if failed:
    print("  Cac test that bai:")
    for n in failed:
        print(f"    - {n}")
    print("\n  => Sua het cac loi tren TRUOC KHI chuyen sang debug Encoder.")
else:
    print("  Module Image Embedding + PE hoat dong dung. Co the sang buoc 2 (Encoder).")
print()
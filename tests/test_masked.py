"""
=============================================================================
 STAGE 2 DIAGNOSTIC: Causal Mask / Teacher Forcing / Inference
=============================================================================
 CACH DUNG: sua phan CONFIG, import model + vocab that cua ban, roi chay.
 Cac test 1-2 khong can checkpoint. Test 3-4 can model da train.
=============================================================================
"""
import torch, torch.nn as nn

# ============================ CONFIG  <<<< SUA O DAY =========================
# from my_model import ImageCaptioningTransformer
# from my_data   import Vocabulary, train_loader
# model = ImageCaptioningTransformer(...).to(DEVICE)
# model.load_state_dict(torch.load("checkpoint.pt", map_location=DEVICE))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PASS, FAIL = "\033[92m[PASS]\033[0m", "\033[91m[FAIL]\033[0m"
def check(name, cond, detail=""):
    print(f"  {PASS if cond else FAIL} {name}" + (f"  ->  {detail}" if detail else ""))
def header(t): print(f"\n{'='*72}\n  {t}\n{'='*72}")


# =============================================================================
# TEST 1 | RO RI THONG TIN TUONG LAI  (kiem tra causal mask)
# Nguyen ly: co dinh prefix, doi cac token PHIA SAU vi tri t.
# Logits tai vi tri t PHAI khong doi. Neu doi -> causal mask bi ro.
# =============================================================================
def test_causal_leak(model, vocab, visual_features, seq_len=10):
    header("TEST 1 | Ro ri thong tin tuong lai (causal mask)")
    model.eval()
    torch.manual_seed(0)
    ids_a = torch.randint(4, vocab.size, (1, seq_len), device=DEVICE)
    ids_b = ids_a.clone()
    t = seq_len // 2
    ids_b[:, t+1:] = torch.randint(4, vocab.size, (1, seq_len-t-1), device=DEVICE)

    with torch.inference_mode():
        pm_a = (ids_a == vocab.pad_token_id)
        pm_b = (ids_b == vocab.pad_token_id)
        la = model.decode_captions(decoder_input_ids=ids_a,
                                   visual_features=visual_features,
                                   decoder_padding_mask=pm_a)
        lb = model.decode_captions(decoder_input_ids=ids_b,
                                   visual_features=visual_features,
                                   decoder_padding_mask=pm_b)

    leak = (la[:, :t+1, :] - lb[:, :t+1, :]).abs().max().item()
    print(f"  Doi token tai vi tri {t+1}..{seq_len-1}, do logits tai vi tri 0..{t}")
    print(f"  max |delta logit| = {leak:.3e}")
    check("Khong ro ri thong tin tuong lai", leak < 1e-5,
          "logits qua khu bi anh huong boi token tuong lai => CAUSAL MASK SAI")
    if leak >= 1e-5:
        print("  >> Kiem tra: nn.Transformer.generate_square_subsequent_mask(L)")
        print("     phai la ma tran TAM GIAC TREN chua -inf, truyen vao tham so")
        print("     `tgt_mask` (KHONG phai memory_mask), va shape phai la [L, L].")


# =============================================================================
# TEST 2 | QUY UOC PADDING MASK
# PyTorch: True = BO QUA vi tri do. Dao nguoc => model chi nhin thay pad.
# =============================================================================
def test_padding_convention(model, vocab, visual_features):
    header("TEST 2 | Quy uoc padding mask (True = bo qua)")
    model.eval()
    real = torch.randint(4, vocab.size, (1, 6), device=DEVICE)
    ids = torch.cat([real, torch.full((1, 4), vocab.pad_token_id, device=DEVICE)], 1)
    with torch.inference_mode():
        correct = model.decode_captions(
            decoder_input_ids=ids, visual_features=visual_features,
            decoder_padding_mask=(ids == vocab.pad_token_id))
        no_pad = model.decode_captions(
            decoder_input_ids=real, visual_features=visual_features,
            decoder_padding_mask=(real == vocab.pad_token_id))
    d = (correct[:, :6, :] - no_pad).abs().max().item()
    print(f"  max |delta| giua 'co pad + mask' va 'khong pad' = {d:.3e}")
    check("Pad token bi mask dung (khong anh huong token that)", d < 1e-4,
          "pad dang ro vao attention => quy uoc mask bi dao nguoc")


# =============================================================================
# TEST 3 | TEACHER FORCING vs AUTOREGRESSIVE tren anh TRONG TAP TRAIN
# Day la test chan doan quan trong nhat.
# =============================================================================
def test_tf_vs_ar(model, vocab, image_tensor, caption_ids, generate_fn, max_length=25):
    header("TEST 3 | Teacher forcing vs Autoregressive (anh trong tap TRAIN)")
    model.eval()
    with torch.inference_mode():
        vf = model.encode_images(image_tensor)
        inp = caption_ids[:, :-1]
        logits = model.decode_captions(
            decoder_input_ids=inp, visual_features=vf,
            decoder_padding_mask=(inp == vocab.pad_token_id))
        tf_ids = logits.argmax(-1)[0].tolist()

    tf_cap = vocab.decode(tf_ids, skip_special_tokens=True)
    ar_cap, _ = generate_fn(model, image_tensor, vocab, max_length)
    gt_cap = vocab.decode(caption_ids[0].tolist(), skip_special_tokens=True)

    print(f"  Ground truth   : {gt_cap}")
    print(f"  Teacher forcing: {tf_cap}")
    print(f"  Autoregressive : {ar_cap}")
    print("\n  CHAN DOAN:")
    print("    TF dung + AR dung  -> embedding/decode OK, van de o hoi tu hoac data")
    print("    TF dung + AR rac   -> causal mask ro, HOAC quen model.eval()")
    print("    TF cung rac        -> model chua hoi tu that, quay lai training loop")


# =============================================================================
# TEST 4 | OVERFIT SINGLE BATCH
# Neu khong overfit noi 8 mau thi co loi logic, khong phai loi hyperparameter.
# =============================================================================
def test_overfit(model, vocab, images, captions, steps=300, lr=1e-4):
    header("TEST 4 | Overfit 1 batch (8 mau)")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss(ignore_index=vocab.pad_token_id)
    inp, tgt = captions[:, :-1], captions[:, 1:]      # <-- TEACHER FORCING SHIFT
    for s in range(steps):
        opt.zero_grad()
        vf = model.encode_images(images)
        logits = model.decode_captions(
            decoder_input_ids=inp, visual_features=vf,
            decoder_padding_mask=(inp == vocab.pad_token_id))
        loss = lossf(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if s % 50 == 0 or s == steps - 1:
            print(f"    step {s:4d}  loss = {loss.item():.4f}")
    check("Loss < 0.1 sau khi overfit", loss.item() < 0.1,
          f"loss = {loss.item():.4f} -> con loi logic o dau do")
    model.eval()


if __name__ == "__main__":
    print(__doc__)
    test_causal_leak()
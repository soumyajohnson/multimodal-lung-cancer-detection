import os
import numpy as np
import torch
import os
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from dataset_mm import LungMM
from models_mm import CTOnly, LateFusion
# from aug3d import simple_aug  # optional

def eval_epoch(model, loader, device):
    model.eval()
    y_true, y_prob = [], []
    loss_fn = torch.nn.CrossEntropyLoss()
    total_loss, n = 0.0, 0

    with torch.no_grad():
        for b in loader:
            ct = b["ct"].to(device)
            x = b["x_clin"].to(device)
            g = b["ehr_present"].to(device)
            y = b["y"].to(device)

            logits = model(ct, x, g)
            loss = loss_fn(logits, y)

            prob = torch.softmax(logits, dim=1)[:, 1]

            bs = y.size(0)
            total_loss += loss.item() * bs
            n += bs

            y_true.append(y.cpu().numpy())
            y_prob.append(prob.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)

    auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    return total_loss / max(n, 1), auc, ap

def train():
    ROOT = r"your-root-directory-path-here"
    MAN_DIR = os.path.join(ROOT, "manifests_mdpi")

    train_csv = os.path.join(MAN_DIR, "train.csv")
    val_csv   = os.path.join(MAN_DIR, "val.csv")
    test_csv  = os.path.join(MAN_DIR, "test.csv")

    clinical_npz = os.path.join(ROOT, "processed_clinical", "clinical_features.npz")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] device:", device)

    # Datasets
    train_ds = LungMM(train_csv, clinical_npz, train=True, augment_fn=None)  # or augment_fn=simple_aug
    val_ds   = LungMM(val_csv, clinical_npz, train=False)
    test_ds  = LungMM(test_csv, clinical_npz, train=False)

    bs = 2  # adjust for GPU
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)

    clinical_dim = train_ds.clinical_dim
    print("[INFO] clinical_dim:", clinical_dim)

    # ===== Choose model =====
    #model = CTOnly()
    model = LateFusion(clinical_dim=clinical_dim, use_ehr_present=True)
    model.to(device)

    # =========================
    # Load CT-only weights into the CT encoder (for faster convergence)
    # =========================
    CT_CKPT_PATH = r"your-ct-only-checkpoint-path-here"
    ct_ckpt = torch.load(CT_CKPT_PATH, map_location=device)

    # IMPORTANT: ct_ckpt["model_state_dict"] belongs to CTOnly model
    # We load it into model.ct (the ResNet part) using strict=False
    model.ct.load_state_dict(ct_ckpt["model_state_dict"], strict=False)

    print("[INFO] Loaded CT-only weights into fusion CT encoder")

    # =========================
    # Freeze CT encoder (saves a lot of time; clinical+head learns fusion)
    # =========================
    for p in model.ct.parameters():
        p.requires_grad = False
    print("[INFO] Frozen CT encoder")

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_auc = -1
    best_state = None

    for ep in range(1, 11):
        model.train()
        total, n = 0.0, 0

        for b in train_loader:
            ct = b["ct"].to(device)
            x  = b["x_clin"].to(device)
            g  = b["ehr_present"].to(device)
            y  = b["y"].to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(ct, x, g)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()

            bs_ = y.size(0)
            total += loss.item() * bs_
            n += bs_

        tr_loss = total / max(n, 1)
        va_loss, va_auc, va_ap = eval_epoch(model, val_loader, device)
        print(f"Epoch {ep:02d} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | val_auc={va_auc:.4f} | val_ap={va_ap:.4f}")



        CKPT_DIR = r"your-checkpoint-directory-path-here"  # change per experiment
        os.makedirs(CKPT_DIR, exist_ok=True)
        CKPT_PATH = os.path.join(CKPT_DIR, "best.pt")

        if va_auc > best_auc:
            best_auc = va_auc
            torch.save(
                {
                    "epoch": ep,
                    "val_auc": va_auc,
                    "model_state_dict": model.state_dict(),
                },
                CKPT_PATH
            )
            print(f"[CHECKPOINT] Saved best model @ epoch {ep} (val_auc={va_auc:.4f})")


    # Load best + test
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"[INFO] Loaded checkpoint from epoch {ckpt['epoch']} (val_auc={ckpt['val_auc']:.4f})")

    # =========================
    # Final test evaluation
    # =========================
    model.eval()
    y_true, y_prob = [], []
    loss_fn = torch.nn.CrossEntropyLoss()
    total_loss, n = 0.0, 0

    with torch.no_grad():
        for b in test_loader:
            ct = b["ct"].to(device)
            x  = b["x_clin"].to(device)
            g  = b["ehr_present"].to(device)
            y  = b["y"].to(device)

            logits = model(ct, x, g)
            loss = loss_fn(logits, y)

            prob = torch.softmax(logits, dim=1)[:, 1]

            bs = y.size(0)
            total_loss += loss.item() * bs
            n += bs

            y_true.append(y.cpu().numpy())
            y_prob.append(prob.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)

    te_loss = total_loss / max(n, 1)
    from sklearn.metrics import roc_auc_score, average_precision_score
    te_auc = roc_auc_score(y_true, y_prob)
    te_ap  = average_precision_score(y_true, y_prob)

    print(f"[TEST] loss={te_loss:.4f} | auc={te_auc:.4f} | ap={te_ap:.4f}")

    # =========================
    # SAVE TEST PREDICTIONS
    # =========================
    np.savez(
        os.path.join(CKPT_DIR, "test_predictions.npz"),
        y_true=y_true,
        y_prob=y_prob
    )
    print("[OK] Saved test predictions to:", os.path.join(CKPT_DIR, "test_predictions.npz"))

if __name__ == "__main__":
    train()

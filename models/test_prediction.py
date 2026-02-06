import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

from dataset_mm import LungMM
from models_mm import CTOnly, LateFusion


# =========================
# CONFIG
# =========================
ROOT = r"your-root-directory-path-here"
MAN_DIR = os.path.join(ROOT, "manifests_mdpi")
TEST_CSV = os.path.join(MAN_DIR, "test.csv")
CLINICAL_NPZ = os.path.join(ROOT, "processed_clinical", "clinical_features.npz")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2


# =========================
# Generic evaluation helper
# =========================
def run_test_eval(model, loader, device):
    model.eval()
    y_true, y_prob = [], []

    with torch.no_grad():
        for b in loader:
            ct = b["ct"].to(device)
            x  = b["x_clin"].to(device)
            g  = b["ehr_present"].to(device)
            y  = b["y"].to(device)

            logits = model(ct, x, g)
            prob = torch.softmax(logits, dim=1)[:, 1]

            y_true.append(y.cpu().numpy())
            y_prob.append(prob.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)

    auc = roc_auc_score(y_true, y_prob)
    ap  = average_precision_score(y_true, y_prob)

    return y_true, y_prob, auc, ap


def main():
    print("[INFO] device:", DEVICE)

    # dataset / loader
    test_ds = LungMM(TEST_CSV, CLINICAL_NPZ, train=False)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    clinical_dim = test_ds.clinical_dim
    print("[INFO] clinical_dim:", clinical_dim)

    # =====================================================
    # 1️⃣ CT-ONLY
    # =====================================================
    print("\n[CT-ONLY] Evaluating...")
    ct_ckpt_dir = os.path.join(ROOT, "checkpoints", "ct-only")
    ct_ckpt_path = os.path.join(ct_ckpt_dir, "best.pt")

    ct_model = CTOnly().to(DEVICE)
    ckpt = torch.load(ct_ckpt_path, map_location=DEVICE)
    ct_model.load_state_dict(ckpt["model_state_dict"])

    y_true, y_prob, auc, ap = run_test_eval(ct_model, test_loader, DEVICE)
    print(f"[CT-ONLY TEST] AUC={auc:.4f} | AP={ap:.4f}")

    np.savez(
        os.path.join(ct_ckpt_dir, "test_predictions_ct2.npz"),
        y_true=y_true,
        y_prob=y_prob
    )
    print("[OK] Saved CT-only test_predictions_ct2.npz")

    # =====================================================
    # 2️⃣ LATE FUSION
    # =====================================================
    print("\n[LATE FUSION] Evaluating...")
    lf_ckpt_dir = os.path.join(ROOT, "checkpoints", "late-fusion")
    lf_ckpt_path = os.path.join(lf_ckpt_dir, "best.pt")

    lf_model = LateFusion(
        clinical_dim=clinical_dim,
        use_ehr_present=True
    ).to(DEVICE)

    ckpt = torch.load(lf_ckpt_path, map_location=DEVICE)
    lf_model.load_state_dict(ckpt["model_state_dict"])

    y_true, y_prob, auc, ap = run_test_eval(lf_model, test_loader, DEVICE)
    print(f"[LATE FUSION TEST] AUC={auc:.4f} | AP={ap:.4f}")

    np.savez(
        os.path.join(lf_ckpt_dir, "test_predictions_ct2.npz"),
        y_true=y_true,
        y_prob=y_prob
    )
    print("[OK] Saved Late-fusion test_predictions_ct2.npz")


if __name__ == "__main__":
    main()

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

# =========================
# CONFIG
# =========================
ROOT = r"your-root-directory-path-here"

CT_PRED_PATH = os.path.join(
    ROOT, "checkpoints", "ct-only", "test_predictions_ct2.npz"
)

LF_PRED_PATH = os.path.join(
    ROOT, "checkpoints", "late-fusion", "test_predictions_ct2.npz"
)

OUT_DIR = os.path.join(ROOT, "plots")
os.makedirs(OUT_DIR, exist_ok=True)


# =========================
# Helper
# =========================
def load_preds(path):
    data = np.load(path)
    return data["y_true"], data["y_prob"]


# =========================
# Load predictions
# =========================
y_true_ct, y_prob_ct = load_preds(CT_PRED_PATH)
y_true_lf, y_prob_lf = load_preds(LF_PRED_PATH)

assert np.array_equal(y_true_ct, y_true_lf), "Mismatch in test labels!"

# =========================
# ROC CURVE
# =========================
fpr_ct, tpr_ct, _ = roc_curve(y_true_ct, y_prob_ct)
fpr_lf, tpr_lf, _ = roc_curve(y_true_lf, y_prob_lf)

auc_ct = auc(fpr_ct, tpr_ct)
auc_lf = auc(fpr_lf, tpr_lf)

plt.figure(figsize=(6, 6))
plt.plot(fpr_ct, tpr_ct, label=f"CT-only (AUC = {auc_ct:.3f})")
plt.plot(fpr_lf, tpr_lf, label=f"Late Fusion (AUC = {auc_lf:.3f})")
plt.plot([0, 1], [0, 1], "--", color="gray")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Lung Cancer Detection")
plt.legend(loc="lower right")
plt.grid(True)

roc_path = os.path.join(OUT_DIR, "roc_curve_ct_vs_fusion_ct2.png")
plt.savefig(roc_path, dpi=300, bbox_inches="tight")
plt.close()

print("[OK] Saved ROC curve to:", roc_path)

# =========================
# PRECISION–RECALL CURVE
# =========================
prec_ct, rec_ct, _ = precision_recall_curve(y_true_ct, y_prob_ct)
prec_lf, rec_lf, _ = precision_recall_curve(y_true_lf, y_prob_lf)

ap_ct = average_precision_score(y_true_ct, y_prob_ct)
ap_lf = average_precision_score(y_true_lf, y_prob_lf)

plt.figure(figsize=(6, 6))
plt.plot(rec_ct, prec_ct, label=f"CT-only (AP = {ap_ct:.3f})")
plt.plot(rec_lf, prec_lf, label=f"Late Fusion (AP = {ap_lf:.3f})")

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve – Lung Cancer Detection")
plt.legend(loc="lower left")
plt.grid(True)

pr_path = os.path.join(OUT_DIR, "pr_curve_ct_vs_fusion_ct2.png")
plt.savefig(pr_path, dpi=300, bbox_inches="tight")
plt.close()

print("[OK] Saved PR curve to:", pr_path)

# =========================
# Console summary
# =========================
print("\n=== TEST SET PERFORMANCE ===")
print(f"CT-only      : AUC={auc_ct:.4f}, AP={ap_ct:.4f}")
print(f"Late Fusion  : AUC={auc_lf:.4f}, AP={ap_lf:.4f}")

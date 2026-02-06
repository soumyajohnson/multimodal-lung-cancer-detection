import os, glob, re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# =========================
# PATHS (yours)
# =========================
DIR_CLIN = r"your-clinical-directory-path-here"
CLIN_CSV = os.path.join(DIR_CLIN, "canonical_clinical_merged.csv")
CLIN_NPZ = os.path.join(DIR_CLIN, "clinical_features.npz")

DIR_MOSMED = r"your-mosmed-directory-path-here"
DIR_RAD = r"your-nsclc-radiomics-npz-directory-path-here"
DIR_RG = r"your-nsclc-radiogenomics-npz-directory-path-here"
DIR_LUNG3 = r"your-lung3-npz-directory-path-here"
DIR_MOSMED_CT2 = r"your-mosmed-ct2-npz-directory-path-here"

OUT_DIR = r"your-output-directory-path-here"
os.makedirs(OUT_DIR, exist_ok=True)

# =========================
# HELPERS
# =========================
def list_npz(folder):
    return sorted([p for p in glob.glob(os.path.join(folder, "*.npz"))])

def stem(path):
    return os.path.splitext(os.path.basename(path))[0]

def normalize_id(x: str) -> str:
    x = x.strip().upper()
    x = x.replace(" ", "")
    x = re.sub(r"__+", "_", x)
    return x

def patient_id_from_npz(npz_path: str) -> str:
    # your filenames are already the IDs (LUNG1-001, AMC-002, LUNG3-01, study_0001)
    return normalize_id(stem(npz_path))

def load_X(npz_path: str):
    z = np.load(npz_path, allow_pickle=True)
    print("[DEBUG] clinical_features.npz keys:", z.files)

    X = z["X"]
    print(f"[DEBUG] Raw X dtype={X.dtype}, shape={X.shape}")

    # Unwrap 0-d object
    if X.dtype == object and X.shape == ():
        X = X.item()
        print("[DEBUG] Unwrapped X via .item(), type:", type(X))

    # If it's a scipy sparse matrix, densify safely
    try:
        import scipy.sparse as sp
        if sp.issparse(X):
            X = X.toarray()
            print("[DEBUG] Converted sparse -> dense:", X.shape, X.dtype)
    except Exception as e:
        print("[DEBUG] scipy sparse check skipped/failed:", e)

    # Ensure numpy float32
    X = np.asarray(X, dtype=np.float32)
    print("[DEBUG] Final X dtype=", X.dtype, "shape=", X.shape)
    return X

def load_patient_ids(npz_path: str):
    z = np.load(npz_path, allow_pickle=True)
    pid = z["patient_id"]
    if pid.dtype == object and pid.shape == ():
        pid = pid.item()
    pid = np.asarray(pid).astype(str)
    return pid



# =========================
# 1) Build clinical row index map: patient_id -> row_idx
# =========================
pid_npz = load_patient_ids(CLIN_NPZ)
pid_npz_norm = [normalize_id(p) for p in pid_npz]

pid_to_rowidx = {}
for i, pid in enumerate(pid_npz_norm):
    pid_to_rowidx.setdefault(pid, i)

X = load_X(CLIN_NPZ)
assert X.shape[0] == len(pid_npz_norm), f"X rows ({X.shape[0]}) != patient_id rows ({len(pid_npz_norm)})"
clinical_dim = X.shape[1]
print(f"[OK] X.shape={X.shape}, clinical_dim={clinical_dim}")


# =========================
# 2) Build rows for each dataset
# =========================
rows = []

def add_nsclc(ct_dir, source_name):
    miss = 0
    for p in list_npz(ct_dir):
        pid = patient_id_from_npz(p)
        row_idx = pid_to_rowidx.get(pid, None)
        if row_idx is None:
            miss += 1
            continue
        rows.append({
            "patient_id": pid,
            "ct_npz_path": p,
            "label": 1,
            "source_dataset": source_name,
            "clinical_row_idx": int(row_idx),
            "ehr_present": 1
        })
    print(f"[INFO] {source_name}: CT files={len(list_npz(ct_dir))}, matched_clinical={len([r for r in rows if r['source_dataset']==source_name])}, missing_match={miss}")

def add_mosmed(ct_dir, source_name="MosMed-CT0"):
    for p in list_npz(ct_dir):
        pid = patient_id_from_npz(p)  # STUDY_0001
        rows.append({
            "patient_id": pid,
            "ct_npz_path": p,
            "label": 0,
            "source_dataset": source_name,
            "clinical_row_idx": -1,
            "ehr_present": 0
        })
    print(f"[INFO] {source_name}: CT files={len(list_npz(ct_dir))}")

add_nsclc(DIR_RAD, "NSCLC-Radiomics")               # TRAIN positives
add_nsclc(DIR_RG, "NSCLC-Radiogenomics")            # VAL positives
add_nsclc(DIR_LUNG3, "NSCLC-Radiomics-Genomics_LUNG3")  # TEST positives
add_mosmed(DIR_MOSMED, "MosMed-CT0")                # negatives (will be split)
add_mosmed(DIR_MOSMED_CT2, "MosMed-CT2")


df = pd.DataFrame(rows)

# =========================
# 3) Source-based split policy you requested
# =========================
# Positives:
train_pos = df[df["source_dataset"] == "NSCLC-Radiomics"].copy()
val_pos   = df[df["source_dataset"] == "NSCLC-Radiogenomics"].copy()
test_pos  = df[df["source_dataset"] == "NSCLC-Radiomics-Genomics_LUNG3"].copy()

# Negatives:
# - Use MosMed CT-0 ONLY for training negatives
# - Use MosMed CT-2 ONLY for val/test hard negatives (no reuse/leakage)

mos_ct0 = df[df["source_dataset"] == "MosMed-CT0"].copy()
mos_ct2 = df[df["source_dataset"] == "MosMed-CT2"].copy()

# # Negatives: split MosMed into train/val/test so metrics are valid everywhere
# mos = df[df["source_dataset"] == "MosMed-CT0"].copy()

# Split CT-0 into train/val negatives (60/40 or 80/20 — choose one)
mos_train, mos_val = train_test_split(
    mos_ct0,
    test_size=0.20,      # 20% CT-0 for validation negatives
    random_state=42,
    shuffle=True
)

# Hard test negatives = all CT-2
ct2_test = mos_ct2.copy()

# # # 60/20/20 split for negatives (adjust if you want)
# mos_train, mos_tmp = train_test_split(mos, test_size=0.40, random_state=42, shuffle=True, stratify=mos["label"])
# mos_val, mos_test  = train_test_split(mos_tmp, test_size=0.50, random_state=42, shuffle=True, stratify=mos_tmp["label"])

# ---- Hard negatives (CT-2) split between val/test
# Option A (recommended): split CT-2 into val/test
# ct2_val, ct2_test = train_test_split(
#     mos_ct2,
#     test_size=0.50,
#     random_state=42,
#     shuffle=True
# )

# Option B (even faster): put ALL CT-2 in TEST, keep VAL easier
# ct2_val = mos_ct2.iloc[0:0].copy()   # empty
# ct2_test = mos_ct2.copy()

train_df = pd.concat([train_pos, mos_train], ignore_index=True)
val_df   = pd.concat([val_pos, mos_val], ignore_index=True)
test_df  = pd.concat([test_pos, ct2_test], ignore_index=True)

# Shuffle rows (optional)
train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
val_df   = val_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
test_df  = test_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

assert (train_df["source_dataset"] == "MosMed-CT2").sum() == 0, "Leak: CT-2 in TRAIN!"
assert (val_df["source_dataset"] == "MosMed-CT2").sum() == 0, "Leak: CT-2 in VAL!"
assert (test_df["source_dataset"] == "MosMed-CT0").sum() == 0, "Leak: CT-0 in TEST!"


print("\n[INFO] Negative policy:")
print(" - Train negatives:", (train_df["source_dataset"] == "MosMed-CT0").sum(), "from MosMed-CT0")
print(" - Val negatives:",   (val_df["source_dataset"] == "MosMed-CT0").sum(), "from MosMed-CT0")
print(" - Test negatives:",  (test_df["source_dataset"] == "MosMed-CT2").sum(), "from MosMed-CT2 (hard)")

# =========================
# 4) Write manifests
# =========================
all_path   = os.path.join(OUT_DIR, "all_patients.csv")
train_path = os.path.join(OUT_DIR, "train.csv")
val_path   = os.path.join(OUT_DIR, "val.csv")
test_path  = os.path.join(OUT_DIR, "test.csv")

df.to_csv(all_path, index=False)
train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)
test_df.to_csv(test_path, index=False)

print("\n[OK] Wrote:")
print(" -", all_path, "n=", len(df))
print(" -", train_path, "n=", len(train_df), "label counts:\n", train_df["label"].value_counts().to_string())
print(" -", val_path, "n=", len(val_df), "label counts:\n", val_df["label"].value_counts().to_string())
print(" -", test_path, "n=", len(test_df), "label counts:\n", test_df["label"].value_counts().to_string())

print("\n[NOTE] clinical_dim =", clinical_dim)

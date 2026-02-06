"""
preprocess_all_nsclc_clinical.py

ONE single script to preprocess ALL your NSCLC clinical CSVs into:
1) a unified canonical clinical table (canonical_clinical_merged.csv)
2) a model-ready clinical feature matrix (clinical_features.npz)
3) a saved sklearn preprocessor (clinical_preproc.joblib)

Place this file in:
\preprocessing code

It uses RELATIVE PATHS from the script location (same style as your CT scripts).
"""

import os
import re
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


# =============================================================================
# RELATIVE PATHS
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# ---- INPUT FILES (edit only these names if you move/rename files) ----
LUNG1_CSV = os.path.join(PROJECT_ROOT, "NSCLC-Radiomics-Lung1.clinical-version3-Oct-2019.csv")
RADIOGENOMICS_CSV = os.path.join(PROJECT_ROOT, "NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv")
LUNG3_CSV = os.path.join(PROJECT_ROOT, "Lung3.metadata.csv")

# ---- OUTPUT DIR ----
OUT_DIR = os.path.join(PROJECT_ROOT, "processed_clinical")
os.makedirs(OUT_DIR, exist_ok=True)

CANONICAL_CSV_OUT = os.path.join(OUT_DIR, "canonical_clinical_merged.csv")
FEATURES_NPZ_OUT = os.path.join(OUT_DIR, "clinical_features.npz")
PREPROC_JOBLIB_OUT = os.path.join(OUT_DIR, "clinical_preproc.joblib")


# =============================================================================
# CANONICAL FEATURE SET (small + robust across datasets)
# =============================================================================
NUMERIC_COLS = ["age"]
CATEGORICAL_COLS = ["sex", "overall_stage", "histology"]
JOIN_KEY = "patient_id"


# =============================================================================
# UTILITIES
# =============================================================================
def _read_csv_robust(path_csv: str) -> pd.DataFrame:
    """
    Robust CSV read for TCIA/Excel-exported clinical files.
    Compatible with pandas >=2.1 (no applymap).
    """
    try:
        df = pd.read_csv(path_csv, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path_csv, encoding="latin1", engine="python")

    # Clean non-breaking spaces ONLY in object/string columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace("\xa0", " ", regex=False)
            .str.strip()
        )

    return df



def _find_first_existing_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _normalize_sex(x: str) -> str:
    if pd.isna(x):
        return "UNK"
    s = str(x).strip().lower()
    if s in {"m", "male", "1"}:
        return "M"
    if s in {"f", "female", "0"}:
        return "F"
    return "UNK"


def _normalize_stage(x: str) -> str:
    """Keep as categorical; normalize common junk."""
    if pd.isna(x):
        return "UNK"
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "na", "n/a"}:
        return "UNK"
    return s


def _normalize_histology(x: str) -> str:
    if pd.isna(x):
        return "UNK"
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none", "na", "n/a"}:
        return "UNK"
    return s


def _normalize_patient_id(pid: str) -> str:
    """
    Ensure the ID matches your CT npz naming scheme as much as possible.

    - LUNG1-014 stays LUNG1-014
    - LUNG3-01 stays LUNG3-01
    - AMC-003 stays AMC-003 (Radiogenomics)
    """
    if pd.isna(pid):
        return "UNKNOWN"
    s = str(pid).strip()

    # Clean whitespace
    s = re.sub(r"\s+", "", s)

    return s


# =============================================================================
# MAPPERS: dataset CSV -> canonical schema
# =============================================================================
def map_lung1(path_csv: str) -> pd.DataFrame:
    df = _read_csv_robust(path_csv)

    id_col = _find_first_existing_col(df, ["PatientID", "patient_id", "Case", "case", "ID"])
    if id_col is None:
        raise ValueError("LUNG1: could not find patient id column.")

    age_col = _find_first_existing_col(df, ["age", "Age", "AGE"])
    sex_col = _find_first_existing_col(df, ["gender", "Gender", "sex", "Sex"])
    stage_col = _find_first_existing_col(df, ["Overall.Stage", "overall_stage", "OverallStage", "Stage", "stage"])
    hist_col = _find_first_existing_col(df, ["Histology", "histology", "HISTOLOGY"])

    out = pd.DataFrame({
        "patient_id": df[id_col].astype(str).map(_normalize_patient_id),
        "age": pd.to_numeric(df[age_col], errors="coerce") if age_col else np.nan,
        "sex": df[sex_col].map(_normalize_sex) if sex_col else "UNK",
        "overall_stage": df[stage_col].map(_normalize_stage) if stage_col else "UNK",
        "histology": df[hist_col].map(_normalize_histology) if hist_col else "UNK",
        "source_dataset": "NSCLC-Radiomics-LUNG1",
        "ehr_present": 1
    })

    return out


def map_radiogenomics(path_csv: str) -> pd.DataFrame:
    df = _read_csv_robust(path_csv)

    # Radiogenomics often uses "Case ID"
    id_col = _find_first_existing_col(df, ["Case ID", "CaseID", "case_id", "PatientID", "Patient ID", "ID"])
    if id_col is None:
        raise ValueError("Radiogenomics: could not find patient id column.")

    age_col = _find_first_existing_col(df, ["Age at Histological Diagnosis", "Age", "age"])
    sex_col = _find_first_existing_col(df, ["Gender", "gender", "Sex", "sex"])
    # Overall stage is not always present in Radiogenomics; keep UNK if absent
    stage_col = _find_first_existing_col(df, ["Overall Stage", "Overall.Stage", "Stage", "stage", "Clinical Stage"])
    # histology column sometimes has trailing space
    hist_col = _find_first_existing_col(df, ["Histology", "Histology ", "histology"])

    out = pd.DataFrame({
        "patient_id": df[id_col].astype(str).map(_normalize_patient_id),
        "age": pd.to_numeric(df[age_col], errors="coerce") if age_col else np.nan,
        "sex": df[sex_col].map(_normalize_sex) if sex_col else "UNK",
        "overall_stage": df[stage_col].map(_normalize_stage) if stage_col else "UNK",
        "histology": df[hist_col].map(_normalize_histology) if hist_col else "UNK",
        "source_dataset": "NSCLC-Radiogenomics",
        "ehr_present": 1
    })

    return out


def map_lung3(path_csv: str) -> pd.DataFrame:
    df = _read_csv_robust(path_csv)

    # Lung3.metadata.csv columns (confirmed):
    # - sample.name (ID)
    # - characteristics.tag.gender
    # - characteristics.tag.histology
    # - characteristics.tag.stage.primary.tumor / nodes / mets

    id_col = _find_first_existing_col(df, [
        "sample.name", "sample_name",
        "Subject ID", "SubjectID",
        "PatientID", "patient_id",
        "Case", "case", "ID"
    ])
    if id_col is None:
        raise ValueError(f"LUNG3: could not find patient id column. Columns: {list(df.columns)}")

    sex_col = _find_first_existing_col(df, [
        "characteristics.tag.gender", "Sex", "sex", "Gender", "gender"
    ])

    hist_col = _find_first_existing_col(df, [
        "characteristics.tag.histology", "Histology", "histology"
    ])

    # Stage in Lung3 is split across TNM columns; build a compact categorical string
    t_col = _find_first_existing_col(df, ["characteristics.tag.stage.primary.tumor"])
    n_col = _find_first_existing_col(df, ["characteristics.tag.stage.nodes"])
    m_col = _find_first_existing_col(df, ["characteristics.tag.stage.mets"])

    def _tnm_row(r):
        t = str(r[t_col]).strip() if t_col and pd.notna(r[t_col]) else ""
        n = str(r[n_col]).strip() if n_col and pd.notna(r[n_col]) else ""
        m = str(r[m_col]).strip() if m_col and pd.notna(r[m_col]) else ""
        s = f"{t}{n}{m}".strip()
        return s if s else "UNK"

    overall_stage = df.apply(_tnm_row, axis=1) if (t_col or n_col or m_col) else "UNK"

    out = pd.DataFrame({
        "patient_id": df[id_col].astype(str).map(_normalize_patient_id),
        "age": np.nan,  # Lung3 metadata does not have age in this file
        "sex": df[sex_col].map(_normalize_sex) if sex_col else "UNK",
        "overall_stage": overall_stage.map(_normalize_stage) if isinstance(overall_stage, pd.Series) else "UNK",
        "histology": df[hist_col].map(_normalize_histology) if hist_col else "UNK",
        "source_dataset": "NSCLC-Radiomics-Genomics-LUNG3",
        "ehr_present": 1
    })

    return out


# =============================================================================
# PREPROCESSOR (fit on all NSCLC clinical rows)
# NOTE: In training, you should ideally fit on TRAIN split only.
# For now we fit on all to create a stable feature space; you can refit later.
# =============================================================================
def build_preprocessor():
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("num", num_pipe, NUMERIC_COLS),
        ("cat", cat_pipe, CATEGORICAL_COLS)
    ])


def main():
    # ---- sanity checks ----
    for p in [LUNG1_CSV, RADIOGENOMICS_CSV, LUNG3_CSV]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing clinical file: {p}")

    # ---- map each dataset -> canonical ----
    lung1 = map_lung1(LUNG1_CSV)
    rg = map_radiogenomics(RADIOGENOMICS_CSV)
    lung3 = map_lung3(LUNG3_CSV)

    clin = pd.concat([lung1, rg, lung3], ignore_index=True)

    # Drop clearly broken IDs
    clin = clin[clin[JOIN_KEY].notna()].copy()
    clin[JOIN_KEY] = clin[JOIN_KEY].astype(str)

    # If duplicates exist (same patient appears multiple times), keep the first non-UNK values
    # (You can improve this later by choosing best row per patient.)
    clin = clin.sort_values(["patient_id", "source_dataset"]).drop_duplicates("patient_id", keep="first")

    # Save canonical merged table
    clin.to_csv(CANONICAL_CSV_OUT, index=False)
    print(f"[OK] Wrote canonical clinical table: {CANONICAL_CSV_OUT}")
    print(f"[INFO] Patients in merged clinical: {clin['patient_id'].nunique()}")

    # ---- build and fit preprocessor (feature space) ----
    preproc = build_preprocessor()
    X = preproc.fit_transform(clin[NUMERIC_COLS + CATEGORICAL_COLS])

    # Save preprocessor
    joblib.dump(preproc, PREPROC_JOBLIB_OUT)
    print(f"[OK] Saved clinical preprocessor: {PREPROC_JOBLIB_OUT}")

    # Save features in NPZ (row order aligned to patient_ids array)
    patient_ids = clin["patient_id"].to_numpy()
    np.savez_compressed(
        FEATURES_NPZ_OUT,
        X=X.astype(np.float32),
        patient_id=patient_ids,
        columns_num=np.array(NUMERIC_COLS, dtype=object),
        columns_cat=np.array(CATEGORICAL_COLS, dtype=object),
    )
    print(f"[OK] Saved clinical feature matrix: {FEATURES_NPZ_OUT}")
    print(f"[INFO] X shape: {X.shape} (rows=patients, cols=encoded clinical features)")

    print("\nNext step in your model:")
    print("- Load clinical_features.npz -> get X and patient_id")
    print("- Join by patient_id with your CT index.csv/npz naming")
    print("- For MosMed negatives: clinical vector = zeros with same feature dim; ehr_present=0")


if __name__ == "__main__":
    main()

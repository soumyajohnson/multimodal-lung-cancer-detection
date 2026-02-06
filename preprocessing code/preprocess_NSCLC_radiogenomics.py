import os
import re
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

# =============================================================================
# CONFIG — RELATIVE PATHS (relative to this .py file location)
# =============================================================================

# Folder where this script lives:
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Project root = two levels up from "code-base/preprocessing code"
# E:\AIM AHEAD Research\code-base\preprocessing code\preprocess_nsclc.py
# -> E:\AIM AHEAD Research
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# ---- YOUR DATASET PATH (RELATIVE) ----
# E:\AIM AHEAD Research\manifest-1770173558383\NSCLC Radiogenomics
RAW_ROOT = os.path.join(PROJECT_ROOT, "manifest-1770173558383", "NSCLC Radiogenomics")

# metadata.csv location (keep it in PROJECT_ROOT OR update this relative path)
METADATA_CSV = os.path.join("manifest-1770173558383", "metadata.csv")

# Output folder (relative)
OUT_DIR = os.path.join(PROJECT_ROOT, "processed_nsclc_radiogenomics_npz_mdpi")

# =============================================================================
# MDPI/Sensors Paper-aligned preprocessing (same as your pipeline)
# =============================================================================
TARGET_SPACING = (1.0, 1.0, 1.0)   # (x, y, z)
HU_CLIP = (-1000, 400)
NORM_MODE = "minmax"
OUT_SHAPE = (128, 192, 192)        # (D, H, W) after SITK->numpy

# =============================================================================
# HELPERS
# =============================================================================

def norm_text(x: str) -> str:
    return str(x).strip().lower()

def is_dicom_file(fname: str) -> bool:
    f = fname.lower()
    return f.endswith(".dcm") or ("." not in f)

def read_dicom_series(series_dir: str) -> sitk.Image:
    """
    Reads the DICOM series using SimpleITK.
    If multiple series IDs exist in the same folder, selects the one with most slices.
    """
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(series_dir)
    if not series_ids:
        raise RuntimeError(f"No DICOM series IDs found in: {series_dir}")

    best_id = None
    best_n = -1
    for sid in series_ids:
        files = reader.GetGDCMSeriesFileNames(series_dir, sid)
        if len(files) > best_n:
            best_n = len(files)
            best_id = sid

    files = reader.GetGDCMSeriesFileNames(series_dir, best_id)
    reader.SetFileNames(files)
    image = reader.Execute()
    return sitk.Cast(image, sitk.sitkFloat32)

def resample_image(image: sitk.Image, target_spacing):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    target_spacing = tuple(float(s) for s in target_spacing)

    new_size = [
        int(round(original_size[i] * (original_spacing[i] / target_spacing[i])))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(sitk.sitkLinear)
    return resampler.Execute(image)

def sitk_to_numpy_DHW(image: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(image).astype(np.float32)  # (D,H,W) = (z,y,x)

def center_crop_or_pad(arr: np.ndarray, out_shape):
    D, H, W = arr.shape
    outD, outH, outW = out_shape

    pad_d = max(0, outD - D)
    pad_h = max(0, outH - H)
    pad_w = max(0, outW - W)

    if pad_d or pad_h or pad_w:
        arr = np.pad(
            arr,
            (
                (pad_d // 2, pad_d - pad_d // 2),
                (pad_h // 2, pad_h - pad_h // 2),
                (pad_w // 2, pad_w - pad_w // 2),
            ),
            mode="constant",
            constant_values=0.0,
        )

    D, H, W = arr.shape
    start_d = (D - outD) // 2
    start_h = (H - outH) // 2
    start_w = (W - outW) // 2

    return arr[start_d:start_d + outD, start_h:start_h + outH, start_w:start_w + outW]

def lung_focus_bbox(arr_hu: np.ndarray):
    """
    Compute a rough lung-focused bounding box using HU thresholding.
    Returns (z0,z1,y0,y1,x0,x1) in (D,H,W).
    """
    D, H, W = arr_hu.shape

    body = arr_hu > -500
    if body.sum() < 1000:
        return 0, D, 0, H, 0, W

    z_any = body.reshape(D, -1).any(axis=1)
    z_idx = np.where(z_any)[0]
    if len(z_idx) == 0:
        return 0, D, 0, H, 0, W
    z0, z1 = int(z_idx[0]), int(z_idx[-1]) + 1

    sub = arr_hu[z0:z1]
    body_sub = body[z0:z1]

    lung_air = (sub < -600) & body_sub

    if lung_air.sum() < 5000:
        ys = np.where(body_sub.any(axis=(0, 2)))[0]
        xs = np.where(body_sub.any(axis=(0, 1)))[0]
        if len(ys) == 0 or len(xs) == 0:
            return 0, D, 0, H, 0, W
        y0, y1 = int(ys[0]), int(ys[-1]) + 1
        x0, x1 = int(xs[0]), int(xs[-1]) + 1
        return z0, z1, y0, y1, x0, x1

    ys = np.where(lung_air.any(axis=(0, 2)))[0]
    xs = np.where(lung_air.any(axis=(0, 1)))[0]
    if len(ys) == 0 or len(xs) == 0:
        return z0, z1, 0, H, 0, W

    y0, y1 = int(ys[0]), int(ys[-1]) + 1
    x0, x1 = int(xs[0]), int(xs[-1]) + 1

    margin_z = 8
    margin_xy = 16

    z0 = max(0, z0 - margin_z)
    z1 = min(D, z1 + margin_z)
    y0 = max(0, y0 - margin_xy)
    y1 = min(H, y1 + margin_xy)
    x0 = max(0, x0 - margin_xy)
    x1 = min(W, x1 + margin_xy)

    return z0, z1, y0, y1, x0, x1

def lung_focus_crop_then_pad(arr_hu: np.ndarray, out_shape):
    z0, z1, y0, y1, x0, x1 = lung_focus_bbox(arr_hu)
    cropped = arr_hu[z0:z1, y0:y1, x0:x1]
    return center_crop_or_pad(cropped, out_shape)

# =============================================================================
# METADATA-BASED SERIES SELECTION (Radiogenomics has multiple scans per subject)
# =============================================================================

def relevance_score(row) -> float:
    """
    Score a series: higher = more likely to be the main axial chest/lung CT.
    Uses Series Description / Study Description / Number of Images.
    """
    sd = norm_text(row.get("Series Description", ""))
    studyd = norm_text(row.get("Study Description", ""))
    text = f"{sd} {studyd}"

    score = 0.0

    # ✅ positives
    if "lung" in text: score += 10
    if "chest" in text: score += 8
    if "thorax" in text: score += 8
    if "axial" in text: score += 4

    # ✅ thin-slice hints
    if re.search(r"\b1mm\b", text): score += 6
    if re.search(r"\b1\.0\b", text): score += 3
    if re.search(r"\b2mm\b", text): score += 2

    # ✅ kernel hints (common lung recon kernels)
    if re.search(r"\bb45f\b", text): score += 6
    if re.search(r"\bb45\b", text): score += 4
    if re.search(r"\bb50\b", text): score += 3

    # ❌ negatives (not the main axial series)
    negatives = [
        "topogram", "scout", "localizer",
        "coronal", "sagittal", "mpr",
        "abd", "abdomen", "pelvis", "abdpel"
    ]
    for bad in negatives:
        if bad in text:
            score -= 20

    # ✅ tie-breaker: more images often means main volume
    n_imgs = row.get("Number of Images", 0)
    try:
        n_imgs = int(n_imgs)
    except:
        n_imgs = 0
    score += min(n_imgs / 100.0, 10.0)

    return score

def file_location_to_series_dir(file_location: str) -> str:
    """
    metadata 'File Location' is usually like:
      .\\NSCLC Radiogenomics\\AMC-002\\...\\<series folder>
    We convert it to absolute path under PROJECT_ROOT by removing leading '.\'.
    """
    rel = str(file_location).replace("/", os.sep).replace("\\", os.sep)
    rel = rel.lstrip(".").lstrip(os.sep)

    # If metadata already starts with "NSCLC Radiogenomics", this will become:
    # PROJECT_ROOT\manifest-...\NSCLC Radiogenomics\AMC-...\...
    # But in your dataset, RAW_ROOT already points to "... \NSCLC Radiogenomics".
    # So we should join from PROJECT_ROOT if metadata includes manifest, or from RAW_ROOT if not.
    #
    # Most TCIA metadata uses paths relative to the manifest root (i.e., includes "NSCLC Radiogenomics").
    # We handle both cases safely.

    if rel.lower().startswith("nsclc radiogenomics".lower()):
        # join with the manifest root folder
        return os.path.join(PROJECT_ROOT, "manifest-1770173558383", rel)
    else:
        # assume rel starts from subject folder (AMC-xxx/...)
        return os.path.join(RAW_ROOT, rel)

def choose_best_series_per_subject(df_meta: pd.DataFrame) -> pd.DataFrame:
    df = df_meta.copy()

    # Keep CT only if Modality exists
    if "Modality" in df.columns:
        df = df[df["Modality"].astype(str).str.upper() == "CT"].copy()

    df["__score"] = df.apply(relevance_score, axis=1)

    # Optional tie-break by Study Date (if present)
    if "Study Date" in df.columns:
        df["__study_date"] = pd.to_datetime(df["Study Date"], errors="coerce")
    else:
        df["__study_date"] = pd.NaT

    chosen = (
        df.sort_values(
            ["Subject ID", "__score", "__study_date", "Number of Images"],
            ascending=[True, False, False, False]
        )
        .groupby("Subject ID", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    return chosen

# =============================================================================
# MAIN
# =============================================================================

def main():
    print("SCRIPT_DIR   :", SCRIPT_DIR)
    print("PROJECT_ROOT :", PROJECT_ROOT)
    print("RAW_ROOT     :", RAW_ROOT)
    print("METADATA_CSV :", METADATA_CSV)
    print("OUT_DIR      :", OUT_DIR)

    if not os.path.isdir(RAW_ROOT):
        raise FileNotFoundError(
            f"RAW_ROOT not found: {RAW_ROOT}\n"
            f"Expected: {os.path.join(PROJECT_ROOT, 'manifest-1770173558383', 'NSCLC Radiogenomics')}"
        )

    if not os.path.isfile(METADATA_CSV):
        raise FileNotFoundError(
            f"metadata.csv not found: {METADATA_CSV}\n"
            f"Put metadata.csv inside PROJECT_ROOT or update METADATA_CSV."
        )

    os.makedirs(OUT_DIR, exist_ok=True)

    df_meta = pd.read_csv(METADATA_CSV)
    chosen = choose_best_series_per_subject(df_meta)

    print(f"Metadata rows: {len(df_meta)}")
    print(f"Chosen (1 per subject): {len(chosen)}")

    # Save chosen series list for verification
    chosen_csv = os.path.join(OUT_DIR, "chosen_series.csv")
    cols = [c for c in ["Subject ID", "Study Description", "Series Description", "Number of Images", "File Location", "__score"] if c in chosen.columns]
    chosen[cols].to_csv(chosen_csv, index=False)
    print("Wrote chosen series list:", chosen_csv)

    rows = []
    for _, rowm in tqdm(chosen.iterrows(), total=len(chosen), desc="Preprocessing subjects"):
        patient_id = str(rowm["Subject ID"]).strip()
        sdir = file_location_to_series_dir(rowm["File Location"])
        out_path = os.path.join(OUT_DIR, f"{patient_id}.npz")

        # skip if already done
        if os.path.exists(out_path):
            continue

        try:
            img = read_dicom_series(sdir)
            orig_spacing = img.GetSpacing()
            orig_size = img.GetSize()

            img = resample_image(img, TARGET_SPACING)
            arr = sitk_to_numpy_DHW(img)  # HU-ish

            # Step 1: clip HU first (needed for lung bbox)
            arr = np.clip(arr, HU_CLIP[0], HU_CLIP[1]).astype(np.float32)

            # Step 2: lung-focused crop, then pad/crop to OUT_SHAPE
            arr = lung_focus_crop_then_pad(arr, OUT_SHAPE)

            # Step 3: minmax normalization to [0,1]
            arr = (arr - HU_CLIP[0]) / float(HU_CLIP[1] - HU_CLIP[0] + 1e-8)

            np.savez_compressed(
                out_path,
                image=arr.astype(np.float32),     # (D,H,W), normalized
                patient_id=patient_id,
                series_dir=sdir,
                orig_spacing=np.array(orig_spacing, dtype=np.float32),
                orig_size=np.array(orig_size, dtype=np.int32),
                new_spacing=np.array(TARGET_SPACING, dtype=np.float32),
                out_shape=np.array(OUT_SHAPE, dtype=np.int32),
                hu_clip=np.array(HU_CLIP, dtype=np.float32),
                norm_mode=NORM_MODE,
                series_description=str(rowm.get("Series Description", "")),
                study_description=str(rowm.get("Study Description", "")),
                num_images=int(rowm.get("Number of Images", 0)) if str(rowm.get("Number of Images", "")).strip() != "" else 0,
                score=float(rowm.get("__score", 0.0)),
            )

            rows.append({
                "patient_id": patient_id,
                "series_dir": sdir,
                "out_npz": out_path,
                "orig_spacing": str(orig_spacing),
                "orig_size": str(orig_size),
                "new_spacing": str(TARGET_SPACING),
                "out_shape": str(OUT_SHAPE),
                "hu_clip": str(HU_CLIP),
                "norm_mode": NORM_MODE,
                "series_description": str(rowm.get("Series Description", "")),
                "study_description": str(rowm.get("Study Description", "")),
                "num_images": int(rowm.get("Number of Images", 0)) if str(rowm.get("Number of Images", "")).strip() != "" else 0,
                "score": float(rowm.get("__score", 0.0)),
            })

        except Exception as e:
            rows.append({
                "patient_id": patient_id,
                "series_dir": sdir,
                "out_npz": None,
                "error": str(e)
            })

    index_csv = os.path.join(OUT_DIR, "index.csv")
    pd.DataFrame(rows).to_csv(index_csv, index=False)
    print(f"Done. Wrote: {index_csv}")
    print(f"Processed NPZ saved in: {OUT_DIR}")

if __name__ == "__main__":
    main()

import os
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

# Raw TCIA download folder (edit only if your folder name changes)
#RAW_ROOT = os.path.join(PROJECT_ROOT, "manifest-1770166404340", "NSCLC-Radiomics")
RAW_ROOT = os.path.join(PROJECT_ROOT, "manifest-1770177421180", "NSCLC-Radiomics-Genomics")


# Output folder
#OUT_DIR = os.path.join(PROJECT_ROOT, "processed_nsclc_radiomics_npz_mdpi")
OUT_DIR = os.path.join(PROJECT_ROOT, "processed_nsclc_radiomics_genomics_npz_mdpi")


# =============================================================================
# MDPI/Sensors Paper-aligned preprocessing
# =============================================================================
# Paper uses 1mm isotropic resampling for CT volumes.
TARGET_SPACING = (1.0, 1.0, 1.0)  # (x, y, z) in SimpleITK spacing

# Paper uses HU clipping [-1000, 400]
HU_CLIP = (-1000, 400)

# Paper uses min-max normalization -> [0, 1]
NORM_MODE = "minmax"

# NOTE:
# The paper uses small ROI patches (e.g., 20×50×50) around nodules.
# Since you're doing whole-volume classification, we keep a fixed input size.
# You can change this if GPU memory is an issue.
OUT_SHAPE = (128, 192, 192)  # (D, H, W) after SITK->numpy


# =============================================================================
# HELPERS
# =============================================================================

def is_dicom_file(fname: str) -> bool:
    f = fname.lower()
    # TCIA sometimes stores DICOMs without extension
    return f.endswith(".dcm") or ("." not in f)


def find_dicom_series_leaf_dirs(root_dir: str, min_slices: int = 40):
    """
    Find likely CT series directories (folders containing many DICOM slices).
    We use min_slices=40 to reduce chance of capturing scouts/localizers.
    """
    leaf_dirs = []
    for dirpath, _, filenames in os.walk(root_dir):
        dicom_files = [f for f in filenames if is_dicom_file(f)]
        if len(dicom_files) >= min_slices:
            leaf_dirs.append(dirpath)
    return leaf_dirs


def read_dicom_series(series_dir: str):
    """
    Reads the DICOM series using SimpleITK.
    If multiple series IDs exist, selects the one with most slices.
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

    # Ensure float32 for processing
    return sitk.Cast(image, sitk.sitkFloat32)


def resample_image(image: sitk.Image, target_spacing):
    """
    Resamples to target spacing while preserving physical size.
    """
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
    resampler.SetInterpolator(sitk.sitkLinear)  # linear OK for CT intensity
    return resampler.Execute(image)


def sitk_to_numpy_DHW(image: sitk.Image) -> np.ndarray:
    """
    SimpleITK -> numpy array, returns (D,H,W) = (z,y,x)
    """
    return sitk.GetArrayFromImage(image).astype(np.float32)


def hu_clip_and_normalize(arr: np.ndarray, hu_clip, mode="minmax") -> np.ndarray:
    """
    Paper-style:
    - Clip HU to [-1000, 400]
    - Min-max normalize to [0,1]
    """
    lo, hi = hu_clip
    arr = np.clip(arr, lo, hi).astype(np.float32)

    if mode == "minmax":
        arr = (arr - lo) / float(hi - lo + 1e-8)
    elif mode == "zscore":
        mu = float(arr.mean())
        sd = float(arr.std()) + 1e-8
        arr = (arr - mu) / sd
    else:
        raise ValueError("mode must be 'minmax' or 'zscore'")
    return arr


def center_crop_or_pad(arr: np.ndarray, out_shape):
    """
    Ensures arr becomes (D,H,W)=out_shape by center-cropping or zero-padding.
    """
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
    Returns (z0,z1,y0,y1,x0,x1) in (D,H,W) coordinates.
    Works without segmentations.
    """

    D, H, W = arr_hu.shape

    # Body mask: everything that's not outside-air
    # Outside-air is typically ~ -1000 HU; body/tissue is much higher
    body = arr_hu > -500

    # Fallback if body mask is too small (bad series/scout)
    if body.sum() < 1000:
        return 0, D, 0, H, 0, W

    # Find z-range where body exists
    z_any = body.reshape(D, -1).any(axis=1)
    z_idx = np.where(z_any)[0]
    if len(z_idx) == 0:
        return 0, D, 0, H, 0, W
    z0, z1 = int(z_idx[0]), int(z_idx[-1]) + 1

    # Restrict to body region
    sub = arr_hu[z0:z1]
    body_sub = body[z0:z1]

    # Lung air: very low HU inside body region
    lung_air = (sub < -600) & body_sub

    # If lung air region too small, fallback to body bbox in x/y
    if lung_air.sum() < 5000:
        ys = np.where(body_sub.any(axis=(0, 2)))[0]
        xs = np.where(body_sub.any(axis=(0, 1)))[0]
        if len(ys) == 0 or len(xs) == 0:
            return 0, D, 0, H, 0, W
        y0, y1 = int(ys[0]), int(ys[-1]) + 1
        x0, x1 = int(xs[0]), int(xs[-1]) + 1
        return z0, z1, y0, y1, x0, x1

    # Lung bbox in y/x within z-range
    ys = np.where(lung_air.any(axis=(0, 2)))[0]
    xs = np.where(lung_air.any(axis=(0, 1)))[0]
    if len(ys) == 0 or len(xs) == 0:
        return z0, z1, 0, H, 0, W

    y0, y1 = int(ys[0]), int(ys[-1]) + 1
    x0, x1 = int(xs[0]), int(xs[-1]) + 1

    # Margins to avoid cutting off pleural lesions
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
    """
    Lung-focused crop first (in HU space), then pad/crop to out_shape.
    """
    z0, z1, y0, y1, x0, x1 = lung_focus_bbox(arr_hu)
    cropped = arr_hu[z0:z1, y0:y1, x0:x1]
    return center_crop_or_pad(cropped, out_shape)



def extract_patient_id(path: str) -> str:
    """
    TCIA patient folders are like .../NSCLC-Radiomics/LUNG1-014/...
    """
    parts = os.path.normpath(path).split(os.sep)
    for p in parts:
        if p.startswith("LUNG3-"):
            return p
    return "UNKNOWN"


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("SCRIPT_DIR :", SCRIPT_DIR)
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("RAW_ROOT   :", RAW_ROOT)
    print("OUT_DIR    :", OUT_DIR)

    if not os.path.isdir(RAW_ROOT):
        raise FileNotFoundError(
            f"RAW_ROOT not found: {RAW_ROOT}\n"
            f"Edit RAW_ROOT or ensure the dataset exists under PROJECT_ROOT."
        )

    os.makedirs(OUT_DIR, exist_ok=True)

    series_dirs = find_dicom_series_leaf_dirs(RAW_ROOT, min_slices=40)
    print(f"Found {len(series_dirs)} candidate DICOM series folders.")

    rows = []
    for sdir in tqdm(series_dirs, desc="Preprocessing series"):
        patient_id = extract_patient_id(sdir)

        out_path = os.path.join(OUT_DIR, f"{patient_id}.npz")

        # Avoid overwriting if multiple series directories exist for same patient
        if os.path.exists(out_path):
            continue

        try:
            img = read_dicom_series(sdir)
            orig_spacing = img.GetSpacing()
            orig_size = img.GetSize()

            img = resample_image(img, TARGET_SPACING)
            arr = sitk_to_numpy_DHW(img)  # (D,H,W) in HU-like intensity

            # Step 1: clip HU first (still HU space; needed for lung bbox)
            arr = np.clip(arr, HU_CLIP[0], HU_CLIP[1]).astype(np.float32)

            # Step 2: lung-focused crop, then pad/crop to OUT_SHAPE
            arr = lung_focus_crop_then_pad(arr, OUT_SHAPE)

            # Step 3: paper-style minmax normalization to [0,1]
            # (since it's already clipped, this is stable)
            arr = (arr - HU_CLIP[0]) / float(HU_CLIP[1] - HU_CLIP[0] + 1e-8)

            np.savez_compressed(
                out_path,
                image=arr.astype(np.float32),     # (D,H,W), already normalized
                patient_id=patient_id,
                series_dir=sdir,
                orig_spacing=np.array(orig_spacing, dtype=np.float32),
                orig_size=np.array(orig_size, dtype=np.int32),
                new_spacing=np.array(TARGET_SPACING, dtype=np.float32),
                out_shape=np.array(OUT_SHAPE, dtype=np.int32),
                hu_clip=np.array(HU_CLIP, dtype=np.float32),
                norm_mode=NORM_MODE,
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
                "norm_mode": NORM_MODE
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

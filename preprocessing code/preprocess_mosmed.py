import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm


# =============================================================================
# CONFIG — RELATIVE PATHS
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# Your MosMed path:
# E:\AIM AHEAD Research\MosMedData\MosMedData\studies\CT-0
MOSMED_ROOT = os.path.join(PROJECT_ROOT, "MosMedData", "MosMedData", "studies", "CT-2")

# Output folder
OUT_DIR = os.path.join(PROJECT_ROOT, "processed_mosmed_ct2_npz_mdpi")

# Paper-style preprocessing alignment
TARGET_SPACING = (1.0, 1.0, 1.0)  # (x,y,z) mm
HU_CLIP = (-1000, 400)
OUT_SHAPE = (128, 192, 192)       # (D,H,W)
USE_LUNG_CROP = True              # recommended


# =============================================================================
# HELPERS
# =============================================================================
def find_nifti_files(root_dir: str):
    nii_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            fl = f.lower()
            if fl.endswith(".nii") or fl.endswith(".nii.gz"):
                nii_files.append(os.path.join(dirpath, f))
    return sorted(nii_files)


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
    # SITK -> numpy gives (z,y,x) = (D,H,W)
    return sitk.GetArrayFromImage(image).astype(np.float32)


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
    sd = (D - outD) // 2
    sh = (H - outH) // 2
    sw = (W - outW) // 2
    return arr[sd:sd + outD, sh:sh + outH, sw:sw + outW]


def lung_focus_bbox(arr_hu: np.ndarray):
    """
    Rough lung-focused bbox using HU thresholds.
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
# MAIN
# =============================================================================
def main():
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("MOSMED_ROOT :", MOSMED_ROOT)
    print("OUT_DIR     :", OUT_DIR)

    if not os.path.isdir(MOSMED_ROOT):
        raise FileNotFoundError(f"MOSMED_ROOT not found: {MOSMED_ROOT}")

    os.makedirs(OUT_DIR, exist_ok=True)

    nii_files = find_nifti_files(MOSMED_ROOT)
    print(f"Found {len(nii_files)} NIfTI files under CT-2.")

    rows = []
    for fpath in tqdm(nii_files, desc="Preprocessing MosMed NIfTI"):
        case_id = os.path.splitext(os.path.basename(fpath))[0].replace(".nii", "")
        out_path = os.path.join(OUT_DIR, f"{case_id}.npz")

        try:
            img = sitk.ReadImage(fpath)
            img = sitk.Cast(img, sitk.sitkFloat32)

            orig_spacing = img.GetSpacing()
            orig_size = img.GetSize()

            img = resample_image(img, TARGET_SPACING)
            arr = sitk_to_numpy_DHW(img)  # (D,H,W) HU-ish

            # Clip HU
            arr = np.clip(arr, HU_CLIP[0], HU_CLIP[1]).astype(np.float32)

            # Optional lung-focused crop
            if USE_LUNG_CROP:
                arr = lung_focus_crop_then_pad(arr, OUT_SHAPE)
            else:
                arr = center_crop_or_pad(arr, OUT_SHAPE)

            # Min-max normalize to [0,1] (paper-aligned)
            arr = (arr - HU_CLIP[0]) / float(HU_CLIP[1] - HU_CLIP[0] + 1e-8)

            np.savez_compressed(
                out_path,
                image=arr.astype(np.float32),  # (D,H,W)
                case_id=case_id,
                source="MosMedData_CT-0",
                nifti_path=fpath,
                orig_spacing=np.array(orig_spacing, dtype=np.float32),
                orig_size=np.array(orig_size, dtype=np.int32),
                new_spacing=np.array(TARGET_SPACING, dtype=np.float32),
                out_shape=np.array(OUT_SHAPE, dtype=np.int32),
                hu_clip=np.array(HU_CLIP, dtype=np.float32),
                norm_mode="minmax",
                lung_crop=bool(USE_LUNG_CROP),
            )

            rows.append({
                "case_id": case_id,
                "nifti_path": fpath,
                "out_npz": out_path,
                "orig_spacing": str(orig_spacing),
                "orig_size": str(orig_size),
                "new_spacing": str(TARGET_SPACING),
                "out_shape": str(OUT_SHAPE),
                "lung_crop": USE_LUNG_CROP,
            })

        except Exception as e:
            rows.append({
                "case_id": case_id,
                "nifti_path": fpath,
                "out_npz": None,
                "error": str(e),
            })

    index_csv = os.path.join(OUT_DIR, "index.csv")
    pd.DataFrame(rows).to_csv(index_csv, index=False)
    print(f"Done. Wrote: {index_csv}")
    print(f"Processed NPZ saved in: {OUT_DIR}")


if __name__ == "__main__":
    main()

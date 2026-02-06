import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

def load_X_dense(clinical_features_npz):
    z = np.load(clinical_features_npz, allow_pickle=True)
    X = z["X"]
    if X.dtype == object and X.shape == ():
        X = X.item()
    try:
        import scipy.sparse as sp
        if sp.issparse(X):
            X = X.toarray()
    except Exception:
        pass
    X = np.asarray(X, dtype=np.float32)
    return X  # (N, clinical_dim)

class LungMM(Dataset):
    def __init__(self, manifest_csv, clinical_features_npz, ct_key="ct", train=False, augment_fn=None):
        self.df = pd.read_csv(manifest_csv)
        self.X = load_X_dense(clinical_features_npz)  # global matrix (722, 56)
        self.clinical_dim = self.X.shape[1]
        self.ct_key = ct_key
        self.train = train
        self.augment_fn = augment_fn

    def __len__(self):
        return len(self.df)

    def _load_ct(self, path):
        z = np.load(path)
        if self.ct_key in z:
            ct = z[self.ct_key]
        else:
            ct = z[z.files[0]]
        ct = ct.astype(np.float32)  # (D,H,W) in [0,1]
        return ct

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        ct = self._load_ct(row["ct_npz_path"])
        if self.train and self.augment_fn is not None:
            ct = self.augment_fn(ct)

        ct = torch.from_numpy(ct).unsqueeze(0)  # (1,D,H,W)

        y = torch.tensor(int(row["label"]), dtype=torch.long)
        ehr_present = torch.tensor(float(row["ehr_present"]), dtype=torch.float32)

        row_idx = int(row["clinical_row_idx"])
        if row_idx >= 0 and int(row["ehr_present"]) == 1:
            x_clin = self.X[row_idx]
        else:
            x_clin = np.zeros((self.clinical_dim,), dtype=np.float32)

        x_clin = torch.from_numpy(x_clin)

        return {
            "ct": ct,
            "x_clin": x_clin,
            "ehr_present": ehr_present,
            "y": y,
            "patient_id": str(row["patient_id"]),
            "source": str(row["source_dataset"]),
        }

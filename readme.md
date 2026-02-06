Multimodal Lung Cancer Detection (CT + Clinical Metadata)
=========================================================

This repository showcases an **end-to-end multimodal machine learning pipeline** that combines **3D chest CT scans** with **structured clinical metadata** using a **late fusion architecture**. The focus is on **clean ML engineering**, including reproducible preprocessing, leakage-aware data splits, and robust evaluation across multiple public datasets.

Built in **PyTorch**, the project compares **CT-only** and **CT + clinical** models and highlights how dataset bias and distribution shift can inflate performance if not handled carefully. The codebase is modular, extensible, and designed as a **portfolio-quality example of applied ML/SWE work**, rather than a production or clinical system.



🎯 Project Goals
----------------

*   Build a **binary lung cancer classifier**
    
*   Compare **CT-only** vs **CT + clinical metadata (late fusion)**
    
*   Use **only public datasets**
    
*   Avoid **data leakage** and improper evaluation
    
*   Study the effect of **dataset bias and negative-class difficulty**
    

📦 Datasets Used (Public Only)
------------------------------

### Cancer (Positive Class)

**TCIA – NSCLC cohorts**

*   **NSCLC-Radiomics** (~422 patients)
    
*   **NSCLC-Radiogenomics** (~54 patients)
    
*   **NSCLC-Radiomics-Genomics (LUNG3)** (~89 patients)
    

Each includes:

*   Chest CT scans (DICOM)
    
*   Structured clinical / EHR-style metadata:
    
    *   age
        
    *   sex
        
    *   cancer stage
        
    *   histology
        

### Non-Cancer (Negative Class)

**MosMedData**

*   **CT-0**: ~254 normal lung CTs
    
*   **CT-2**: ~125 abnormal but non-cancer CTs (pneumonia-like)
    

> Note: MosMedData does not include clinical metadata. These features are treated as missing.

🧠 Preprocessing
----------------

### CT preprocessing

*   Resampled to **1×1×1 mm**
    
*   HU clipped to **\[-1000, 400\]**
    
*   Min-max normalized to **\[0, 1\]**
    
*   Automatic lung-focused crop (no segmentation)
    
*   Final shape: **(128 × 192 × 192)**
    
*   Saved as .npz
    

### Clinical preprocessing

*   Canonical schema across NSCLC datasets:
    
    *   age, sex, overall\_stage, histology, ehr\_present
        
*   Numeric: median imputation + standardization
    
*   Categorical: mode imputation + one-hot encoding
    
*   Saved as:
    
    *   clinical\_features.npz
        
    *   clinical\_preproc.joblib

Models
------

### 1️⃣ CT-Only Model

*   3D ResNet-18 (PyTorch)
    
*   Input: CT volume only
    

### 2️⃣ Late Fusion Model

*   CT encoder + MLP clinical branch
    
*   \[CT\_embedding || Clinical\_embedding || ehr\_present\]
    
*   CT encoder initialized from CT-only model
    
*   Clinical + fusion layers trained separately
    

📊 Evaluation
-------------

Metrics:

*   ROC-AUC
    
*   Average Precision (PR-AUC)
    

Artifacts:

*   Saved predictions (test\_predictions.npz)
    
*   ROC and PR curves
    
*   Clean vs hard negative comparison
    

📌 Key Findings
---------------

*   Models achieve **near-perfect AUC** on clean public datasets.
    
*   Performance remains high even on abnormal non-cancer CTs (MosMed CT-2).
    
*   This indicates **strong dataset bias and separability**, not true screening-level difficulty.
    
*   Late fusion provides limited benefit under current data distributions.
    

These results are treated as **feasibility findings**, not clinical claims.

📌 How to Run (High Level)
---------------

\# create manifests

python make\_source\_based\_splits.py

\# train CT-only

python train\_ct\_only.py

\# train late fusion

python train\_late\_fusion.py

\# evaluate on hard test

python test\_predictions.py
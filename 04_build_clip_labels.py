# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Build per-clip training labels for CM fine-tuning
# MAGIC
# MAGIC Takes the tensor index from `01_dicom_to_tensor` (one row per video clip)
# MAGIC and joins each clip to its patient's binary label from `labeled_patients.csv`.
# MAGIC
# MAGIC Output: a per-clip CSV the fine-tuning script trains on, with columns
# MAGIC `tensor_file_path`, `PMBB_ID`, `label`.
# MAGIC
# MAGIC The negative-downsampling switch lives here — flip it off to keep every
# MAGIC negative patient instead of trimming to a ratio.

# COMMAND ----------

# ────────────────────────────────────────────────────────────────────────────
# SECTION 0 │ Config
# ────────────────────────────────────────────────────────────────────────────

TARGET_COLUMN        = "HCM_PLP"     # HCM_PLP | DCM_PLP | any_CM_PLP | <any 0/1 column>

# tensor index from 01_dicom_to_tensor — one row per clip.
# Must contain a path column and a patient-ID column (rename below if different).
TENSOR_INDEX_CSV     = "/Workspace/VermaLab/Sahil_EchoCV/tensor_index.csv"
TENSOR_PATH_COL      = "tensor_file_path"
TENSOR_PATIENT_COL   = "PMBB_ID"     # the patient ID column inside the tensor index

LABELS_CSV           = "/Workspace/VermaLab/Sahil_EchoCV/labeled_patients.csv"
OUTPUT_CSV           = f"/Workspace/VermaLab/Sahil_EchoCV/clip_labels_{TARGET_COLUMN}.csv"

# ── Negative handling — the on/off switch you asked for ──────────────────────
DOWNSAMPLE_NEGATIVES = True          # True = trim negatives to a ratio; False = keep ALL negatives
NEG_RATIO            = 4             # negatives kept per positive (only used if downsampling)
SEED                 = 42

# COMMAND ----------

import numpy as np
import pandas as pd

np.random.seed(SEED)

# ────────────────────────────────────────────────────────────────────────────
# SECTION 1 │ Load tensor index + labels
# ────────────────────────────────────────────────────────────────────────────

tensors = pd.read_csv(TENSOR_INDEX_CSV)
labels  = pd.read_csv(LABELS_CSV)

# normalize column names so the join is clean
tensors = tensors.rename(columns={
    TENSOR_PATH_COL:    "tensor_file_path",
    TENSOR_PATIENT_COL: "PMBB_ID",
})

labels["label"] = labels[TARGET_COLUMN].astype(int)
labels = labels[["PMBB_ID", "label"]]

print(f"Clips in tensor index: {len(tensors):,}")
print(f"Patients with labels:  {len(labels):,}")

# ────────────────────────────────────────────────────────────────────────────
# SECTION 2 │ Join — every clip inherits its patient's binary label
# ────────────────────────────────────────────────────────────────────────────

clips = tensors.merge(labels, on="PMBB_ID", how="inner")

n_pos_pat = labels[labels.label == 1].PMBB_ID.nunique()
n_neg_pat = labels[labels.label == 0].PMBB_ID.nunique()
print(f"\nPatients — positives: {n_pos_pat} | negatives: {n_neg_pat}")
print(f"Clips after join: {len(clips):,}")

# ────────────────────────────────────────────────────────────────────────────
# SECTION 3 │ Negative downsampling (patient-level, before expanding to clips)
# ────────────────────────────────────────────────────────────────────────────

if DOWNSAMPLE_NEGATIVES:
    pos_patients = labels[labels.label == 1].PMBB_ID.unique()
    neg_patients = labels[labels.label == 0].PMBB_ID.unique()

    keep_n = min(len(neg_patients), len(pos_patients) * NEG_RATIO)
    keep_neg = np.random.choice(neg_patients, size=keep_n, replace=False)

    keep_patients = set(pos_patients) | set(keep_neg)
    clips = clips[clips.PMBB_ID.isin(keep_patients)].reset_index(drop=True)

    print(f"\nDownsampled negatives to {NEG_RATIO}:1 "
          f"({len(pos_patients)} pos patients / {keep_n} neg patients)")
else:
    print("\nKeeping ALL negatives (downsampling off)")

# ────────────────────────────────────────────────────────────────────────────
# SECTION 4 │ Save per-clip training CSV
# ────────────────────────────────────────────────────────────────────────────

out = clips[["tensor_file_path", "PMBB_ID", "label"]]

print(f"\n── Final clip-level dataset ──────────────────────────────────────")
print(f"Total clips:    {len(out):,}")
print(f"Positive clips: {(out.label == 1).sum():,}")
print(f"Negative clips: {(out.label == 0).sum():,}")
print(f"Patients:       {out.PMBB_ID.nunique():,}")

out.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved to {OUTPUT_CSV}")

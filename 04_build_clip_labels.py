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
# NOTE: 01 outputs only `dicom_file_path` + `tensor_file_path` (no patient ID).
# The DICOM folder carries the PMBB_RAD_ID (imaging ID, e.g. PMBBA + 10 digits),
# which is a DIFFERENT id space than the labels' PMBB_ID (PMBB + 14 digits).
# So we: parse PMBB_RAD_ID from the path -> map to PMBB_ID via the crosswalk
# table -> join to labels on PMBB_ID.
TENSOR_INDEX_CSV     = "/Workspace/VermaLab/Sahil_EchoCV/tensor_index.csv"
DICOM_PATH_COL       = "dicom_file_path"   # column holding the original DICOM path
TENSOR_PATH_COL      = "tensor_file_path"

# Crosswalk Databricks table with PMBB_RAD_ID <-> PMBB_ID (the one with EF/GLS/... cols).
CROSSWALK_TABLE      = "biobank_analytics.dev.<crosswalk_table_name>"   # SET THIS

# Regex that pulls the PMBB_RAD_ID out of the DICOM path (the imaging ID in the folder).
# VERIFY it matches the PMBB_RAD_ID format you see in the folders.
RAD_ID_REGEX         = r"(PMBBA\d+)"

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

tensors = tensors.rename(columns={TENSOR_PATH_COL: "tensor_file_path"})

# 1. parse PMBB_RAD_ID (imaging ID) out of the DICOM path
tensors["PMBB_RAD_ID"] = tensors[DICOM_PATH_COL].str.extract(RAD_ID_REGEX)
n_unparsed = tensors["PMBB_RAD_ID"].isna().sum()
if n_unparsed:
    print(f"WARNING: {n_unparsed:,} clips — could not parse PMBB_RAD_ID from path; dropping them")
    tensors = tensors.dropna(subset=["PMBB_RAD_ID"]).reset_index(drop=True)

# 2. map PMBB_RAD_ID -> PMBB_ID via the crosswalk table
crosswalk = (spark.table(CROSSWALK_TABLE)
                  .select("PMBB_RAD_ID", "PMBB_ID")
                  .dropDuplicates()
                  .toPandas())
tensors = tensors.merge(crosswalk, on="PMBB_RAD_ID", how="inner")
print(f"Clips after RAD_ID -> PMBB_ID crosswalk: {len(tensors):,}")
if len(tensors) == 0:
    raise ValueError("Crosswalk matched nothing — check RAD_ID_REGEX and CROSSWALK_TABLE")

labels["label"] = labels[TARGET_COLUMN].astype(int)
labels = labels[["PMBB_ID", "label"]]

# 3. sanity check: do the mapped PMBB_IDs match labeled patients?
matched = tensors["PMBB_ID"].isin(set(labels["PMBB_ID"])).sum()
print(f"Clips whose PMBB_ID matches a labeled patient: {matched:,} / {len(tensors):,}")
if matched == 0:
    raise ValueError("No mapped PMBB_IDs matched labels — check ID spaces line up")

print(f"\nClips in tensor index: {len(tensors):,}")
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

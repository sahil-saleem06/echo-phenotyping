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
# All project CSVs/outputs live under this new project volume.
# UC volume path is /Volumes/<catalog>/<schema>/<volume>/ — CONFIRM the volume name.
PROJECT_DIR          = "/Volumes/biobank_analytics/vl_echo_genetic_cm_finetuning/files"

# tensor index is an EXTERNAL input (coworker's 01 output) — set to its real location.
TENSOR_INDEX_CSV     = "/Volumes/biobank_analytics/pmbb_imaging_prepared/echo/july25/video_tensors_test/tensor_index.csv"
DICOM_PATH_COL       = "dicom_file_path"   # column holding the original DICOM path
TENSOR_PATH_COL      = "tensor_file_path"

# Crosswalk Databricks table with PMBB_RAD_ID <-> PMBB_ID (the one with EF/GLS/... cols).
CROSSWALK_TABLE      = "biobank_analytics.pmbb_imaging_prepared.july25_echo_id_trait_map"

# PMBB_RAD_ID is a numeric folder in the DICOM path (the one just above the dicoms/
# report subfolders). Rather than a fragile regex (the path has other numbers too),
# we identify it by matching path components against the known RAD_IDs in the crosswalk.

LABELS_CSV           = f"{PROJECT_DIR}/labeled_echo_patients.csv"
OUTPUT_CSV           = f"{PROJECT_DIR}/clip_labels_{TARGET_COLUMN}.csv"

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

# 1. load the crosswalk (PMBB_RAD_ID <-> PMBB_ID)
crosswalk = (spark.table(CROSSWALK_TABLE)
                  .select("PMBB_RAD_ID", "PMBB_ID")
                  .dropDuplicates()
                  .toPandas())
crosswalk["PMBB_RAD_ID"] = crosswalk["PMBB_RAD_ID"].astype(str)
rad_id_set = set(crosswalk["PMBB_RAD_ID"])

# 2. find each clip's PMBB_RAD_ID = the path component that is a known RAD_ID
def extract_rad_id(path):
    for part in str(path).split("/"):
        if part in rad_id_set:
            return part
    return None

tensors["PMBB_RAD_ID"] = tensors[DICOM_PATH_COL].apply(extract_rad_id)
n_unmatched = tensors["PMBB_RAD_ID"].isna().sum()
if n_unmatched:
    print(f"WARNING: {n_unmatched:,} clips — no RAD_ID in path matched the crosswalk; dropping them")
    tensors = tensors.dropna(subset=["PMBB_RAD_ID"]).reset_index(drop=True)

# 3. map PMBB_RAD_ID -> PMBB_ID
tensors = tensors.merge(crosswalk, on="PMBB_RAD_ID", how="inner")
print(f"Clips after RAD_ID -> PMBB_ID crosswalk: {len(tensors):,}")
if len(tensors) == 0:
    raise ValueError("Crosswalk matched nothing — check CROSSWALK_TABLE and the DICOM paths")

labels["label"] = labels[TARGET_COLUMN].astype(int)
labels = labels[["PMBB_ID", "label"]]

# 4. sanity check: do the mapped PMBB_IDs match labeled patients?
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

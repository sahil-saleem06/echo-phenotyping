# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Map Filtered Patients to DICOM Folders
# MAGIC
# MAGIC Takes a patient ID list from the LPC labeling steps (02_label_patients.R —
# MAGIC e.g. labeled_patients.csv or positive_patients.csv) and finds each patient's
# MAGIC DICOM folder in the echo data directory.
# MAGIC
# MAGIC Output: CSV with PMBB_ID and path to their DICOM folder.

# COMMAND ----------

# ────────────────────────────────────────────────────────────────────────────
# SECTION 0 │ Config
# ────────────────────────────────────────────────────────────────────────────

PATIENT_LIST  = "/Workspace/VermaLab/Sahil_EchoCV/filtered_patients.csv"
ECHO_BASE_DIR = "/Volumes/biobank_analytics/pmbb_imaging_prepared/echo/july25/"
OUTPUT_PATH   = "/Workspace/VermaLab/Sahil_EchoCV/patient_dicom_map.csv"

# ────────────────────────────────────────────────────────────────────────────
# SECTION 1 │ Load filtered patient list
# ────────────────────────────────────────────────────────────────────────────

import os
import pandas as pd
from tqdm import tqdm

patient_df = pd.read_csv(PATIENT_LIST)
patient_ids = set(patient_df["PMBB_ID"].astype(str).tolist())
print(f"Patients to find: {len(patient_ids):,}")

# ────────────────────────────────────────────────────────────────────────────
# SECTION 2 │ Build lookup of PMBB_ID → DICOM folder
# ────────────────────────────────────────────────────────────────────────────

records = []

for batch in sorted(os.listdir(ECHO_BASE_DIR)):
    batch_path = os.path.join(ECHO_BASE_DIR, batch)
    if not os.path.isdir(batch_path):
        continue

    for accession in os.listdir(batch_path):
        acc_path = os.path.join(batch_path, accession)
        if not os.path.isdir(acc_path):
            continue

        for patient_id in os.listdir(acc_path):
            if patient_id not in patient_ids:
                continue

            patient_path = os.path.join(acc_path, patient_id)

            # Find all DICOMs recursively regardless of subfolder structure
            dicoms = [
                os.path.join(root, f)
                for root, dirs, files in os.walk(patient_path)
                for f in files
                if f.endswith(".dcm")
            ]

            if dicoms:
                records.append({
                    "PMBB_ID":     patient_id,
                    "dicom_dir":   patient_path,
                    "n_dicoms":    len(dicoms),
                })

# ────────────────────────────────────────────────────────────────────────────
# SECTION 3 │ Results
# ────────────────────────────────────────────────────────────────────────────

results_df = pd.DataFrame(records)

print(f"\n── Results ──────────────────────────────────────────────────────")
print(f"Patients in filter list:      {len(patient_ids):,}")
print(f"Patients with DICOMs found:   {len(results_df):,}")
print(f"Patients with no DICOMs:      {len(patient_ids) - len(results_df):,}")
print(f"Avg DICOMs per patient:       {results_df['n_dicoms'].mean():.1f}")

results_df.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved to {OUTPUT_PATH}")

display(results_df.head(10))

# echo-phenotyping

Fine-tune a pretrained echocardiography foundation model (PanEcho or EchoPrime)
to predict a **binary phenotype** from echo videos — a disease, a genetic variant,
or any 0/1 label. Built and validated for detecting hypertrophic and dilated
cardiomyopathy (HCM / DCM) variant carriers in PMBB, but the target is fully
configurable.

---

## The idea

A pretrained echo model already knows how to *see* a heart — it learned that from
millions of videos. We **freeze** that knowledge and train just a small "head" on
top to answer one new yes/no question (e.g. *does this patient carry an HCM
variant?*). We never re-teach it to see hearts — only to attach a new label to what
it already sees. That's why a few hundred positive patients is enough to fine-tune,
when training from scratch would need orders of magnitude more.

**One clip at a time.** Each patient has ~100 video clips. Instead of one prediction
per patient, every clip becomes a training example carrying its patient's label —
turning, say, 149 HCM patients into ~15,000 HCM clips (far more signal). At
evaluation we average a patient's clip predictions back into a single patient-level
answer, which is the clinically meaningful unit.

---

## Pipeline

The project spans two environments. Scripts are numbered in run order.

**LPC (R) — define the cohort and labels:**

| Script | Does | Output |
|---|---|---|
| `01_filter_patients.R` | Select patients with abnormal echo measurements (LVEF / IVS / LVIDd), with sex-specific thresholds and outlier cleaning | `filtered_patients.csv` |
| `02_label_patients.R` | Join genetic variant data; convert to 0/1 labels; report class balance | `labeled_patients.csv`, `positive_patients.csv` |

**Databricks (Python) — prepare videos and train:**

| Script | Does | Output |
|---|---|---|
| `03_map_patients.py` | Find each patient's DICOM folder under the echo volume | `patient_dicom_map.csv` |
| *(external)* `01_dicom_to_tensor` | Convert DICOM clips to `.pt` tensors — **coworker's script, prerequisite** | `tensor_index.csv` |
| `04_build_clip_labels.py` | Join each clip tensor to its patient's 0/1 label; optional negative downsampling | `clip_labels_<TARGET>.csv` |
| `05_finetune_cm.py` | Fine-tune the head, evaluate at clip and patient level | trained model + predictions |

> **Note:** `04` consumes `tensor_index.csv` from the coworker's `01_dicom_to_tensor`
> step, which is not part of this repo. It expects a path column and a patient-ID
> column — adjust `TENSOR_PATH_COL` / `TENSOR_PATIENT_COL` at the top of `04` if the
> names differ.

---

## How each script works

### `01_filter_patients.R`
Loads the PMBB echo measurement tables (CUPID + PROSOLV), standardizes column names,
cleans values (strips units, averages ranges, drops impossible entries), and flags
patients whose **LVEF**, **IVS**, or **LVIDd** cross clinical thresholds. Thresholds
are sex-specific (e.g. IVS > 1.3 cm in men, > 1.2 in women) and editable at the top.
A patient qualifies on **any** abnormal measure (OR logic). Outputs the unique
qualifying patient IDs.

### `02_label_patients.R`
Inner-joins the filtered patients with the genetic variant table
(`HCM_PLP` / `DCM_PLP` / `any_CM_PLP` — pathogenic/likely-pathogenic flags),
converts true/false to 1/0, and prints the class balance (how many positives you
actually have). Writes the full labeled table plus a positives-only file for quickly
checking DICOM availability.

### `03_map_patients.py`
Walks the echo data volume and, for each patient in your list, recursively finds all
`.dcm` files (handling inconsistent sub-folder layouts). Outputs each patient's DICOM
directory and clip count — your real ceiling on usable data.

### `04_build_clip_labels.py`
Turns the per-patient labels into a **per-clip** training table: every clip tensor
inherits its patient's 0/1 label. The negative-handling switch lives here —
`DOWNSAMPLE_NEGATIVES = True` trims negatives to `NEG_RATIO` per positive;
`False` keeps them all.

### `05_finetune_cm.py`
The core training script (adapted from the lab's proven per-clip fine-tuning):

- Loads the chosen backbone as a **feature extractor only**
  (`backbone_only=True` for PanEcho → 768-dim; EchoPrime's MViT encoder → 512-dim)
- Attaches a small classifier head
- **Patient-level train/val/test split** — no patient spans two splits, so the test
  score can't be inflated by memorizing a patient
- **Class-weighted BCE loss** (`pos_weight`) so the model can't win by always
  predicting "no"
- Judged by **AUC / precision / recall**, never raw accuracy (useless at ~4%
  positives)
- Early stopping on validation AUC; saves the best checkpoint
- Reports both clip-level and **patient-level** results (clip probabilities averaged
  per patient)

---

## Configuration switches

Set at the top of the scripts:

| Switch | Where | Meaning |
|---|---|---|
| `TARGET_COLUMN` | `04`, `05` | Which label to train: `HCM_PLP`, `DCM_PLP`, `any_CM_PLP`, or any 0/1 column |
| `ENCODER` | `05` | `panecho` (proven) or `echoprime` |
| `FREEZE_BACKBONE` | `05` | `True` = train head only (safer with few positives); `False` = full fine-tune at a tiny LR |
| `DOWNSAMPLE_NEGATIVES` | `04` | `True` = trim negatives to `NEG_RATIO`; `False` = keep all |

Train one target at a time; re-run with a different `TARGET_COLUMN` for each. Swap
`ENCODER` to compare PanEcho vs EchoPrime on the same task.

#!/usr/bin/env Rscript
# Filter PMBB echo patients for HCM/DCM study
#
# Uses lab's standardized echo processing pipeline to identify patients
# with abnormal LVEF, IVS, or LVIDd values suggesting HCM or DCM.
# Applies sex-specific thresholds for IVS and LVIDd.
# Output: CSV of unique PMBB IDs meeting at least one threshold.
#
# Run on LPC:
#   Rscript filter_patients.R

library(dplyr)
library(tidyr)
library(ggplot2)
library(Routliers)
library(scales)
library(purrr)
library(readr)
library(vroom)

# ── Config — edit paths and thresholds here ─────────────────────────────────

CUPID_FILE   <- "/project/damrauer_shared/Phenotypes/echo_data/Echo_Result_discrete_cupid_clean_01_09_2025.csv"
PROSOLV_FILE <- "/project/damrauer_shared/Phenotypes/echo_data/All_Echo_Result_Prosolv_6_27_25_long.csv"
OUTPUT_PATH  <- "/project/damrauer_shared/Phenotypes/echo_data/Cleaned_TTE_merged/hcm_dcm_patient_ids.csv"

# Sex/demographics file — fill in path once you have it from boss
SEX_FILE     <- "PASTE_PATH_HERE"  # e.g. "/project/damrauer_shared/Phenotypes/demographics.csv"
SEX_COL      <- "PASTE_COL_NAME"   # e.g. "sex" or "gender" (must have PMBB_ID to join on)

# Abnormality thresholds — patients meeting ANY of these qualify
LVEF_THRESHOLD      <- 40      # LVEF < 40% → DCM
IVS_THRESHOLD_M     <- 1.3     # IVS > 1.3cm in men → HCM
IVS_THRESHOLD_F     <- 1.2     # IVS > 1.2cm in women → HCM
LVIDD_THRESHOLD_M   <- 6.3     # LVIDd > 6.3cm in men → DCM
LVIDD_THRESHOLD_F   <- 5.6     # LVIDd > 5.6cm in women → DCM

# ── Exclusion lists (from lab pipeline) ──────────────────────────────────────

TEE_STRESS_CUPID <- c(
  "ECHOCARDIOGRAM STRESS TEST",
  "TRANSESOPHAGEAL ECHO (TEE)",
  "CONGENITAL TRANSESOPHAGEAL ECHO (TEE)",
  "TRANSESOPHAGEAL ECHO (TEE) WITH POSSIBLE CARDIOVERSION",
  "RESEARCH ECHOCARDIOGRAM STRESS TEST",
  "INTRACARDIAC ECHO (ICE) (93662)",
  "ECHOCARDIOGRAM STRESS TEST W CPET",
  "ECHOCARDIOGRAM STRESS TEST WITH CARDIOPULMONARY EXERCISE TEST",
  "TRANSESOPHAGEAL ECHOCARDIOGRAM IMAGES ONLY",
  "CONGENITAL TRANSESOPHAGEAL ECHO ANES (TEE)",
  "TREADMILL STRESS ECHOCARDIOGRAM"
)

KEEP_ECHO_PROSOLV <- c("ECHO TTE", "ECHO TTE LIMITED")

# ── Lab functions ─────────────────────────────────────────────────────────────

pmbb_echo_format <- function(file_path,
                             measure_name_col = "COMMON_NAME",
                             measure_value_col = "ORD_VALUE",
                             IID_col = "PMBB_ID",
                             measure_unit_col = "REFERENCE_UNIT",
                             procedure_name_col = "DESCRIPTION",
                             exam_id = "ORDER_PROC_ID",
                             exam_date_col = "ORDERING_DATE_SHIFTED") {
  vroom::vroom(file_path) %>%
    dplyr::select(
      IID          = !!sym(IID_col),
      measure_name = !!sym(measure_name_col),
      measure_value = !!sym(measure_value_col),
      measure_unit  = !!sym(measure_unit_col),
      procedure_name = !!sym(procedure_name_col),
      exam_date     = !!sym(exam_date_col),
      exam_id       = !!sym(exam_id)
    )
}

process_measure_value <- function(df, measure_type = "numeric") {
  if (measure_type == "numeric") {
    df <- df %>%
      mutate(
        measure_value_cleaned = gsub(
          paste0(" ?(", paste(unique(measure_unit), collapse = "|"), ")"),
          "", measure_value
        ),
        measure_value_cleaned = case_when(
          grepl("^-?[0-9]+\\.?[0-9]*-[0-9]+\\.?[0-9]*$", measure_value_cleaned) ~
            sapply(strsplit(measure_value_cleaned, "-"),
                   function(x) { nums <- as.numeric(x); if (any(is.na(nums))) NA_real_ else mean(nums) }),
          grepl("^-?[0-9]+(\\.[0-9]+)?$", measure_value_cleaned) ~
            as.numeric(measure_value_cleaned),
          TRUE ~ NA_real_
        )
      )
  }
  return(df)
}

pmbb_echo_filter <- function(df, outcome_name, outcome_name_list,
                             exclude_procedure_list = NULL) {
  df %>%
    filter(measure_name %in% outcome_name_list) %>%
    { if (!is.null(exclude_procedure_list)) filter(., !(procedure_name %in% exclude_procedure_list)) else . } %>%
    mutate(outcome_name = outcome_name) %>%
    process_measure_value() %>%
    drop_na(measure_value_cleaned)
}

pmbb_MAD_filter <- function(df, threshold_number = 5) {
  out_MAD <- Routliers::outliers_mad(df$measure_value_cleaned, threshold = threshold_number)
  df %>% filter(measure_value_cleaned >= out_MAD$limits[1] &
                measure_value_cleaned <= out_MAD$limits[2])
}

pmbb_echo_manual_filter <- function(df, min = 0, max = 100) {
  df %>% filter(measure_value_cleaned >= min & measure_value_cleaned <= max)
}

# ── Load data ─────────────────────────────────────────────────────────────────

cat("Loading CUPID data...\n")
df_cupid <- pmbb_echo_format(CUPID_FILE) %>%
  filter(!procedure_name %in% TEE_STRESS_CUPID)

cat("Loading PROSOLV data...\n")
df_prosolv <- pmbb_echo_format(PROSOLV_FILE,
                               exam_id = "exam_id",
                               exam_date_col = "exam_date_shifted") %>%
  filter(procedure_name %in% KEEP_ECHO_PROSOLV)

df_echo <- rbind(df_cupid, df_prosolv)
cat("Combined rows:", nrow(df_echo), "\n\n")

# ── Load sex/demographics ────────────────────────────────────────────────────

cat("Loading sex/demographics data...\n")
df_sex <- vroom::vroom(SEX_FILE) %>%
  dplyr::select(IID = PMBB_ID, sex = !!sym(SEX_COL))

df_echo <- df_echo %>% left_join(df_sex, by = "IID")
cat("Sex information loaded\n\n")

# ── Process LVEF ──────────────────────────────────────────────────────────────

cat("Processing LVEF...\n")
final_list_EF <- c("EJECTION FRACTION (ECHO)", "ECHOEF", "3DECHOEF", "2DEF", "lvef_min")

df_LVEF <- df_echo %>%
  pmbb_echo_filter(outcome_name = "LV_EF", outcome_name_list = final_list_EF) %>%
  pmbb_echo_manual_filter(min = 0, max = 100) %>%
  pmbb_MAD_filter()

# Patients with low LVEF
low_ef_ids <- df_LVEF %>%
  filter(measure_value_cleaned < LVEF_THRESHOLD) %>%
  distinct(IID)

cat("Patients with LVEF <", LVEF_THRESHOLD, ":", nrow(low_ef_ids), "\n\n")

# ── Process IVS ───────────────────────────────────────────────────────────────

cat("Processing IVS...\n")
list_IVS <- c("IVS", "ivs_diastolic_thickness")

df_IVS <- df_echo %>%
  pmbb_echo_filter(outcome_name = "IVS", outcome_name_list = list_IVS) %>%
  pmbb_echo_manual_filter(min = 0, max = 100) %>%
  pmbb_MAD_filter()

# Patients with thick IVS — apply sex-specific thresholds
thick_ivs_ids <- df_IVS %>%
  filter(
    (tolower(sex) %in% c("m", "male") & measure_value_cleaned > IVS_THRESHOLD_M) |
    (tolower(sex) %in% c("f", "female") & measure_value_cleaned > IVS_THRESHOLD_F)
  ) %>%
  distinct(IID)

cat("Patients with abnormal IVS (sex-specific):", nrow(thick_ivs_ids), "\n\n")

# ── Process LVIDd ─────────────────────────────────────────────────────────────

cat("Processing LVIDd...\n")
list_LVIDD <- c("LVIDD", "LV DIAMETER, END-DIASTOLIC", "lvidd")

df_LVIDD <- df_echo %>%
  pmbb_echo_filter(outcome_name = "LVIDD", outcome_name_list = list_LVIDD) %>%
  pmbb_echo_manual_filter(min = 0, max = 100) %>%
  pmbb_MAD_filter()

# Patients with dilated LV — apply sex-specific thresholds
dilated_lv_ids <- df_LVIDD %>%
  filter(
    (tolower(sex) %in% c("m", "male") & measure_value_cleaned > LVIDD_THRESHOLD_M) |
    (tolower(sex) %in% c("f", "female") & measure_value_cleaned > LVIDD_THRESHOLD_F)
  ) %>%
  distinct(IID)

cat("Patients with abnormal LVIDd (sex-specific):", nrow(dilated_lv_ids), "\n\n")

# ── Combine — patients meeting ANY threshold ──────────────────────────────────

all_patient_ids <- bind_rows(low_ef_ids, thick_ivs_ids, dilated_lv_ids) %>%
  distinct(IID)

cat("── Results ──────────────────────────────────────────────────────\n")
cat("Total qualifying patients (any threshold):", nrow(all_patient_ids), "\n")

# ── Save ──────────────────────────────────────────────────────────────────────

write_csv(all_patient_ids %>% rename(PMBB_ID = IID), OUTPUT_PATH)
cat("Saved to:", OUTPUT_PATH, "\n")

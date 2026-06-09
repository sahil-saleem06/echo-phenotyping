#!/usr/bin/env Rscript
# Cross-reference filtered echo patients with genetic variant data
#
# Joins the abnormal-echo patient list (output of filter_patients.R) with the
# genetic variant table to produce a labeled training table: one row per
# patient with binary HCM/DCM variant labels.
#
# Run on LPC:
#   Rscript label_patients.R

library(dplyr)
library(readr)
library(vroom)

# ── Config — edit paths here ─────────────────────────────────────────────────

FILTERED_LIST <- "/home/saleemsa/filtered_echo_patients"   # output of filter_patients.R (no .csv ext, reads fine)
GENETIC_FILE  <- "PASTE_GENETIC_PATH_HERE"                  # fill in the genetic table path
OUTPUT_PATH   <- "/home/saleemsa/labeled_patients.csv"

# ── Helper: convert true/false (logical OR string) to 1/0 ────────────────────

to_binary <- function(x) {
  if (is.logical(x)) return(as.integer(x))
  as.integer(tolower(as.character(x)) %in% c("true", "t", "1", "yes"))
}

# ── Load both tables ─────────────────────────────────────────────────────────

cat("Loading filtered patient list...\n")
# single-column file (PMBB_ID only) — set delim explicitly so vroom doesn't guess
filtered <- vroom::vroom(FILTERED_LIST, delim = ",")
cat("  Filtered patients:", nrow(filtered), "\n\n")

cat("Loading genetic data...\n")
genetic <- vroom::vroom(GENETIC_FILE) %>%
  mutate(
    HCM_PLP    = to_binary(HCM_PLP),
    DCM_PLP    = to_binary(DCM_PLP),
    any_CM_PLP = to_binary(any_CM_PLP),
    both       = to_binary(both)
  )
cat("  Patients with genetic data:", nrow(genetic), "\n\n")

# ── Join — keep only patients with BOTH echo filter and genetic data ─────────

labeled <- filtered %>%
  inner_join(genetic, by = "PMBB_ID")

# ── Results ──────────────────────────────────────────────────────────────────

cat("── Cross-reference results ──────────────────────────────────────\n")
cat("Filtered (abnormal echo) patients:  ", nrow(filtered), "\n")
cat("  ...with genetic data (trainable): ", nrow(labeled), "\n")
cat("  ...without genetic data (dropped):", nrow(filtered) - nrow(labeled), "\n\n")

cat("Class balance among trainable patients:\n")
cat("  HCM variant (HCM_PLP=1):  ", sum(labeled$HCM_PLP, na.rm = TRUE), "\n")
cat("  DCM variant (DCM_PLP=1):  ", sum(labeled$DCM_PLP, na.rm = TRUE), "\n")
cat("  Any CM variant:           ", sum(labeled$any_CM_PLP, na.rm = TRUE), "\n")
cat("  Both HCM and DCM:         ", sum(labeled$both, na.rm = TRUE), "\n")
cat("  No variant (negatives):   ", sum(labeled$any_CM_PLP == 0, na.rm = TRUE), "\n\n")

# ── Export tables for Databricks DICOM check ─────────────────────────────────

# 1. Full labeled table — all trainable patients with their 0/1 labels
write_csv(labeled, OUTPUT_PATH)
cat("Saved full labeled table:", nrow(labeled), "patients ->", OUTPUT_PATH, "\n")

# 2. Positives only (any CM variant) — to check DICOM availability on Databricks
positives <- labeled %>% filter(any_CM_PLP == 1)
write_csv(positives, "/home/saleemsa/positive_patients.csv")
cat("Saved positives:", nrow(positives), "patients -> /home/saleemsa/positive_patients.csv\n")

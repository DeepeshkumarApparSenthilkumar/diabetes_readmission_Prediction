library(tidyverse)

df <- read_csv("data/processed/diabetes_clean.csv")

cat("=== FEATURE ENGINEERING ===\n")
cat("Input:", nrow(df), "rows x", ncol(df), "cols\n")

df <- df %>%
  mutate(
    # 1. high utilizer flag — strong prior signal for readmission
    high_utilizer   = as.integer(number_inpatient >= 3),
    # 2. polypharmacy flag — medication complexity
    polypharmacy    = as.integer(num_medications > 10),
    # 3. total visits — combined utilization score
    total_visits    = number_outpatient + number_emergency + number_inpatient,
    # 4. diabetes as primary diagnosis
    diab_primary    = as.integer(diag_1 == "Diabetes"),
    # 5. any medication changed during encounter
    any_change      = as.integer(change == "Ch"),
    # 6. age x inpatient interaction — old patients with frequent admissions
    age_x_inpatient = age_numeric * number_inpatient,
    # 7. medications x diagnoses interaction — complexity composite
    med_x_diagnoses = num_medications * number_diagnoses,
    # 8. emergency ratio — divide-by-zero guarded
    emergency_ratio = ifelse(total_visits > 0, number_emergency / total_visits, 0)
  )

cat("New features added: high_utilizer, polypharmacy, total_visits,\n")
cat("  diab_primary, any_change, age_x_inpatient, med_x_diagnoses, emergency_ratio\n")
cat("Output:", nrow(df), "rows x", ncol(df), "cols\n")

write_csv(df, "data/processed/diabetes_featured.csv")
cat("Saved: data/processed/diabetes_featured.csv\n")

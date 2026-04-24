library(tidyverse)

df <- read_csv("data/processed/diabetes_clean.csv")

cat("=== FEATURE ENGINEERING ===\n")
cat("Input:", nrow(df), "rows x", ncol(df), "cols\n")

# 1. high utilizer flag — strong prior signal for readmission
df <- df %>%
  mutate(high_utilizer = as.integer(number_inpatient >= 3))

# 2. polypharmacy flag — medication complexity
df <- df %>%
  mutate(polypharmacy = as.integer(num_medications > 10))

# 3. total visits — combined utilization score
df <- df %>%
  mutate(total_visits = number_outpatient + number_emergency + number_inpatient)

# 4. diabetes as primary diagnosis
df <- df %>%
  mutate(diab_primary = as.integer(diag_1 == "Diabetes"))

# 5. any medication changed during encounter
df <- df %>%
  mutate(any_change = as.integer(change == "Ch"))

# 6. age x inpatient interaction — old patients with frequent admissions
df <- df %>%
  mutate(age_x_inpatient = age_numeric * number_inpatient)

# 7. medications x diagnoses interaction — complexity composite
df <- df %>%
  mutate(med_x_diagnoses = num_medications * number_diagnoses)

# 8. emergency ratio — proportion of visits that were emergencies
df <- df %>%
  mutate(emergency_ratio = ifelse(
    total_visits > 0, number_emergency / total_visits, 0))

cat("New features added: high_utilizer, polypharmacy, total_visits,\n")
cat("  diab_primary, any_change, age_x_inpatient, med_x_diagnoses, emergency_ratio\n")
cat("Output:", nrow(df), "rows x", ncol(df), "cols\n")

write_csv(df, "data/processed/diabetes_featured.csv")
cat("Saved: data/processed/diabetes_featured.csv\n")

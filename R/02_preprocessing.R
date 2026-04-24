library(tidyverse)
library(janitor)

df <- read_csv("data/raw/diabetic_data.csv", na = c("", "NA", "?"))
df <- df %>% clean_names()

cat("=== PREPROCESSING PIPELINE ===\n")
cat("Start:", nrow(df), "rows\n")

# Step 1: remove deceased/hospice patients (cannot be readmitted — label noise)
dead_ids <- c(11, 13, 14, 19, 20, 21)
df <- df %>% filter(!discharge_disposition_id %in% dead_ids)
cat("After removing deceased/hospice:", nrow(df), "rows\n")

# Step 2: keep only first encounter per patient (prevent data leakage)
df <- df %>%
  arrange(patient_nbr) %>%
  distinct(patient_nbr, .keep_all = TRUE) %>%
  select(-patient_nbr, -encounter_id)
cat("After dedup (first encounter):", nrow(df), "rows\n")

# Step 3: drop high-missing columns
df <- df %>% select(-weight, -payer_code)

# Step 4: impute remaining missing values
df <- df %>%
  mutate(
    race              = replace_na(race, "Unknown"),
    medical_specialty = replace_na(medical_specialty, "Unknown")
  )

# Step 5: binary target — <30 = readmitted within 30 days
df <- df %>%
  mutate(readmitted_binary = as.integer(readmitted == "<30")) %>%
  select(-readmitted)

cat("Class distribution:\n")
print(prop.table(table(df$readmitted_binary)) %>% round(3))

# Step 6: age range -> numeric midpoint
df <- df %>%
  mutate(age_numeric = case_when(
    age == "[0-10)"   ~ 5,  age == "[10-20)"  ~ 15,
    age == "[20-30)"  ~ 25, age == "[30-40)"  ~ 35,
    age == "[40-50)"  ~ 45, age == "[50-60)"  ~ 55,
    age == "[60-70)"  ~ 65, age == "[70-80)"  ~ 75,
    age == "[80-90)"  ~ 85, age == "[90-100)" ~ 95,
    TRUE ~ 65
  )) %>%
  select(-age)

# Step 7: ICD-9 grouping (diag_1, diag_2, diag_3)
group_diag <- function(x) {
  x_num <- suppressWarnings(as.numeric(x))
  case_when(
    is.na(x)                              ~ "Other",
    str_detect(x, "^[Vv]|^[Ee]")         ~ "Other",
    x_num >= 390 & x_num <= 459           ~ "Circulatory",
    x_num >= 460 & x_num <= 519           ~ "Respiratory",
    x_num >= 520 & x_num <= 579           ~ "Digestive",
    x_num >= 250 & x_num <  251           ~ "Diabetes",
    x_num >= 800 & x_num <= 999           ~ "Injury",
    x_num >= 710 & x_num <= 739           ~ "Musculoskeletal",
    x_num >= 580 & x_num <= 629           ~ "Genitourinary",
    x_num >= 140 & x_num <= 239           ~ "Neoplasms",
    TRUE                                   ~ "Other"
  )
}
df <- df %>%
  mutate(across(c(diag_1, diag_2, diag_3), group_diag))

# Step 8: medical_specialty — keep top 10, collapse rest
top_spec <- df %>% count(medical_specialty) %>%
  slice_max(n, n = 10) %>% pull(medical_specialty)
df <- df %>%
  mutate(medical_specialty = ifelse(medical_specialty %in% top_spec,
                                    medical_specialty, "Other"))

# Step 9: A1C and glucose result — ordinal encoding
df <- df %>%
  mutate(
    a1cresult  = case_when(
      a1cresult == "None" ~ 0L, a1cresult == "Norm" ~ 1L,
      a1cresult == ">7"   ~ 2L, a1cresult == ">8"   ~ 3L, TRUE ~ 0L),
    max_glu_serum = case_when(
      max_glu_serum == "None"  ~ 0L, max_glu_serum == "Norm"  ~ 1L,
      max_glu_serum == ">200"  ~ 2L, max_glu_serum == ">300"  ~ 3L, TRUE ~ 0L)
  )

# Step 10: admission/discharge/source IDs -> factor
df <- df %>%
  mutate(
    admission_type_id        = as.factor(admission_type_id),
    discharge_disposition_id = as.factor(discharge_disposition_id),
    admission_source_id      = as.factor(admission_source_id)
  )

# Step 11: convert all remaining character columns to factors
df <- df %>% mutate(across(where(is.character), as.factor))

# final check
cat("\nFinal dimensions:", nrow(df), "rows x", ncol(df), "cols\n")
cat("Missing values remaining:", sum(is.na(df)), "\n")
cat("Positive rate:", round(mean(df$readmitted_binary) * 100, 1), "%\n")

write_csv(df, "data/processed/diabetes_clean.csv")
cat("Saved: data/processed/diabetes_clean.csv\n")

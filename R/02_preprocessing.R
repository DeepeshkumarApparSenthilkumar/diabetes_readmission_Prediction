library(tidyverse)
library(janitor)

df <- read_csv("data/raw/diabetic_data.csv", na = c("", "NA", "?"))
df <- df %>% clean_names()

# drop columns that are mostly missing or not useful for modeling
df <- df %>%
  select(-weight, -payer_code, -encounter_id)

# keep only first encounter per patient to avoid leakage
df <- df %>%
  arrange(patient_nbr) %>%
  distinct(patient_nbr, .keep_all = TRUE) %>%
  select(-patient_nbr)

nrow(df)  # should be 71518

# fix missing values
df <- df %>%
  mutate(
    race             = replace_na(race, "Unknown"),
    medical_specialty = replace_na(medical_specialty, "Unknown")
  )

# binary target - what we actually care about is <30 day readmission
df <- df %>%
  mutate(readmitted_binary = ifelse(readmitted == "<30", 1, 0)) %>%
  select(-readmitted)

table(df$readmitted_binary)
round(prop.table(table(df$readmitted_binary)) * 100, 1)

# age is a range like [50-60) - convert to numeric midpoint
df <- df %>%
  mutate(age_numeric = case_when(
    age == "[0-10)"   ~ 5,
    age == "[10-20)"  ~ 15,
    age == "[20-30)"  ~ 25,
    age == "[30-40)"  ~ 35,
    age == "[40-50)"  ~ 45,
    age == "[50-60)"  ~ 55,
    age == "[60-70)"  ~ 65,
    age == "[70-80)"  ~ 75,
    age == "[80-90)"  ~ 85,
    age == "[90-100)" ~ 95
  )) %>%
  select(-age)

# diag codes have 700+ levels - group them into broad icd-9 categories
# this is standard practice for this dataset
group_diag <- function(x) {
  case_when(
    is.na(x)                          ~ "Other",
    str_detect(x, "^[Vv]")           ~ "Other",
    str_detect(x, "^[Ee]")           ~ "Other",
    as.numeric(x) >= 390 & as.numeric(x) <= 459 ~ "Circulatory",
    as.numeric(x) >= 460 & as.numeric(x) <= 519 ~ "Respiratory",
    as.numeric(x) >= 520 & as.numeric(x) <= 579 ~ "Digestive",
    as.numeric(x) >= 250 & as.numeric(x) < 251   ~ "Diabetes",
    as.numeric(x) >= 800 & as.numeric(x) <= 999  ~ "Injury",
    as.numeric(x) >= 710 & as.numeric(x) <= 739  ~ "Musculoskeletal",
    as.numeric(x) >= 580 & as.numeric(x) <= 629  ~ "Genitourinary",
    as.numeric(x) >= 140 & as.numeric(x) <= 239  ~ "Neoplasms",
    TRUE                                           ~ "Other"
  )
}

df <- df %>%
  mutate(
    diag_1 = group_diag(diag_1),
    diag_2 = group_diag(diag_2),
    diag_3 = group_diag(diag_3)
  )

# medical_specialty has 73 levels - keep top ones, collapse rest to Other
top_specialties <- df %>%
  count(medical_specialty) %>%
  slice_max(n, n = 10) %>%
  pull(medical_specialty)

df <- df %>%
  mutate(medical_specialty = ifelse(
    medical_specialty %in% top_specialties,
    medical_specialty,
    "Other"
  ))

# admission_type_id, discharge_disposition_id, admission_source_id
# are numeric codes but they're actually categories
df <- df %>%
  mutate(
    admission_type_id        = as.factor(admission_type_id),
    discharge_disposition_id = as.factor(discharge_disposition_id),
    admission_source_id      = as.factor(admission_source_id)
  )

# convert all remaining character columns to factors for modeling
df <- df %>%
  mutate(across(where(is.character), as.factor))

# final check
glimpse(df)
dim(df)
sum(is.na(df))  # should be 0

# save cleaned data
write_csv(df, "data/processed/diabetes_clean.csv")
cat("saved to data/processed/diabetes_clean.csv\n")
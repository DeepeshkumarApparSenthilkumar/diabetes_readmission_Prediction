library(tidyverse)
library(skimr)
library(janitor)

# load the data - ? is how missing values are coded in this dataset
df <- read_csv("data/raw/diabetic_data.csv", na = c("", "NA", "?"))

dim(df)
head(df)
names(df)

df <- df %>% clean_names()

# get a proper overview of everything
skim(df)

# checking which columns have missing data and how bad it is
missing <- df %>%
  summarise(across(everything(), ~sum(is.na(.)))) %>%
  pivot_longer(everything(), names_to = "col", values_to = "n_missing") %>%
  mutate(pct_missing = round(n_missing / nrow(df) * 100, 1)) %>%
  filter(n_missing > 0) %>%
  arrange(desc(pct_missing))

missing

# how does the target variable look - expecting imbalance
df %>%
  count(readmitted) %>%
  mutate(pct = round(n / sum(n) * 100, 1))

# same patient can show up multiple times across encounters
# this is a problem for modeling - need to decide how to handle it
n_distinct(df$patient_nbr)
n_distinct(df$encounter_id)
nrow(df)

# how many patients have more than one visit in the data
df %>%
  count(patient_nbr) %>%
  filter(n > 1) %>%
  nrow()

# numeric columns
df %>% select(where(is.numeric)) %>% summary()

# cardinality of categorical columns - some might be too high to use directly
df %>%
  select(where(is.character)) %>%
  summarise(across(everything(), n_distinct)) %>%
  pivot_longer(everything(), names_to = "col", values_to = "unique_vals") %>%
  arrange(desc(unique_vals))

# save missing value summary - will need this when deciding on imputation
write_csv(missing, "outputs/results/missing_audit.csv")
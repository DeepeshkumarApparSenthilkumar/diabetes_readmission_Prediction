library(tidyverse)
library(ggplot2)

df <- read_csv("data/processed/diabetes_clean.csv")

# make sure binary target is a factor for plotting
df <- df %>%
  mutate(readmitted_binary = as.factor(readmitted_binary))

# 1. target class distribution
df %>%
  count(readmitted_binary) %>%
  mutate(pct = round(n / sum(n) * 100, 1)) %>%
  ggplot(aes(x = readmitted_binary, y = n, fill = readmitted_binary)) +
  geom_col(width = 0.5) +
  geom_text(aes(label = paste0(pct, "%")), vjust = -0.5, size = 4) +
  scale_fill_manual(values = c("0" = "#7F77DD", "1" = "#D85A30")) +
  labs(title = "readmission class distribution",
       x = "readmitted within 30 days", y = "count") +
  theme_minimal() +
  theme(legend.position = "none")

ggsave("outputs/figures/01_class_distribution.png", width = 6, height = 4)

# 2. age vs readmission
df %>%
  ggplot(aes(x = age_numeric, fill = readmitted_binary)) +
  geom_histogram(binwidth = 10, position = "dodge", alpha = 0.8) +
  scale_fill_manual(values = c("0" = "#7F77DD", "1" = "#D85A30"),
                    labels = c("not readmitted", "readmitted <30")) +
  labs(title = "age distribution by readmission status",
       x = "age", y = "count", fill = "") +
  theme_minimal()

ggsave("outputs/figures/02_age_distribution.png", width = 7, height = 4)

# 3. time in hospital
df %>%
  ggplot(aes(x = readmitted_binary, y = time_in_hospital, fill = readmitted_binary)) +
  geom_boxplot(alpha = 0.8) +
  scale_fill_manual(values = c("0" = "#7F77DD", "1" = "#D85A30")) +
  labs(title = "time in hospital vs readmission",
       x = "readmitted within 30 days", y = "days in hospital") +
  theme_minimal() +
  theme(legend.position = "none")

ggsave("outputs/figures/03_time_in_hospital.png", width = 6, height = 4)

# 4. number of inpatient visits before this encounter
df %>%
  ggplot(aes(x = readmitted_binary, y = number_inpatient, fill = readmitted_binary)) +
  geom_boxplot(alpha = 0.8) +
  scale_fill_manual(values = c("0" = "#7F77DD", "1" = "#D85A30")) +
  labs(title = "prior inpatient visits vs readmission",
       x = "readmitted within 30 days", y = "number of prior inpatient visits") +
  theme_minimal() +
  theme(legend.position = "none")

ggsave("outputs/figures/04_prior_inpatient.png", width = 6, height = 4)

# 5. number of medications
df %>%
  ggplot(aes(x = num_medications, fill = readmitted_binary)) +
  geom_density(alpha = 0.5) +
  scale_fill_manual(values = c("0" = "#7F77DD", "1" = "#D85A30"),
                    labels = c("not readmitted", "readmitted <30")) +
  labs(title = "number of medications by readmission status",
       x = "number of medications", y = "density", fill = "") +
  theme_minimal()

ggsave("outputs/figures/05_num_medications.png", width = 7, height = 4)

# 6. readmission rate by primary diagnosis
df %>%
  group_by(diag_1) %>%
  summarise(readmit_rate = mean(as.numeric(as.character(readmitted_binary))),
            n = n()) %>%
  filter(n > 100) %>%
  arrange(desc(readmit_rate)) %>%
  ggplot(aes(x = reorder(diag_1, readmit_rate), y = readmit_rate)) +
  geom_col(fill = "#D85A30", alpha = 0.8) +
  coord_flip() +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "readmission rate by primary diagnosis",
       x = "", y = "readmission rate") +
  theme_minimal()

ggsave("outputs/figures/06_readmit_by_diag.png", width = 7, height = 5)

# 7. insulin use vs readmission
df %>%
  group_by(insulin) %>%
  summarise(readmit_rate = mean(as.numeric(as.character(readmitted_binary))),
            n = n()) %>%
  ggplot(aes(x = reorder(insulin, readmit_rate), y = readmit_rate)) +
  geom_col(fill = "#1D9E75", alpha = 0.8) +
  coord_flip() +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "readmission rate by insulin prescription",
       x = "insulin", y = "readmission rate") +
  theme_minimal()

ggsave("outputs/figures/07_insulin_readmit.png", width = 6, height = 4)

# 8. correlation among numeric features
num_cols <- df %>%
  select(time_in_hospital, num_lab_procedures, num_procedures,
         num_medications, number_outpatient, number_emergency,
         number_inpatient, number_diagnoses, age_numeric)

cor_matrix <- cor(num_cols)
print(round(cor_matrix, 2))

cat("all plots saved to outputs/figures/\n")
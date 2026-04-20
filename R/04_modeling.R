library(tidyverse)
library(caret)
library(randomForest)
library(xgboost)
library(pROC)

df <- read_csv("data/processed/diabetes_clean.csv")

df <- df %>%
  mutate(across(where(is.character), as.factor)) %>%
  mutate(readmitted_binary = as.factor(ifelse(readmitted_binary == 1, "yes", "no")))

# 70/30 stratified split
set.seed(42)
train_idx <- createDataPartition(df$readmitted_binary, p = 0.7, list = FALSE)
train <- df[train_idx, ]
test  <- df[-train_idx, ]

cat("train size:", nrow(train), "\n")
cat("test size: ", nrow(test), "\n")

# manually balance training set - 10k each class
set.seed(42)
train_no  <- train %>% filter(readmitted_binary == "no")  %>% sample_n(10000, replace = FALSE)
train_yes <- train %>% filter(readmitted_binary == "yes") %>% sample_n(10000, replace = TRUE)
train_balanced <- bind_rows(train_no, train_yes) %>% sample_frac(1)

cat("balanced size:", nrow(train_balanced), "\n")
print(prop.table(table(train_balanced$readmitted_binary)))

# remove near-zero variance columns
nzv <- nearZeroVar(train_balanced)
if (length(nzv) > 0) {
  cat("removing", length(nzv), "near-zero variance columns\n")
  train_balanced <- train_balanced[, -nzv]
  test <- test[, names(test) %in% names(train_balanced)]
}
cat("columns remaining:", ncol(train_balanced), "\n")

# 5-fold CV setup for LR and RF
ctrl <- trainControl(
  method          = "cv",
  number          = 5,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = TRUE
)

# logistic regression
cat("\ntraining logistic regression...\n")
set.seed(42)
model_lr <- train(
  readmitted_binary ~ .,
  data      = train_balanced,
  method    = "glm",
  family    = "binomial",
  trControl = ctrl,
  metric    = "ROC"
)
cat("LR cross-val ROC:", round(max(model_lr$results$ROC), 3), "\n")

# random forest
cat("\ntraining random forest...\n")
set.seed(42)
model_rf <- train(
  readmitted_binary ~ .,
  data      = train_balanced,
  method    = "rf",
  trControl = ctrl,
  metric    = "ROC",
  tuneGrid  = expand.grid(mtry = 8),
  ntree     = 100
)
cat("RF cross-val ROC:", round(max(model_rf$results$ROC), 3), "\n")

# xgboost - needs numeric matrix, convert factors to dummies first
cat("\ntraining xgboost...\n")

xgb_features <- train_balanced %>% select(-readmitted_binary)
xgb_label    <- ifelse(train_balanced$readmitted_binary == "yes", 1, 0)

# one-hot encode all factor columns
xgb_features <- model.matrix(~ . - 1, data = xgb_features)

test_features <- test %>% select(-readmitted_binary)
test_label    <- ifelse(test$readmitted_binary == "yes", 1, 0)
test_features <- model.matrix(~ . - 1, data = test_features)

# align columns between train and test matrix
common_cols   <- intersect(colnames(xgb_features), colnames(test_features))
xgb_features  <- xgb_features[, common_cols]
test_features <- test_features[, common_cols]

dtrain <- xgb.DMatrix(data = xgb_features, label = xgb_label)
dtest  <- xgb.DMatrix(data = test_features, label = test_label)

set.seed(42)
model_xgb <- xgb.train(
  data             = dtrain,
  nrounds          = 100,
  max_depth        = 3,
  eta              = 0.1,
  subsample        = 0.8,
  colsample_bytree = 0.8,
  objective        = "binary:logistic",
  eval_metric      = "auc",
  verbose          = 0
)

# get ROC on training set to report cross-val equivalent
xgb_train_probs <- predict(model_xgb, dtrain)
xgb_train_roc   <- roc(xgb_label, xgb_train_probs, quiet = TRUE)
cat("XGB train AUC:", round(auc(xgb_train_roc), 3), "\n")

# save everything needed for evaluation
saveRDS(model_lr,   "outputs/results/model_lr.rds")
saveRDS(model_rf,   "outputs/results/model_rf.rds")
saveRDS(model_xgb,  "outputs/results/model_xgb.rds")
saveRDS(test,       "outputs/results/test_set.rds")
saveRDS(dtest,      "outputs/results/dtest.rds")
saveRDS(test_label, "outputs/results/test_label.rds")

cat("\nall models saved\n")
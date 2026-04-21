library(tidyverse)
library(caret)
library(pROC)
library(xgboost)

# load saved models and test set
model_lr  <- readRDS("outputs/results/model_lr.rds")
model_rf  <- readRDS("outputs/results/model_rf.rds")
model_xgb <- readRDS("outputs/results/model_xgb.rds")
test      <- readRDS("outputs/results/test_set.rds")

# fix factor columns
test <- test %>%
  mutate(across(where(is.character), as.factor)) %>%
  mutate(readmitted_binary = as.factor(readmitted_binary))

# rebuild xgboost test matrix using exact training columns
xgb_col_names <- readRDS("outputs/results/xgb_col_names.rds")
test_label    <- ifelse(test$readmitted_binary == "yes", 1, 0)
test_features <- test %>% select(-readmitted_binary)
test_features <- model.matrix(~ . - 1, data = test_features)

missing_cols <- setdiff(xgb_col_names, colnames(test_features))
if (length(missing_cols) > 0) {
  extra <- matrix(0,
                  nrow = nrow(test_features),
                  ncol = length(missing_cols),
                  dimnames = list(NULL, missing_cols))
  test_features <- cbind(test_features, extra)
}
test_features <- test_features[, xgb_col_names]
dtest <- xgb.DMatrix(data = test_features, label = test_label)

# logistic regression predictions
lr_probs <- predict(model_lr, newdata = test, type = "prob")[, "yes"]
lr_preds <- predict(model_lr, newdata = test)
lr_roc   <- roc(test$readmitted_binary, lr_probs,
                levels = c("no", "yes"), quiet = TRUE)

# random forest predictions
rf_probs <- predict(model_rf, newdata = test, type = "prob")[, "yes"]
rf_preds <- predict(model_rf, newdata = test)
rf_roc   <- roc(test$readmitted_binary, rf_probs,
                levels = c("no", "yes"), quiet = TRUE)

# xgboost predictions
xgb_probs         <- predict(model_xgb, dtest)
xgb_preds_factor  <- factor(ifelse(xgb_probs > 0.5, "yes", "no"),
                            levels = c("no", "yes"))
test_label_factor <- factor(ifelse(test_label == 1, "yes", "no"),
                            levels = c("no", "yes"))
xgb_roc <- roc(test_label, xgb_probs, quiet = TRUE)

# confusion matrices
cat("\n--- logistic regression ---\n")
cm_lr <- confusionMatrix(lr_preds, test$readmitted_binary, positive = "yes")
print(cm_lr)

cat("\n--- random forest ---\n")
cm_rf <- confusionMatrix(rf_preds, test$readmitted_binary, positive = "yes")
print(cm_rf)

cat("\n--- xgboost ---\n")
cm_xgb <- confusionMatrix(xgb_preds_factor, test_label_factor, positive = "yes")
print(cm_xgb)

# AUC comparison
cat("\n--- AUC comparison ---\n")
cat("LR  AUC:", round(auc(lr_roc),  3), "\n")
cat("RF  AUC:", round(auc(rf_roc),  3), "\n")
cat("XGB AUC:", round(auc(xgb_roc), 3), "\n")

# ROC curves
png("outputs/figures/08_roc_curves.png", width = 700, height = 600)
plot(lr_roc,  col = "#7F77DD", lwd = 2,
     main = "ROC curves - model comparison")
plot(rf_roc,  col = "#1D9E75", lwd = 2, add = TRUE)
plot(xgb_roc, col = "#D85A30", lwd = 2, add = TRUE)
legend("bottomright",
       legend = c(
         paste("LR  AUC =", round(auc(lr_roc),  3)),
         paste("RF  AUC =", round(auc(rf_roc),  3)),
         paste("XGB AUC =", round(auc(xgb_roc), 3))
       ),
       col = c("#7F77DD", "#1D9E75", "#D85A30"),
       lwd = 2)
dev.off()
cat("ROC curve saved\n")

# feature importance from random forest
rf_imp <- varImp(model_rf)$importance %>%
  rownames_to_column("feature") %>%
  arrange(desc(Overall)) %>%
  head(15)

ggplot(rf_imp, aes(x = reorder(feature, Overall), y = Overall)) +
  geom_col(fill = "#1D9E75", alpha = 0.8) +
  coord_flip() +
  labs(title = "top 15 features - random forest",
       x = "", y = "importance") +
  theme_minimal()

ggsave("outputs/figures/09_feature_importance.png", width = 7, height = 6)

# final results table
results <- tibble(
  model     = c("Logistic Regression", "Random Forest", "XGBoost"),
  auc       = round(c(auc(lr_roc), auc(rf_roc), auc(xgb_roc)), 3),
  precision = round(c(cm_lr$byClass["Precision"],
                      cm_rf$byClass["Precision"],
                      cm_xgb$byClass["Precision"]), 3),
  recall    = round(c(cm_lr$byClass["Recall"],
                      cm_rf$byClass["Recall"],
                      cm_xgb$byClass["Recall"]), 3),
  f1        = round(c(cm_lr$byClass["F1"],
                      cm_rf$byClass["F1"],
                      cm_xgb$byClass["F1"]), 3)
)

print(results)
write_csv(results, "outputs/results/model_comparison.csv")
cat("results saved\n")
################################################################################
# Libraries
################################################################################

library(randomForest)
library(caret)
library(doParallel)
library(ggplot2)
library(smotefamily)
library(pROC)
library(MLmetrics)
library(tidyverse)
library(gt)
library(reshape2)
library(RColorBrewer)
library(themis)

################################################################################
# Data Loading & Preparation
################################################################################

#Higher perfomance (only if necessary!!!)
#cl <- makeCluster(detectCores()-3)
#registerDoParallel(cl)

# Load raw data
data <- read.csv("data/processed/df_final.csv")

# Inspect structure
summary(data)
str(data)

# Convert binary integers to factors
data$female          <- factor(data$female, levels = c(0,1), labels = c("male","female"))
data$urban           <- factor(data$urban, levels = c(0,1), labels = c("rural","urban"))
data$employed        <- factor(data$employed, levels = c(0,1), labels = c("unemployed","employed"))
data$rec_gov_transfer<- factor(data$rec_gov_transfer, levels = c(0,1), labels = c("No","Yes"))
data$rec_gov_pension <- factor(data$rec_gov_pension, levels = c(0,1), labels = c("No","Yes"))
data$rec_agri_payment<- factor(data$rec_agri_payment, levels = c(0,1), labels = c("No","Yes"))
data$paid_ut_bill    <- factor(data$paid_ut_bill, levels = c(0,1), labels = c("No","Yes"))
data$internetaccess  <- factor(data$internetaccess, levels = c(0,1), labels = c("No","Yes"))
data$mobileowner     <- factor(data$mobileowner, levels = c(0,1), labels = c("No","Yes"))
data$has_debit_card  <- factor(data$has_debit_card, levels = c(0,1), labels = c("No","Yes"))

# Convert ordinal variables
data$educ <- factor(data$educ, 
                    levels = c(1,2,3), 
                    labels = c("primary_or_less","secondary","tertiary_or_more"), 
                    ordered = TRUE)

data$income_q <- factor(data$income_q, 
                        level = c(1,2,3,4,5), 
                        labels = c("poorest_20","second_20","middle_20","fourth_20","richest_20"), 
                        ordered = TRUE)

# Keep only complete cases
data_clean <- na.omit(data)


################################################################################
# Data splitting
################################################################################

#70 % training 30 % testing
set.seed(67)
train_index <- createDataPartition(data_clean$has_debit_card, p = 0.7, list = FALSE)
data_train <- data_clean[train_index, ]
data_test  <- data_clean[-train_index, ]

#set distributions
cat("training distribution:\n")
print(table(data_train$has_debit_card))
cat("test distribution:\n")
print(table(data_test$has_debit_card))

#Visualization training distribution
train_dist <- data_train %>%
  count(has_debit_card, name = "Count") %>%
  mutate(Percent = Count/sum(Count)) %>%
  rename(Class = has_debit_card) %>%
  gt() %>%
  fmt_percent(columns = "Percent", decimals = 1) %>%
  cols_label() %>%
  tab_header(title = "Training Set Distribution")

train_dist


################################################################################
# Tree stabilization analysis OOB
################################################################################

set.seed(67)

# Range of trees to test
tree_counts <- c(500, 1000, 5000)

rf_models <- list()

for (nt in tree_counts) {
  rf_models[[as.character(nt)]] <- randomForest(
    has_debit_card ~ ., data = data_train,
    importance = TRUE, keep.forest = TRUE, keep.inbag = TRUE,
    ntree = nt
  )
}

# Plot OOB error for each ntree
par(mfrow = c(2,2))  # 4 plots on one page

for (nt in tree_counts) {
  err <- rf_models[[as.character(nt)]]$err.rate
  
  # Linear scale
  plot(1:nrow(err), err[,"OOB"], type="l", lwd=2, col="blue",
       xlab="Number of trees", ylab="OOB Error rate",
       main=paste("OOB Error (", nt, " trees, linear)", sep=""))
  
  # Log scale
  plot(1:nrow(err), err[,"OOB"], type="l", lwd=2, col="blue", log="x",
       xlab="Number of trees (log scale)", ylab="OOB Error rate",
       main=paste("OOB Error (", nt, " trees, log)", sep=""))
  
  # Class-specific error (linear)
  matplot(1:nrow(err), err, type="l", lty=1, lwd=2,
          col=c("blue","red","darkgreen"),
          xlab="Number of trees", ylab="Error rate",
          main=paste("Class Error (", nt, " trees, linear)", sep=""))
  legend("topright", legend=colnames(err),
         col=c("blue","red","darkgreen"), lty=1, lwd=2, cex=0.8)
  
  # Class-specific error (log)
  matplot(1:nrow(err), err, type="l", lty=1, lwd=2, log="x",
          col=c("blue","red","darkgreen"),
          xlab="Number of trees (log scale)", ylab="Error rate",
          main=paste("Class Error (", nt, " trees, log)", sep=""))
  legend("topright", legend=colnames(err),
         col=c("blue","red","darkgreen"), lty=1, lwd=2, cex=0.8)
}

par(mfrow=c(1,1))  # reset


################################################################################
# Optimization of mtry and trees with OOB (with Accuracy)
################################################################################
set.seed(67)
p <- ncol(data_train) - 1
tunegrid <- expand.grid(.mtry = 1:p)
ntree_values <- c(100, 200, 500, 750, 1000)

control_oob <- trainControl(method = "oob")

results_oob <- list()

for (nt in ntree_values) {
  cat("==== OOB: ntree =", nt, "====\n")
  
  rf_oob <- train(
    has_debit_card ~ ., data = data_train,
    method = "rf",
    metric = "Accuracy",
    tuneGrid = tunegrid,
    trControl = control_oob,
    ntree = nt
  )
  
  rf_oob$results$ntree <- nt
  rf_oob$results$type  <- "OOB"
  
  results_oob[[paste0("ntree_", nt)]] <- rf_oob$results
}

results_oob <- dplyr::bind_rows(results_oob)

cols <- brewer.pal(6, "BuPu")[2:6]
ggplot(results_oob, aes(x = mtry, y = Accuracy, color = factor(ntree))) +
  geom_line() + geom_point() +
  scale_x_continuous(breaks = seq(min(results_oob$mtry), max(results_oob$mtry), 1)) +
  scale_color_manual(values = cols) +
  labs(title = "Random Forest (OOB): Accuracy by mtry × ntree",
       x = "mtry", y = "Accuracy", color = "ntree") +
  theme_minimal(base_size = 14) 


################################################################################
# Optimization of mtry and trees with 10-fold CV (with ROC)
################################################################################
set.seed(67)
control_cv <- trainControl(method="cv", number=10, search="grid", classProbs = TRUE, summaryFunction = twoClassSummary)

results_cv <- list()

for (nt in ntree_values) {
  cat("==== CV: ntree =", nt, "====\n")
  
  rf_cv <- train(
    has_debit_card ~ ., data = data_train,
    method = "rf",
    metric = "ROC",
    tuneGrid = tunegrid,
    trControl = control_cv,
    ntree = nt
  )
  
  rf_cv$results$ntree <- nt
  rf_cv$results$type  <- "CV_10fold"
  
  results_cv[[paste0("ntree_", nt)]] <- rf_cv$results
}

results_cv <- dplyr::bind_rows(results_cv)

ggplot(results_cv, aes(x = mtry, y = ROC, color = factor(ntree))) +
  geom_line() + geom_point() +
  scale_x_continuous(breaks = seq(min(results_oob$mtry), max(results_cv$mtry), 1)) +
  scale_color_manual(values = cols) +
  labs(title = "Random Forest (10-fold CV): ROC AUC by mtry × ntree",
       x = "mtry", y = "AUC (ROC)", color = "ntree") +
  theme_minimal(base_size = 14)


################################################################################
# Optimization of mtry and trees with LOOCV (several hours with parallelization) (with ROC)
################################################################################
set.seed(67)
control_loocv <- trainControl(method="LOOCV", search="grid", classProbs = TRUE, summaryFunction = twoClassSummary)

results_loocv <- list()

for (nt in ntree_values) {
  cat("==== LOOCV: ntree =", nt, "====\n")
  
  rf_loocv <- train(
    has_debit_card ~ ., data = data_train,
    method = "rf",
    metric = "ROC",
    tuneGrid = tunegrid,
    trControl = control_loocv,
    ntree = nt
  )
  
  rf_loocv$results$ntree <- nt
  rf_loocv$results$type  <- "LOOCV"
  
  results_loocv[[paste0("ntree_", nt)]] <- rf_loocv$results
}

results_loocv <- dplyr::bind_rows(results_loocv)


ggplot(results_loocv, aes(x = mtry, y = ROC, color = factor(ntree))) +
  geom_line() + geom_point() +
  scale_x_continuous(breaks = seq(min(results_loocv$mtry), max(results_cv$mtry), 1)) +
  labs(title = "Random Forest (LOOCV): AUC ROC by mtry × ntree",
       x = "mtry", y = "AUC (ROC) ", color = "ntree") +
  theme_minimal(base_size = 14)

################################################################################
# Model Evaluation on test data: Default vs Tuned
################################################################################

### Default RF (500 trees, mtry=3)
set.seed(67)
rf_default <- randomForest(has_debit_card ~ ., data = data_train,
                           importance = TRUE, keep.forest = TRUE, keep.inbag = TRUE, ntree = 500)

cat("\n--- Default RF (500 trees, mtry=3) ---\n")
print(rf_default)   # OOB error

# Predictions (class + probability)
rf_pred_class_default <- predict(rf_default, newdata = data_test, type = "response")
rf_pred_prob_default  <- predict(rf_default, newdata = data_test, type = "prob")[,"Yes"]

# Confusion matrix with detailed stats
cm_default <- confusionMatrix(rf_pred_class_default, data_test$has_debit_card, positive="Yes")
print(cm_default)

# Visualization confusion matrix
cm_d <- as.data.frame(cm_default$table)

ggplot(cm_d, aes(x = Prediction, y = Reference, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "white", size = 6, fontface = "bold") +
  scale_fill_gradient(low = "plum1", high = "plum4", name = "Freq") +
  coord_equal() +
  labs(
    title = "Confusion Matrix - Default (Test Set)",
    x = "Predicted Class",
    y = "Actual Class"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5)
  )

# Extra metrics
roc_obj_default <- roc(response = data_test$has_debit_card, predictor = rf_pred_prob_default,
                       levels = c("No","Yes"), direction = "<")
auc_default <- auc(roc_obj_default)
cat("AUC (ROC) Default RF:", auc_default, "\n")

### Tuned RF (Optimal: 500 trees, mtry=1)
set.seed(67)
rf_optimal <- randomForest(has_debit_card ~ ., data = data_train,
                           importance = TRUE, keep.forest = TRUE, keep.inbag = TRUE,
                           ntree = 500, mtry = 1)

cat("\n--- Tuned RF (500 trees, mtry=1) ---\n")
print(rf_optimal)   # OOB error

# Predictions (class + probability)
rf_pred_class_opt <- predict(rf_optimal, newdata = data_test, type = "response")
rf_pred_prob_opt  <- predict(rf_optimal, newdata = data_test, type = "prob")[,"Yes"]

# Confusion matrix with detailed stats
cm_opt <- confusionMatrix(rf_pred_class_opt, data_test$has_debit_card, positive="Yes")
print(cm_opt)

# Visualization confusion matrix on test data
cm_optimal <- as.data.frame(cm_opt$table)
  
ggplot(cm_optimal, aes(x = Prediction, y = Reference, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "white", size = 6, fontface = "bold") +
  scale_fill_gradient(low = "plum1", high = "plum4", name = "Freq") +
  coord_equal() +
  labs(
    title = "Confusion Matrix - Tuned (Test Set)",
    x = "Predicted Class",
    y = "Actual Class"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5)
  )
  
# Extra metrics
roc_obj_opt <- roc(response = data_test$has_debit_card, predictor = rf_pred_prob_opt,
                   levels = c("No","Yes"), direction = "<")
auc_opt <- auc(roc_obj_opt)
cat("AUC (ROC) Tuned RF:", auc_opt, "\n")

# end the higher perfomance setting (if you started it)
#stopCluster(cl)
#registerDoSEQ()


################################################################################
# Model Comparison Summary (with F1)
################################################################################

# Extract metrics for both models
extract_metrics <- function(cm, auc_val, model_name) {
  precision <- cm$byClass["Pos Pred Value"]
  recall    <- cm$byClass["Sensitivity"]
  f1        <- ifelse((precision + recall) == 0, 0, 2 * (precision * recall) / (precision + recall))
  
  data.frame(
    Model = model_name,
    Accuracy = cm$overall["Accuracy"],
    Kappa = cm$overall["Kappa"],
    Precision = precision,
    Recall = recall,
    F1 = f1,
    Specificity = cm$byClass["Specificity"],
    Balanced_Accuracy = cm$byClass["Balanced Accuracy"],
    AUC = auc_val
  )
}

metrics_default <- extract_metrics(cm_default, auc_default, "Default RF (500 trees, mtry=3)")
metrics_optimal <- extract_metrics(cm_opt, auc_opt, "Tuned RF (500 trees, mtry=1)")

comparison_table <- rbind(metrics_default, metrics_optimal) %>%
  mutate(across(where(is.numeric), round, 4))

comparison_table_select <- comparison_table %>%
  select(Accuracy, Balanced_Accuracy, AUC) %>%
  rename('Bal. Acc.' = Balanced_Accuracy) %>%
  mutate(Model = c("Default", "Tuned")) %>%
  gt(rowname_col = "Model")

comparison_table_select

cat("\n==================== MODEL COMPARISON (Extended) ====================\n")
print(comparison_table)
cat("=====================================================================\n")

# Visualization 
library(reshape2)
comparison_long <- melt(comparison_table, id.vars = "Model")

ggplot(comparison_long, aes(x = variable, y = value, fill = Model)) +
  geom_bar(stat = "identity", position = position_dodge(), width = 0.7) +
  scale_fill_manual(values = cols[c(2,4)]) +
  labs(
    title = "Random Forest Comparison: \nDefault vs Tuned",
    subtitle = "Including F1, Precision, Recall & AUC",
    x = "Metric",
    y = "Value",
    fill = "Model"
  ) +
  theme_minimal(base_size = 14) +
  coord_cartesian(ylim = c(0.4, 0.85)) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        plot.title = element_text(face = "bold"))


################################################################################
# Calculating predictive importance of predictors 
################################################################################

## Permutation importance

# For optimal Random Forest
set.seed(67)
per_imp <- importance(rf_optimal, type = 1, scale = TRUE)

per_imp <- as.data.frame(per_imp) %>%
  rownames_to_column(var ="Variable") %>%
  arrange(desc(MeanDecreaseAccuracy))
  
ggplot(per_imp, aes(x = MeanDecreaseAccuracy, y = reorder(Variable, MeanDecreaseAccuracy))) +
  geom_col(fill = "plum4") +
  labs(
    title = "Permutation Importance (Optimal RF)",
    subtitle = "Mean decrease in accuracy, scaled by se",
    x = "Permutation Importance",
    y = "Variable"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

# For default Random Forest (to check robustness)
per_imp_default <- importance(rf_default, type = 1, scale = TRUE)

per_imp_default <- as.data.frame(per_imp_default) %>%
  rownames_to_column(var ="Variable") %>%
  arrange(desc(MeanDecreaseAccuracy))

ggplot(per_imp_default, aes(x = MeanDecreaseAccuracy, y = reorder(Variable, MeanDecreaseAccuracy))) +
  geom_col(fill = "plum4") +
  labs(
    title = "Permutation Importance (Default RF)",
    subtitle = "Mean decrease in accuracy, scaled by se",
    x = "Permutation Importance",
    y = "Variable"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

## Impurity-based (Gini) importance

#For optimal RF
imp <- importance(rf_optimal, type = 2, scale = TRUE)

imp <- as.data.frame(imp) %>%
  rownames_to_column(var = "Variable") %>%
  arrange(desc(MeanDecreaseGini)) 

ggplot(imp, aes(x = MeanDecreaseGini, y = reorder(Variable, MeanDecreaseGini))) +
  geom_col(fill = "plum") +
  labs(
    title = "Impurity importance (Optimal RF)",
    subtitle = "Mean decrease in node impurity (Gini)",
    x = "Impurity Importance",
    y = "Variable"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

#For default RF (to check robustness)
imp_default <- importance(rf_default, type = 2, scale = TRUE)

imp_default <- as.data.frame(imp_default) %>%
  rownames_to_column(var = "Variable") %>%
  arrange(desc(MeanDecreaseGini)) 

ggplot(imp_default, aes(x = MeanDecreaseGini, y = reorder(Variable, MeanDecreaseGini))) +
  geom_col(fill = "plum") +
  labs(
    title = "Impurity importance (Default RF)",
    subtitle = "Mean decrease in node impurity (Gini)",
    x = "Impurity Importance",
    y = "Variable"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

#RF without rec_agri_payment and rec_gov_pension
set.seed(67)
data_train_select <- data_train %>%
  select(- rec_agri_payment, - rec_gov_pension)

data_test_select <- data_test %>%
  select(- rec_agri_payment, -rec_gov_pension)

rf_optimal_select <- randomForest(has_debit_card ~ ., data = data_train_select,
                           importance = TRUE, keep.forest = TRUE, keep.inbag = TRUE,
                           ntree = 750, mtry = 2)

cat("\n--- Tuned RF (500 trees, mtry=1) ---\n")
print(rf_optimal_select)   # OOB error

# Predictions (class + probability)
rf_pred_class_opt_select <- predict(rf_optimal_select, newdata = data_test_select, type = "response")
rf_pred_prob_opt_select  <- predict(rf_optimal_select, newdata = data_test_select, type = "prob")[,"Yes"]

# Confusion matrix with detailed stats
cm_opt_select <- confusionMatrix(rf_pred_class_opt_select, data_test_select$has_debit_card, positive="Yes")
print(cm_opt_select)

# Visualization confusion matrix on test data
cm_optimal_select <- as.data.frame(cm_opt_select$table)

ggplot(cm_optimal_select, aes(x = Prediction, y = Reference, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "white", size = 6, fontface = "bold") +
  scale_fill_gradient(low = "plum1", high = "plum4", name = "Freq") +
  coord_equal() +
  labs(
    title = "Confusion Matrix - Tuned (Selected Test Set)",
    subtitle = "Evaluated after SMOTE-NC balancing",
    x = "Predicted Class",
    y = "Actual Class"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5)
  )

################################################################################
# Apply SMOTE 
################################################################################

set.seed(67)

cat("\nBefore SMOTE:\n")
print(table(data_train$has_debit_card))

# Convert data to data.frame (themis works with tibbles or data.frames)
df_train <- as.data.frame(data_train)

# Run SMOTENC
data_train_bal <- smotenc(
  df = df_train,
  var = "has_debit_card",  # target variable (must be factor)
  k = 5,                   # number of neighbors
  over_ratio = 1          # balance classes 1:0.8 because if 1:1 -> overfit
)

cat("\nAfter SMOTE (themis::smotenc):\n")
print(table(data_train_bal$has_debit_card))

# Check structure
str(data_train_bal)

# Replace training dataset
data_train <- data_train_bal

#Visualization training distribution
train_dist_smote <- data_train_bal %>%
  count(has_debit_card, name = "Count") %>%
  mutate(Percent = Count/sum(Count)) %>%
  rename(Class = has_debit_card) %>%
  gt() %>%
  fmt_percent(columns = "Percent", decimals = 1) %>%
  cols_label() %>%
  tab_header(title = "Training Set Distribution (SMOTE-NC)")

train_dist_smote

set.seed(67)
rf_optimal_smote <- randomForest(has_debit_card ~ ., data = data_train,
                           importance = TRUE, keep.forest = TRUE, keep.inbag = TRUE,
                           ntree = 500, mtry = 6)

rf_pred_class_optimal_smote <- predict(rf_optimal_smote, newdata = data_test, type = "response")

cm_optimal_smote <- confusionMatrix(rf_pred_class_optimal_smote, data_test$has_debit_card, positive="Yes")
print(cm_optimal_smote)

# Visualization confusion matrix
cm_o_s <- as.data.frame(cm_optimal_smote$table)

ggplot(cm_o_s, aes(x = Prediction, y = Reference, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "white", size = 6, fontface = "bold") +
  scale_fill_gradient(low = "plum1", high = "plum4", name = "Freq") +
  coord_equal() +
  labs(
    title = "Confusion Matrix - Tuned (Test Set)",
    subtitle = "Evaluated after SMOTE-NC balancing",
    x = "Predicted Class",
    y = "Actual Class"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5)
  )





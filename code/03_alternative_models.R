################################################################################
# Task 3 - Alternative Models (Decision Tree & Neural Network)
# Goal: Predict has_debit_card (Yes/No) 
# Author: Jonah-B. Lohmann
# Random Seed: 67
# Runtime: ca. 45 seconds (excl. package installation)
################################################################################

################################################################################
# 0) SET-UP
################################################################################
cat("\n===== 0) Basic Set-Up =====\n")

# -------- 0.1) Load packages & set seed ----

cat("\n===== 0) Setup: Packages, Seed, Data, Manage conflics =====\n")

# Fixed CRAN mirror (avoid interactive prompt)
options(repos = c(CRAN = "https://cloud.r-project.org"))

# Needed packages
pkgs <- c(
  "tidyverse",     # data wrangling & ggplot
  "tidymodels",    # packages for modeling and machine learning
  "scales",        # scale functions for visualization 
  "caret",         # unified training/tuning
  "pROC",          # calculate AUC/ROC
  "rpart",         # decision tree engine used by caret
  "rpart.plot",    # visualizing decision trees by rpart
  "nnet",          # neural networks engine used by caret
  "NeuralNetTools" # visualizing neural networks by nnet
  )

# Install needed packages only if not installed already
to_install <- setdiff(pkgs, rownames(installed.packages()))
if (length(to_install)) install.packages(to_install, dependencies = TRUE, quiet = TRUE)

# Load core libraries
suppressPackageStartupMessages({
  library(tidyverse)
  library(tidymodels)
  library(scales)
  library(caret)
  library(pROC)
  library(rpart)
  library(rpart.plot)
  library(nnet)
  library(NeuralNetTools)
})

# Set seed
set.seed(67)


# -------- 0.2) Load data (already preprocessed elsewhere) --------
cat("\n===== 0.1) Load Data =====\n")

# Load pre-processed data
data_na <- readRDS("/Users/jonah-baptiste/Documents/2_Ausbildung/2_HSG_UdeSA/7_Sem_VWL/3_Machine Learning_Finance/0_3_Homework/ML-in-Finance/data/processed/df_final.rds")

# Keep only complete cases, to ensure same data used for Decision Tree & NN
# ANN can't handle NAs and the small number thereof justifies dropping them
data <- na.omit(data_na)

# Inspect data
glimpse(data) #Columns: 13 (10 fct, 2 ord, 1 int)
summary(data) #Rows: 2305
str(data) #no NA values

# For ML later: Store count of predictors
p <- ncol(train) - 1

# -------- 0.3) Train / Validation / Test split (70 / 15 / 15) --------
cat("\n===== 0.2) Create 70/15/15 split (stratified) =====\n")

# Creates a 70% stratified split by the target of having a debit card
set.seed(67)
init_split_obj <- initial_split(data, prop = 0.7, strata = has_debit_card)
train <- training(init_split_obj)
remain  <- testing(init_split_obj)

# Divide remaining 30% stratified split into validation/test (15% each)
set.seed(67)
val_test_split <- initial_split(remain, prop = 0.5, strata = has_debit_card)
validation <- training(val_test_split)
test       <- testing(val_test_split)

# Check proportion of data to ensure correctness of split
cat("\n===== Check class balance of split =====\n")
map(list(train=train, validation=validation, test=test),
    ~ count(.x, has_debit_card) %>% 
      mutate(proportion = n/sum(n))
    ) %>% print()
cat("\nNote: There is a small amount of data entries with has_debit_card = Yes
      Weaker performance in correctly identifying has_debit_card = Yes is expected\n")

# Seperate combination of training & validation data for retrain before testing
train_val <- bind_rows(train, validation)

# -------- 0.4) Define consistent helper functions for metrics and plots --------

#Helper function to plot confusion matrix
plot_cm_dt <- function(cm, title_txt) {
  cm_df <- as.data.frame(cm$table)
  colnames(cm_df) <- c("Predicted", "Actual", "Freq")
  ggplot(cm_df, aes(x = Predicted, y = Actual, fill = Freq)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Freq), color = "white", size = 6, fontface = "bold") +
    scale_fill_gradient(low = "steelblue1", high = "steelblue4", name = "Freq") +
    coord_equal() +
    labs(title = title_txt, x = "Predicted", y = "Actual") +
    theme_minimal(base_size = 14)
}

plot_cm_nn <- function(cm, title_txt) {
  cm_df <- as.data.frame(cm$table)
  colnames(cm_df) <- c("Predicted", "Actual", "Freq")
  ggplot(cm_df, aes(x = Predicted, y = Actual, fill = Freq)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Freq), color = "white", size = 6, fontface = "bold") +
    scale_fill_gradient(low = "olivedrab1", high = "olivedrab4", name = "Freq") +
    coord_equal() +
    labs(title = title_txt, x = "Predicted", y = "Actual") +
    theme_minimal(base_size = 14)
}

# TrainControl for using 10-fold CV when tuning to optimize ROC
ctrl_cv <- trainControl(
  method = "cv", number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

################################################################################
# 1) Task 3a: Decision Tree (with caret & rpart)
################################################################################
cat("\n===== 1) Decision Tree (with rpart) =====\n")

# ---- 1.1) DT DEFAULT: Train with default params; predict on VALIDATION ----
cat("\n===== 1.1) Train default DT (cp = 0.01) and score on VALIDATION =====\n")

# Train default DT (with CARET)
set.seed(67)
dt_default <- caret::train(
  has_debit_card ~ ., data = train,
  method = "rpart",
  trControl = trainControl(method = "none", classProbs = TRUE), #no resampling
  tuneGrid = data.frame(cp = 0.01) # default cp value
)

# Predict default DT on VALIDATION
dt_pred_val  <- predict(dt_default, 
                        newdata = validation)
# Return probabilities behind classification prediction
dt_prob_val  <- predict(dt_default, 
                        newdata = validation,
                        type = "prob")[,"Yes"] 

cat("\n===== Calculate metrics of the default DT (Validation) =====\n")

# DT default: Confusion Matrix
dt_cm_val_default <- confusionMatrix(dt_pred_val, 
                                     validation$has_debit_card, 
                                     positive = "Yes")
# DT default: Highlight Accuracy (from Confusion Matrix)
dt_acc_val_default <- as.numeric(dt_cm_val_default$overall
                                 ["Accuracy"])
# DT default: Highlight AUC (from prediction probabilities)
dt_auc_val_default <- as.numeric(pROC::auc(pROC::roc(response = validation$has_debit_card,
                                                     predictor = dt_prob_val,
                                                     levels = c("No","Yes"))))

cat("\n===== Extract statistics of the default DT (Validation) =====\n")
print(dt_cm_val_default)

cat("\n===== Extract focus metrics of the default DT (Validation) =====\n")
cat(sprintf("Accuracy (VAL): %.4f | ROC AUC (VAL): %.4f\n", dt_acc_val_default, dt_auc_val_default))

cat("\n===== Plot Confusion Matrix of the default DT (Validation) =====\n")
# Uses Helper: Confusion Matrix predefined in Section 0.4)
plot_cm_dt(dt_cm_val_default, "Confusion Matrix 
Model: Decision Tree (Default) 
Scored on: Validation")


# ---- 1.2) DT TUNE: CV on TRAIN optimizing ROC (with cp; pruning) ----
cat("\n===== 1.2) Tune & train DT and score on VALIDATION  =====\n")

# Train tuned DT (with CARET; pruning range cp = [0.00001;0.002])
set.seed(67)
dt_tuned <- caret::train(
  has_debit_card ~ ., data = train,
  method = "rpart",
  trControl = ctrl_cv, # Uses Helper: 10-fold CV predefined in Section 0.4)
  tuneGrid = expand.grid(cp = seq(0.00001, 0.002, length.out = 10)), #Pruning range to determine best cp
  metric = "ROC" #Best pruning range is determined by maximising ROC
)

cat("\n The hyperparameters for tuned DT, maximising ROC are: \n")
dt_tuned$bestTune %>% print()

# Predict tuned DT on VALIDATION
dt_pred_val_tuned <- predict(dt_tuned, 
                             newdata = validation)
# Return probabilities behind classification prediction
dt_prob_val_tuned  <- predict(dt_tuned, 
                        newdata = validation,
                        type = "prob")[,"Yes"] 

cat("\n===== Calculate metrics of the tuned DT (Validation) =====\n")

# DT tuned: Confusion Matrix
dt_cm_val_tuned <- confusionMatrix(dt_pred_val_tuned, 
                                     validation$has_debit_card, 
                                     positive = "Yes")
# DT tuned: Highlight Accuracy (from Confusion Matrix)
dt_acc_val_tuned <- as.numeric(dt_cm_val_tuned$overall
                               ["Accuracy"])
# DT tuned: Highlight AUC (from prediction probabilities)
dt_auc_val_tuned <- as.numeric(pROC::auc(pROC::roc(response = validation$has_debit_card,
                                                   predictor = dt_prob_val_tuned,
                                                   levels = c("No","Yes"))))

cat("\n===== Extract statistics of the default DT (Validation) =====\n")
print(dt_cm_val_tuned)

cat("\n===== Extract focus metrics of the default DT (Validation) =====\n")
cat(sprintf("Accuracy (VAL): %.4f | ROC AUC (VAL): %.4f\n", dt_acc_val_tuned, dt_auc_val_tuned))

cat("\n===== Plot Confusion Matrix of the default DT (Validation) =====\n")
# Uses Helper: Confusion Matrix predefined in Section 0.4)
plot_cm_dt(dt_cm_val_tuned, "Confusion Matrix 
Model: Decision Tree (Tuned) 
Scored on: Validation")

# ---- 1.3) DT LOCK-IN: Retrain tuned on TRAIN+VALIDATION, predict on TEST ----
cat("\n===== 1.3) Retrain optimal DT parameters and TEST =====\n")

# Train final DT with best hyperparameters on cobined training & validation set
set.seed(67)
dt_final <- caret::train(
  has_debit_card ~ ., data = train_val, #combined training + validation set 
  method = "rpart",
  trControl = trainControl(method = "none", classProbs = TRUE),
  tuneGrid = dt_tuned$bestTune  # Lock in best hyperparameter
)

# Predict final DT on TEST
dt_pred_test <- predict(dt_final, 
                        newdata = test)
# Return probabilities behind classification prediction
dt_prob_test <- predict(dt_final, 
                        newdata = test, 
                        type = "prob")[,"Yes"]

cat("\n===== Calculate metrics of the final DT (TEST) =====\n")

# DT final: Confusion Matrix
dt_cm_test   <- confusionMatrix(dt_pred_test, 
                                test$has_debit_card, 
                                positive = "Yes")
# DT final: Highlight Accuracy (from Confusion Matrix)
dt_acc_test  <- as.numeric(dt_cm_test$overall
                           ["Accuracy"])
# DT final: Highlight AUC (from prediction probabilities)
dt_auc_test  <- as.numeric(pROC::auc(pROC::roc(response = test$has_debit_card,
                                               predictor = dt_prob_test,
                                               levels = c("No","Yes"))))

cat("\n===== Extract statistics of the final DT (Test) =====\n")
print(dt_cm_test)

cat("\n===== Extract focus metrics of the final DT (Test) =====\n")
cat(sprintf("Accuracy (TEST): %.4f | ROC AUC (TEST): %.4f\n", dt_acc_test, dt_auc_test))

cat("\n===== Plot Confusion Matrix of the final DT (Test) =====\n")
plot_cm_dt(dt_cm_test, "Confusion Matrix
Model: Decision Tree (Final) 
Scored on: Test")

# Store final DT metrics for later comparison
metrics_dt_final <- c(Accuracy = dt_acc_test, ROC = dt_auc_test)


# ---- 1.4) DT INTERPRETABILITY: Visualize the final decision tree ----

# Variable Importance in decision tree
vi_dt <- caret::varImp(dt_final)$importance %>%
  tibble::rownames_to_column("Variable") %>%
  arrange(desc(Overall)) %>%
  slice_head(n = 12)   

# Plot Variable Importance in decision tree
cat("\n===== Plotting Variable Importance =====\n")
ggplot(vi_dt, aes(x = reorder(Variable, Overall), y = Overall)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(
    title = "Decision Tree — Top Variable Importance",
    x = "Variable",
    y = "Importance"
  ) +
  theme_minimal(base_size = 14)

# Plot the final tree structure - DT's USP
cat("\n===== Plotting the final Decision Tree structure =====\n")
rpart.plot(
  dt_final$finalModel,
  type = 2,                # split labels on branches; leaves on same level
  extra = 104,             # show fitted class, prob of class, and % of observations
  under = TRUE,            # put node numbers/extra info under the boxes
  fallen.leaves = TRUE,    # leaves at bottom for readability
  tweak = 1.5,             # slightly larger boxes
  branch.lty = 3,          # dashed branches for readability
  shadow.col = 0,          # no shadows
  main = "Decision Tree (Final): Structure & Node Visualisation"
)

################################################################################
# 2) Task 3b: Neural Network (with caret & nnet)
################################################################################
cat("\n===== 2) Neural Network (nnet) =====\n")

# ---- 2.1) NN DEFAULT: Train with default params; predict on VALIDATION ----
cat("\n===== 1.1) Train default NN (size=5, decay=0) and score on VALIDATION =====\n")

# Train default NN (with CARET & nnet)
set.seed(67)
nn_default <- caret::train(
  has_debit_card ~ ., data = train,
  method = "nnet", # Single-hidden-layer neural net, classification via softmax
  trControl = trainControl(method = "none", classProbs = TRUE),
  tuneGrid = data.frame(size = 5, decay = 0),  # simple default
  preProcess = c("center","scale"), # z-score transformation
  trace = FALSE,
  MaxNWts = 5000, # default is 1000; x5 safety margin because predictors are factors
  maxit = 200 # default is 100; doubled to give more room for conversion
)

# Predict default NN on VALIDATION
nn_pred_val <- predict(nn_default, 
                       newdata = validation)
# Return probabilities behind classification prediction
nn_prob_val <- predict(nn_default, 
                       newdata = validation, 
                       type = "prob")[,"Yes"]

cat("\n===== Calculate metrics of the default NN (Validation) =====\n")

# NN default: Confusion Matrix
nn_cm_val_default <- confusionMatrix(nn_pred_val, 
                                     validation$has_debit_card, 
                                     positive = "Yes")
# NN default: Highlight Accuracy (from Confusion Matrix)
nn_acc_val_default <- as.numeric(nn_cm_val_default$overall
                                 ["Accuracy"])
# NN default: Highlight AUC (from prediction probabilities)
nn_auc_val_default <- as.numeric(pROC::auc(pROC::roc(response = validation$has_debit_card,
                                                     predictor = nn_prob_val,
                                                     levels = c("No","Yes"))))

cat("\n===== Extract statistics of the default NN (Validation) =====\n")
print(nn_cm_val_default)

cat("\n===== Extract focus metrics of the default NN (Validation) =====\n")
cat(sprintf("Accuracy (VAL): %.4f | ROC AUC (VAL): %.4f\n", nn_acc_val_default, nn_auc_val_default))

cat("\n===== Plot Confusion Matrix of the default NN (Validation) =====\n")
plot_cm_nn(nn_cm_val_default, "Confusion Matrix
Model: Neural Net (Default)
Scored on: Validation")

# ---- 2.2) NN TUNE: CV on TRAIN optimizing ROC (with size x decay) ----
cat("\n===== 2.2) Tune & train NN and score on VALIDATION  =====\n")

# Tune size and decay with ranges (optimal range determined in manual iterations)
# Runtime: ca. 50 seconds
set.seed(67)
nn_tuned <- caret::train(
  has_debit_card ~ ., data = train,
  method = "nnet",
  trControl = ctrl_cv, # 10-fold CV defined in section 0.4)
  tuneGrid = expand.grid(
    size = 1:5, 
    decay = 10^seq(-2, -1, length.out = 10)), # log-spaced, wider than default
  preProcess = c("center","scale"),
  metric = "ROC", # Tuning optimises ROC
  trace = FALSE,
  MaxNWts = 5000,
  maxit = 200
)

cat("\n The hyperparameters for tuned NN, maximising ROC are: \n")
nn_tuned$bestTune %>% print()

# Predict tuned NN on VALIDATION
nn_pred_val_tuned <- predict(nn_tuned, 
                             newdata = validation)
# Return probabilities behind classification prediction
nn_prob_val_tuned <- predict(nn_tuned, 
                             newdata = validation, 
                             type = "prob")[,"Yes"]

cat("\n===== Calculate metrics of the tuned DT (Validation) =====\n")

# NN tuned: Confusion Matrix
nn_cm_val_tuned <- confusionMatrix(nn_pred_val_tuned, 
                                   validation$has_debit_card, 
                                   positive = "Yes")
# NN tuned: Highlight Accuracy (from Confusion Matrix)
nn_acc_val_tuned <- as.numeric(nn_cm_val_tuned$overall
                               ["Accuracy"])
# NN tuned: Highlight AUC (from prediction probabilities)
nn_auc_val_tuned <- as.numeric(pROC::auc(pROC::roc(response = validation$has_debit_card,
                                                   predictor = nn_prob_val_tuned,
                                                   levels = c("No","Yes"))))

cat("\n===== Extract statistics of the tuned NN (Validation) =====\n")
print(nn_cm_val_tuned)

cat("\n===== Extract focus metrics of the tuned NN (Validation) =====\n")
cat(sprintf("Accuracy (VAL): %.4f | ROC AUC (VAL): %.4f\n", nn_acc_val_tuned, nn_auc_val_tuned))

cat("\n===== Plot Confusion Matrix of the tuned NN (Validation) =====\n")
plot_cm_nn(nn_cm_val_tuned, "Confusion Matrix
Model: Neural Net (Tuned)
Scored on: Validation")

# ---- 2.3) NN LOCK-IN: Retrain tuned NN on TRAIN+VALIDATION, predict on TEST ----
cat("\n===== 2.3) Retrain optimal NN parameters and TEST =====\n")

# Train final NN with best hyperparameters on combined training & validation set
set.seed(67)
nn_final <- train(
  has_debit_card ~ ., data = train_val, # combination of TRAIN + VALUATION
  method = "nnet",
  trControl = trainControl(method = "none", classProbs = TRUE),
  tuneGrid = nn_tuned$bestTune, # use tuned parameters
  preProcess = c("center","scale"),
  trace = FALSE,
  MaxNWts = 5000,
  maxit = 200
)

# Predict final NN on TEST
nn_pred_test <- predict(nn_final, 
                        newdata = test)
# Return probabilities behind classification prediction
nn_prob_test <- predict(nn_final, 
                        newdata = test, 
                        type = "prob")[,"Yes"]

cat("\n===== Calculate metrics of the final NN (TEST) =====\n")

# NN final: Confusion Matrix
nn_cm_test   <- confusionMatrix(nn_pred_test, 
                                test$has_debit_card, 
                                positive = "Yes")
# NN final: Highlight Accuracy (from Confusion Matrix)
nn_acc_test  <- as.numeric(nn_cm_test$overall
                           ["Accuracy"])
# NN final: Highlight AUC (from prediction probabilities)
nn_auc_test  <- as.numeric(pROC::auc(pROC::roc(response = test$has_debit_card,
                                               predictor = nn_prob_test,
                                               levels = c("No","Yes"))))

cat("\n===== Extract statistics of the final NN (Test) =====\n")
print(nn_cm_test)

cat("\n===== Extract focus metrics of the final NN (Test) =====\n")
cat(sprintf("Accuracy (TEST): %.4f | ROC AUC (TEST): %.4f\n", nn_acc_test, nn_auc_test))

cat("\n===== Plot Confusion Matrix of the final NN (Test) =====\n")
plot_cm_nn(nn_cm_test, "Confusion Matrix
Model: Neural Net (Final)
Scored on: Test")

# Store final NN metrics for later comparison
metrics_nn_final <- c(Accuracy = nn_acc_test, ROC = nn_auc_test)


# ---- 2.4) NN INTERPRETABILITY: Visualize the final neural network ----

# Plot Neural Network architecture diagram (inputs → hidden → outputs)
cat("\n-- Plotting NN architecture (nnet) --\n")
plotnet(
  nn_final$finalModel,
  alpha      = 0.6,      # lighter edges
  circle_cex = 1.3,      # node size
  cex_val    = 0.8,      # smaller text labels
  pos_col    = "green3", # positive weights (green lines)
  neg_col    = "red3",   # negative weights (red lines)
  max_spread  = TRUE,    # spread input/output nodes vertically
  node_labs  = TRUE,     # show node labels
  var_labs   = TRUE      # show input labels
)

################################################################################
# 3) Model Comparison: Decision Tree vs. Random Forest vs. Neural Network
################################################################################
cat("\n===== 3) Model Comparison (TEST, final models) =====\n")


# ---- [Delete later] Placeholder for RF test metrics ----
# Command: To ensure compatibility with this script, follow this naming convention
# Final RF Confusion Matrix = rf_cm_test
# Final RF Accuracy = rf_acc_test
# Final RF Area under Curve (AUC) = rf_auc_test

# If you RF test metrics in this environment, this block will be skipped.
if (!exists("rf_acc_test")) {
  rf_acc_test <- as.numeric(NA)  # replace with numeric (e.g., 0.78)
}
if (!exists("rf_auc_test")) {
  rf_auc_test <- as.numeric(NA)  # replace with numeric (e.g., 0.83)
}
# If RF confusion matrix inexistent this will be set to NULL and secondary metrics for RF will be NA.
if (!exists("rf_cm_test")) {
  rf_cm_test <- NULL
}


# ---- 3.0) Extract secondary metrics from confusion matrices for plotting ----

# helper function to extract secondary metrics from caret::confusionMatrix()
extract_cm_metrics <- function(cm_obj) {
  if (is.null(cm_obj)) {
    return(tibble(
      Accuracy = NA_real_,         # overall correctness, easy to interpret
      Sensitivity = NA_real_,      # how well we find true positives
      Specificity = NA_real_,      # how well we avoid false positives
      B_Accuracy = NA_real_) # average of Sensitivity and Specificity
    )
  }
  tibble(
    Accuracy         = as.numeric(cm_obj$overall["Accuracy"]),
    Sensitivity      = as.numeric(cm_obj$byClass["Sensitivity"]),
    Specificity      = as.numeric(cm_obj$byClass["Specificity"]),
    B_Accuracy = as.numeric(cm_obj$byClass["Balanced Accuracy"])
  )
}

# Extract secondary metrics from confusion matrices for benchmarking later
dt_sec  <- extract_cm_metrics(dt_cm_test)
nn_sec  <- extract_cm_metrics(nn_cm_test)
rf_sec  <- extract_cm_metrics(rf_cm_test)   # NA until you provide rf_cm_test


# ---- 3.1) Pairwise comparisons: DT vs. RF / NN vs. RF / DT vs. NN ----

# DT vs. RF (primary metrics: Accuracy & AUC)
comp_dt_rf <- tibble(
  Metric       = c("Accuracy", "ROC"),
  DecisionTree = c(dt_acc_test, dt_auc_test),
  RandomForest = c(rf_acc_test, rf_auc_test)
)
cat("\n===== Decision Tree vs Random Forest (TEST) =====\n")
print(comp_dt_rf)

# NN vs. RF (primary metrics: Accuracy & AUC)
comp_nn_rf <- tibble(
  Metric     = c("Accuracy", "ROC"),
  NeuralNet  = c(nn_acc_test, nn_auc_test),
  RandomForest = c(rf_acc_test, rf_auc_test)
)
cat("\n===== Neural Net vs Random Forest (TEST) =====\n")
print(comp_nn_rf)

# DT vs. NN (primary metrics: Accuracy & AUC)
comp_dt_nn <- tibble(
  Metric     = c("Accuracy", "ROC"),
  DecisionTree = c(dt_acc_test, dt_auc_test),
  NeuralNet  = c(nn_acc_test, nn_auc_test)
)
cat("\n===== Decision Tree vs Neural Net (TEST) =====\n")
print(comp_dt_nn)


# ---- 3.2) Benchmarking: DT vs. RF vs. NN (all metrics) ----

# Build tidy frame with all metrics
all_metrics <- bind_rows(
  dt_sec  %>% mutate(Model = "Decision Tree", ROC = dt_auc_test),
  rf_sec  %>% mutate(Model = "Random Forest", ROC = rf_auc_test),
  nn_sec  %>% mutate(Model = "Neural Net",    ROC = nn_auc_test)
) %>%
  select(Model, 
         Accuracy, 
         ROC, 
         Sensitivity, 
         Specificity, 
         B_Accuracy) %>%
  pivot_longer(cols = -Model, 
               names_to = "Metric", 
               values_to = "Value")

# Convert to percentages for the chart labels if you like; we’ll keep 0–1 scale for y
# but show % in the axis labels via scale_y_continuous(labels = scales::percent_format()).

# Color scheme for barchart
model_colors <- c(
  "Decision Tree" = "steelblue1",
  "Random Forest" = "plum2",
  "Neural Net"    = "olivedrab2"
)

# Order metrics: emphasize ROC and Accuracy early
metric_order <- c("ROC", "Accuracy", "B_Accuracy", "Sensitivity", "Specificity")
all_metrics$Metric <- factor(all_metrics$Metric, levels = metric_order)

# Plot barchart for comparison 
# This will throw a warning, if RF is missing and it removes them from the chart
cat("\n===== Benchmark bar chart across models (TEST) =====\n")
ggplot(all_metrics, aes(x = Metric, y = Value, fill = Model)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.8) +
  scale_fill_manual(values = model_colors) +
  coord_cartesian(ylim = c(0.5, 1)) +  # just zooms; doesn’t cut data
  labs(
    title = "Benchmarking DT vs. RF vs. NN",
    subtitle = "Primary: ROC & Accuracy 
Secondary: Balanced Accuracy, Sensitivity, Specificity",
    x = "Metrics",
    y = "Score",
    fill = "Model"
  ) +
  theme_minimal(base_size = 14)

cat("\n===== 03_alternative_models.R is DONE =====\n")
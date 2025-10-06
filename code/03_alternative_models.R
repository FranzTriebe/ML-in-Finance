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
# -------- 0.1) Load packages & set seed ----
cat("\n 0.1) Load packages & set seed \n")

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

cat("\n===== Success: 0.1) Load packages & set seed =====\n")


# -------- 0.2) Load data (already preprocessed elsewhere) --------
cat("\n 0.2) Load data (already preprocessed elsewhere) \n")

# Load pre-processed data
data_na <- readRDS("/Users/jonah-baptiste/Documents/2_Ausbildung/2_HSG_UdeSA/7_Sem_VWL/3_Machine Learning_Finance/0_3_Homework/ML-in-Finance/data/processed/df_final.rds")

# Keep only complete cases, to ensure same data used for Decision Tree & NN
# ANN can't handle NAs and the small number thereof justifies dropping them
data <- na.omit(data_na)

# Inspect data - Columns: 13 (10 fct, 2 ord, 1 int), Rows: 2305, no NAs
cat("\n===== Data & predictor overview =====\n")
glimpse(data)
cat("\n===== Data summary =====\n")
summary(data)

# For later: Store count of predictors
p <- ncol(train) - 1

cat("\n===== 0.2) Sucess: Load Data =====\n")


# -------- 0.3) Data Splitting (70% train / 30% test) --------
cat("\n 0.3) Data Splitting (70% train / 30% test) \n")

# Creates a 70% / 30% stratified split by the target of having a debit card
set.seed(67)
init_split_obj <- initial_split(data, prop = 0.7, strata = has_debit_card)
train <- training(init_split_obj)
test  <- testing(init_split_obj)

# Check proportion of data to ensure correctness of split
cat("\n===== Check class balance of split =====\n")
map(list(train=train, test=test),
    ~ count(.x, has_debit_card) %>% 
      mutate(proportion = n/sum(n))
    ) %>% print()
#Note: There is a small amount of data entries with has_debit_card = Yes
#Weakness in identifying true positives (has_debit_card = Yes) is expected

cat("\n===== Success: Data Splitting (70% train / 30% test) =====\n")


# -------- 0.4) Define helper functions for plots & cross-validation --------
cat("\n 0.4) Define helper functions for plots & cross-validation \n")

#Helper functions to plot confusion matrix
# Decision Trees are blue
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
# Neural Networks are green 
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

# Predetermine 10-fold cross-validation while tuning
ctrl_cv <- trainControl(
  method = "cv", number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

cat("\n===== Success: 0.4) Define helper functions =====\n")

################################################################################
# 1) Task 3a: Decision Tree (with caret & rpart)
################################################################################
# ---- 1.1) DT DEFAULT: Train DT with default parameters (cp = 0.01) ----
cat("\n 1.1) DT DEFAULT: Train DT with default parameters (cp = 0.01) \n")

# Train default DT (with CARET's limited hyperparameters: cp = 0.01)
set.seed(67)
dt_default <- caret::train(
  has_debit_card ~ ., data = train,
  method = "rpart",
  trControl = trainControl(method = "none", #no tuning
                           classProbs = TRUE), # keep probabilities for metrics
  tuneGrid = data.frame(cp = 0.01) # default value
)

# Score default DT on TEST
dt_default_pred  <- predict(dt_default, 
                        newdata = test)
# DT Default: Return probabilities of classification for ROC calculation
dt_default_prob  <- predict(dt_default, 
                        newdata = test,
                        type = "prob")[,"Yes"] 

#Calculate the Metrics of DT default
# DT default: Confusion Matrix
dt_cm_default <- confusionMatrix(dt_default_pred, 
                                     test$has_debit_card, 
                                     positive = "Yes")
# DT default: Highlight Accuracy (from Confusion Matrix)
dt_acc_default <- as.numeric(dt_cm_default$overall
                                 ["Accuracy"])
# DT default: Highlight ROC AUC (from prediction probabilities)
dt_auc_default <- as.numeric(pROC::auc(pROC::roc(response = test$has_debit_card,
                                                     predictor = dt_default_prob,
                                                     levels = c("No","Yes"))))

cat("\n===== Extract statistics of the default DT =====\n")
print(dt_cm_default)

cat("\n===== Extract primary metrics of the default DT =====\n")
cat(sprintf("ROC AUC (DT Default): %.4f | Accuracy (DT Default): %.4f\n", dt_auc_default, dt_acc_default))

cat("\n===== Plot Confusion Matrix of the default DT =====\n")
# Uses Helper: Confusion Matrix predefined in Section 0.4)
plot_cm_dt(dt_cm_default, "Confusion Matrix 
Model: Decision Tree (Default)")

cat("\n==== Success: 1.1) DT DEFAULT: Train DT with default parameters (cp = 0.01) ====\n")


# ---- 1.2) DT TUNED: 10-fold CV, tuning cp to optimize for ROC ----
cat("\n 1.2) DT TUNED: 10-fold CV, tuning cp to optimize for ROC \n")

# Tune DT (pruning cp on linear tuning grid = [0.001;0.003])
set.seed(67)
dt_tuned <- caret::train(
  has_debit_card ~ ., data = train,
  method = "rpart",
  trControl = ctrl_cv, # Uses Helper: 10-fold CV predefined in Section 0.4)
  tuneGrid = expand.grid(cp = seq(0.001, 0.003, length.out = 11)),
  metric = "ROC" #Best pruning range is determined by maximizing ROC
)

cat("\n ===== Tuned DT: Best hyperparameter for optimizing ROC ===== \n")
dt_tuned$bestTune %>% print()

# Score tuned DT on TEST
dt_pred_tuned <- predict(dt_tuned, 
                             newdata = test)
# DT Tuned: Return probabilities of classification for ROC calculation
dt_prob_tuned  <- predict(dt_tuned, 
                        newdata = test,
                        type = "prob")[,"Yes"] 

#Calculate Metrics of DT Tuned
# DT tuned: Confusion Matrix
dt_cm_tuned <- confusionMatrix(dt_pred_tuned, 
                                     test$has_debit_card, 
                                     positive = "Yes")
# DT tuned: Highlight Accuracy (from Confusion Matrix)
dt_acc_tuned <- as.numeric(dt_cm_tuned$overall
                               ["Accuracy"])
# DT tuned: Highlight ROC AUC (from prediction probabilities)
dt_auc_tuned <- as.numeric(pROC::auc(pROC::roc(response = test$has_debit_card,
                                                   predictor = dt_prob_tuned,
                                                   levels = c("No","Yes"))))

cat("\n===== Extract statistics of the tuned DT =====\n")
print(dt_cm_tuned)

cat("\n===== Extract primary metrics of the tuned DT =====\n")
cat(sprintf("ROC AUC: %.4f | Accuracy: %.4f\n", dt_auc_tuned, dt_acc_tuned))

cat("\n===== Plot Confusion Matrix of the tuned DT =====\n")
# Uses Helper: Confusion Matrix predefined in Section 0.4)
plot_cm_dt(dt_cm_tuned, "Confusion Matrix 
Model: Decision Tree (Tuned)")

# Store final NN metrics for later comparison
metrics_dt_final <- c(ROC = dt_auc_tuned, Accuracy = dt_acc_tuned)

cat("\n===== Success: 1.2) DT TUNED: 10-fold CV, tuning cp to optimize for ROC =====\n")


# ---- 1.3) DT INTERPRETABILITY: Visualize the final decision tree ----
cat("\n 1.3) DT INTERPRETABILITY: Visualize the final decision tree \n")

# Variable Importance in decision tree
vi_dt <- caret::varImp(dt_tuned)$importance %>%
  tibble::rownames_to_column("Variable") %>%
  arrange(desc(Overall)) %>%
  slice_head(n = 12)   

cat("\n===== Plot Variable Importance in DT =====\n")
# Plot Variable Importance in decision tree
ggplot(vi_dt, aes(x = reorder(Variable, Overall), y = Overall)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(
    title = "Decision Tree (tuned): Variable Importance",
    x = "Variable",
    y = "Importance"
  ) +
  theme_minimal(base_size = 14)

cat("\n===== Plot the DT structure =====\n")
# Plot the tree structure - DT's USP
rpart.plot(dt_tuned$finalModel,
  type = 2,                # split labels on branches; leaves on same level
  under = TRUE,            # put node numbers/extra info under the boxes
  fallen.leaves = TRUE,    # leaves at bottom for readability
  cex = 0.4,               # node text (affects p(Yes)/%Obs)
  split.cex = 0.9,           # smaller split labels to avoid collisions
  tweak = 1.1,            # larger boxes/spacing without exploding layout
  faclen = 12,             # abbreviate long variable names
  branch.lty = 2,          # dashed branches for readability
  shadow.col = 0,          # no shadows
  main = "Decision Tree (Tuned): Structure & Node Visualisation"
)

cat("\n===== Success: 1.3) DT INTERPRETABILITY: Visualize the final decision tree =====\n")

################################################################################
# 2) Task 3b: Neural Network (with caret & nnet)
################################################################################
# ---- 2.1) NN DEFAULT: Train with default parameters (size=5, decay=0) ----
cat("\n 2.1) NN DEFAULT: Train with default parameters (size=5, decay=0) \n")

# Train default NN (with CARET & nnet; size=5, decay=0)
# Single-hidden-layer neural net, classification via softmax
set.seed(67)
nn_default <- caret::train(
  has_debit_card ~ ., data = train,
  method = "nnet", 
  trControl = trainControl(method = "none", classProbs = TRUE), 
  tuneGrid = data.frame(size = 5, decay = 0),
  preProcess = c("center","scale"), # z-score transformation
  trace = FALSE,
  MaxNWts = 5000, # default is 1000; x5 safety margin because predictors are factors
  maxit = 200 # default is 100; doubled to give more room for conversion
)

# Predict default NN on VALIDATION
nn_default_pred <- predict(nn_default, 
                       newdata = test)
# Return probabilities behind classification prediction
nn_default_prob <- predict(nn_default, 
                       newdata = test, 
                       type = "prob")[,"Yes"]

# Calculate the Metrics of NN default
# NN default: Confusion Matrix
nn_cm_default <- confusionMatrix(nn_default_pred, 
                                     test$has_debit_card, 
                                     positive = "Yes")
# NN default: Highlight Accuracy (from Confusion Matrix)
nn_acc_default <- as.numeric(nn_cm_default$overall
                                 ["Accuracy"])
# NN default: Highlight ROC AUC (from prediction probabilities)
nn_auc_default <- as.numeric(pROC::auc(pROC::roc(response = test$has_debit_card,
                                                     predictor = nn_default_prob,
                                                     levels = c("No","Yes"))))

cat("\n===== Extract statistics of the default NN =====\n")
print(nn_cm_default)

cat("\n===== Extract primary metrics of the default NN =====\n")
cat(sprintf("ROC AUC: %.4f | Accuracy: %.4f\n", nn_auc_default, nn_acc_default))

cat("\n===== Plot Confusion Matrix of the default NN (Validation) =====\n")
plot_cm_nn(nn_cm_default, "Confusion Matrix
Model: Neural Net (Default)")

cat("\n===== Success: 2.1) NN DEFAULT: Train with default parameters (size=5, decay=0) =====\n")


# ---- 2.2) NN TUNED: 10-fold CV, tuning size & decay to optimize for ROC ----
cat("\n 2.2) NN TUNED: 10-fold CV, tuning size & decay to optimize for ROC \n")

# Tune size and decay with ranges (optimal range determined in manual iterations)
# Runtime: ca. 50 seconds
set.seed(67)
nn_tuned <- caret::train(
  has_debit_card ~ ., data = train,
  method = "nnet",
  trControl = ctrl_cv, # 10-fold CV defined in section 0.4)
  tuneGrid = expand.grid(
    size = 1:5, 
    decay = 10^seq(-3, -1, length.out = 10)), # log-spaced, wider than default
  preProcess = c("center","scale"),
  metric = "ROC", # Tuning optimises ROC
  trace = FALSE,
  MaxNWts = 5000,
  maxit = 200
)

cat("\n ===== NN Tuned: Best hyperparameter for optimizing ROC ===== \n")
nn_tuned$bestTune %>% print()

# Score tuned NN on TEST
nn_pred_tuned <- predict(nn_tuned, 
                             newdata = test)
# NN Tuned: Return probabilities of classification for ROC calculation
nn_prob_tuned <- predict(nn_tuned, 
                             newdata = test, 
                             type = "prob")[,"Yes"]

# #Calculate Metrics of NN Tuned
# NN tuned: Confusion Matrix
nn_cm_tuned <- confusionMatrix(nn_pred_tuned, 
                                   test$has_debit_card, 
                                   positive = "Yes")
# NN tuned: Highlight Accuracy (from Confusion Matrix)
nn_acc_tuned <- as.numeric(nn_cm_tuned$overall
                               ["Accuracy"])
# NN tuned: Highlight ROC AUC (from prediction probabilities)
nn_auc_tuned <- as.numeric(pROC::auc(pROC::roc(response = test$has_debit_card,
                                                   predictor = nn_prob_tuned,
                                                   levels = c("No","Yes"))))

cat("\n===== Extract statistics of tuned NN =====\n")
print(nn_cm_tuned)

cat("\n===== Extract primary metrics of tuned NN =====\n")
cat(sprintf("ROC AUC: %.4f | Accuracy: %.4f\n", nn_auc_tuned, nn_acc_tuned))

cat("\n===== Plot Confusion Matrix of tuned NN =====\n")
plot_cm_nn(nn_cm_tuned, "Confusion Matrix
Model: Neural Net (Tuned)")

# Store final NN metrics for later comparison
metrics_nn_final <- c(ROC = nn_auc_tuned, Accuracy = nn_acc_tuned)

cat("\n===== Success: 2.2) NN TUNED: 10-fold CV, tuning size & decay to optimize for ROC =====\n")


# ---- 2.3) NN INTERPRETABILITY: Visualize the final neural network ----
cat("\n 2.3) NN INTERPRETABILITY: Visualize the final neural network \n")

# Plot Neural Network architecture diagram (inputs → hidden → outputs)
cat("\n==== Plot NN architecture ====\n")
plotnet(
  nn_tuned$finalModel,
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
# ---- [Delete later] Placeholder for RF test metrics ----
# Command: To ensure compatibility with this script, follow this naming convention
# Final RF Confusion Matrix = rf_cm_tuned
# Final RF Accuracy = rf_acc_test
# Final RF Area under Curve (AUC) = rf_auc_test

# If you RF test metrics in this environment, this block will be skipped.
if (!exists("rf_acc_tuned")) {
  rf_acc_tuned <- as.numeric(NA)  # replace with numeric (e.g., 0.78)
}
if (!exists("rf_auc_tuned")) {
  rf_auc_tuned <- as.numeric(NA)  # replace with numeric (e.g., 0.83)
}
# If RF confusion matrix inexistent this will be set to NULL and secondary metrics for RF will be NA.
if (!exists("rf_cm_tuned")) {
  rf_cm_tuned <- NULL
}


# ---- 3.1) Extract secondary metrics from confusion matrices for plotting ----

# helper function to extract secondary metrics from caret::confusionMatrix()
extract_cm_metrics <- function(cm_obj) {
  if (is.null(cm_obj)) {
    return(tibble(
      Accuracy = NA_real_,         # overall correctness, easy to interpret
      Sensitivity = NA_real_,      # how well we find true positives
      Specificity = NA_real_,      # how well we avoid false positives
      B_Accuracy = NA_real_)       # average of Sensitivity and Specificity
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
dt_sec  <- extract_cm_metrics(dt_cm_tuned)
nn_sec  <- extract_cm_metrics(nn_cm_tuned)
rf_sec  <- extract_cm_metrics(rf_cm_tuned)   # NA until you provide rf_cm_tuned


# ---- 3.2) Pairwise comparisons: DT vs. RF / NN vs. RF / DT vs. NN ----

# DT vs. RF (primary metrics: ROC & Accuracy)
comp_dt_rf <- tibble(
  Metric       = c("ROC", "Accuracy"),
  DecisionTree = c(dt_auc_tuned, dt_acc_tuned),
  RandomForest = c(rf_auc_tuned, rf_acc_tuned)
)
cat("\n===== Decision Tree vs Random Forest (both tuned for ROC) =====\n")
print(comp_dt_rf)

# NN vs. RF (primary metrics: ROC & Accuracy)
comp_nn_rf <- tibble(
  Metric     = c("ROC", "Accuracy"),
  NeuralNet  = c(nn_auc_tuned, nn_acc_tuned),
  RandomForest = c(rf_auc_tuned, rf_acc_tuned)
)
cat("\n===== Neural Net vs Random Forest (both tuned for ROC) =====\n")
print(comp_nn_rf)

# DT vs. NN (primary metrics: ROC & Accuracy)
comp_dt_nn <- tibble(
  Metric     = c("ROC", "Accuracy"),
  DecisionTree = c(dt_auc_tuned, dt_acc_tuned),
  NeuralNet  = c(nn_auc_tuned, nn_acc_tuned)
)
cat("\n===== Decision Tree vs Neural Net (both tuned for ROC) =====\n")
print(comp_dt_nn)

cat("\n===== Success: 3.2) Pairwise comparisons: DT vs. RF / NN vs. RF / DT vs. NN =====\n")


# ---- 3.3) Benchmarking: DT vs. RF vs. NN (all metrics) ----

# Build tidy frame with all metrics
all_metrics <- bind_rows(
  dt_sec  %>% mutate(Model = "Decision Tree", ROC = dt_auc_tuned),
  rf_sec  %>% mutate(Model = "Random Forest", ROC = rf_auc_tuned),
  nn_sec  %>% mutate(Model = "Neural Net",    ROC = nn_auc_tuned)
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

cat("\n===== Plot Model Benchmarking =====\n")
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

cat("\n===== Success: 3.3) Benchmarking: DT vs. RF vs. NN (all metrics) =====\n")

cat("\n===== 03_alternative_models.R is DONE =====\n")
################################################################################
# Task 3 - Alternative Models (Decision Tree & Neural Network)
# Goal: Predict has_debit_card (Yes/No) 
# Author: Jonah. |  Seed: 67
################################################################################

################################################################################
# 0) SET-UP
################################################################################

cat("\n===== 0) Setup: Packages, Seed, Data, Manage conflics =====\n")

# Fixed CRAN mirror (avoid interactive prompt)
options(repos = c(CRAN = "https://cloud.r-project.org"))

# Needed packages
pkgs <- c(
  "tidyverse",   # data wrangling & ggplot
  "tidymodels",  # packages for modeling and machine learning
  "caret",       # unified training/tuning
  "pROC",        # calculate AUC/ROC
  "rpart",       # decision tree engine used by caret
  "nnet"         # neural networks engine used by caret
  )

# Install needed packages only if not installed already
to_install <- setdiff(pkgs, rownames(installed.packages()))
if (length(to_install)) install.packages(to_install, dependencies = TRUE, quiet = TRUE)

# Load core libraries
suppressPackageStartupMessages({
  library(tidyverse)
  library(tidymodels)
  library(caret)
  library(pROC)
  library(rpart)
  library(nnet)
})

# Set seed
set.seed(67)


# -------- 0.1) Load data (already preprocessed elsewhere) --------
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

# -------- 0.2) Train / Validation / Test split (70 / 15 / 15) --------
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


# -------- 0.3) Define consistent helper functions for metrics and plots --------

#Helper function to plot confusion matrix
plot_cm <- function(cm, title_txt) {
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

# TrainControl for using 5-fold CV when tuning to optimize ROC
ctrl_cv <- trainControl(
  method = "cv", number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

################################################################################
# 1) Task 3a: Decision Tree (with caret & rpart)
################################################################################
cat("\n===== 1) Decision Tree (with rpart) =====\n")

# ---- 1.1) Train DT with default params on TRAIN and predict on VALIDATION
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

# Confusion Matrix
dt_cm_val_default <- confusionMatrix(dt_pred_val, 
                                     validation$has_debit_card, 
                                     positive = "Yes")

# Highlight Accuracy (from Confusion Matrix)
dt_acc_val_default <- as.numeric(dt_cm_val_default$overall
                                 ["Accuracy"])

# Highlight AUC (from prediction probabilities)
dt_auc_val_default <- as.numeric(pROC::auc(pROC::roc(response = validation$has_debit_card,
                                                     predictor = dt_prob_val,
                                                     levels = c("No","Yes"))))

cat("\n===== Extract statistics of the default DT (Validation) =====\n")
print(dt_cm_val_default)

cat("\n===== Extract focus metrics of the default DT (Validation) =====\n")
cat(sprintf("Accuracy (VAL): %.4f | ROC AUC (VAL): %.4f\n", dt_acc_val_default, dt_auc_val_default))

cat("\n===== Plot Confusion Matrix of the default DT (Validation) =====\n")
# Uses Helper: Confusion Matrix predefined in Section 0.3)
plot_cm(dt_cm_val_default, "Confusion Matrix 
Model: Decision Tree (Default) 
Scored on: Validation")


# ---- 1.2) Train DT with tuned parameters on TRAIN and predict on VALIDATION ----
cat("\n===== 1.2) Tune & train DT and score on VALIDATION  =====\n")

# Train tuned DT (with CARET; pruning range cp = [0.00001;0.002])
set.seed(67)
dt_tuned <- caret::train(
  has_debit_card ~ ., data = train,
  method = "rpart",
  trControl = ctrl_cv, # Uses Helper: 5-fold CV predefined in Section 0.3)
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

# Confusion Matrix
dt_cm_val_tuned <- confusionMatrix(dt_pred_val_tuned, 
                                     validation$has_debit_card, 
                                     positive = "Yes")

# Highlight Accuracy (from Confusion Matrix)
dt_acc_val_tuned <- as.numeric(dt_cm_val_tuned$overall
                               ["Accuracy"])

# Highlight AUC (from prediction probabilities)
dt_auc_val_tuned <- as.numeric(pROC::auc(pROC::roc(response = validation$has_debit_card,
                                                   predictor = dt_prob_val_tuned,
                                                   levels = c("No","Yes"))))

cat("\n===== Extract statistics of the default DT (Validation) =====\n")
print(dt_cm_val_tuned)

cat("\n===== Extract focus metrics of the default DT (Validation) =====\n")
cat(sprintf("Accuracy (VAL): %.4f | ROC AUC (VAL): %.4f\n", dt_acc_val_tuned, dt_auc_val_tuned))

cat("\n===== Plot Confusion Matrix of the default DT (Validation) =====\n")
# Uses Helper: Confusion Matrix predefined in Section 0.3)
plot_cm(dt_cm_val_tuned, "Confusion Matrix 
Model: Decision Tree (Tuned) 
Scored on: Validation")

# ---- 1.3) LOCK-IN: retrain on TRAIN+VALIDATION with best cp, test on TEST ----
cat("\n===== 1.3) Retrain optimal parameters and TEST  =====\n")

# Combine training & validation set to prepare retrain before test
train_val <- bind_rows(train, validation)

# Train final DT with best hyperparameters; no CV because that was used for tuning
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

# Confusion Matrix
dt_cm_test   <- confusionMatrix(dt_pred_test, test$has_debit_card, positive = "Yes")

# Highlight Accuracy (from Confusion Matrix)
dt_acc_test  <- as.numeric(dt_cm_test$overall["Accuracy"])

# Highlight AUC (from prediction probabilities)
dt_auc_test  <- as.numeric(pROC::auc(pROC::roc(response = test$has_debit_card,
                                               predictor = dt_prob_test,
                                               levels = c("No","Yes"))))

cat("\n===== Extract statistics of the final DT (Test) =====\n")
print(dt_cm_test)

cat("\n===== Extract focus metrics of the final DT (Test) =====\n")
cat(sprintf("Accuracy (TEST): %.4f | ROC AUC (TEST): %.4f\n", dt_acc_test, dt_auc_test))

cat("\n===== Plot Confusion Matrix of the final DT (Test) =====\n")
plot_cm(dt_cm_test, "Confusion Matrix
Model: Decision Tree (Final) 
Scored on: Test")

# Store final DT metrics for later comparison
metrics_dt_final <- c(Accuracy = dt_acc_test, ROC = dt_auc_test)


################################################################################
# 2) Task 3b: Neural Network (with caret & nnet)
################################################################################



# Old script
--------------------------------------------------------------------------------
# 1.1) Create recipe for decision tree model

rec_tree <- recipe(has_debit_card ~ ., data = train) %>%
  step_zv(all_predictors())


# 1.2) Specify the decision tree model with tunable parameters

tree_spec <- decision_tree(
  mode = "classification",
  cost_complexity = tune(),  # tune for optimal pruning strength
  tree_depth     = tune(),   # tune for optimal depth
  min_n          = tune()    # tune for optimal min observations per leaf
) %>%
  set_engine("rpart")


# 1.3) Create workflow that links recipie with decision tree model

wf_tree <- workflow() %>%
  add_model(tree_spec) %>%
  add_recipe(rec_tree)


# 1.4) Define a grid with the range of hyper parameters that will be assessed

tree_grid <- grid_regular(
  cost_complexity(range = c(-4, -1)), # pruning penalty: 10^-4 to 10^-1
  tree_depth(range = c(2L, 20L)), # max splits from root to leaf: 2-20
  min_n(range = c(5L, 30L)), # min. observations per leaf: 5-30
  levels = 4 # create 4 evenly spaced values for each parameter within its range
)


# 1.5) Grid search (fit on train, score on validation) & collect results in one table
# Grid Search has shown to yield better results for roc_auc than cross validation

set.seed(67)
tree_results <- map_dfr(1:nrow(tree_grid), function(i){
  params <- tree_grid[i,]
  #finalize_workflow() will replace tune() in model 
  fitted <- finalize_workflow(wf_tree, params) %>% fit(train)
  #applies the helper eval function to score the model on the validation set
  mets   <- eval_cls(fitted, validation) %>% mutate(.config = i)
  #combines parameter values and the evaluation metrics into one row.
  bind_cols(params, mets)
})


# 1.6) Choose best hyperparameters config by ROC AUC (primary metric) & train

# Find the best hyper parameter configuration by ROC AUC
best_row_tree <- tree_results %>%
  filter(.metric == "roc_auc") %>%
  arrange(desc(.estimate)) %>%
  slice(1)

# Retrieve the best hyper parameters
best_params_tree <- tree_grid[best_row_tree$.config, , drop = FALSE]

# Build & fit the final model on "train" with best hyper parameters
best_tree <- finalize_workflow(wf_tree, best_params_tree) %>% fit(train)


# 1.7) Evaluate the best hyper parameters for the report

# Look up the best hyper parameters
best_params_tree 
# === Results: cost_complexity = 10^-4, tree-depth = 8, min_n = 21 ===
# Tuning successful because hyper parameters do not lie at the edge of grid
# Exception: cost_complexity, however didn't yield better results at 10^-5

# Look up the evaluation metrics of the best hyperparameters
val_metrics_tree <- eval_cls(best_tree, validation) %>% mutate(model = "Decision Tree")
val_metrics_tree 
# === Results: roc_auc = 0.735, outperforming other metrics ===

# Plot the confusion matrix on the validation set
cm_val <- best_tree %>%
  augment(new_data = validation) %>%
  conf_mat(truth = has_debit_card, estimate = .pred_class)
plot_confusion_matrix(cm_val, "Confusion Matrix — Decision Tree on Validation")

################################################################################
# 2) Task 3b: Neural Network: tune & eval
################################################################################

# 2.1) Create recipie for Artificial Neural Network (ANN; numeric matrix needed)

rec_ann <- recipe(has_debit_card ~ ., data = train) %>%
  step_zv(all_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%        # one-hot
  step_normalize(all_numeric_predictors())        # scale

prep_ann <- prep(rec_ann)
train_nn <- bake(prep_ann, new_data = train)
val_nn   <- bake(prep_ann, new_data = validation)

# Convert target to numeric 0/1 column named 'y' (neuralnet expects numeric)
train_nn <- train_nn %>% 
  mutate(y = as.integer(has_debit_card == "Yes")) %>% 
  select(-has_debit_card)
val_nn <- val_nn %>% 
  mutate(y = as.integer(has_debit_card == "Yes")) %>% 
  select(-has_debit_card)

# Build formula: y ~ x1 + x2 + ...
nn_formula <- as.formula(
  paste("y ~", paste(setdiff(names(train_nn),"y"), collapse = " + ")))


# 2.2) Build tuning grid
# 1) Proper grid (hidden as list-column)
nn_grid <- tibble::tibble(
  hidden    = list(c(16), c(32), c(32,16), c(64,32)),
  act       = c("logistic","logistic","logistic","logistic"),
  stepmax   = c(2e5, 4e5, 4e5, 6e5),
  threshold = c(0.05, 0.03, 0.03, 0.03)
) %>% 
  dplyr::mutate(.config = dplyr::row_number())

# Helper: fit one model and score VALIDATION
fit_eval_nn <- function(hidden, act, stepmax, threshold) {
  stopifnot(is.numeric(stepmax), length(stepmax) == 1, !is.na(stepmax))
  stopifnot(is.character(act), length(act) == 1)
  stopifnot(is.numeric(hidden), length(hidden) >= 1)
  
  set.seed(67)
  nn <- neuralnet(
    formula        = nn_formula,
    data           = train_nn,
    hidden         = hidden,
    act.fct        = act,
    linear.output  = FALSE,
    lifesign       = "minimal",
    stepmax        = stepmax,
    threshold      = threshold
  )
  
  val_probs <- neuralnet::compute(nn, val_nn |> dplyr::select(-y))$net.result[,1]
  val_cls   <- factor(ifelse(val_probs > 0.5, "Yes", "No"), levels = c("No","Yes"))
  
  out <- tibble::tibble(
    has_debit_card = factor(ifelse(val_nn$y == 1, "Yes", "No"), levels = c("No","Yes")),
    .pred_class    = val_cls,
    .pred_Yes      = val_probs
  )
  list(
    model   = nn,
    metrics = metric_set_cls(out, truth = has_debit_card, estimate = .pred_class, .pred_Yes = .pred_Yes)
  )
}


# 2.3) Grid search on VALIDATION (Runtime: ca. 1 minute)
set.seed(67)
nn_results <- purrr::pmap_dfr(
  nn_grid,
  function(hidden, act, stepmax, threshold, .config) {
    res <- fit_eval_nn(hidden, act, stepmax, threshold)
    res$metrics %>%
      dplyr::mutate(
        .config    = .config,
        hidden_str = paste(hidden, collapse = "-"),
        act        = act,
        stepmax    = stepmax,
        threshold  = threshold
      )
  }
)


# 2.4) Pick the best hyper parameter by accuracy

best_cfg_id <- nn_results %>%
  dplyr::filter(.metric == "accuracy") %>%
  dplyr::arrange(dplyr::desc(.estimate)) %>%
  dplyr::slice(1) %>%
  dplyr::pull(.config)

best_params_nn <- nn_grid %>% dplyr::filter(.config == best_cfg_id)

# 2.5) Refit the best ANN on "train" and keep validation metrics for the report

best_fit_val <- fit_eval_nn(
  hidden    = best_params_nn$hidden[[1]],
  act       = best_params_nn$act,
  stepmax   = best_params_nn$stepmax,
  threshold = best_params_nn$threshold
)
val_metrics_ann <- best_fit_val$metrics %>% dplyr::mutate(model = "ANN (neuralnet)")
val_metrics_ann

# Define val-probs using the best fitted model from fit_eval_nn()
best_nn <- best_fit_val$model  # extract the trained neuralnet object

val_probs <- neuralnet::compute(best_nn, val_nn %>% select(-y))$net.result[, 1]
val_cls   <- factor(ifelse(val_probs > 0.5, "Yes", "No"), levels = c("No", "Yes"))
val_truth <- factor(ifelse(val_nn$y == 1, "Yes", "No"), levels = c("No", "Yes"))

# Compute and plot confusion matrix
cm_ann_val <- conf_mat(
  tibble(has_debit_card = val_truth, .pred_class = val_cls),
  truth    = has_debit_card,
  estimate = .pred_class
)

plot_confusion_matrix(cm_ann_val, title = "Confusion Matrix - ANN on Validation")

################################################################################
# Task 3 - Alternative Models (Decision Tree vs ANN, Keras)
# Goal: Predict has_debit_card (Yes/No) 
# Author: Jonah    Seed: 67
################################################################################

################################################################################
# 0.1) Set-Up: Load packages, manage conflicts & set seed
################################################################################

# Use a fixed CRAN mirror to prevent interactive prompt when running script
options(repos = c(CRAN = "https://cloud.r-project.org"))

# Needed Packages
pkgs <- c(
  "tidyverse", 
  "tidymodels", 
  "rpart", 
  "keras")

# Install only missing packages (quietly)
to_install <- setdiff(pkgs, rownames(installed.packages()))
if (length(to_install)) install.packages(to_install, dependencies = TRUE, quiet = TRUE)

# Load core libraries
suppressPackageStartupMessages({
  library(tidyverse)
  library(tidymodels)
  library(rpart)
})

# Load Keras **only if** it’s installed AND TensorFlow is available
USE_ANN <- requireNamespace("keras", quietly = TRUE) && keras::is_keras_available()
if (USE_ANN) {
  library(keras)  # ANN sections ready
} else {
  message("Keras/TensorFlow not available: ANN sections will be skipped.\n",
          "Run once: keras::install_keras()  # then re-run the script.")
}

# Prefer tidymodels versions of masked functions
tidymodels_prefer()

# Set seed
set.seed(67)

################################################################################
# 0.2) Set-Up: Data Loading & Inspection
################################################################################

# Load pre-processed data
data <- readRDS("/Users/jonah-baptiste/Documents/2_Ausbildung/2_HSG_UdeSA/7_Sem_VWL/3_Machine Learning_Finance/0_3_Homework/ML-in-Finance/data/processed/df_final.rds")

# Keep only complete cases, to ensure same data used for Decision Tree & ANN
# ANN can't handle NAs and the small number thereof justifies dropping them
data_clean <- na.omit(data)

# Inspect data
glimpse(data_clean) #Columns: 13 (Kind: 10 fct, 2 ord, 1 int)
summary(data_clean) #Rows: 2305
str(data_clean) #no NA values

################################################################################
# 0.3) Set-Up: Train/Validation/Test split (70/15/15)
################################################################################

# Creates a 70% stratified split by the target of having a debit card
set.seed(67)
init_split_obj <- initial_split(data_clean, prop = 0.7, strata = has_debit_card)
train <- training(init_split_obj) # "train" is now training data
rest  <- testing(init_split_obj)

# Divide the remaining 30% stratified split into validation/test (15% each)
set.seed(67)
val_test_split <- initial_split(rest, prop = 0.5, strata = has_debit_card)
validation <- training(val_test_split) # "validation" is now validation data
test       <- testing(val_test_split) # "test" is now testing data

# Check proportion of data to ensure correctness of split (success)
map(list(train=train, validation=validation, test=test),
    ~ count(.x, has_debit_card) %>% 
      mutate(proportion = n/sum(n))
    )

################################################################################
# 0.4) Set-Up: Define helper utility - Computing metrics
################################################################################

#Define helper function to return metrics to compare classification models
metric_set_cls <- metric_set(accuracy, precision, recall, f_meas) #no roc_auc

# Define helper function to evaluate any fitted workflow on a dataset with metrics
eval_cls <- function(fitted_wf, new_data) {
  probs   <- predict(fitted_wf, new_data, type = "prob")
  classes <- predict(fitted_wf, new_data, type = "class")
  
  out <- bind_cols(
    new_data %>% select(has_debit_card),
    classes,       # already has .pred_class
    probs          # has .pred_No, .pred_Yes
    )
  
  m_cls <- metric_set_cls(
    out,
    truth    = has_debit_card,
    estimate = .pred_class,
    event_level = "second"   # if levels are c("No","Yes")
    )
  
  m_auc <- roc_auc(
    out,
    truth = has_debit_card,
    .pred_Yes,
    event_level = "second"
  )
  
  bind_rows(m_cls, m_auc)
}

# Function to plot confusion matrix as a heatmap
plot_confusion_matrix <- function(cm_obj, title = "Confusion Matrix") {
  cm_data <- as.data.frame(cm_obj$table)
  colnames(cm_data) <- c("Prediction", "Reference", "Freq")
  
  ggplot(cm_data, aes(x = Prediction, y = Reference, fill = Freq)) +
    geom_tile() +
    geom_text(aes(label = Freq), color = "white", size = 5) +
    scale_fill_gradient(low = "steelblue1", high = "steelblue4") +
    labs(title = title, x = "Predicted", y = "Actual") +
    theme_minimal()
}

################################################################################
# 1) Task 3a: Decision Tree - Tuning & Evaluation
################################################################################

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


# 1.3) Create the workflow that links recipie with the decision tree model

wf_tree <- workflow() %>%
  add_model(tree_spec) %>%
  add_recipe(rec_tree)


# 1.4) Define 4x4x4 grid with range of hyperparameters that will be assessed

tree_grid <- grid_regular(
  cost_complexity(range = c(-4, -1)), # pruning penalty: 10^-4 to 10^-1
  tree_depth(range = c(2L, 20L)), # max splits from root to leaf: 2-20
  min_n(range = c(5L, 30L)), # min. observations per leaf: 5-30
  levels = 4 # create 4 evenly spaced values for each parameter within its range
)


# 1.5) Grid search (fit on train, score on validation) & collect results in one table

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
plot_confusion_matrix(cm_val, "Confusion Matrix — Validation")


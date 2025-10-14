################################################################################
# MACHINE LEARNING IN FINANCE
# India Microdata - Data Preparation and EDA
# Authors: Heejung Jung & Luc Wuethrich
# Runtime: ca. 0min 5s (excl. packages installation)
################################################################################

# ──────────────────────────────────────────────────────────────────────────────
#### 1. Load required libraries ####
# ──────────────────────────────────────────────────────────────────────────────

cat("\n 0.1) Load packages & set seed \n")

# Use a fixed CRAN mirror to avoid interactive prompts
options(repos = c(CRAN = "https://cloud.r-project.org"))

# ──────────────────────────────────────────────────────────────────────────────
# Required packages for the full ML workflow (EDA + RF + DT + NN)
# ──────────────────────────────────────────────────────────────────────────────
pkgs <- c(
  # Core wrangling & visualization
  "tidyverse", "readr", "skimr", "gt", "scales",
  
  # Modeling frameworks
  "caret", "tidymodels",
  
  # Evaluation & metrics
  "pROC", "RColorBrewer",
  
  # Resampling & balancing
  "themis",
  
  # Specific model engines
  "randomForest", "rpart", "rpart.plot", "nnet", "NeuralNetTools",
  
  # Reshaping and data transformation
  "reshape2", 
  
  # Optional (parallel computing)
  "doParallel"
)

# ──────────────────────────────────────────────────────────────────────────────
# Install any missing packages
# ──────────────────────────────────────────────────────────────────────────────
to_install <- setdiff(pkgs, rownames(installed.packages()))
if (length(to_install)) {
  cat("\nInstalling missing packages:\n")
  print(to_install)
  install.packages(to_install, dependencies = TRUE, quiet = TRUE)
}

# ──────────────────────────────────────────────────────────────────────────────
# Load all libraries quietly
# ──────────────────────────────────────────────────────────────────────────────
suppressPackageStartupMessages({
  library(tidyverse)       # Wrangling + plotting
  library(readr)           # Fast CSV import
  library(skimr)           # Data overview
  library(gt)              # Summary and result tables
  library(scales)          # Plot scaling and percent formatting
  library(caret)           # Unified ML training/tuning
  library(tidymodels)      # Data splitting and resampling
  library(pROC)            # ROC and AUC metrics
  library(RColorBrewer)    # Color palettes
  library(themis)          # SMOTE-NC resampling
  library(randomForest)    # Random Forests
  library(rpart)           # Decision Trees
  library(rpart.plot)      # Decision Tree visualization
  library(nnet)            # Neural Networks
  library(NeuralNetTools)  # NN visualization
  library(reshape2)      # Data transformation
  library(doParallel)      # Parallel processing (optional)
})

# ──────────────────────────────────────────────────────────────────────────────
#### 2. Import dataset ####
# ──────────────────────────────────────────────────────────────────────────────

df <- read_csv("data/raw/microdata_india_raw.csv")

# ──────────────────────────────────────────────────────────────────────────────
#### 3. Select relevant variables ####
# ──────────────────────────────────────────────────────────────────────────────

df_selected <- df %>%
  select(
    female, age, educ, inc_q, emp_in, urbanicity_f2f,
    account_fin, account_mob,
    fin2, fin4, fin9, fin10,
    fin14_1, fin14a, fin14a1, fin14b,
    fin16, fin17a, fin17b,
    fin20, fin22a, fin22b,
    fin24, fin26, fin28, fin30,
    fin37, fin38, fin42,
    fin44a, fin44b, fin44c, fin44d, fin45_1,
    saved, borrowed,
    receive_wages, receive_transfers, receive_pension,
    receive_agriculture, pay_utilities, remittances,
    mobileowner, internetaccess, anydigpayment, merchantpay_dig
  )

# ──────────────────────────────────────────────────────────────────────────────
#### 4. Rename variables for clarity ####
# ──────────────────────────────────────────────────────────────────────────────

df_renamed <- df_selected %>%
  rename(
    income_q         = inc_q,          # within-economy income quantile
    employed         = emp_in,         # respondent is employed
    urban            = urbanicity_f2f, # urban/rural area
    has_debit_card   = fin2,           # owns a debit card
    used_debit_card  = fin4,           # used a debit card
    deposited        = fin9,           # made deposit
    withdrew         = fin10,          # made withdrawal
    mob_instore_pay  = fin14_1,        # mobile in-store payment
    bill_paid_int    = fin14a,         # paid bills online
    sent_money_int   = fin14a1,        # sent money online
    bought_on_int    = fin14b,         # bought something online
    saved_old_age    = fin16,          # saved for old age
    saved_using_acc  = fin17a,         # saved using account
    saved_inf_club   = fin17b,         # saved using informal club
    borrow_med       = fin20,          # borrowed for medical needs
    borrow_fin       = fin22a,         # borrowed from financial institution
    borrow_friends   = fin22b,         # borrowed from family/friends
    source_emergency = fin24,          # source of emergency funds
    sent_dom_rem     = fin26,          # sent domestic remittance
    received_dom_rem = fin28,          # received domestic remittance
    paid_ut_bill     = fin30,          # paid utility bill
    rec_gov_transfer = fin37,          # received government transfer
    rec_gov_pension  = fin38,          # received government pension
    rec_agri_payment = fin42,          # received agricultural payment
    fin_worried_old  = fin44a,         # worried: old age
    fin_worried_med  = fin44b,         # worried: medical
    fin_worried_bil  = fin44c,         # worried: bills
    fin_worried_edu  = fin44d,         # worried: education
    fin_worried_cov  = fin45_1         # worried: COVID
  )

# ──────────────────────────────────────────────────────────────────────────────
#### 5. Filter only respondents with financial accounts ####
# ──────────────────────────────────────────────────────────────────────────────

df_filtered <- df_renamed %>%
  filter(account_fin == 1)

# ──────────────────────────────────────────────────────────────────────────────
#### 6. Select relevant predictors (government-known variables) ####
# ──────────────────────────────────────────────────────────────────────────────

df_final <- df_filtered %>%
  select(
    female, age, educ, urban, income_q, employed,
    rec_gov_transfer, rec_gov_pension, rec_agri_payment,
    paid_ut_bill, internetaccess, mobileowner, has_debit_card
  )

# ──────────────────────────────────────────────────────────────────────────────
#### 7. Overview of the selected data ####
# ──────────────────────────────────────────────────────────────────────────────

skim(df_final)

# ──────────────────────────────────────────────────────────────────────────────
#### 8. Function to summarize variable ranges and missing values ####
# ──────────────────────────────────────────────────────────────────────────────

variable_summary <- function(data, var) {
  data %>%
    summarise(
      min_value = min({{ var }}, na.rm = TRUE),
      max_value = max({{ var }}, na.rm = TRUE),
      num_NAs   = sum(is.na({{ var }}))
    )
}

# Check variable ranges and missing values
variable_summary(df_final, female)
variable_summary(df_final, age)
variable_summary(df_final, educ)
variable_summary(df_final, urban)
variable_summary(df_final, income_q)
variable_summary(df_final, employed)
variable_summary(df_final, rec_gov_transfer)
variable_summary(df_final, rec_gov_pension)
variable_summary(df_final, rec_agri_payment)
variable_summary(df_final, paid_ut_bill)
variable_summary(df_final, internetaccess)
variable_summary(df_final, mobileowner)
variable_summary(df_final, has_debit_card)

# ──────────────────────────────────────────────────────────────────────────────
#### 9. Clean binary and ordinal variables, handle missing values ####
# ──────────────────────────────────────────────────────────────────────────────

df_final <- df_final %>%
  mutate(
    female           = if_else(female == 2, 0L, female),
    educ             = if_else(educ > 3, NA_integer_, educ),
    urban            = if_else(urban == 2, 1L, 0L),
    employed         = if_else(employed == 2, 0L, employed),
    rec_gov_transfer = if_else(rec_gov_transfer > 2, NA_integer_, rec_gov_transfer),
    rec_gov_transfer = if_else(rec_gov_transfer == 2, 0L, rec_gov_transfer),
    rec_gov_pension  = if_else(rec_gov_pension > 2, NA_integer_, rec_gov_pension),
    rec_gov_pension  = if_else(rec_gov_pension == 2, 0L, rec_gov_pension),
    rec_agri_payment = if_else(rec_agri_payment > 2, NA_integer_, rec_agri_payment),
    rec_agri_payment = if_else(rec_agri_payment == 2, 0L, rec_agri_payment),
    paid_ut_bill     = if_else(paid_ut_bill > 2, NA_integer_, paid_ut_bill),
    paid_ut_bill     = if_else(paid_ut_bill == 2, 0L, paid_ut_bill),
    internetaccess   = if_else(internetaccess > 2, NA_integer_, internetaccess),
    internetaccess   = if_else(internetaccess == 2, 0L, internetaccess),
    mobileowner      = if_else(mobileowner > 2, NA_integer_, mobileowner),
    mobileowner      = if_else(mobileowner == 2, 0L, mobileowner),
    has_debit_card   = if_else(has_debit_card > 2, NA_integer_, has_debit_card),
    has_debit_card   = if_else(has_debit_card == 2, 0L, has_debit_card)
  ) %>%
  mutate(across(everything(), as.integer)) %>%
  filter(!is.na(has_debit_card))  # drop missing targets

# ──────────────────────────────────────────────────────────────────────────────
#### 10. Summary statistics ####
# ──────────────────────────────────────────────────────────────────────────────

df_summary <- df_final %>%
  pivot_longer(cols = everything(), names_to = "Variable", values_to = "Value") %>%
  group_by(Variable) %>%
  summarise(
    Observations        = sum(!is.na(Value)),
    Mean                = mean(Value, na.rm = TRUE),
    `Standard Deviation`= sd(Value, na.rm = TRUE),
    Min                 = min(Value, na.rm = TRUE),
    Max                 = max(Value, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(across(where(is.numeric), round, 2))

summary_table <- df_summary %>%
  gt() %>%
  fmt_number(columns = where(is.numeric), decimals = 2) %>%
  opt_row_striping() %>%
  tab_style(
    style = cell_text(weight = "bold"),
    locations = cells_column_labels()
  ) %>%
  tab_header(
    title = md("**Summary Statistics – Cleaned India Microdata**"),
    subtitle = md("Variables related to debit card ownership")
  )

print(summary_table)

# ──────────────────────────────────────────────────────────────────────────────
#### 11. Proportion barplots ####
# ──────────────────────────────────────────────────────────────────────────────

# Convert continuous variables into factor groups
df_barplot <- df_final %>%
  mutate(
    age_group = factor(
      if_else(age < 25, "under_25", "25_and_above"),
      levels = c("under_25", "25_and_above")
    ),
    income_group = factor(
      if_else(income_q < 3, "under_3", "3_and_above"),
      levels = c("under_3", "3_and_above")
    )
  )

# Define binary predictors
binary_cols <- c(
  "female", "urban", "employed",
  "rec_gov_transfer", "rec_gov_pension", "rec_agri_payment",
  "paid_ut_bill", "internetaccess", "mobileowner"
)

# Convert binary variables to factors
df_barplot <- df_barplot %>%
  mutate(across(all_of(binary_cols),
                ~ factor(.x, levels = c(0, 1), labels = c("No", "Yes")))) %>%
  mutate(has_debit_card = factor(has_debit_card,
                                 levels = c(0, 1),
                                 labels = c("No", "Yes")))

# Variables for plotting
vars_to_plot <- c(binary_cols, "age_group", "income_group")

# Create stacked proportion barplots
plots <- map(vars_to_plot, function(v) {
  ggplot(df_barplot, aes(x = has_debit_card, fill = .data[[v]])) +
    geom_bar(position = "fill") +
    scale_y_continuous(labels = scales::percent) +
    labs(
      title = v,
      x = "Has Debit Card",
      y = "Percentage",
      fill = v
    ) +
    theme_minimal(base_size = 13)
})

# Print plots sequentially
walk(plots, print)

# ──────────────────────────────────────────────────────────────────────────────
#### 12. Correlation matrix (numeric variables only) ####
# ──────────────────────────────────────────────────────────────────────────────

corr_mat <- cor(df_final, use = "pairwise.complete.obs")

# Order by hierarchical clustering
ord <- hclust(as.dist(1 - abs(corr_mat)))$order
vars_ordered <- rownames(corr_mat)[ord]
corr_ord <- corr_mat[vars_ordered, vars_ordered]

# Long form correlation heatmap
corr_ord %>%
  as.data.frame() %>%
  rownames_to_column("row") %>%
  pivot_longer(-row, names_to = "col", values_to = "value") %>%
  mutate(
    row = factor(row, levels = vars_ordered),
    col = factor(col, levels = vars_ordered)
  ) %>%
  filter(as.integer(row) >= as.integer(col)) %>%
  ggplot(aes(x = col, y = row, fill = value)) +
  geom_tile(color = "grey80") +
  geom_text(aes(label = sprintf("%.2f", value)), size = 3) +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red", limits = c(-1, 1)) +
  coord_equal() +
  labs(
    title = "Correlation Matrix – Cleaned India Microdata",
    x = NULL,
    y = NULL
  ) +
  theme_minimal(base_size = 13) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# ──────────────────────────────────────────────────────────────────────────────
#### 13. Barplot of absolute correlations with target variable ####
# ──────────────────────────────────────────────────────────────────────────────

target_col <- "has_debit_card"
predictors <- setdiff(names(df_final), target_col)

cor_with_target <- tibble(
  variable = predictors,
  correlation = map_dbl(predictors,
                        ~ cor(df_final[[.x]], df_final[[target_col]],
                              use = "pairwise.complete.obs"))
) %>%
  mutate(abs_corr = abs(correlation)) %>%
  arrange(desc(abs_corr))

cor_with_target %>%
  slice_head(n = 20) %>%
  mutate(variable = fct_reorder(variable, abs_corr)) %>%
  ggplot(aes(x = abs_corr, y = variable)) +
  geom_col() +
  labs(
    title = "Top Predictors by |Correlation| with Debit Card Ownership",
    x = "|Correlation|",
    y = NULL
  ) +
  theme_minimal(base_size = 13)

# ──────────────────────────────────────────────────────────────────────────────
#### 14. Preparing data for modelling ####
# ──────────────────────────────────────────────────────────────────────────────

# Convert binary integers to factors
df_final$female          <- factor(df_final$female, levels = c(0,1), labels = c("male","female"))
df_final$urban           <- factor(df_final$urban, levels = c(0,1), labels = c("rural","urban"))
df_final$employed        <- factor(df_final$employed, levels = c(0,1), labels = c("unemployed","employed"))
df_final$rec_gov_transfer<- factor(df_final$rec_gov_transfer, levels = c(0,1), labels = c("No","Yes"))
df_final$rec_gov_pension <- factor(df_final$rec_gov_pension, levels = c(0,1), labels = c("No","Yes"))
df_final$rec_agri_payment<- factor(df_final$rec_agri_payment, levels = c(0,1), labels = c("No","Yes"))
df_final$paid_ut_bill    <- factor(df_final$paid_ut_bill, levels = c(0,1), labels = c("No","Yes"))
df_final$internetaccess  <- factor(df_final$internetaccess, levels = c(0,1), labels = c("No","Yes"))
df_final$mobileowner     <- factor(df_final$mobileowner, levels = c(0,1), labels = c("No","Yes"))
df_final$has_debit_card  <- factor(df_final$has_debit_card, levels = c(0,1), labels = c("No","Yes"))

# Convert ordinal variables
df_final$educ <- factor(df_final$educ, 
                        levels = c(1,2,3), 
                        labels = c("primary_or_less","secondary","tertiary_or_more"), 
                        ordered = TRUE)

df_final$income_q <- factor(df_final$income_q, 
                            level = c(1,2,3,4,5), 
                            labels = c("poorest_20","second_20","middle_20","fourth_20","richest_20"), 
                            ordered = TRUE)

# Keep only complete cases
df_final <- na.omit(df_final)
################################################################################
# END OF SCRIPT 1
################################################################################

################################################################################
# MACHINE LEARNING IN FINANCE
# India Microdata - Random Forest Modeling
# Authors: Alena Kohl & Franz Triebe
# Seed: 67
# Runtime: ca. 8min 45s (excl. 5. Optimization with LOOCV) 
################################################################################

# ──────────────────────────────────────────────────────────────────────────────
#### 1. Data splitting ####
# ──────────────────────────────────────────────────────────────────────────────

# Optionally enable parallelization for heavy training loops
# cl <- makeCluster(detectCores() - 3)
# registerDoParallel(cl)

# 70% training / 30% testing split
set.seed(67)

train_index <- createDataPartition(df_final$has_debit_card, p = 0.7, list = FALSE)
data_train  <- df_final[train_index, ]
data_test   <- df_final[-train_index, ]

# Display training and test distributions
cat("Training distribution:\n")
print(table(data_train$has_debit_card))

cat("Test distribution:\n")
print(table(data_test$has_debit_card))

# Visualize training set distribution
train_dist <- data_train %>%
  count(has_debit_card, name = "Count") %>%
  mutate(
    Percent = Count / sum(Count),
    Class = has_debit_card
  ) %>%
  gt() %>%
  fmt_percent(columns = "Percent", decimals = 1) %>%
  cols_label() %>%
  tab_header(title = "Training Set Distribution")

train_dist

# ──────────────────────────────────────────────────────────────────────────────
#### 2. Tree stabilization analysis (OOB error) ####
# ──────────────────────────────────────────────────────────────────────────────

set.seed(67)

tree_counts <- c(500, 1000, 5000)
rf_models <- list()

# Train models with different tree counts
for (nt in tree_counts) {
  rf_models[[as.character(nt)]] <- randomForest(
    has_debit_card ~ ., 
    data = data_train,
    importance = TRUE,
    keep.forest = TRUE,
    keep.inbag = TRUE,
    ntree = nt
  )
}

# Plot OOB and class-specific errors
par(mfrow = c(2, 2))  # 4 plots per page

for (nt in tree_counts) {
  err <- rf_models[[as.character(nt)]]$err.rate
  
  # Linear scale OOB error
  plot(
    1:nrow(err), err[, "OOB"], 
    type = "l", lwd = 2, col = "blue",
    xlab = "Number of Trees", ylab = "OOB Error Rate",
    main = paste0("OOB Error (", nt, " trees, linear)")
  )
  
  # Log scale OOB error
  plot(
    1:nrow(err), err[, "OOB"],
    type = "l", lwd = 2, col = "blue", log = "x",
    xlab = "Number of Trees (log scale)", ylab = "OOB Error Rate",
    main = paste0("OOB Error (", nt, " trees, log)")
  )
  
  # Class-specific error (linear)
  matplot(
    1:nrow(err), err, type = "l", lty = 1, lwd = 2,
    col = c("blue", "red", "darkgreen"),
    xlab = "Number of Trees", ylab = "Error Rate",
    main = paste0("Class Error (", nt, " trees, linear)")
  )
  legend("topright", legend = colnames(err),
         col = c("blue", "red", "darkgreen"), lty = 1, lwd = 2, cex = 0.8)
  
  # Class-specific error (log)
  matplot(
    1:nrow(err), err, type = "l", lty = 1, lwd = 2, log = "x",
    col = c("blue", "red", "darkgreen"),
    xlab = "Number of Trees (log scale)", ylab = "Error Rate",
    main = paste0("Class Error (", nt, " trees, log)")
  )
  legend("topright", legend = colnames(err),
         col = c("blue", "red", "darkgreen"), lty = 1, lwd = 2, cex = 0.8)
}

par(mfrow = c(1, 1))  # Reset plot layout

# ──────────────────────────────────────────────────────────────────────────────
#### 3. Optimisation of mtry and ntree with OOB (Accuracy) ####
# ──────────────────────────────────────────────────────────────────────────────

set.seed(67)

p <- ncol(data_train) - 1
tune_grid <- expand.grid(.mtry = 1:p)
ntree_values <- c(100, 200, 500, 750, 1000)

control_oob <- trainControl(method = "oob")
results_oob <- list()

for (nt in ntree_values) {
  cat("==== OOB: ntree =", nt, "====\n")
  
  rf_oob <- train(
    has_debit_card ~ ., data = data_train,
    method = "rf",
    metric = "Accuracy",
    tuneGrid = tune_grid,
    trControl = control_oob,
    ntree = nt
  )
  
  rf_oob$results$ntree <- nt
  rf_oob$results$type  <- "OOB"
  
  results_oob[[paste0("ntree_", nt)]] <- rf_oob$results
}

results_oob <- bind_rows(results_oob)

cols <- brewer.pal(6, "BuPu")[2:6]

ggplot(results_oob, aes(x = mtry, y = Accuracy, color = factor(ntree))) +
  geom_line() +
  geom_point() +
  scale_x_continuous(breaks = seq(min(results_oob$mtry), max(results_oob$mtry), 1)) +
  scale_color_manual(values = cols) +
  labs(
    title = "Random Forest (OOB): Accuracy by mtry × ntree",
    x = "mtry",
    y = "Accuracy",
    color = "ntree"
  ) +
  theme_minimal(base_size = 14)

# ──────────────────────────────────────────────────────────────────────────────
#### 4. Optimisation of mtry and ntree with 10-Fold CV (ROC) ####
# ──────────────────────────────────────────────────────────────────────────────

set.seed(67)

control_cv <- trainControl(
  method = "cv",
  number = 10,
  search = "grid",
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

results_cv <- list()

for (nt in ntree_values) {
  cat("==== CV: ntree =", nt, "====\n")
  
  rf_cv <- train(
    has_debit_card ~ ., data = data_train,
    method = "rf",
    metric = "ROC",
    tuneGrid = tune_grid,
    trControl = control_cv,
    ntree = nt
  )
  
  rf_cv$results$ntree <- nt
  rf_cv$results$type  <- "CV_10fold"
  
  results_cv[[paste0("ntree_", nt)]] <- rf_cv$results
}

results_cv <- bind_rows(results_cv)

ggplot(results_cv, aes(x = mtry, y = ROC, color = factor(ntree))) +
  geom_line() +
  geom_point() +
  scale_x_continuous(breaks = seq(min(results_oob$mtry), max(results_cv$mtry), 1)) +
  scale_color_manual(values = cols) +
  labs(
    title = "Random Forest (10-fold CV): ROC AUC by mtry × ntree",
    x = "mtry",
    y = "AUC (ROC)",
    color = "ntree"
  ) +
  theme_minimal(base_size = 14)

# ──────────────────────────────────────────────────────────────────────────────
#### 5. Optimisation of mtry and ntree with LOOCV (ROC); WARNING - Runtime! ####
# ──────────────────────────────────────────────────────────────────────────────

# Warning: Very slow (can take several hours without parallelization)
set.seed(67)

control_loocv <- trainControl(
  method = "LOOCV",
  search = "grid",
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

results_loocv <- list()

for (nt in ntree_values) {
  cat("==== LOOCV: ntree =", nt, "====\n")
  
  rf_loocv <- train(
    has_debit_card ~ ., data = data_train,
    method = "rf",
    metric = "ROC",
    tuneGrid = tune_grid,
    trControl = control_loocv,
    ntree = nt
  )
  
  rf_loocv$results$ntree <- nt
  rf_loocv$results$type  <- "LOOCV"
  
  results_loocv[[paste0("ntree_", nt)]] <- rf_loocv$results
}

results_loocv <- bind_rows(results_loocv)

ggplot(results_loocv, aes(x = mtry, y = ROC, color = factor(ntree))) +
  geom_line() +
  geom_point() +
  scale_x_continuous(breaks = seq(min(results_loocv$mtry), max(results_cv$mtry), 1)) +
  labs(
    title = "Random Forest (LOOCV): AUC ROC by mtry × ntree",
    x = "mtry",
    y = "AUC (ROC)",
    color = "ntree"
  ) +
  theme_minimal(base_size = 14)

# ──────────────────────────────────────────────────────────────────────────────
#### 6. Model Evaluation on Test Data: Default vs Tuned ####
# ──────────────────────────────────────────────────────────────────────────────

### Default RF (500 trees, mtry = 3)
set.seed(67)
rf_default <- randomForest(
  has_debit_card ~ ., data = data_train,
  importance = TRUE, keep.forest = TRUE, keep.inbag = TRUE,
  ntree = 500
)

cat("\n--- Default RF (500 trees, mtry = 3) ---\n")
print(rf_default)

# Predictions (class + probability)
rf_pred_class_default <- predict(rf_default, newdata = data_test, type = "response")
rf_pred_prob_default  <- predict(rf_default, newdata = data_test, type = "prob")[, "Yes"]

# Confusion matrix with detailed stats
cm_default <- confusionMatrix(rf_pred_class_default, data_test$has_debit_card, positive = "Yes")
print(cm_default)

# Confusion matrix visualization
cm_d <- as.data.frame(cm_default$table)

ggplot(cm_d, aes(x = Prediction, y = Reference, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "white", size = 6, fontface = "bold") +
  scale_fill_gradient(low = "plum1", high = "plum4", name = "Freq") +
  coord_equal() +
  labs(
    title = "Confusion Matrix – Default",
    subtitle = "Baseline Random Forest (500 trees, mtry = 3)",
    x = "Predicted Class", y = "Actual Class"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5)
  )

# ROC + AUC
roc_obj_default <- roc(
  response = data_test$has_debit_card,
  predictor = rf_pred_prob_default,
  levels = c("No", "Yes"), direction = "<"
)
auc_default <- auc(roc_obj_default)
cat("AUC (ROC) Default RF:", auc_default, "\n")

### Tuned RF (500 trees, mtry = 1)
set.seed(67)
rf_optimal <- randomForest(
  has_debit_card ~ ., data = data_train,
  importance = TRUE, keep.forest = TRUE, keep.inbag = TRUE,
  ntree = 500, mtry = 1
)

cat("\n--- Tuned RF (500 trees, mtry = 1) ---\n")
print(rf_optimal)

# Predictions (class + probability)
rf_pred_class_opt <- predict(rf_optimal, newdata = data_test, type = "response")
rf_pred_prob_opt  <- predict(rf_optimal, newdata = data_test, type = "prob")[, "Yes"]

# Confusion matrix with detailed stats
cm_opt <- confusionMatrix(rf_pred_class_opt, data_test$has_debit_card, positive = "Yes")
print(cm_opt)

# Confusion matrix visualization
cm_optimal <- as.data.frame(cm_opt$table)

ggplot(cm_optimal, aes(x = Prediction, y = Reference, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "white", size = 6, fontface = "bold") +
  scale_fill_gradient(low = "plum1", high = "plum4", name = "Freq") +
  coord_equal() +
  labs(
    title = "Confusion Matrix – Tuned",
    subtitle = "Optimized Random Forest (500 trees, mtry = 1)",
    x = "Predicted Class", y = "Actual Class"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5)
  )

# ROC + AUC
roc_obj_opt <- roc(
  response = data_test$has_debit_card,
  predictor = rf_pred_prob_opt,
  levels = c("No", "Yes"), direction = "<"
)
auc_opt <- auc(roc_obj_opt)
cat("AUC (ROC) Tuned RF:", auc_opt, "\n")

# stopCluster(cl); registerDoSEQ()  # end parallel if started

# ──────────────────────────────────────────────────────────────────────────────
#### 7. Model Comparison Summary (with F1) ####
# ──────────────────────────────────────────────────────────────────────────────

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
    AUC = as.numeric(auc_val)   
  )
}

metrics_default <- extract_metrics(cm_default, auc_default, "Default RF (500 trees, mtry = 3)")
metrics_optimal <- extract_metrics(cm_opt, auc_opt, "Tuned RF (500 trees, mtry = 1)")

comparison_table <- bind_rows(metrics_default, metrics_optimal) %>%
  mutate(across(where(is.numeric), round, 4))

comparison_table_select <- comparison_table %>%
  select(Accuracy, Balanced_Accuracy, AUC) %>%
  rename(`Bal. Acc.` = Balanced_Accuracy) %>%
  mutate(Model = c("Default", "Tuned")) %>%
  gt(rowname_col = "Model")

comparison_table_select

cat("\n==================== MODEL COMPARISON (Extended) ====================\n")
print(comparison_table)
cat("=====================================================================\n")

# Visualization of model comparison
comparison_long <- melt(comparison_table, id.vars = "Model")

ggplot(comparison_long, aes(x = variable, y = value, fill = Model)) +
  geom_bar(stat = "identity", position = position_dodge(), width = 0.7) +
  scale_fill_manual(values = cols[c(2, 4)]) +
  labs(
    title = "Random Forest Comparison: Default vs Tuned",
    subtitle = "Including F1, Precision, Recall & AUC",
    x = "Metric",
    y = "Value",
    fill = "Model"
  ) +
  theme_minimal(base_size = 14) +
  coord_cartesian(ylim = c(0.4, 0.85)) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    plot.title = element_text(face = "bold")
  )

# ──────────────────────────────────────────────────────────────────────────────
#### 8. Feature Importance (Permutation & Gini) ####
# ──────────────────────────────────────────────────────────────────────────────

## Permutation importance – optimal RF
set.seed(67)
per_imp <- importance(rf_optimal, type = 1, scale = TRUE) %>%
  as.data.frame() %>%
  rownames_to_column("Variable") %>%
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

## Permutation importance – default RF
per_imp_default <- importance(rf_default, type = 1, scale = TRUE) %>%
  as.data.frame() %>%
  rownames_to_column("Variable") %>%
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

## Gini importance – optimal RF
imp <- importance(rf_optimal, type = 2, scale = TRUE) %>%
  as.data.frame() %>%
  rownames_to_column("Variable") %>%
  arrange(desc(MeanDecreaseGini))

ggplot(imp, aes(x = MeanDecreaseGini, y = reorder(Variable, MeanDecreaseGini))) +
  geom_col(fill = "plum") +
  labs(
    title = "Impurity Importance (Optimal RF)",
    subtitle = "Mean decrease in node impurity (Gini)",
    x = "Impurity Importance",
    y = "Variable"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

## Gini importance – default RF
imp_default <- importance(rf_default, type = 2, scale = TRUE) %>%
  as.data.frame() %>%
  rownames_to_column("Variable") %>%
  arrange(desc(MeanDecreaseGini))

ggplot(imp_default, aes(x = MeanDecreaseGini, y = reorder(Variable, MeanDecreaseGini))) +
  geom_col(fill = "plum") +
  labs(
    title = "Impurity Importance (Default RF)",
    subtitle = "Mean decrease in node impurity (Gini)",
    x = "Impurity Importance",
    y = "Variable"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

# ──────────────────────────────────────────────────────────────────────────────
#### 9. Robustness Check – Remove Non-Significant Predictors ####
# ──────────────────────────────────────────────────────────────────────────────

set.seed(67)

data_train_select <- data_train %>%
  select(-rec_agri_payment, -rec_gov_pension)

data_test_select <- data_test %>%
  select(-rec_agri_payment, -rec_gov_pension)

rf_optimal_select <- randomForest(
  has_debit_card ~ ., data = data_train_select,
  importance = TRUE, keep.forest = TRUE, keep.inbag = TRUE,
  ntree = 500, mtry = 1
)

cat("\n--- Tuned RF (500 trees, mtry = 1) – Reduced Predictors ---\n")
print(rf_optimal_select)

rf_pred_class_opt_select <- predict(rf_optimal_select, newdata = data_test_select, type = "response")
rf_pred_prob_opt_select  <- predict(rf_optimal_select, newdata = data_test_select, type = "prob")[, "Yes"]

cm_opt_select <- confusionMatrix(rf_pred_class_opt_select, data_test_select$has_debit_card, positive = "Yes")
print(cm_opt_select)

# Confusion matrix visualization
cm_optimal_select <- as.data.frame(cm_opt_select$table)

ggplot(cm_optimal_select, aes(x = Prediction, y = Reference, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "white", size = 6, fontface = "bold") +
  scale_fill_gradient(low = "plum1", high = "plum4", name = "Freq") +
  coord_equal() +
  labs(
    title = "Confusion Matrix – Tuned (Reduced Feature Set)",
    subtitle = "After omitting rec_agri_payment & rec_gov_pension",
    x = "Predicted Class", y = "Actual Class"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5)
  )

# ──────────────────────────────────────────────────────────────────────────────
#### 10. Apply SMOTE for Class Balancing ####
# ──────────────────────────────────────────────────────────────────────────────

set.seed(67)

cat("\nBefore SMOTE:\n")
print(table(data_train$has_debit_card))

# Convert to data frame (for themis compatibility)
df_train <- as.data.frame(data_train)

# Apply SMOTENC balancing
data_train_bal <- smotenc(
  df = df_train,
  var = "has_debit_card",  # Target variable
  k = 5,                   # Number of neighbors
  over_ratio = 1            # 1:1 ratio (watch for potential overfitting)
)

cat("\nAfter SMOTE (themis::smotenc):\n")
print(table(data_train_bal$has_debit_card))
str(data_train_bal)

# Replace training data but storing the old one
data_train_old <- data_train
data_train <- data_train_bal

# Visualize class distribution after SMOTE
train_dist_smote <- data_train_bal %>%
  count(has_debit_card, name = "Count") %>%
  mutate(Percent = Count / sum(Count), Class = has_debit_card) %>%
  gt() %>%
  fmt_percent(columns = "Percent", decimals = 1) %>%
  cols_label() %>%
  tab_header(title = "Training Set Distribution (SMOTE-NC)")

train_dist_smote

# Train tuned RF after SMOTE
set.seed(67)
rf_optimal_smote <- randomForest(
  has_debit_card ~ ., data = data_train,
  importance = TRUE, keep.forest = TRUE, keep.inbag = TRUE,
  ntree = 500, mtry = 6
)

rf_pred_class_optimal_smote <- predict(rf_optimal_smote, newdata = data_test, type = "response")
rf_pred_prob_optimal_smote  <- predict(rf_optimal_smote, newdata = data_test, type = "prob")[, "Yes"]
cm_optimal_smote <- confusionMatrix(rf_pred_class_optimal_smote, data_test$has_debit_card, positive = "Yes")
print(cm_optimal_smote)

#AUC ROC
roc_smote <- roc(data_test$has_debit_card, rf_pred_prob_optimal_smote, levels = c("No", "Yes"))
auc_smote <- auc(roc_smote)
cat("\nAUC (ROC) – Tuned RF (After SMOTE):", round(auc_smote, 4), "\n")
plot(roc_smote, col = "purple4", lwd = 3, main = sprintf("ROC Curve – Tuned RF (After SMOTE)\nAUC = %.3f", auc_smote))

# Confusion matrix visualization after SMOTE
cm_o_s <- as.data.frame(cm_optimal_smote$table)

ggplot(cm_o_s, aes(x = Prediction, y = Reference, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "white", size = 6, fontface = "bold") +
  scale_fill_gradient(low = "plum1", high = "plum4", name = "Freq") +
  coord_equal() +
  labs(
    title = "Confusion Matrix – Tuned RF (After SMOTE)",
    subtitle = "Evaluated on Test Set after SMOTE-NC Balancing",
    x = "Predicted", y = "Actual"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5)
  )

# ──────────────────────────────────────────────────────────────────────────────
#### 11. Adapt SMOTE for Class Balancing without overfitting ####
# ──────────────────────────────────────────────────────────────────────────────

set.seed(67)

cat("\nBefore SMOTE:\n")
print(table(data_train_old$has_debit_card))

# Convert to data frame (for themis compatibility)
df_train_old <- as.data.frame(data_train_old)

# Apply SMOTENC balancing
data_train_bal <- smotenc(
  df = df_train_old,
  var = "has_debit_card",  # Target variable
  k = 5,                   # Number of neighbors
  over_ratio = 806/1006    #oversampling ratio
)

set.seed(67)
no_data  <- data_train_bal %>% filter(has_debit_card == "No")
yes_data <- data_train_bal %>% filter(has_debit_card == "Yes")

no_data_sub <- no_data %>% sample_n(nrow(no_data) - 200, replace = FALSE)

# Neu zusammenfügen & mischen
data_train_bal <- bind_rows(no_data_sub, yes_data) %>%
  sample_frac(1) %>%
  mutate(has_debit_card = factor(has_debit_card, levels = c("No", "Yes")))

cat("\nAfter SMOTE (themis::smotenc):\n")
print(table(data_train_bal$has_debit_card))
str(data_train_bal)

# Replace training data 
data_train <- data_train_bal

# Visualize class distribution after SMOTE
train_dist_smote <- data_train_bal %>%
  count(has_debit_card, name = "Count") %>%
  mutate(Percent = Count / sum(Count), Class = has_debit_card) %>%
  gt() %>%
  fmt_percent(columns = "Percent", decimals = 1) %>%
  cols_label() %>%
  tab_header(title = "Training Set Distribution (SMOTE-NC)")

train_dist_smote

# Train tuned RF after SMOTE
set.seed(67)
rf_optimal_smote <- randomForest(
  has_debit_card ~ ., data = data_train,
  importance = TRUE, keep.forest = TRUE, keep.inbag = TRUE,
  ntree = 500, mtry = 5
)

rf_pred_class_optimal_smote <- predict(rf_optimal_smote, newdata = data_test, type = "response")
rf_pred_prob_optimal_smote  <- predict(rf_optimal_smote, newdata = data_test, type = "prob")[, "Yes"]
cm_optimal_smote <- confusionMatrix(rf_pred_class_optimal_smote, data_test$has_debit_card, positive = "Yes")
print(cm_optimal_smote)

#AUC ROC
roc_smote <- roc(data_test$has_debit_card, rf_pred_prob_optimal_smote, levels = c("No", "Yes"))
auc_smote <- auc(roc_smote)
cat("\nAUC (ROC) – Tuned RF (After SMOTE):", round(auc_smote, 4), "\n")
plot(roc_smote, col = "purple4", lwd = 3, main = sprintf("ROC Curve – Tuned RF (After SMOTE)\nAUC = %.3f", auc_smote))

# Confusion matrix visualization after SMOTE
cm_o_s <- as.data.frame(cm_optimal_smote$table)

ggplot(cm_o_s, aes(x = Prediction, y = Reference, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "white", size = 6, fontface = "bold") +
  scale_fill_gradient(low = "plum1", high = "plum4", name = "Freq") +
  coord_equal() +
  labs(
    title = "Confusion Matrix – Tuned RF (After SMOTE)",
    subtitle = "Evaluated on Test Set after SMOTE-NC Balancing",
    x = "Predicted", y = "Actual"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5)
  )

# stopCluster(cl); registerDoSEQ()  # if parallel was used

# ──────────────────────────────────────────────────────────────────────────────
#### 12. Generalization check of tuned Random Forest on Nepal ####
# ──────────────────────────────────────────────────────────────────────────────

### Data Preparation 

# Import data set
df_nepal <- read_csv("data/microdata_nepal_raw.csv")

# Select relevant variables
df_nepal_selected <- df_nepal %>%
  select(
    female, age, educ, inc_q, emp_in, urbanicity_f2f,
    account_fin, account_mob,
    fin2, fin4, fin9, fin10,
    fin14_1, fin14a, fin14a1, fin14b,
    fin16, fin17a, fin17b,
    fin20, fin22a, fin22b,
    fin24, fin26, fin28, fin30,
    fin37, fin38, fin42,
    fin44a, fin44b, fin44c, fin44d, fin45_1,
    saved, borrowed,
    receive_wages, receive_transfers, receive_pension,
    receive_agriculture, pay_utilities, remittances,
    mobileowner, internetaccess, anydigpayment, merchantpay_dig
  )

# Rename variables for clarity
df_nepal_renamed <- df_nepal_selected %>%
  rename(
    income_q         = inc_q,          # within-economy income quantile
    employed         = emp_in,         # respondent is employed
    urban            = urbanicity_f2f, # urban/rural area
    has_debit_card   = fin2,           # owns a debit card
    used_debit_card  = fin4,           # used a debit card
    deposited        = fin9,           # made deposit
    withdrew         = fin10,          # made withdrawal
    mob_instore_pay  = fin14_1,        # mobile in-store payment
    bill_paid_int    = fin14a,         # paid bills online
    sent_money_int   = fin14a1,        # sent money online
    bought_on_int    = fin14b,         # bought something online
    saved_old_age    = fin16,          # saved for old age
    saved_using_acc  = fin17a,         # saved using account
    saved_inf_club   = fin17b,         # saved using informal club
    borrow_med       = fin20,          # borrowed for medical needs
    borrow_fin       = fin22a,         # borrowed from financial institution
    borrow_friends   = fin22b,         # borrowed from family/friends
    source_emergency = fin24,          # source of emergency funds
    sent_dom_rem     = fin26,          # sent domestic remittance
    received_dom_rem = fin28,          # received domestic remittance
    paid_ut_bill     = fin30,          # paid utility bill
    rec_gov_transfer = fin37,          # received government transfer
    rec_gov_pension  = fin38,          # received government pension
    rec_agri_payment = fin42,          # received agricultural payment
    fin_worried_old  = fin44a,         # worried: old age
    fin_worried_med  = fin44b,         # worried: medical
    fin_worried_bil  = fin44c,         # worried: bills
    fin_worried_edu  = fin44d,         # worried: education
    fin_worried_cov  = fin45_1         # worried: COVID
  )

# Filter only respondents with financial accounts ####
df_nepal_filtered <- df_nepal_renamed %>%
  filter(account_fin == 1)

# Select relevant predictors (government-known variables)
df_nepal_final <- df_nepal_filtered %>%
  select(
    female, age, educ, urban, income_q, employed,
    rec_gov_transfer, rec_gov_pension, rec_agri_payment,
    paid_ut_bill, internetaccess, mobileowner, has_debit_card
  )

# Overview of the selected data

skim(df_nepal_final)

# Check variable ranges and missing values
variable_summary(df_nepal_final, female)
variable_summary(df_nepal_final, age)
variable_summary(df_nepal_final, educ)
variable_summary(df_nepal_final, urban)
variable_summary(df_nepal_final, income_q)
variable_summary(df_nepal_final, employed)
variable_summary(df_nepal_final, rec_gov_transfer)
variable_summary(df_nepal_final, rec_gov_pension)
variable_summary(df_nepal_final, rec_agri_payment)
variable_summary(df_nepal_final, paid_ut_bill)
variable_summary(df_nepal_final, internetaccess)
variable_summary(df_nepal_final, mobileowner)
variable_summary(df_nepal_final, has_debit_card)

# Clean binary and ordinal variables, handle missing values ####
df_nepal_final <- df_nepal_final %>%
  mutate(
    female           = if_else(female == 2, 0L, female),
    educ             = if_else(educ > 3, NA_integer_, educ),
    urban            = if_else(urban == 2, 1L, 0L),
    employed         = if_else(employed == 2, 0L, employed),
    rec_gov_transfer = if_else(rec_gov_transfer > 2, NA_integer_, rec_gov_transfer),
    rec_gov_transfer = if_else(rec_gov_transfer == 2, 0L, rec_gov_transfer),
    rec_gov_pension  = if_else(rec_gov_pension > 2, NA_integer_, rec_gov_pension),
    rec_gov_pension  = if_else(rec_gov_pension == 2, 0L, rec_gov_pension),
    rec_agri_payment = if_else(rec_agri_payment > 2, NA_integer_, rec_agri_payment),
    rec_agri_payment = if_else(rec_agri_payment == 2, 0L, rec_agri_payment),
    paid_ut_bill     = if_else(paid_ut_bill > 2, NA_integer_, paid_ut_bill),
    paid_ut_bill     = if_else(paid_ut_bill == 2, 0L, paid_ut_bill),
    internetaccess   = if_else(internetaccess > 2, NA_integer_, internetaccess),
    internetaccess   = if_else(internetaccess == 2, 0L, internetaccess),
    mobileowner      = if_else(mobileowner > 2, NA_integer_, mobileowner),
    mobileowner      = if_else(mobileowner == 2, 0L, mobileowner),
    has_debit_card   = if_else(has_debit_card > 2, NA_integer_, has_debit_card),
    has_debit_card   = if_else(has_debit_card == 2, 0L, has_debit_card)
  ) %>%
  mutate(across(everything(), as.integer)) %>%
  filter(!is.na(has_debit_card))  # drop missing targets

# Summary statistics
df_nepal_summary <- df_nepal_final %>%
  pivot_longer(cols = everything(), names_to = "Variable", values_to = "Value") %>%
  group_by(Variable) %>%
  summarise(
    Observations        = sum(!is.na(Value)),
    Mean                = mean(Value, na.rm = TRUE),
    `Standard Deviation`= sd(Value, na.rm = TRUE),
    Min                 = min(Value, na.rm = TRUE),
    Max                 = max(Value, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(across(where(is.numeric), round, 2))

summary_table_nepal <- df_nepal_summary %>%
  gt() %>%
  fmt_number(columns = where(is.numeric), decimals = 2) %>%
  opt_row_striping() %>%
  tab_style(
    style = cell_text(weight = "bold"),
    locations = cells_column_labels()
  ) %>%
  tab_header(
    title = md("**Summary Statistics – Cleaned India Microdata**"),
    subtitle = md("Variables related to debit card ownership")
  )

print(summary_table_nepal)

### Preparing data for modelling #

# Convert binary integers to factors
df_nepal_final$female          <- factor(df_nepal_final$female, levels = c(0,1), labels = c("male","female"))
df_nepal_final$urban           <- factor(df_nepal_final$urban, levels = c(0,1), labels = c("rural","urban"))
df_nepal_final$employed        <- factor(df_nepal_final$employed, levels = c(0,1), labels = c("unemployed","employed"))
df_nepal_final$rec_gov_transfer<- factor(df_nepal_final$rec_gov_transfer, levels = c(0,1), labels = c("No","Yes"))
df_nepal_final$rec_gov_pension <- factor(df_nepal_final$rec_gov_pension, levels = c(0,1), labels = c("No","Yes"))
df_nepal_final$rec_agri_payment<- factor(df_nepal_final$rec_agri_payment, levels = c(0,1), labels = c("No","Yes"))
df_nepal_final$paid_ut_bill    <- factor(df_nepal_final$paid_ut_bill, levels = c(0,1), labels = c("No","Yes"))
df_nepal_final$internetaccess  <- factor(df_nepal_final$internetaccess, levels = c(0,1), labels = c("No","Yes"))
df_nepal_final$mobileowner     <- factor(df_nepal_final$mobileowner, levels = c(0,1), labels = c("No","Yes"))
df_nepal_final$has_debit_card  <- factor(df_nepal_final$has_debit_card, levels = c(0,1), labels = c("No","Yes"))

# Convert ordinal variables
df_nepal_final$educ <- factor(df_nepal_final$educ, 
                        levels = c(1,2,3), 
                        labels = c("primary_or_less","secondary","tertiary_or_more"), 
                        ordered = TRUE)

df_nepal_final$income_q <- factor(df_nepal_final$income_q, 
                            level = c(1,2,3,4,5), 
                            labels = c("poorest_20","second_20","middle_20","fourth_20","richest_20"), 
                            ordered = TRUE)

# Keep only complete cases
df_nepal_final <- na.omit(df_nepal_final)

### Data splitting

# 70% training / 30% testing split
set.seed(67)

train_index_nepal <- createDataPartition(df_nepal_final$has_debit_card, p = 0.7, list = FALSE)
data_train_nepal  <- df_nepal_final[train_index_nepal, ]
data_test_nepal   <- df_nepal_final[-train_index_nepal, ]

# Display training and test distributions
cat("Training distribution:\n")
print(table(data_train_nepal$has_debit_card))

cat("Test distribution:\n")
print(table(data_test_nepal$has_debit_card))

# Visualize training set distribution
train_dist_nepal <- data_train_nepal %>%
  count(has_debit_card, name = "Count") %>%
  mutate(
    Percent = Count / sum(Count),
    Class = has_debit_card
  ) %>%
  gt() %>%
  fmt_percent(columns = "Percent", decimals = 1) %>%
  cols_label() %>%
  tab_header(title = "Training Set Distribution")

print(train_dist_nepal)

### Testing tuned RF on Nepal data

# Predictions (class + probability)
rf_pred_class_opt_nepal <- predict(rf_optimal, newdata = data_test_nepal, type = "response")
rf_pred_prob_opt_nepal  <- predict(rf_optimal, newdata = data_test_nepal, type = "prob")[, "Yes"]

# Confusion matrix with detailed stats
cm_opt_nepal <- confusionMatrix(rf_pred_class_opt_nepal, data_test_nepal$has_debit_card, positive = "Yes")
print(cm_opt_nepal)

# Confusion matrix visualization
cm_optimal_nepal <- as.data.frame(cm_opt_nepal$table)

ggplot(cm_optimal_nepal, aes(x = Prediction, y = Reference, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), color = "white", size = 6, fontface = "bold") +
  scale_fill_gradient(low = "plum1", high = "plum4", name = "Freq") +
  coord_equal() +
  labs(
    title = "Confusion Matrix – Nepal",
    subtitle = "Optimized Random Forest (500 trees, mtry = 1)",
    x = "Predicted", y = "Actual"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5)
  )

# ROC + AUC
roc_obj_opt_nepal <- roc(
  response = data_test_nepal$has_debit_card,
  predictor = rf_pred_prob_opt_nepal,
  levels = c("No", "Yes"), direction = "<"
)
auc_opt_nepal <- auc(roc_obj_opt_nepal)
cat("AUC (ROC) Tuned RF:", auc_opt, "\n")

################################################################################
# End of Script 2
################################################################################

################################################################################
# MACHINE LEARNING IN FINANCE
# India Microdata – Decision Tree and Neural Network Modelling
# Author: Jonah-Baptiste Lohmann
# Seed: 67
# Runtime: ca. 0min 40s
################################################################################

# ──────────────────────────────────────────────────────────────────────────────
#### 1. Data Splitting (70 % train / 30 % test) ####
# ──────────────────────────────────────────────────────────────────────────────

cat("\n 1. Data Splitting (70% train / 30% test) \n")

# Stratified split to preserve class balance
set.seed(67)
split_obj <- initial_split(df_final, prop = 0.7, strata = has_debit_card)
train <- training(split_obj)
test  <- testing(split_obj)

# Confirm proportions of target classes
cat("\n===== Check class balance of split =====\n")
map(list(train = train, test = test),
    ~ count(.x, has_debit_card) %>%
      mutate(proportion = n / sum(n))
) %>% print()

cat("\n===== Success: 1. Data split created =====\n")

# ──────────────────────────────────────────────────────────────────────────────
#### 2. Helper Functions for Plots & Cross-Validation ####
# ──────────────────────────────────────────────────────────────────────────────

cat("\n 2. Define helper functions for plots & cross-validation \n")

# Generic confusion-matrix plotter for Decision Trees (blue palette)
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

# Confusion-matrix plotter for Neural Networks (green palette)
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

# Define 10-fold cross-validation control
ctrl_cv <- trainControl(
  method = "cv", number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

cat("\n===== Success: 2. Helper functions ready =====\n")

# ──────────────────────────────────────────────────────────────────────────────
#### 3. Decision Tree DEFAULT (cp = 0.01) ####
# ──────────────────────────────────────────────────────────────────────────────

cat("\n 3. Decision Tree DEFAULT (cp = 0.01) \n")

set.seed(67)
dt_default <- caret::train(
  has_debit_card ~ ., data = train,
  method = "rpart",
  trControl = trainControl(method = "none", classProbs = TRUE),
  tuneGrid = data.frame(cp = 0.01)
)

# Predictions (classes + probabilities)
dt_pred_default <- predict(dt_default, newdata = test)
dt_prob_default <- predict(dt_default, newdata = test, type = "prob")[, "Yes"]

# Confusion Matrix + Metrics
dt_cm_default <- confusionMatrix(dt_pred_default, test$has_debit_card, positive = "Yes")
dt_acc_default <- as.numeric(dt_cm_default$overall["Accuracy"])
dt_auc_default <- as.numeric(
  pROC::auc(
    pROC::roc(response = test$has_debit_card, predictor = dt_prob_default,
              levels = c("No", "Yes"))
  )
)

cat("\n===== DT Default – Performance =====\n")
cat(sprintf("ROC AUC = %.4f | Accuracy = %.4f\n", dt_auc_default, dt_acc_default))
plot_cm_dt(dt_cm_default, "Confusion Matrix – Decision Tree (Default)")

cat("\n==== Success: 3. DT DEFAULT trained ====\n")

# ──────────────────────────────────────────────────────────────────────────────
#### 4. Decision Tree TUNED: 10-Fold CV on cp [0.001;0.003] ####
# ──────────────────────────────────────────────────────────────────────────────

cat("\n 4. Decision Tree TUNED (10-fold CV on cp) \n")

set.seed(67)
dt_tuned <- caret::train(
  has_debit_card ~ ., data = train,
  method = "rpart",
  trControl = ctrl_cv,
  tuneGrid = expand.grid(cp = seq(0.001, 0.003, length.out = 11)),
  metric = "ROC"
)

cat("\n===== Tuned DT – Best cp parameter =====\n")
print(dt_tuned$bestTune)

# Predictions (classes + probabilities)
dt_pred_tuned <- predict(dt_tuned, newdata = test)
dt_prob_tuned <- predict(dt_tuned, newdata = test, type = "prob")[, "Yes"]

# Confusion Matrix + Metrics
dt_cm_tuned <- confusionMatrix(dt_pred_tuned, test$has_debit_card, positive = "Yes")
dt_acc_tuned <- as.numeric(dt_cm_tuned$overall["Accuracy"])
dt_auc_tuned <- as.numeric(
  pROC::auc(
    pROC::roc(response = test$has_debit_card, predictor = dt_prob_tuned,
              levels = c("No", "Yes"))
  )
)

cat("\n===== DT Tuned – Performance =====\n")
cat(sprintf("ROC AUC = %.4f | Accuracy = %.4f\n", dt_auc_tuned, dt_acc_tuned))
plot_cm_dt(dt_cm_tuned, "Confusion Matrix – Decision Tree (Tuned)")

# Store metrics for later model comparison
metrics_dt_final <- c(ROC = dt_auc_tuned, Accuracy = dt_acc_tuned)

cat("\n===== Success: 4. DT TUNED completed =====\n")

# ──────────────────────────────────────────────────────────────────────────────
#### 5. Decision Tree: Interpretability ####
# ──────────────────────────────────────────────────────────────────────────────

cat("\n 5. DT INTERPRETABILITY: Variable Importance & Structure \n")

# Variable importance (top 12)
vi_dt <- caret::varImp(dt_tuned)$importance %>%
  rownames_to_column("Variable") %>%
  arrange(desc(Overall)) %>%
  slice_head(n = 12)

ggplot(vi_dt, aes(x = reorder(Variable, Overall), y = Overall)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(
    title = "Decision Tree (Tuned): Top Variables",
    x = "Variable", y = "Importance"
  ) +
  theme_minimal(base_size = 14)

# Plot tree structure using rpart.plot
rpart.plot(
  dt_tuned$finalModel,
  type = 2, under = TRUE, fallen.leaves = TRUE,
  cex = 0.4, split.cex = 0.9, tweak = 1.1, faclen = 12,
  branch.lty = 2, shadow.col = 0,
  main = "Decision Tree (Tuned): Structure & Node Visualization"
)

cat("\n===== Success: 5. DT INTERPRETABILITY complete =====\n")

# ──────────────────────────────────────────────────────────────────────────────
#### 6. Neural Network DEFAULT: size = 5, decay = 0 ####
# ──────────────────────────────────────────────────────────────────────────────

cat("\n 6. Neural Network DEFAULT (size = 5, decay = 0) \n")

set.seed(67)
nn_default <- caret::train(
  has_debit_card ~ ., data = train,
  method = "nnet",
  trControl = trainControl(method = "none", classProbs = TRUE),
  tuneGrid = data.frame(size = 5, decay = 0),
  preProcess = c("center", "scale"),   # z-score transformation
  trace = FALSE,
  MaxNWts = 5000,                      # default 1000 → allow for factor expansion
  maxit = 200                          # default 100 → more iterations for convergence
)

# Predictions (class + probability)
nn_pred_default <- predict(nn_default, newdata = test)
nn_prob_default <- predict(nn_default, newdata = test, type = "prob")[, "Yes"]

# Confusion matrix and metrics
nn_cm_default <- confusionMatrix(nn_pred_default, test$has_debit_card, positive = "Yes")
nn_acc_default <- as.numeric(nn_cm_default$overall["Accuracy"])
nn_auc_default <- as.numeric(
  pROC::auc(
    pROC::roc(response = test$has_debit_card,
              predictor = nn_prob_default,
              levels = c("No", "Yes"))
  )
)

cat("\n===== NN Default – Performance =====\n")
print(nn_cm_default)
cat(sprintf("ROC AUC = %.4f  |  Accuracy = %.4f\n", nn_auc_default, nn_acc_default))

# Visualize confusion matrix
plot_cm_nn(nn_cm_default, "Confusion Matrix – Neural Network (Default)")

cat("\n===== Success: 6. NN DEFAULT completed =====\n")

# ──────────────────────────────────────────────────────────────────────────────
#### 7. Neural Network TUNED: 10-Fold CV on size & decay ####
# ──────────────────────────────────────────────────────────────────────────────

cat("\n 7. Neural Network TUNED (10-fold CV on size & decay) \n")

set.seed(67)
nn_tuned <- caret::train(
  has_debit_card ~ ., data = train,
  method = "nnet",
  trControl = ctrl_cv,                           # defined in section 0.4
  tuneGrid = expand.grid(
    size  = 1:5,
    decay = 10^seq(-3, -1, length.out = 10)      # log-spaced grid
  ),
  preProcess = c("center", "scale"),
  metric = "ROC",
  trace = FALSE,
  MaxNWts = 5000,
  maxit = 200
)

cat("\n===== NN Tuned – Best Hyperparameters =====\n")
print(nn_tuned$bestTune)

# Predictions (class + probability)
nn_pred_tuned <- predict(nn_tuned, newdata = test)
nn_prob_tuned <- predict(nn_tuned, newdata = test, type = "prob")[, "Yes"]

# Confusion matrix and metrics
nn_cm_tuned <- confusionMatrix(nn_pred_tuned, test$has_debit_card, positive = "Yes")
nn_acc_tuned <- as.numeric(nn_cm_tuned$overall["Accuracy"])
nn_auc_tuned <- as.numeric(
  pROC::auc(
    pROC::roc(response = test$has_debit_card,
              predictor = nn_prob_tuned,
              levels = c("No", "Yes"))
  )
)

cat("\n===== NN Tuned – Performance =====\n")
print(nn_cm_tuned)
cat(sprintf("ROC AUC = %.4f  |  Accuracy = %.4f\n", nn_auc_tuned, nn_acc_tuned))

plot_cm_nn(nn_cm_tuned, "Confusion Matrix – Neural Network (Tuned)")

# Store final metrics
metrics_nn_final <- c(ROC = nn_auc_tuned, Accuracy = nn_acc_tuned)

cat("\n===== Success: 7. NN TUNED completed =====\n")

# ──────────────────────────────────────────────────────────────────────────────
#### 8. Neural Network: Interpretability ####
# ──────────────────────────────────────────────────────────────────────────────

cat("\n 8. NN INTERPRETABILITY: Visualize the final network \n")

# Plot neural-network architecture (inputs → hidden → output)
plotnet(
  nn_tuned$finalModel,
  alpha       = 0.6,
  circle_cex  = 1.3,
  cex_val     = 0.8,
  pos_col     = "green3",
  neg_col     = "red3",
  max_spread  = TRUE,
  node_labs   = TRUE,
  var_labs    = TRUE
)

# ──────────────────────────────────────────────────────────────────────────────
#### 9. Model Comparison: Decision Tree vs Random Forest vs Neural Network ####
# ──────────────────────────────────────────────────────────────────────────────

## Ensure RF metrics exist for fair comparison
rf_prob_tuned <- rf_pred_prob_opt
rf_cm_tuned   <- cm_opt

rf_acc_tuned <- as.numeric(rf_cm_tuned$overall["Accuracy"])
rf_auc_tuned <- as.numeric(
  pROC::auc(
    pROC::roc(response = data_test$has_debit_card,
              predictor = rf_prob_tuned,
              levels = c("No", "Yes"))
  )
)

# Fallbacks if RF not loaded
if (!exists("rf_acc_tuned")) rf_acc_tuned <- NA_real_
if (!exists("rf_auc_tuned")) rf_auc_tuned <- NA_real_
if (!exists("rf_cm_tuned"))  rf_cm_tuned  <- NULL

# ──────────────────────────────────────────────────────────────────────────────
#### 10. Extract Secondary Metrics from Confusion Matrices ####
# ──────────────────────────────────────────────────────────────────────────────

extract_cm_metrics <- function(cm_obj) {
  if (is.null(cm_obj)) {
    return(tibble(
      Accuracy     = NA_real_,
      Sensitivity  = NA_real_,
      Specificity  = NA_real_,
      B_Accuracy   = NA_real_
    ))
  }
  tibble(
    Accuracy     = as.numeric(cm_obj$overall["Accuracy"]),
    Sensitivity  = as.numeric(cm_obj$byClass["Sensitivity"]),
    Specificity  = as.numeric(cm_obj$byClass["Specificity"]),
    B_Accuracy   = as.numeric(cm_obj$byClass["Balanced Accuracy"])
  )
}

dt_sec <- extract_cm_metrics(dt_cm_tuned)
nn_sec <- extract_cm_metrics(nn_cm_tuned)
rf_sec <- extract_cm_metrics(rf_cm_tuned)

# ──────────────────────────────────────────────────────────────────────────────
#### 11. Pairwise Comparisons (DT vs RF, NN vs RF, DT vs NN) ####
# ──────────────────────────────────────────────────────────────────────────────

comp_dt_rf <- tibble(
  Metric       = c("ROC", "Accuracy"),
  DecisionTree = c(dt_auc_tuned, dt_acc_tuned),
  RandomForest = c(rf_auc_tuned, rf_acc_tuned)
)
cat("\n===== Decision Tree vs Random Forest =====\n")
print(comp_dt_rf)

comp_nn_rf <- tibble(
  Metric       = c("ROC", "Accuracy"),
  NeuralNet    = c(nn_auc_tuned, nn_acc_tuned),
  RandomForest = c(rf_auc_tuned, rf_acc_tuned)
)
cat("\n===== Neural Net vs Random Forest =====\n")
print(comp_nn_rf)

comp_dt_nn <- tibble(
  Metric       = c("ROC", "Accuracy"),
  DecisionTree = c(dt_auc_tuned, dt_acc_tuned),
  NeuralNet    = c(nn_auc_tuned, nn_acc_tuned)
)
cat("\n===== Decision Tree vs Neural Net =====\n")
print(comp_dt_nn)

cat("\n===== Success: 11. Pairwise comparisons complete =====\n")

# ──────────────────────────────────────────────────────────────────────────────
#### 12. Benchmarking of all Models ####
# ──────────────────────────────────────────────────────────────────────────────

all_metrics <- bind_rows(
  dt_sec %>% mutate(Model = "Decision Tree", ROC = dt_auc_tuned),
  rf_sec %>% mutate(Model = "Random Forest", ROC = rf_auc_tuned),
  nn_sec %>% mutate(Model = "Neural Net",    ROC = nn_auc_tuned)
) %>%
  select(Model, Accuracy, ROC, Sensitivity, Specificity, B_Accuracy) %>%
  pivot_longer(cols = -Model, names_to = "Metric", values_to = "Value")

# Color scheme and order
model_colors <- c(
  "Decision Tree" = "steelblue2",
  "Random Forest" = "plum2",
  "Neural Net"    = "olivedrab2"
)
metric_order <- c("ROC", "Accuracy", "B_Accuracy", "Sensitivity", "Specificity")
all_metrics$Metric <- factor(all_metrics$Metric, levels = metric_order)

cat("\n===== Plot Benchmark Bar Chart Across Models =====\n")

ggplot(all_metrics, aes(x = Metric, y = Value, fill = Model)) +
  geom_col(position = position_dodge(width = 0.8), width = 0.8) +
  scale_fill_manual(values = model_colors) +
  coord_cartesian(ylim = c(0.5, 0.85)) +
  labs(
    title = "Benchmarking Decision Tree vs Random Forest vs Neural Net",
    subtitle = "Primary: ROC & Accuracy • Secondary: Balanced Accuracy, Sensitivity, Specificity",
    x = "Metric", y = "Score", fill = "Model"
  ) +
  theme_minimal(base_size = 14)

cat("\n===== Success: 12. Benchmarking complete =====\n")

################################################################################
# End of Script 3
################################################################################
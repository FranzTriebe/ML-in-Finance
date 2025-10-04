################################################################################
# Task 3 - Alternative Models (Decision Tree vs ANN, Keras)
# Goal: Predict has_debit_card (Yes/No) 
# Author: Jonah    Seed: 67
################################################################################

################################################################################
# 0.1) Set-Up: Load packages, manage conflicts, set seed & helper funtions
################################################################################

# Install packages
install.packages(c(
  "tidyverse",   # data wrangling
  "tidymodels",  # recipes, models & metrics in a unified package
  "rpart",       # decision tree engine
  "keras"        # neural network backend
  )
)

# Load packages
library(tidyverse)
library(tidymodels)
library(rpart)
library(keras)

# Precautionary measure to manage conflicts of tidyverse with other packages
tidymodels_prefer()

# Set consistent seed to make randomness replicable
set.seed(67)

#Define helper function to return metrics to compare classification models
metric_set_cls <- metric_set(accuracy, roc_auc, precision, recall, f_meas)

################################################################################
# 0.2) Set-Up: Data Loading & Inspection
################################################################################

# Load pre-processed data
data <- readRDS("/Users/jonah-baptiste/Documents/2_Ausbildung/2_HSG_UdeSA/7_Sem_VWL/3_Machine Learning_Finance/0_3_Homework/ML-in-Finance/data/processed/df_final.rds")

# Keep only complete cases (small number of NA's justifies dropping them to use same data for Decision Tree & NN)
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


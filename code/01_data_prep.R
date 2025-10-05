#install packages

#load libraries
library(readr)
library(tidyr)
library(tidyverse)
library(dplyr)
library(skimr)
library(ggcorrplot)

#import dataset
df <- read_csv("~/OneDrive - Universität St.Gallen/Semester 5/Machine Learning/Project/micro_ind.csv")

#select possible variables to use
df_select <- df %>% select(female, 
                    age, 
                    educ, 
                    inc_q, 
                    emp_in, 
                    urbanicity_f2f, 
                    account_fin, 
                    account_mob, 
                    fin2, 
                    fin4, 
                    fin9, 
                    fin10, 
                    fin14_1, 
                    fin14a, 
                    fin14a1, 
                    fin14b, 
                    fin16, 
                    fin17a, 
                    fin17b, 
                    fin20, 
                    fin22a, 
                    fin22b, 
                    fin24, 
                    fin26, 
                    fin28, 
                    fin30, 
                    fin37, 
                    fin38, 
                    fin42, 
                    fin44a, 
                    fin44b, 
                    fin44c, 
                    fin44d, 
                    fin45_1, 
                    saved, 
                    borrowed, 
                    receive_wages, 
                    receive_transfers, 
                    receive_pension, 
                    receive_agriculture, 
                    pay_utilities, 
                    remittances, 
                    mobileowner, 
                    internetaccess, 
                    anydigpayment, 
                    merchantpay_dig)

#rename the variables
df_select_renamed <- df_select %>% rename( 
  income_q         = "inc_q",          #within-economy household income quantile
  employed         = "emp_in",         #respondent is in workforce
  urban            = "urbanicity_f2f", #respondant lives in rural area
  has_debit_card   = "fin2",           #has a debit card
  used_debit_card  = "fin4",           #used a debit card
  deposited        = "fin9",           #made any deposit into the account
  withdrew         = "fin10",          #withdrew from the account
  mob_instore_pay  = "fin14_1",        #used mobile phone to pay for a purchase in-store
  bill_paid_int    = "fin14a",         #made bill payments 
  sent_money_int   = "fin14a1",        #sent money to relative or friend using internet
  bought_on_int    = "fin14b",         #bought something online using internet
  saved_old_age    = "fin16",          #saved for old age
  saved_using_acc  = "fin17a",         #saved using account at financial institution
  saved_inf_club   = "fin17b",         #saved using informal savings club
  borrow_med       = "fin20",          #borrowed for medical purposes
  borrow_fin       = "fin22a",         #borrowed from financial institution
  borrow_friends   = "fin22b",         #borrowed from family or friends 
  source_emergency = "fin24",          #main source of emergency fund in 30 days
  sent_dom_rem     = "fin26",          #sent domestic remittances
  received_dom_rem = "fin28",          #received domestic remittances
  paid_ut_bill     = "fin30",          #paid a utility bill
  rec_gov_transfer = "fin37",          #received a government transfer (binary)
  rec_gov_pension  = "fin38",          #received a government pension (binary)
  rec_agri_payment = "fin42",          #received an agricultural payment (binary)
  fin_worried_old  = "fin44a",         #financially worried: old age 
  fin_worried_med  = "fin44b",         #financially worried: medical costs
  fin_worried_bil  = "fin44c",         #financially worried: bills
  fin_worried_edu  = "fin44d",         #financially worried: education
  fin_worried_Cov  = "fin45_1")        #financially worried: education


#filter rows based on condition that account_fin == 1
df_filtered <- df_select_renamed %>% 
                  filter(account_fin == 1)

#select the predictors we want to use (information that the government has)
df_final <- df_filtered %>% select(female,            #known; binary
                                   age,               #known; numeric
                                   educ,              #known; ordinal scale
                                   urban,             #known; binary
                                   income_q,          #known; ordinal scale
                                   employed,          #known; binary
                                   rec_gov_transfer,  #known; binary
                                   rec_gov_pension,   #known; binary
                                   rec_agri_payment,  #known; binary
                                   paid_ut_bill,      #known if state-issued; binary
                                   internetaccess,    #maybe known; binary
                                   mobileowner,       #maybe known; binary
                                   has_debit_card)    #output variable; binary

#get overview of data for cleaning
skim(df_final)

#create function to get min and max values of variable
variable_summary <- function(data, var) {
  data %>%
    summarise(
      min_value = min({{ var }}, na.rm = TRUE),
      max_value = max({{ var }}, na.rm = TRUE),
      num_NAs = sum(is.na({{ var }}))
    )
}

#get min and max and understand how NAs are shown
variable_summary(df_final, female)            #no NAs 
variable_summary(df_final, age)               #no NAs
variable_summary(df_final, educ)              #11 NAs (shown as 4s and 5s)
variable_summary(df_final, urban)             #no NAs
variable_summary(df_final, income_q)          #no NAs
variable_summary(df_final, employed)          #no NAs
variable_summary(df_final, rec_gov_transfer)  #20 NAs (shown as 3s and 4s)
variable_summary(df_final, rec_gov_pension)   #4 NAs (shown as 3s and 4s)
variable_summary(df_final, rec_agri_payment)  #14 NAs (shown as 3s and 4s)
variable_summary(df_final, paid_ut_bill)      #12 NAs (shown as 3s and 4s)
variable_summary(df_final, internetaccess)    #24 NAs (shown as 3s)
variable_summary(df_final, mobileowner)       #2 NAs (shown as 3s)
variable_summary(df_final, has_debit_card)    #35 NAs (shown as 3s and 4s)

#clean binary data and deal with missing values
df_final <- df_final %>%
  mutate(female = ifelse(female == 2, 0, female)) %>%.                               #set binary 0 or 1
  mutate(educ = ifelse(educ > 3, NA, educ)) %>%                                      #set 4s and 5s to NA; set binary to 0 or 1
  mutate(urban = ifelse(urban == 2, 1, 0)) %>%                                       #set binary 0 or 1
  mutate(employed = ifelse(employed == 2, 0, employed)) %>%.                         #set binary 0 or 1
  mutate(rec_gov_transfer = ifelse(rec_gov_transfer > 2, NA, rec_gov_transfer)) %>%  #set 3s and 4s to NA
  mutate(rec_gov_transfer = ifelse(rec_gov_transfer == 2, 0, rec_gov_transfer)) %>%  #set binary to 0 or 1
  mutate(rec_gov_pension = ifelse(rec_gov_pension > 2, NA, rec_gov_pension)) %>%     #set 3s and 4s to NA
  mutate(rec_gov_pension = ifelse(rec_gov_pension == 2, 0, rec_gov_pension)) %>%     #set binary to 0 or 1
  mutate(rec_agri_payment = ifelse(rec_agri_payment > 2, NA, rec_agri_payment)) %>%. #set 3s and 4s to NA
  mutate(rec_agri_payment = ifelse(rec_agri_payment == 2, 0, rec_agri_payment)) %>%. #set binary to 0 or 1
  mutate(paid_ut_bill = ifelse(paid_ut_bill > 2, NA, paid_ut_bill)) %>%              #set 3s and 4s to NA
  mutate(paid_ut_bill = ifelse(paid_ut_bill == 2, 0, paid_ut_bill)) %>%              #set binary to 0 or 1
  mutate(internetaccess = ifelse(internetaccess > 2, NA, internetaccess)) %>%        #set 3s to NA
  mutate(internetaccess = ifelse(internetaccess == 2, 0, internetaccess)) %>%        #set binary to 0 or 1
  mutate(mobileowner = ifelse(mobileowner > 2, NA, mobileowner)) %>%                 #set 3s to NA
  mutate(mobileowner = ifelse(mobileowner == 2, 0, mobileowner)) %>%                 #set binary to 0 or 1
  mutate(has_debit_card = ifelse(has_debit_card > 2, NA, has_debit_card)) %>%        #set 3s and 4s to NA
  mutate(has_debit_card = ifelse(has_debit_card == 2, 0, has_debit_card))            #set binary to 0 or 1
  
  
#convert all variables to integers instead of floating point numbers
df_final <- df_final %>%
    mutate(across(everything(), as.integer))

#drop observations with missing data in the target variable
df_final <- df_final %>%
  filter(!is.na(has_debit_card))

########################summary statistics################################

dim(df_final)                                 #number of observations and variables
mean(df_final$female, na.rm = TRUE)           #ratio of female
mean(df_final$age, na.rm = TRUE)              #mean of age
min(df_final$age, na.rm = TRUE)               #minimum of age
max(df_final$age, na.rm = TRUE)               #maximum of age
mean(df_final$income_q, na.rm = TRUE)         #mean of income quintile
min(df_final$income_q, na.rm = TRUE)          #minimum of income quintile
max(df_final$income_q, na.rm = TRUE)          #maximum of income quintile

######################proportion barplots#################################

#convert age and income to binary factor groups
df_barplot <- df_final %>%
  mutate(
    age_group = factor(ifelse(age < 25, "under_25", "25_and_above"),
                       levels = c("under_25", "25_and_above")),
    income_group = factor(ifelse(income_q < 3, "under_3", "3_and_above"),
                          levels = c("under_3", "3_and_above"))
  )

#list of other binary columns to convert
binary_cols <- c("female", "urban", "employed", 
                 "rec_gov_transfer", "rec_gov_pension", "rec_agri_payment",
                 "paid_ut_bill", "internetaccess", "mobileowner")

#convert binary numeric columns to factors with labels
df_barplot <- df_barplot %>%
  mutate(across(all_of(binary_cols),
                ~ factor(.x, levels = c(0, 1), labels = c("No", "Yes"))))

#also convert has_debit_card to factor for the x-axis
df_barplot <- df_barplot %>%
  mutate(has_debit_card = factor(has_debit_card, levels = c(0, 1), labels = c("No", "Yes")))

#define the variables to plot
vars <- c(binary_cols, "age_group", "income_group")

#loop to create and print each percent-stacked bar plot
plots <- map(vars, function(v) {
  ggplot(df_barplot, aes(x = has_debit_card, fill = .data[[v]])) +
    geom_bar(position = "fill") +
    scale_y_continuous(labels = scales::percent) +
    labs(title = v, x = "has_debit_card", y = "percentage", fill = v) +
    theme_minimal()
})

#print each plot
walk(plots, print)

#######################correlation matrix ###########################

#compute and order correlation matrix
corr_mat <- cor(df_final, use = "pairwise.complete.obs")
ord <- hclust(as.dist(1 - abs(corr_mat)))$order
vars <- rownames(corr_mat)[ord]
corr_ord <- corr_mat[vars, vars]

#long form and plot
corr_ord %>%
  as.data.frame() %>%
  rownames_to_column("row") %>%
  pivot_longer(-row, names_to = "col", values_to = "value") %>%
  mutate(row = factor(row, levels = vars),
         col = factor(col, levels = vars)) %>%
  filter(as.integer(row) >= as.integer(col)) %>%
  ggplot(aes(x = col, y = row, fill = value)) +
  geom_tile(color = "grey80") +
  geom_text(aes(label = sprintf("%.2f", value)), size = 3) +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red", limits = c(-1, 1)) +
  coord_equal() +
  labs(title = "Correlation matrix", x = NULL, y = NULL) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#barplot of absolute correlations
target_col <- "has_debit_card"
predictors <- setdiff(names(df_final), target_col)

cor_with_target <- 
  tibble(
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
    title = "Top predictors by absolute Pearson correlation with has_debit_card",
    x = "|correlation|",
    y = NULL
  ) +
  theme_minimal()

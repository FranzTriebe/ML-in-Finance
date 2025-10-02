#loading the data 
data <- read.csv("data/processed/df_final.csv")

#looking at the data in a descriptive way
summary(data)
str(data)

#making a binary factor out of binary variables stored as integer
data$female <- factor(data$female, levels = c(0,1), labels = c("male","female"))
data$urban <- factor(data$urban, levels = c(0,1), labels = c("rural","urban"))
data$employed <- factor(data$employed, levels = c(0,1), labels = c("unemployed","employed"))
data$rec_gov_transfer <- factor(data$rec_gov_transfer, levels = c(0,1), labels = c("No","Yes"))
data$rec_gov_pension <- factor(data$rec_gov_pension, levels = c(0,1), labels = c("No","Yes"))
data$rec_agri_payment <- factor(data$rec_agri_payment, levels = c(0,1), labels = c("No","Yes"))
data$paid_ut_bill <- factor(data$paid_ut_bill, levels = c(0,1), labels = c("No","Yes"))
data$internetaccess <- factor(data$internetaccess, levels = c(0,1), labels = c("No","Yes"))
data$mobileowner <- factor(data$mobileowner, levels = c(0,1), labels = c("No","Yes"))
data$has_debit_card <- factor(data$has_debit_card, levels = c(0,1), labels = c("No","Yes"))

#making ordinary factors out of factors stored as integer (ordered = TRUE because there is a logical order)
data$educ <- factor(data$educ, levels = c(1,2,3), labels = c("primary_or_less","secondary","tertiary_or_more"), ordered = TRUE)
data$income_q <- factor(data$income_q, level = c(1,2,3,4,5), labels = c("poorest_20","second_20","middle_20","fourth_20","richest_20"), ordered = TRUE)


################################################################################
#                   INSTALL AND IMPORT REQUIRED PACKAGES
################################################################################
knitr::opts_chunk$set(echo = TRUE)

if(!require(formatR)) install.packages("formatR", repos = "http://cran.us.r-project.org")
if(!require(tinytex)) tinytex::install_tinytex()
if(!require(kableExtra)) install.packages("kableExtra", repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages('corrplot', repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(xgboost)) install.packages("xgboost", repos = "http://cran.us.r-project.org")

library(tidyverse, warn.conflicts=F, quietly=T)
library(caret, warn.conflicts=F, quietly=T)
library(data.table, warn.conflicts=F, quietly=T)
library(dplyr, warn.conflicts = F, quietly=T)
library(kableExtra, warn.conflicts = F, quietly=T)
library(formatR, warn.conflicts=F, quietly=T)
library(knitr, warn.conflicts=F, quietly=T)
library(corrplot)
library(rpart)
library(xgboost)

################################################################################
#                           DOWNLOAD DATASET
################################################################################
url <- "https://raw.githubusercontent.com/ushaakumaar/diabetes_prediction/main/dataset/diabetes.csv"
diabetes_df <- fread(url)
head(as_tibble(diabetes_df))


################################################################################
#               SPLIT DATASET INTO TRAIN AND TEST DATASETS
################################################################################
# Training dataset will be 10% of Diabetes data
set.seed(1)
test_index <- createDataPartition(y = diabetes_df$Outcome, times = 1, p = 0.1, list = FALSE)
train_set <- diabetes_df[-test_index,]
test_set <- diabetes_df[test_index,]

str(train_set)
str(test_set)

################################################################################
#                           DATA EXPLORATION
################################################################################

# Print dataset structure
str(diabetes_df)

# Print Summary Statistics
summary(diabetes_df)

# Find missing data
total_rows <- diabetes_df %>% nrow()
glucose_missing_percent <- diabetes_df %>% filter(Glucose == 0) %>% nrow()/total_rows*100
bp_missing_percent <- diabetes_df %>% filter(BloodPressure == 0) %>% nrow()/total_rows*100
skinthickness_missing_percent <- diabetes_df %>% filter(SkinThickness == 0) %>% nrow()/total_rows*100
insulin_missing_percent <- diabetes_df %>% filter(Insulin == 0) %>% nrow()/total_rows*100
bmi_missing_percent <- diabetes_df %>% filter(BMI == 0) %>% nrow()/total_rows*100

# Print the percentage of variable having missing data
results <- tibble(Variable = "Glucose - Missing Percentage", Percentage = glucose_missing_percent)
results <- bind_rows(results, tibble(Variable = "BloodPressure - Missing Percentage", Percentage = bp_missing_percent))
results <- bind_rows(results, tibble(Variable = "SkinThickness - Missing Percentage", Percentage = skinthickness_missing_percent))
results <- bind_rows(results, tibble(Variable = "Insulin - Missing Percentage", Percentage = insulin_missing_percent))
results <- bind_rows(results, tibble(Variable = "BMI - Missing Percentage", Percentage = bmi_missing_percent))
results %>%
  kbl(caption="Percentage of Variable having Missing data", digits=4) %>%
  kable_classic_2(full_width = T, latex_options = "hold_position")

################################################################################
#                           MEDIAN IMPUTATION
################################################################################

# Find median values in training set
median_glucose <- as.integer(median(train_set$Glucose))
median_bp <- as.integer(median(train_set$BloodPressure))
median_bmi <- median(train_set$BMI)

# Replace missing data in training set with Median
train_set$Glucose[train_set$Glucose == 0] <- median_glucose
train_set$BloodPressure[train_set$BloodPressure == 0] <- median_bp
train_set$BMI[train_set$BMI == 0] <- median_bmi

# Replace missing data in test set with Median
test_set$Glucose[test_set$Glucose == 0] <- median_glucose
test_set$BloodPressure[test_set$BloodPressure == 0] <- median_bp
test_set$BMI[test_set$BMI == 0] <- median_bmi

# Print summary statistics
summary(train_set)
summary(test_set)

################################################################################
#                       FEATURE CORRELATION MATRIX
################################################################################

# Print correlation matrix
corrplot(cor(diabetes_df), 
         main="\n\nFeature Correlation Matrix", 
         order = "hclust", 
         tl.col = "black", 
         tl.srt=45, 
         tl.cex=0.8, 
         cl.cex=0.8)
box(which = "outer", lty = "solid")

barplot(table(diabetes_df$Outcome), main="Class Distribution", xlab="Outcome")

################################################################################
#                        MULTI LINEAR REGRESSION MODEL
################################################################################

# Create multi linear regression model
mlr_model <- lm(Outcome ~ . - SkinThickness - Insulin, data=train_set)

# Print the model summary statistics
summary(mlr_model)

# Predict outcome in the testing set
mlr_prediction <- predict(mlr_model,newdata=test_set,type='response')
mlr_prediction <- ifelse(mlr_prediction > 0.5,1,0)

# Print confusion matrix
(confusion_matrix_mlr<-table(mlr_prediction, test_set$Outcome))

# Find model accuracy
mlr_accuracy <- mean(mlr_prediction == test_set$Outcome)

# Print the results
results <- tibble(Method = "Model 1: Multi Linear Regression Model", Accuracy = mlr_accuracy)
results %>% 
  kbl(caption="MODEL PERFORMANCE RESULTS", digits=4) %>%
  kable_classic_2(full_width = T, latex_options = "hold_position")

################################################################################
#                       LOGISTIC REGRESSION MODEL
################################################################################

# Create the model
logistic_reg_model <- glm(Outcome ~ . - SkinThickness - Insulin, data=train_set, family="binomial")

# Print the model summary statistics
summary(logistic_reg_model)

# Predict outcome in the testing set
logistic_reg_prediction <- predict(logistic_reg_model,newdata=test_set,type='response')
logistic_reg_prediction <- ifelse(logistic_reg_prediction > 0.5,1,0)

# Print confusion matrix
(confusion_matrix_logistic_reg<-table(logistic_reg_prediction, test_set$Outcome))

# Find model accuracy
logistic_reg_accuracy <- mean(logistic_reg_prediction == test_set$Outcome)

# Print the results
results <- bind_rows(results, tibble(Method = "Model 2: Logistic Regression", Accuracy = logistic_reg_accuracy))
results %>% 
  kbl(caption="MODEL PERFORMANCE RESULTS", digits=4) %>%
  kable_classic_2(full_width = T, latex_options = "hold_position")

################################################################################
#                           DECISION TREE MODEL
################################################################################

# Create decision tree model
decision_tree_model <- rpart(Outcome ~ . - SkinThickness - Insulin, data=train_set,method="class")

# Plot the tree
plot(decision_tree_model, uniform=TRUE, main="Classification Tree for Diabetes")

text(decision_tree_model, use.n=TRUE, all=TRUE, cex=.8)

box(which = "outer", lty = "solid")

# Predict outcome in the testing set
decision_tree_prediction <- predict(decision_tree_model, test_set, type = 'class')
(confusion_matrix_decision_tree<-table(decision_tree_prediction, test_set$Outcome))

# Find model accuracy
decision_tree_accuracy <- mean(decision_tree_prediction == test_set$Outcome)

# Print the results
results <- bind_rows(results, tibble(Method = "Model 3: Decision Tree", Accuracy = decision_tree_accuracy))
results %>% 
  kbl(caption="MODEL PERFORMANCE RESULTS", digits=4) %>%
  kable_classic_2(full_width = T, latex_options = "hold_position")

################################################################################
#                             XGBOOST MODEL
################################################################################

# Define predictor and response variables in training set
train_x = data.matrix(train_set[,-9])
train_y = as.matrix(train_set[,9])

# Define predictor and response variables in testing set
test_x = data.matrix(test_set[, -9])
test_y = as.matrix(test_set[, 9])

# Define final training and testing sets
xgb_train = xgb.DMatrix(data = train_x, label = train_y)
xgb_test = xgb.DMatrix(data = test_x, label = test_y)

# Define watchlist
watchlist = list(train=xgb_train, test=xgb_test)

# Fit XGBoost model and display training and testing data at each round
model = xgb.train(data = xgb_train, max.depth = 3, watchlist=watchlist, nrounds = 70)

# Define final model
final_xgboost = xgboost(data = xgb_train, max.depth = 3, nrounds = 24, verbose = 0)

# Predict the outcome on testing set
xgboost_prediction <- predict(final_xgboost, xgb_test)
xgboost_prediction <- ifelse(xgboost_prediction > 0.5,1,0)

# Find model accuracy
xgboost_accuracy <- mean(xgboost_prediction == test_set$Outcome)

# Print the results
results <- bind_rows(results, tibble(Method = "Model 4: XGBoost", Accuracy = xgboost_accuracy))
results %>% 
  kbl(caption="MODEL PERFORMANCE RESULTS", digits=4) %>%
  kable_classic_2(full_width = T, latex_options = "hold_position")

################################################################################
#                           MACHINE LEARNING MODEL RESULTS
################################################################################

# Print the results
results %>% 
  kbl(caption="MODEL PERFORMANCE RESULTS", digits=4) %>%
  kable_classic_2(full_width = T, latex_options = "hold_position")


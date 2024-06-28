###                          Master Thesis:                      ###
## Evaluation of structural factors influencing Machine Learning Performance ##

library(dplyr)
library(tidyverse)
library(reshape) # melt function for long data 
library(corrplot) # plotting package for matrix
library(ggplot2) # plotting package
library(corpcor) # partial correlation from correlation matrix
library(caret) # ML package
library(e1071) # ML package
library(MASS) # multivariate simulation etc. 
library(stats) # standard stats functions
library(rosetta) # scattermatrix plots
library(VGAM) # logistic regression inside caret
library(randomForest) # rf for insidde caret
library(plotly) # for scatterMatrix
library(GGally) # for scatterMatrix
library(Matrix) # for positive definite matrix function
library(patchwork)

 


                    ### 1. Generate Linear Data  ###

# 1.1 Function that generates linear data with adjustable correlation ---------# 

# The function  takes two correlations (with outcome and between predictors)
generate_data <- function(n, p, correlation, intracorrelation, 
                          min_mean = 3, max_mean = 5, 
                          min_sd = 1, max_sd = 2) {
  # Generate means
  means_vector <- runif(p, min = min_mean, max = max_mean)
  # Create correlation matrix
  cor_m <- matrix(correlation, nrow = p, ncol = p)
  cor_m[-nrow(cor_m), -ncol(cor_m)] <- intracorrelation
  diag(cor_m) <- 1
  # Ensure positive definiteness of the correlation matrix
  cor_m <- nearPD(cor_m)
  # Generate standard deviations
  stdevs <- runif(p, min = min_sd, max = max_sd)
  # Create covariance matrix using Cholesky decomposition
  chol_matrix <- chol(cor_m$mat)
  cov_m <- chol_matrix %*% diag(stdevs) %*% t(chol_matrix)
  # Generate data from multivariate normal distribution
  data <- mvrnorm(n = n, mu = means_vector, Sigma = cov_m, empirical = TRUE)
  data <- pmax(data, 0)
  variable_names <- paste("pred", 1:(p-1), sep = "")
  colnames(data) <- c(variable_names, "outcome")
  data <- as.data.frame(data)
  data$outcome <- ifelse(data$outcome > mean(data$outcome), 1, 0)
  data$outcome <- as.factor(data$outcome)
  data <- as.data.frame(data)
  return(data)
}


# 1.2 We generate linear data sets: 2 effect sizes at different sample sizes

set.seed(1271)
df_100_01 <- generate_data(n = 100, p = 5, correlation = 0.25, intracorrelation = 0) 
scatterMatrix(df_100_01) # at p = 5 cor 21, 09, 17, 19 

# set.seed(1012)
# df_100_03 <- generate_data(n = 100, p = 5, correlation = 0.45, intracorrelation = 0)
set.seed(1230)
df_100_05 <- generate_data(n = 100, p = 5, correlation = 0.60, intracorrelation = 0)
scatterMatrix(df_100_05) # at p = 5 cor 38, 38, 39, 38

set.seed(1017)
df_200_01 <- generate_data(n = 200, p = 5, correlation = 0.25, intracorrelation = 0) 
scatterMatrix(df_200_01) # at p = 5 cor 2, 2, 15, 2,

# set.seed(1012)
# df_200_03 <- generate_data(n = 200, p = 5, correlation = 0.45, intracorrelation = 0)

set.seed(1017)
df_200_05 <- generate_data(n = 200, p = 5, correlation = 0.6, intracorrelation = 0)
scatterMatrix(df_200_05) # at p = 5 cor 4, 39, 4, 35

set.seed(1013)
df_500_01 <- generate_data(n = 500, p = 5, correlation = 0.25, intracorrelation = 0)
scatterMatrix(df_500_01) # at p = 5 cor 19, 16, 2, 2

# set.seed(1014)
# df_500_03 <- generate_data(n = 500, p = 5, correlation = 0.40, intracorrelation = 0)

set.seed(2021) # 1016 is a good seed
df_500_05 <- generate_data(n = 500, p = 5, correlation = 0.6, intracorrelation = 0)
scatterMatrix(df_500_05) # at p = 5 cor .4, .4 , .42, .39

set.seed(1014)
df_1000_01 <- generate_data(n = 1000, p = 5, correlation = 0.25, intracorrelation = 0)
scatterMatrix(df_1000_01) # at p = 5 cor .2, .2, .22, .2
# set.seed(6021)
# df_1000_03 <- generate_data(n = 1000, p = 5, correlation = 0.40, intracorrelation = 0)
set.seed(6022)
df_1000_05 <- generate_data(n = 1000, p = 5, correlation = 0.61, intracorrelation = 0)
scatterMatrix(df_1000_05) # cor .39, .38, .41 .44 

# set.seed(1014)
# df_2000_01 <- generate_data(n = 2000, p = 5, correlation = 0.27, intracorrelation = 0)
# set.seed(106)
# df_2000_03 <- generate_data(n = 2000, p = 5, correlation = 0.42, intracorrelation = 0)
# set.seed(2022)
# df_2000_05 <- generate_data(n = 2000, p = 5, correlation = 0.6, intracorrelation = 0)

# Ensure that no value is lower than 0
apply(df_100_01, 2, min) 

# Inspect original linear data frames and  distributions

# scatterMatrix(df_100_03) # at p = 5 cor ca .3
# scatterMatrix(df_200_03) # at p = 5 cor ca .3
# scatterMatrix(df_500_03) # at p = 5 cor ca .31
# catterMatrix(df_1000_03) # at p = 5 cor ca 31
# scatterMatrix(df_2000_01) # at p = 5 cor ca .21
# scatterMatrix(df_2000_03) # at p = 5 cor ca .31
# scatterMatrix(df_2000_05) # at p = 5 cor ca .41


                         ### 2. Nonlinear Data ###

# 2.1 Function that generates a "weak" nonlinear pattern

nonlinear_data_weak <- function(n, v, p) {
  # Generate predictor variables
  x <- matrix(NA, nrow = n, ncol = v)
  for (i in 1:v) {
    x[, i] <- seq(2, 3 * pi, length.out = n) + rnorm(n, 0, 0.3)  # Generate sine-shaped predictors
  }
  # Generate outcome variable based on predictors
  amplitude <- 0.33 # in strong function, amplitude is = .7
  phases <- rep(pi / 1.4, v)  # in strong function value is 1.4  
  outcome <- apply(x, 1, function(row) {
    sum(amplitude * sin(row + phases))
  })
  # Add some noise to the outcome variable if desired
  noise <- rnorm(n, mean = 0, sd = 0.3)
  if (p == 1) {
    random_variables <- rnorm(n, 4, 0.8)
  }
  else if (p == 2) {
    p1 <- rnorm(n, 4, 0.5)
    p2 <- rnorm(n, 4, 0.6)
    random_variables <- data.frame(p1, p2)
  }
  else if (p == 3) {
    p1 <- rnorm(n, 4, 0.8)
    p2 <- rnorm(n, 4, 0.8)
    p3 <- rnorm(n, 4, 0.8)
    random_variables <- data.frame(p1, p2, p3)
  }
  outcome <- outcome + noise
  # Create dataframe with predictors and outcome
  data <- as.data.frame(cbind(x, random_variables, Outcome = outcome))
  colnames(data) <- c(paste0("Pred", 1:(ncol(data)-1)), "outcome")
  data$outcome <- ifelse(data$outcome > mean(data$outcome), 1, 0)
  data$outcome <- as.factor(data$outcome)
  return(data)
}

# 2.2 function that generates a strong nonlinear pattern
nonlinear_data_strong <- function(n, v, p) {
  # Generate predictor variables
  x <- matrix(NA, nrow = n, ncol = v)
  for (i in 1:v) {
    x[, i] <- seq(2, 3 * pi, length.out = n) + rnorm(n, 0, 0.3)  # Generate sine-shaped predictors
  }
  # Generate outcome variable based on predictors
  amplitude <- 0.8
  phases <- rep(pi / 1.4, v)  # Phases for predictors (e.g., pi/1.5)
  outcome <- apply(x, 1, function(row) {
    sum(amplitude * sin(row + phases))
  })
  # Add some noise to the outcome variable if desired
  noise <- rnorm(n, mean = 0, sd = 0.3)
  if (p == 1) {
    random_variables <- rnorm(n, 4, 0.8)
  }
  else if (p == 2) {
    p1 <- rnorm(n, 4, 0.5)
    p2 <- rnorm(n, 4, 0.6)
    random_variables <- data.frame(p1, p2)
  }
  else if (p == 3) {
    p1 <- rnorm(n, 4, 0.8)
    p2 <- rnorm(n, 4, 0.8)
    p3 <- rnorm(n, 4, 0.8)
    random_variables <- data.frame(p1, p2, p3)
  }
  outcome <- outcome + noise
  # Create dataframe with predictors and outcome
  data <- as.data.frame(cbind(x, random_variables, Outcome = outcome))
  colnames(data) <- c(paste0("Pred", 1:(ncol(data)-1)), "outcome")
  data$outcome <- ifelse(data$outcome > mean(data$outcome), 1, 0)
  data$outcome <- as.factor(data$outcome)
  return(data)
}

#--------- 2.3 Generate nonlinear data: strong and weak effect sizes ------#

# Effect size strong
set.seed(1015)
nl_100_strong <- nonlinear_data_strong(100, 1, 2)
scatterMatrix(nl_100_strong) 

set.seed(1016)
nl_200_strong <- nonlinear_data_strong(200, 1, 2) 
scatterMatrix(nl_200_strong) 

set.seed(1012)
nl_500_strong <- nonlinear_data_strong(500, 1, 2)
scatterMatrix(nl_500_strong) 

set.seed(1413)
nl_1000_strong <- nonlinear_data_strong(1000, 1, 2)
scatterMatrix(nl_1000_strong) 

# effect size weak 

set.seed(1015)
nl_100_weak <- nonlinear_data_weak(100, 1, 2)
scatterMatrix(nl_100_weak) 

set.seed(1016)
nl_200_weak <- nonlinear_data_weak(200, 1, 2) 
scatterMatrix(nl_200_weak) 


set.seed(1012)
nl_500_weak <- nonlinear_data_weak(500, 1, 2)
scatterMatrix(nl_500_weak) 

set.seed(1413)
nl_1000_weak <- nonlinear_data_weak(1000, 1, 2)
scatterMatrix(nl_1000_weak) 


# 2.4 additional inspection and comparison of transformed distributions
histogram(df_nl_500_03[, 2])
histogram(df_500_03[, 1])
histogram(df_nl_500_05[, 2])
histogram(df_500_05[, 1])
histogram(df_nl_1000_03[, 2])
histogram(df_1000_03[, 1])
histogram(df_nl_1000_05[, 2])  
histogram(df_1000_03[, 1])





                   ###        3. Interactions        ###
# 3.1 Different functions to create the interaction 
# (interaction function, data generating function, and binarization function

interaction <- function(df, n, noise) {
  variables <- ncol(df - 1)
  for (i in 1:variables) {
    df[, i] <- scale(df[, i], center = TRUE, scale = FALSE)
  }
  interaction <- apply(df, 1, prod)
  model <- lm(interaction ~ ., data = df)
  residuals <- model$residuals
  interaction <- scale(interaction, center = TRUE, scale = TRUE) # z-transform
  # product (interaction) is stored as new column 
  df <- cbind(df, interaction)
  # outcome is defined + some noise + the interaction to remove predictor-outcome correlation
  outcome <- residuals + (noise * rnorm(n, 0, 1))  # or multiply with a new vector
  # residuals plus noise is stored as new outcome 
  df <- cbind(df, outcome)
  colnames(df) <- c(paste0("pred", 1:(ncol(df)-2)), "interaction", "outcome")
  return(df)
}

generate_data_2 <- function(n, p, correlation, intracorrelation, 
                          min_mean = 3, max_mean = 5, 
                          min_sd = 1, max_sd = 2) {
  # Generate means
  means_vector <- runif(p, min = min_mean, max = max_mean)
  # Create correlation matrix
  cor_m <- matrix(correlation, nrow = p, ncol = p)
  cor_m[-nrow(cor_m), -ncol(cor_m)] <- intracorrelation
  diag(cor_m) <- 1
  # Ensure positive definiteness of the correlation matrix
  cor_m <- nearPD(cor_m)
  # Generate standard deviations
  stdevs <- runif(p, min = min_sd, max = max_sd)
  # Create covariance matrix using Cholesky decomposition
  chol_matrix <- chol(cor_m$mat)
  cov_m <- chol_matrix %*% diag(stdevs) %*% t(chol_matrix)
  # Generate data from multivariate normal distribution
  data <- mvrnorm(n = n, mu = means_vector, Sigma = cov_m, empirical = TRUE)
  data <- pmax(data, 0)
  variable_names <- paste("pred", 1:(p-1), sep = "")
  colnames(data) <- c(variable_names, "outcome")
  data <- as.data.frame(data)
  return(data)
}

# binarized outcome variable
binarize_outcome_2 <- function(data) {
  # Dichotomize outcome variable
  data$outcome <- ifelse(data$outcome > mean(data$outcome), 1, 0)
  data$outcome <- as.factor(data$outcome)
  return(data)
}



# 3.2 Create 2-way interactions with strong effect size 

set.seed(2024)
df_100_strong <- generate_data_2(n = 100, p = 2, correlation = 0, intracorrelation = 0) 
int_100_strong <- interaction(df_100_strong, n = 100, noise = 2.5) 
int_100_strong <- binarize_outcome_2(int_100_strong)
scatterMatrix(int_100_strong) # cor .41

set.seed(1017)
df_200_strong <- generate_data_2(n = 200, p = 2, correlation = 0, intracorrelation = 0) 
int_200_strong <- interaction(df_200_strong, n = 200, noise = 2.5) # 3
int_200_strong <- binarize_outcome_2(int_200_strong)
scatterMatrix(int_200_strong) # .41

set.seed(1017)
df_500_strong <- generate_data_2(n = 500, p = 2, correlation = 0, intracorrelation = 0) 
int_500_strong <- interaction(df_500_strong, n = 500, noise = 2.0) 
int_500_strong <- binarize_outcome_2(int_500_strong)
scatterMatrix(int_500_strong) # .4

set.seed(1017)
df_1000_strong <- generate_data_2(n = 1000, p = 2, correlation = 0, intracorrelation = 0) 
int_1000_strong <- interaction(df_1000_strong, n = 1000, noise = 2.5)
int_1000_strong <- binarize_outcome_2(int_1000_strong)
scatterMatrix(int_1000_strong) # .41

# 3.3 Create 2-way interaction data sets with weak effect size

set.seed(1434)
df_100_weak <- generate_data_2(n = 100, p = 2, correlation = 0, intracorrelation = 0) 
int_100_weak <- interaction(df_100_weak, n = 100, noise = 4.40) 
int_100_weak <- binarize_outcome_2(int_100_weak)
scatterMatrix(int_100_weak) # .18

set.seed(2021)
df_200_weak <- generate_data_2(n = 200, p = 2, correlation = 0, intracorrelation = 0) 
int_200_weak <- interaction(df_200_weak, n = 200, noise = 3.8) # 3
int_200_weak <- binarize_outcome_2(int_200_weak)
scatterMatrix(int_200_weak) # .19

set.seed(1017)
df_500_weak <- generate_data_2(n = 500, p = 2, correlation = 0, intracorrelation = 0) 
int_500_weak <- interaction(df_500_weak, n = 500, noise = 5.6) 
int_500_weak <- binarize_outcome_2(int_500_weak)
scatterMatrix(int_500_weak) # .2


set.seed(1017)
df_1000_weak <- generate_data_2(n = 1000, p = 2, correlation = 0, intracorrelation = 0) 
int_1000_weak <- interaction(df_1000_weak, n = 1000, noise = 8.0)
int_1000_weak <- binarize_outcome_2(int_1000_weak)
scatterMatrix(int_1000_weak) # .21



# 3.4 Remove the interaction term from the data sets

int_100_strong <- int_100_strong[, -ncol(int_100_strong) + 1]
int_200_strong <- int_200_strong[, -ncol(int_200_strong) + 1]
int_500_strong <- int_500_strong[, -ncol(int_500_strong) + 1]
int_1000_strong <- int_1000_strong[, -ncol(int_1000_strong) + 1]

int_100_weak <- int_100_weak[, -ncol(int_100_weak) + 1]
int_200_weak <- int_200_weak[, -ncol(int_200_weak) + 1]
int_500_weak <- int_500_weak[, -ncol(int_500_weak) + 1]
int_1000_weak <- int_1000_weak[, -ncol(int_1000_weak) + 1]




# 3.5 Create 4-way interactions with strong effect size 

set.seed(1119) # 1119, 1918
df_100_strong <- generate_data_2(n = 100, p = 4, correlation = 0, intracorrelation = 0) 
int4_100_strong <- interaction(df_100_strong, n = 100, noise = 0.6) 
int4_100_strong <- binarize_outcome_2(int4_100_strong)
scatterMatrix(int4_100_strong) # cor .38

set.seed(2022)
df_200_strong <- generate_data_2(n = 200, p = 4, correlation = 0, intracorrelation = 0) 
int4_200_strong <- interaction(df_200_strong, n = 200, noise = 0.5) 
int4_200_strong <- binarize_outcome_2(int4_200_strong)
scatterMatrix(int4_200_strong) # .40


set.seed(2042) #
df_500_strong <- generate_data_2(n = 500, p = 4, correlation = 0, intracorrelation = 0) 
int4_500_strong <- interaction(df_500_strong, n = 500, noise = 0.5) 
int4_500_strong <- binarize_outcome_2(int4_500_strong)
scatterMatrix(int4_500_strong) # .4

set.seed(1017)
df_1000_strong <- generate_data_2(n = 1000, p = 4, correlation = 0, intracorrelation = 0) 
int4_1000_strong <- interaction(df_1000_strong, n = 1000, noise = 0.5)
int4_1000_strong <- binarize_outcome_2(int4_1000_strong)
scatterMatrix(int4_1000_strong) # .42

# 3.6 Create 4-way interaction data sets with weak effect size

set.seed(1818)
df_100_weak <- generate_data_2(n = 100, p = 4, correlation = 0, intracorrelation = 0) 
int4_100_weak <- interaction(df_100_weak, n = 100, noise = 8.0) 
int4_100_weak <- binarize_outcome_2(int4_100_weak)
scatterMatrix(int4_100_weak) # .2


set.seed(2040) # 2037 has .08 and 0.6
df_200_weak <- generate_data_2(n = 200, p = 4, correlation = 0, intracorrelation = 0) 
int4_200_weak <- interaction(df_200_weak, n = 200, noise = 7.0) 
int4_200_weak <- binarize_outcome_2(int4_200_weak)
scatterMatrix(int4_200_weak) # .20

set.seed(1017)
df_500_weak <- generate_data_2(n = 500, p = 4, correlation = 0, intracorrelation = 0) 
int4_500_weak <- interaction(df_500_weak, n = 500, noise = 6.3) # 5.6
int4_500_weak <- binarize_outcome_2(int4_500_weak)
scatterMatrix(int4_500_weak) # .20

set.seed(1017)
df_1000_weak <- generate_data_2(n = 1000, p = 4, correlation = 0, intracorrelation = 0) 
int4_1000_weak <- interaction(df_1000_weak, n = 1000, noise = 7.5) 
int4_1000_weak <- binarize_outcome_2(int4_1000_weak)
scatterMatrix(int4_1000_weak) # .20



# 3.7 Remove the interaction term from the data sets

int4_100_strong <- int4_100_strong[, -ncol(int4_100_strong) + 1]
int4_200_strong <- int4_200_strong[, -ncol(int4_200_strong) + 1]
int4_500_strong <- int4_500_strong[, -ncol(int4_500_strong) + 1]
int4_1000_strong <- int4_1000_strong[, -ncol(int4_1000_strong) + 1]

int4_100_weak <- int4_100_weak[, -ncol(int4_100_weak) + 1]
int4_200_weak <- int4_200_weak[, -ncol(int4_200_weak) + 1]
int4_500_weak <- int4_500_weak[, -ncol(int4_500_weak) + 1]
int4_1000_weak <- int4_1000_weak[, -ncol(int4_1000_weak) + 1]










                 ### 4. Classification and Performance ###

# ----- List of the model names used inside the caret train function------ ##

# Logistic Regression = v glmAdjCat 
# SVM with Linear Kernel = svmLinear 
# SVM polynomial Kernal = svmPoly
# Decision tree = rpart 
# Random forest = rf 
# Model Averaged Neural Network = avNNet


# 4.1 Function that runs the classification algorithm ------------------------#

# Function that saves the result of each iteration of CV
classify_data <- function(data, model) {
  # Define cross-validation scheme
  fit_control <- trainControl(method = "repeatedcv", number = 5, repeats = 50, 
                              savePredictions = TRUE, classProbs = FALSE)
  
  # Train the model
  trainmodel <- train(outcome ~ ., data = data, 
                      method = model, 
                      trControl = fit_control,
                      metric = "Accuracy")
  
  # Extract the cross-validation results
  cv_results <- trainmodel$resample
  
  # Calculate the mean and standard deviation of the accuracy scores
  mean_accuracy <- mean(cv_results$Accuracy)
  sd_accuracy <- sd(cv_results$Accuracy)
  
  # Return a list containing the mean accuracy and standard deviation
  return(c(mean_accuracy = mean_accuracy, sd_accuracy = sd_accuracy))
}




#--------------- 4.1 Performance in linear data sets ------------------------###

# 4.1.1 ----- accuracy at sample size 100 
LR_accuracy_100 <- 0
LR_accuracy_100 <- classify_data(df_100_01, model = "vglmAdjCat")
LR_accuracy_100[3:4] <- classify_data(df_100_05, model = "vglmAdjCat")

SVM_lin_accuracy_100 <- 0
SVM_lin_accuracy_100 <- classify_data(df_100_01, model = "svmLinear")
SVM_lin_accuracy_100[3:4] <- classify_data(df_100_05, model = "svmLinear")

SVM_pol_accuracy_100 <- 0
SVM_pol_accuracy_100 <- classify_data(df_100_01, model = "svmPoly")
SVM_pol_accuracy_100[3:4] <- classify_data(df_100_05, model = "svmPoly")

DT_accuracy_100 <- 0
DT_accuracy_100 <- classify_data(df_100_01, model = "rpart")
DT_accuracy_100[3:4] <- classify_data(df_100_05, model = "rpart")

RF_accuracy_100 <- 0
RF_accuracy_100 <- classify_data(df_100_01, model = "rf")
RF_accuracy_100[3:4] <- classify_data(df_100_05, model = "rf")

NN_accuracy_100 <- 0
NN_accuracy_100 <- classify_data(df_100_01, model = "avNNet")
NN_accuracy_100[3:4] <- classify_data(df_100_05, model = "avNNet")

# 4.1.2 ---------- accuracy at sample size 200 
LR_accuracy_200 <- 0
LR_accuracy_200 <- classify_data(df_200_01, model = "vglmAdjCat")
LR_accuracy_200[3:4] <- classify_data(df_200_05, model = "vglmAdjCat")

SVM_lin_accuracy_200 <- 0
SVM_lin_accuracy_200 <- classify_data(df_200_01, model = "svmLinear")
SVM_lin_accuracy_200[3:4] <- classify_data(df_200_05, model = "svmLinear")

SVM_pol_accuracy_200 <- 0
SVM_pol_accuracy_200 <- classify_data(df_200_01, model = "svmPoly")
SVM_pol_accuracy_200[3:4] <- classify_data(df_200_05, model = "svmPoly")

DT_accuracy_200 <- 0
DT_accuracy_200 <- classify_data(df_200_01, model = "rpart")
DT_accuracy_200[3:4] <- classify_data(df_200_05, model = "rpart")

RF_accuracy_200 <- 0
RF_accuracy_200 <- classify_data(df_200_01, model = "rf")
RF_accuracy_200[3:4] <- classify_data(df_200_05, model = "rf")

NN_accuracy_200 <- 0
NN_accuracy_200 <- classify_data(df_200_01, model = "avNNet")
NN_accuracy_200[3:4] <- classify_data(df_200_05, model = "avNNet")

# 4.1.3 --------- Accuracy at sample size 500
LR_accuracy_500 <- 0
LR_accuracy_500 <- classify_data(df_500_01, model = "vglmAdjCat")
LR_accuracy_500[3:4] <- classify_data(df_500_05, model = "vglmAdjCat")

SVM_lin_accuracy_500 <- 0
SVM_lin_accuracy_500 <- classify_data(df_500_01, model = "svmLinear")
SVM_lin_accuracy_500[3:4] <- classify_data(df_500_05, model = "svmLinear")

SVM_pol_accuracy_500 <- 0
SVM_pol_accuracy_500 <- classify_data(df_500_01, model = "svmPoly")
SVM_pol_accuracy_500[3:4] <- classify_data(df_500_05, model = "svmPoly")

DT_accuracy_500 <- 0
DT_accuracy_500 <- classify_data(df_500_01, model = "rpart")
DT_accuracy_500[3:4] <- classify_data(df_500_05, model = "rpart")

RF_accuracy_500 <- 0
RF_accuracy_500 <- classify_data(df_500_01, model = "rf")
RF_accuracy_500[3:4] <- classify_data(df_500_05, model = "rf")

NN_accuracy_500 <- 0
NN_accuracy_500 <- classify_data(df_500_01, model = "avNNet")
NN_accuracy_500[3:4] <- classify_data(df_500_05, model = "avNNet")

# 4.1.4 ------------ accuracy at sample size 1000
LR_accuracy_1000 <- 0
LR_accuracy_1000 <- classify_data(df_1000_01, model = "vglmAdjCat")
LR_accuracy_1000[3:4] <- classify_data(df_1000_05, model = "vglmAdjCat")

SVM_lin_accuracy_1000 <- 0
SVM_lin_accuracy_1000 <- classify_data(df_1000_01, model = "svmLinear")
SVM_lin_accuracy_1000[3:4] <- classify_data(df_1000_05, model = "svmLinear")

SVM_pol_accuracy_1000 <- 0
SVM_pol_accuracy_1000 <- classify_data(df_1000_01, model = "svmPoly")
SVM_pol_accuracy_1000[3:4] <- classify_data(df_1000_05, model = "svmPoly")

DT_accuracy_1000 <- 0
DT_accuracy_1000 <- classify_data(df_1000_01, model = "rpart")
DT_accuracy_1000[3:4] <- classify_data(df_1000_05, model = "rpart")

RF_accuracy_1000 <- 0
RF_accuracy_1000 <- classify_data(df_1000_01, model = "rf")
RF_accuracy_1000[3:4] <- classify_data(df_1000_05, model = "rf")

NN_accuracy_1000 <- 0
NN_accuracy_1000 <- classify_data(df_1000_01, model = "avNNet")
NN_accuracy_1000[3:4] <- classify_data(df_1000_05, model = "avNNet")


### ----------- 4.3 Performance in non-linear data sets -------------##

# 4.3.1 -------- accuracy at sample size 100 
LR_accuracy_100_nl_2 <- 0
LR_accuracy_100_nl_2 <- classify_data(nl_100_weak, model = "vglmAdjCat")
LR_accuracy_100_nl_2[3:4] <- classify_data(nl_100_strong, model = "vglmAdjCat")

SVM_lin_accuracy_100_nl_2 <- 0
SVM_lin_accuracy_100_nl_2 <- classify_data(nl_100_weak, model = "svmLinear")
SVM_lin_accuracy_100_nl_2[3:4] <- classify_data(nl_100_strong, model = "svmLinear")

SVM_pol_accuracy_100_nl_2 <- 0
SVM_pol_accuracy_100_nl_2 <- classify_data(nl_100_weak, model = "svmPoly")
SVM_pol_accuracy_100_nl_2[3:4] <- classify_data(nl_100_strong, model = "svmPoly")

DT_accuracy_100_nl_2 <- 0
DT_accuracy_100_nl_2 <- classify_data(nl_100_weak, model = "rpart")
DT_accuracy_100_nl_2[3:4] <- classify_data(nl_100_strong, model = "rpart")

RF_accuracy_100_nl_2 <- 0
RF_accuracy_100_nl_2 <- classify_data(nl_100_weak, model = "rf")
RF_accuracy_100_nl_2[3:4] <- classify_data(nl_100_strong, model = "rf")

NN_accuracy_100_nl_2 <- 0
NN_accuracy_100_nl_2 <- classify_data(nl_100_weak, model = "avNNet")
NN_accuracy_100_nl_2[3:4] <- classify_data(nl_100_strong, model = "avNNet")

# 4.3.2 -------- accuracy at sample size 200 
LR_accuracy_200_nl_2 <- 0
LR_accuracy_200_nl_2 <- classify_data(nl_200_weak, model = "vglmAdjCat")
LR_accuracy_200_nl_2[3:4] <- classify_data(nl_200_strong, model = "vglmAdjCat")

SVM_lin_accuracy_200_nl_2 <- 0
SVM_lin_accuracy_200_nl_2 <- classify_data(nl_200_weak, model = "svmLinear")
SVM_lin_accuracy_200_nl_2[3:4] <- classify_data(nl_200_strong, model = "svmLinear")

SVM_pol_accuracy_200_nl_2 <- 0
SVM_pol_accuracy_200_nl_2 <- classify_data(nl_200_weak, model = "svmPoly")
SVM_pol_accuracy_200_nl_2[3:4] <- classify_data(nl_200_strong, model = "svmPoly")

DT_accuracy_200_nl_2 <- 0
DT_accuracy_200_nl_2 <- classify_data(nl_200_weak, model = "rpart")
DT_accuracy_200_nl_2[3:4] <- classify_data(nl_200_strong, model = "rpart")

RF_accuracy_200_nl_2 <- 0
RF_accuracy_200_nl_2 <- classify_data(nl_200_weak, model = "rf")
RF_accuracy_200_nl_2[3:4] <- classify_data(nl_200_strong, model = "rf")

NN_accuracy_200_nl_2 <- 0
NN_accuracy_200_nl_2 <- classify_data(nl_200_weak, model = "avNNet")
NN_accuracy_200_nl_2[3:4] <- classify_data(nl_200_strong, model = "avNNet")

# 4.3.3 -------- accuracy at sample size 500
LR_accuracy_500_nl_2 <- 0
LR_accuracy_500_nl_2 <- classify_data(nl_500_weak, model = "vglmAdjCat")
LR_accuracy_500_nl_2[3:4] <- classify_data(nl_500_strong, model = "vglmAdjCat")

SVM_lin_accuracy_500_nl_2 <- 0
SVM_lin_accuracy_500_nl_2 <- classify_data(nl_500_weak, model = "svmLinear")
SVM_lin_accuracy_500_nl_2[3:4] <- classify_data(nl_500_strong, model = "svmLinear")

SVM_pol_accuracy_500_nl_2 <- 0
SVM_pol_accuracy_500_nl_2 <- classify_data(nl_500_weak, model = "svmPoly")
SVM_pol_accuracy_500_nl_2[3:4] <- classify_data(nl_500_strong, model = "svmPoly")

DT_accuracy_500_nl_2 <- 0
DT_accuracy_500_nl_2 <- classify_data(nl_500_weak, model = "rpart")
DT_accuracy_500_nl_2[3:4] <- classify_data(nl_500_strong, model = "rpart")

RF_accuracy_500_nl_2 <- 0
RF_accuracy_500_nl_2 <- classify_data(nl_500_weak, model = "rf")
RF_accuracy_500_nl_2[3:4] <- classify_data(nl_500_strong, model = "rf")

NN_accuracy_500_nl_2 <- 0
NN_accuracy_500_nl_2 <- classify_data(nl_500_weak, model = "avNNet")
NN_accuracy_500_nl_2[3:4] <- classify_data(nl_500_strong, model = "avNNet")

# 4.3.4 -------- accuracy at sample size 1000
LR_accuracy_1000_nl_2 <- 0
LR_accuracy_1000_nl_2 <- classify_data(nl_1000_weak, model = "vglmAdjCat")
LR_accuracy_1000_nl_2[3:4] <- classify_data(nl_1000_strong, model = "vglmAdjCat")

SVM_lin_accuracy_1000_nl_2 <- 0
SVM_lin_accuracy_1000_nl_2 <- classify_data(nl_1000_weak, model = "svmLinear")
SVM_lin_accuracy_1000_nl_2[3:4] <- classify_data(nl_1000_strong, model = "svmLinear")

SVM_pol_accuracy_1000_nl_2 <- 0
SVM_pol_accuracy_1000_nl_2 <- classify_data(nl_1000_weak, model = "svmPoly")
SVM_pol_accuracy_1000_nl_2[3:4] <- classify_data(nl_1000_strong, model = "svmPoly")

DT_accuracy_1000_nl_2 <- 0
DT_accuracy_1000_nl_2 <- classify_data(nl_1000_weak, model = "rpart")
DT_accuracy_1000_nl_2[3:4] <- classify_data(nl_1000_strong, model = "rpart")

RF_accuracy_1000_nl_2 <- 0
RF_accuracy_1000_nl_2 <- classify_data(nl_1000_weak, model = "rf")
RF_accuracy_1000_nl_2[3:4] <- classify_data(nl_1000_strong, model = "rf")

NN_accuracy_1000_nl_2 <- 0
NN_accuracy_1000_nl_2 <- classify_data(nl_1000_weak, model = "avNNet")
NN_accuracy_1000_nl_2[3:4] <- classify_data(nl_1000_strong, model = "avNNet")




### ----------- 4.4 Performance in 2-way interaction data sets ---------------##

# 4.4.1 accuracy at sample size 100 
LR_accuracy_100_int <- 0
LR_accuracy_100_int <- classify_data(int_100_weak, model = "vglmAdjCat")
LR_accuracy_100_int[3:4] <- classify_data(int_100_strong, model = "vglmAdjCat")

SVM_lin_accuracy_100_int <- 0
SVM_lin_accuracy_100_int <- classify_data(int_100_weak, model = "svmLinear")
SVM_lin_accuracy_100_int[3:4] <- classify_data(int_100_strong, model = "svmLinear")

SVM_pol_accuracy_100_int <- 0
SVM_pol_accuracy_100_int <- classify_data(int_100_weak, model = "svmPoly")
SVM_pol_accuracy_100_int[3:4] <- classify_data(int_100_strong, model = "svmPoly")

DT_accuracy_100_int <- 0
DT_accuracy_100_int <- classify_data(int_100_weak, model = "rpart")
DT_accuracy_100_int[3:4] <- classify_data(int_100_strong, model = "rpart")

RF_accuracy_100_int <- 0
RF_accuracy_100_int <- classify_data(int_100_weak, model = "rf")
RF_accuracy_100_int[3:4] <- classify_data(int_100_strong, model = "rf")

NN_accuracy_100_int <- 0
NN_accuracy_100_int <- classify_data(int_100_weak, model = "avNNet")
NN_accuracy_100_int[3:4] <- classify_data(int_100_strong, model = "avNNet")


# 4.4.2 accuracy at sample size 200 
LR_accuracy_200_int <- 0
LR_accuracy_200_int <- classify_data(int_200_weak, model = "vglmAdjCat")
LR_accuracy_200_int[3:4] <- classify_data(int_200_strong, model = "vglmAdjCat")

SVM_lin_accuracy_200_int <- 0
SVM_lin_accuracy_200_int <- classify_data(int_200_weak, model = "svmLinear")
SVM_lin_accuracy_200_int[3:4] <- classify_data(int_200_strong, model = "svmLinear")

SVM_pol_accuracy_200_int <- 0
SVM_pol_accuracy_200_int <- classify_data(int_200_weak, model = "svmPoly")
SVM_pol_accuracy_200_int[3:4] <- classify_data(int_200_strong, model = "svmPoly")

DT_accuracy_200_int <- 0
DT_accuracy_200_int <- classify_data(int_200_weak, model = "rpart")
DT_accuracy_200_int[3:4] <- classify_data(int_200_strong, model = "rpart")

RF_accuracy_200_int <- 0
RF_accuracy_200_int <- classify_data(int_200_weak, model = "rf")
RF_accuracy_200_int[3:4] <- classify_data(int_200_strong, model = "rf")

NN_accuracy_200_int <- 0
NN_accuracy_200_int <- classify_data(int_200_weak, model = "avNNet")
NN_accuracy_200_int[3:4] <- classify_data(int_200_strong, model = "avNNet")


# 4.4.3 accuracy at sample size 500 
LR_accuracy_500_int <- 0
LR_accuracy_500_int <- classify_data(int_500_weak, model = "vglmAdjCat")
LR_accuracy_500_int[3:4] <- classify_data(int_500_strong, model = "vglmAdjCat")

SVM_lin_accuracy_500_int <- 0
SVM_lin_accuracy_500_int <- classify_data(int_500_weak, model = "svmLinear")
SVM_lin_accuracy_500_int[3:4] <- classify_data(int_500_strong, model = "svmLinear")

SVM_pol_accuracy_500_int <- 0
SVM_pol_accuracy_500_int <- classify_data(int_500_weak, model = "svmPoly")
SVM_pol_accuracy_500_int[3:4] <- classify_data(int_500_strong, model = "svmPoly")

DT_accuracy_500_int <- 0
DT_accuracy_500_int <- classify_data(int_500_weak, model = "rpart")
DT_accuracy_500_int[3:4] <- classify_data(int_500_strong, model = "rpart")

RF_accuracy_500_int <- 0
RF_accuracy_500_int <- classify_data(int_500_weak, model = "rf")
RF_accuracy_500_int[3:4] <- classify_data(int_500_strong, model = "rf")

NN_accuracy_500_int <- 0
NN_accuracy_500_int <- classify_data(int_500_weak, model = "avNNet")
NN_accuracy_500_int[3:4] <- classify_data(int_500_strong, model = "avNNet")

# 4.4.4 accuracy at sample size 1000 
LR_accuracy_1000_int <- 0
LR_accuracy_1000_int <- classify_data(int_1000_weak, model = "vglmAdjCat")
LR_accuracy_1000_int[3:4] <- classify_data(int_1000_strong, model = "vglmAdjCat")

SVM_lin_accuracy_1000_int <- 0
SVM_lin_accuracy_1000_int <- classify_data(int_1000_weak, model = "svmLinear")
SVM_lin_accuracy_1000_int[3:4] <- classify_data(int_1000_strong, model = "svmLinear")

SVM_pol_accuracy_1000_int <- 0
SVM_pol_accuracy_1000_int <- classify_data(int_1000_weak, model = "svmPoly")
SVM_pol_accuracy_1000_int[3:4] <- classify_data(int_1000_strong, model = "svmPoly")

DT_accuracy_1000_int <- 0
DT_accuracy_1000_int <- classify_data(int_1000_weak, model = "rpart")
DT_accuracy_1000_int[3:4] <- classify_data(int_1000_strong, model = "rpart")

RF_accuracy_1000_int <- 0
RF_accuracy_1000_int <- classify_data(int_1000_weak, model = "rf")
RF_accuracy_1000_int[3:4] <- classify_data(int_1000_strong, model = "rf")

NN_accuracy_1000_int <- 0
NN_accuracy_1000_int <- classify_data(int_1000_weak, model = "avNNet")
NN_accuracy_1000_int[3:4] <- classify_data(int_1000_strong, model = "avNNet")




### ----------- 4.5 Performance in 4-way interaction data sets -------------------##

# 4.4.1 accuracy at sample size 100 
LR_accuracy_100_int4 <- 0
LR_accuracy_100_int4 <- classify_data(int4_100_weak, model = "vglmAdjCat")
LR_accuracy_100_int4[3:4] <- classify_data(int4_100_strong, model = "vglmAdjCat")

SVM_lin_accuracy_100_int4 <- 0
SVM_lin_accuracy_100_int4 <- classify_data(int4_100_weak, model = "svmLinear")
SVM_lin_accuracy_100_int4[3:4] <- classify_data(int4_100_strong, model = "svmLinear")

SVM_pol_accuracy_100_int4 <- 0
SVM_pol_accuracy_100_int4 <- classify_data(int4_100_weak, model = "svmPoly")
SVM_pol_accuracy_100_int4[3:4] <- classify_data(int4_100_strong, model = "svmPoly")

DT_accuracy_100_int4 <- 0
DT_accuracy_100_int4 <- classify_data(int4_100_weak, model = "rpart")
DT_accuracy_100_int4[3:4] <- classify_data(int4_100_strong, model = "rpart")

RF_accuracy_100_int4 <- 0
RF_accuracy_100_int4 <- classify_data(int4_100_weak, model = "rf")
RF_accuracy_100_int4[3:4] <- classify_data(int4_100_strong, model = "rf")

NN_accuracy_100_int4 <- 0
NN_accuracy_100_int4 <- classify_data(int4_100_weak, model = "avNNet")
NN_accuracy_100_int4[3:4] <- classify_data(int4_100_strong, model = "avNNet")


# 4.4.2 accuracy at sample size 200 
LR_accuracy_200_int4 <- 0
LR_accuracy_200_int4 <- classify_data(int4_200_weak, model = "vglmAdjCat")
LR_accuracy_200_int4[3:4] <- classify_data(int4_200_strong, model = "vglmAdjCat")

SVM_lin_accuracy_200_int4 <- 0
SVM_lin_accuracy_200_int4 <- classify_data(int4_200_weak, model = "svmLinear")
SVM_lin_accuracy_200_int4[3:4] <- classify_data(int4_200_strong, model = "svmLinear")

SVM_pol_accuracy_200_int4 <- 0
SVM_pol_accuracy_200_int4 <- classify_data(int4_200_weak, model = "svmPoly")
SVM_pol_accuracy_200_int4[3:4] <- classify_data(int4_200_strong, model = "svmPoly")

DT_accuracy_200_int4 <- 0
DT_accuracy_200_int4 <- classify_data(int4_200_weak, model = "rpart")
DT_accuracy_200_int4[3:4] <- classify_data(int4_200_strong, model = "rpart")

RF_accuracy_200_int4 <- 0
RF_accuracy_200_int4 <- classify_data(int4_200_weak, model = "rf")
RF_accuracy_200_int4[3:4] <- classify_data(int4_200_strong, model = "rf")

NN_accuracy_200_int4 <- 0
NN_accuracy_200_int4 <- classify_data(int4_200_weak, model = "avNNet")
NN_accuracy_200_int4[3:4] <- classify_data(int4_200_strong, model = "avNNet")


# 4.4.3 accuracy at sample size 500 
LR_accuracy_500_int4 <- 0
LR_accuracy_500_int4 <- classify_data(int4_500_weak, model = "vglmAdjCat")
LR_accuracy_500_int4[3:4] <- classify_data(int4_500_strong, model = "vglmAdjCat")

SVM_lin_accuracy_500_int4 <- 0
SVM_lin_accuracy_500_int4 <- classify_data(int4_500_weak, model = "svmLinear")
SVM_lin_accuracy_500_int4[3:4] <- classify_data(int4_500_strong, model = "svmLinear")

SVM_pol_accuracy_500_int4 <- 0
SVM_pol_accuracy_500_int4 <- classify_data(int4_500_weak, model = "svmPoly")
SVM_pol_accuracy_500_int4[3:4] <- classify_data(int4_500_strong, model = "svmPoly")

DT_accuracy_500_int4 <- 0
DT_accuracy_500_int4 <- classify_data(int4_500_weak, model = "rpart")
DT_accuracy_500_int4[3:4] <- classify_data(int4_500_strong, model = "rpart")

RF_accuracy_500_int4 <- 0
RF_accuracy_500_int4 <- classify_data(int4_500_weak, model = "rf")
RF_accuracy_500_int4[3:4] <- classify_data(int4_500_strong, model = "rf")

NN_accuracy_500_int4 <- 0
NN_accuracy_500_int4 <- classify_data(int4_500_weak, model = "avNNet")
NN_accuracy_500_int4[3:4] <- classify_data(int4_500_strong, model = "avNNet")

# 4.4.4 accuracy at sample size 1000 
LR_accuracy_1000_int4 <- 0
LR_accuracy_1000_int4 <- classify_data(int4_1000_weak, model = "vglmAdjCat")
LR_accuracy_1000_int4[3:4] <- classify_data(int4_1000_strong, model = "vglmAdjCat")

SVM_lin_accuracy_1000_int4 <- 0
SVM_lin_accuracy_1000_int4 <- classify_data(int4_1000_weak, model = "svmLinear")
SVM_lin_accuracy_1000_int4[3:4] <- classify_data(int4_1000_strong, model = "svmLinear")

SVM_pol_accuracy_1000_int4 <- 0
SVM_pol_accuracy_1000_int4 <- classify_data(int4_1000_weak, model = "svmPoly")
SVM_pol_accuracy_1000_int4[3:4] <- classify_data(int4_1000_strong, model = "svmPoly")

DT_accuracy_1000_int4 <- 0
DT_accuracy_1000_int4 <- classify_data(int4_1000_weak, model = "rpart")
DT_accuracy_1000_int4[3:4] <- classify_data(int4_1000_strong, model = "rpart")

RF_accuracy_1000_int4 <- 0
RF_accuracy_1000_int4 <- classify_data(int4_1000_weak, model = "rf")
RF_accuracy_1000_int4[3:4] <- classify_data(int4_1000_strong, model = "rf")

NN_accuracy_1000_int4 <- 0
NN_accuracy_1000_int4 <- classify_data(int4_1000_weak, model = "avNNet")
NN_accuracy_1000_int4[3:4] <- classify_data(int4_1000_strong, model = "avNNet")


                          ### 5. Learning curves ###

# 5.1 Create the data frames with accuracies and standard devaiation   -------##

sample_sizes <- c(100, 200, 500, 1000)

# 5.1.5 ---------- Data frame for correlation on X-axis -----------------------#

# Create a data frame for linear data at correlation .2
lc_correlation_02 <- data.frame(SampleSize = sample_sizes,
 LogisticRegression = c(LR_accuracy_100[1], LR_accuracy_200[1], LR_accuracy_500[1], LR_accuracy_1000[1]),
 LinearSVM = c(SVM_lin_accuracy_100[1], SVM_lin_accuracy_200[1], SVM_lin_accuracy_500[1], SVM_lin_accuracy_1000[1]),
 PolySVM = c(SVM_pol_accuracy_100[1], SVM_pol_accuracy_200[1], SVM_pol_accuracy_500[1], SVM_pol_accuracy_1000[1]),
 DecisionTree = c(DT_accuracy_100[1], DT_accuracy_200[1], DT_accuracy_500[1], DT_accuracy_1000[1]),
 RandomForest = c(RF_accuracy_100[1], RF_accuracy_200[1], RF_accuracy_500[1], RF_accuracy_1000[1]),
 NeuralNetwork = c(NN_accuracy_100[1], NN_accuracy_200[1], NN_accuracy_500[1], NN_accuracy_1000[1]))
# Melt the data frame to long format
lc_correlation_02_long <- melt(lc_correlation_02, id.vars = "SampleSize", variable.name = "Model", value.name = "Accuracy")
names(lc_correlation_02_long) = c("SampleSize", "Model", "Accuracy") # name the variables


lc_correlation_02_long$SD <- c(LR_accuracy_100[2], LR_accuracy_200[2], LR_accuracy_500[2], LR_accuracy_1000[2], 
                               SVM_lin_accuracy_100[2], SVM_lin_accuracy_200[2], SVM_lin_accuracy_500[2], SVM_lin_accuracy_1000[2],
                               SVM_pol_accuracy_100[2], SVM_pol_accuracy_200[2], SVM_pol_accuracy_500[2], SVM_pol_accuracy_1000[2],
                               DT_accuracy_100[2], DT_accuracy_200[2], DT_accuracy_500[2], DT_accuracy_1000[2],
                               RF_accuracy_100[2], RF_accuracy_200[2], RF_accuracy_500[2], RF_accuracy_1000[2],
                               NN_accuracy_100[2], NN_accuracy_200[2], NN_accuracy_500[2], NN_accuracy_1000[2])

# Add lower and upper bounds for error bars
lc_correlation_02_long <- lc_correlation_02_long %>%
  mutate(Accuracy_lower = Accuracy - SD, 
         Accuracy_upper = Accuracy + SD)

# Create a data frame at correlation .4
lc_correlation_04 <- data.frame(SampleSize = sample_sizes,
                                LogisticRegression = c(LR_accuracy_100[3], LR_accuracy_200[3], LR_accuracy_500[3], LR_accuracy_1000[3]),
                                LinearSVM = c(SVM_lin_accuracy_100[3], SVM_lin_accuracy_200[3], SVM_lin_accuracy_500[3], SVM_lin_accuracy_1000[3]),
                                PolySVM = c(SVM_pol_accuracy_100[3], SVM_pol_accuracy_200[3], SVM_pol_accuracy_500[3], SVM_pol_accuracy_1000[3]),
                                DecisionTree = c(DT_accuracy_100[3], DT_accuracy_200[3], DT_accuracy_500[3], DT_accuracy_1000[3]),
                                RandomForest = c(RF_accuracy_100[3], RF_accuracy_200[3], RF_accuracy_500[3], RF_accuracy_1000[3]),
                                NeuralNetwork = c(NN_accuracy_100[3], NN_accuracy_200[3], NN_accuracy_500[3], NN_accuracy_1000[3]))
# Melt the data frame to long format
lc_correlation_04_long <- melt(lc_correlation_04, id.vars = "SampleSize", variable.name = "Model", value.name = "Accuracy")
names(lc_correlation_04_long) = c("SampleSize", "Model", "Accuracy") # name the variables

lc_correlation_04_long$SD <- c(LR_accuracy_100[4], LR_accuracy_200[4], LR_accuracy_500[4], LR_accuracy_1000[4], 
                               SVM_lin_accuracy_100[4], SVM_lin_accuracy_200[4], SVM_lin_accuracy_500[4], SVM_lin_accuracy_1000[4],
                               SVM_pol_accuracy_100[4], SVM_pol_accuracy_200[4], SVM_pol_accuracy_500[4], SVM_pol_accuracy_1000[4],
                               DT_accuracy_100[4], DT_accuracy_200[4], DT_accuracy_500[4], DT_accuracy_1000[4],
                               RF_accuracy_100[4], RF_accuracy_200[4], RF_accuracy_500[4], RF_accuracy_1000[4],
                               NN_accuracy_100[4], NN_accuracy_200[4], NN_accuracy_500[4], NN_accuracy_1000[4])

# Add lower and upper bounds for error bars
lc_correlation_04_long <- lc_correlation_04_long %>%
  mutate(Accuracy_lower = Accuracy - SD, 
         Accuracy_upper = Accuracy + SD)

## ---------------Create data frames for nonlinear data --------------- ##

# Nonlinear data weak effect size
lc_nonlinear_weak <- data.frame(SampleSize = sample_sizes,
                                    LogisticRegression = c(LR_accuracy_100_nl_2[1], LR_accuracy_200_nl_2[1], LR_accuracy_500_nl_2[1], LR_accuracy_1000_nl_2[1]),
                                    LinearSVM = c(SVM_lin_accuracy_100_nl_2[1], SVM_lin_accuracy_200_nl_2[1], SVM_lin_accuracy_500_nl_2[1], SVM_lin_accuracy_1000_nl_2[1]),
                                    PolySVM = c(SVM_pol_accuracy_100_nl_2[1], SVM_pol_accuracy_200_nl_2[1], SVM_pol_accuracy_500_nl_2[1], SVM_pol_accuracy_1000_nl_2[1]),
                                    DecisionTree = c(DT_accuracy_100_nl_2[1], DT_accuracy_200_nl_2[1], DT_accuracy_500_nl_2[1], DT_accuracy_1000_nl_2[1]),
                                    RandomForest = c(RF_accuracy_100_nl_2[1], RF_accuracy_200_nl_2[1], RF_accuracy_500_nl_2[1], RF_accuracy_1000_nl_2[1]),
                                    NeuralNetwork = c(NN_accuracy_100_nl_2[1], NN_accuracy_200_nl_2[1], NN_accuracy_500_nl_2[1], NN_accuracy_1000_nl_2[1]))
# Melt the data frame to long format
lc_nonlinear_weak_long <- melt(lc_nonlinear_weak, id.vars = "SampleSize", variable.name = "Model", value.name = "Accuracy")
names(lc_nonlinear_weak_long) = c("SampleSize", "Model", "Accuracy") # name the variables

lc_nonlinear_weak_long$SD <- c(LR_accuracy_100_nl_2[2], LR_accuracy_200_nl_2[2], LR_accuracy_500_nl_2[2], LR_accuracy_1000_nl_2[2], 
                               SVM_lin_accuracy_100_nl_2[2], SVM_lin_accuracy_200_nl_2[2], SVM_lin_accuracy_500_nl_2[2], SVM_lin_accuracy_1000_nl_2[2],
                               SVM_pol_accuracy_100_nl_2[2], SVM_pol_accuracy_200_nl_2[2], SVM_pol_accuracy_500_nl_2[2], SVM_pol_accuracy_1000_nl_2[2],
                               DT_accuracy_100_nl_2[2], DT_accuracy_200_nl_2[2], DT_accuracy_500_nl_2[2], DT_accuracy_1000_nl_2[2],
                               RF_accuracy_100_nl_2[2], RF_accuracy_200_nl_2[2], RF_accuracy_500_nl_2[2], RF_accuracy_1000_nl_2[2],
                               NN_accuracy_100_nl_2[2], NN_accuracy_200_nl_2[2], NN_accuracy_500_nl_2[2], NN_accuracy_1000_nl_2[2])


# Add lower and upper bounds for error bars
lc_nonlinear_weak_long <- lc_nonlinear_weak_long %>%
  mutate(Accuracy_lower = Accuracy - SD, 
         Accuracy_upper = Accuracy + SD)


# Nonlinear data with strong effect size
lc_nonlinear_strong <- data.frame(SampleSize = sample_sizes,
                                    LogisticRegression = c(LR_accuracy_100_nl_2[3], LR_accuracy_200_nl_2[3], LR_accuracy_500_nl_2[3], LR_accuracy_1000_nl_2[3]),
                                    LinearSVM = c(SVM_lin_accuracy_100_nl_2[3], SVM_lin_accuracy_200_nl_2[3], SVM_lin_accuracy_500_nl_2[3], SVM_lin_accuracy_1000_nl_2[3]),
                                    PolySVM = c(SVM_pol_accuracy_100_nl_2[3], SVM_pol_accuracy_200_nl_2[3], SVM_pol_accuracy_500_nl_2[3], SVM_pol_accuracy_1000_nl_2[3]),
                                    DecisionTree = c(DT_accuracy_100_nl_2[3], DT_accuracy_200_nl_2[3], DT_accuracy_500_nl_2[3], DT_accuracy_1000_nl_2[3]),
                                    RandomForest = c(RF_accuracy_100_nl_2[3], RF_accuracy_200_nl_2[3], RF_accuracy_500_nl_2[3], RF_accuracy_1000_nl_2[3]),
                                    NeuralNetwork = c(NN_accuracy_100_nl_2[3], NN_accuracy_200_nl_2[3], NN_accuracy_500_nl_2[3], NN_accuracy_1000_nl_2[3]))
# Melt the data frame to long format
lc_nonlinear_strong_long <- melt(lc_nonlinear_strong, id.vars = "SampleSize", variable.name = "Model", value.name = "Accuracy")
names(lc_nonlinear_strong_long) = c("SampleSize", "Model", "Accuracy") # name the variables


lc_nonlinear_strong_long$SD <- c(LR_accuracy_100_nl_2[4], LR_accuracy_200_nl_2[4], LR_accuracy_500_nl_2[4], LR_accuracy_1000_nl_2[4], 
                               SVM_lin_accuracy_100_nl_2[4], SVM_lin_accuracy_200_nl_2[4], SVM_lin_accuracy_500_nl_2[4], SVM_lin_accuracy_1000_nl_2[4],
                               SVM_pol_accuracy_100_nl_2[4], SVM_pol_accuracy_200_nl_2[4], SVM_pol_accuracy_500_nl_2[4], SVM_pol_accuracy_1000_nl_2[4],
                               DT_accuracy_100_nl_2[4], DT_accuracy_200_nl_2[4], DT_accuracy_500_nl_2[4], DT_accuracy_1000_nl_2[4],
                               RF_accuracy_100_nl_2[4], RF_accuracy_200_nl_2[4], RF_accuracy_500_nl_2[4], RF_accuracy_1000_nl_2[4],
                               NN_accuracy_100_nl_2[4], NN_accuracy_200_nl_2[4], NN_accuracy_500_nl_2[4], NN_accuracy_1000_nl_2[4])


# Add lower and upper bounds for error bars
lc_nonlinear_strong_long <- lc_nonlinear_strong_long %>%
  mutate(Accuracy_lower = Accuracy - SD, 
         Accuracy_upper = Accuracy + SD)




## ------------------------- Interaction data sets -------------------  ##

# interaction weak effect size
lc_interaction_02 <- data.frame(SampleSize = sample_sizes,
                                    LogisticRegression = c(LR_accuracy_100_int[1], LR_accuracy_200_int[1], LR_accuracy_500_int[1], LR_accuracy_1000_int[1]),
                                    LinearSVM = c(SVM_lin_accuracy_100_int[1], SVM_lin_accuracy_200_int[1], SVM_lin_accuracy_500_int[1], SVM_lin_accuracy_1000_int[1]),
                                    PolySVM = c(SVM_pol_accuracy_100_int[1], SVM_pol_accuracy_200_int[1], SVM_pol_accuracy_500_int[1], SVM_pol_accuracy_1000_int[1]),
                                    DecisionTree = c(DT_accuracy_100_int[1], DT_accuracy_200_int[1], DT_accuracy_500_int[1], DT_accuracy_1000_int[1]),
                                    RandomForest = c(RF_accuracy_100_int[1], RF_accuracy_200_int[1], RF_accuracy_500_int[1], RF_accuracy_1000_int[1]),
                                    NeuralNetwork = c(NN_accuracy_100_int[1], NN_accuracy_200_int[1], NN_accuracy_500_int[1], NN_accuracy_1000_int[1]))
# Melt the data frame to long format
lc_interaction_02_long <- melt(lc_interaction_02, id.vars = "SampleSize", variable.name = "Model", value.name = "Accuracy")
names(lc_interaction_02_long) = c("SampleSize", "Model", "Accuracy") # name the variables

lc_interaction_02_long$SD <- c(LR_accuracy_100_int[2], LR_accuracy_200_int[2], LR_accuracy_500_int[2], LR_accuracy_1000_int[2], 
                                 SVM_lin_accuracy_100_int[2], SVM_lin_accuracy_200_int[2], SVM_lin_accuracy_500_int[2], SVM_lin_accuracy_1000_int[2],
                                 SVM_pol_accuracy_100_int[2], SVM_pol_accuracy_200_int[2], SVM_pol_accuracy_500_int[2], SVM_pol_accuracy_1000_int[2],
                                 DT_accuracy_100_int[2], DT_accuracy_200_int[2], DT_accuracy_500_int[2], DT_accuracy_1000_int[2],
                                 RF_accuracy_100_int[2], RF_accuracy_200_int[2], RF_accuracy_500_int[2], RF_accuracy_1000_int[2],
                                 NN_accuracy_100_int[2], NN_accuracy_200_int[2], NN_accuracy_500_int[2], NN_accuracy_1000_int[2])


# Add lower and upper bounds for error bars
lc_interaction_02_long <- lc_interaction_02_long %>%
  mutate(Accuracy_lower = Accuracy - SD, 
         Accuracy_upper = Accuracy + SD)


# Interaction strong effect size
lc_interaction_04 <- data.frame(SampleSize = sample_sizes,
                              LogisticRegression = c(LR_accuracy_100_int[3], LR_accuracy_200_int[3], LR_accuracy_500_int[3], LR_accuracy_1000_int[3]),
                              LinearSVM = c(SVM_lin_accuracy_100_int[3], SVM_lin_accuracy_200_int[3], SVM_lin_accuracy_500_int[3], SVM_lin_accuracy_1000_int[3]),
                              PolySVM = c(SVM_pol_accuracy_100_int[3], SVM_pol_accuracy_200_int[3], SVM_pol_accuracy_500_int[3], SVM_pol_accuracy_1000_int[3]),
                              DecisionTree = c(DT_accuracy_100_int[3], DT_accuracy_200_int[3], DT_accuracy_500_int[3], DT_accuracy_1000_int[3]),
                              RandomForest = c(RF_accuracy_100_int[3], RF_accuracy_200_int[3], RF_accuracy_500_int[3], RF_accuracy_1000_int[3]),
                              NeuralNetwork = c(NN_accuracy_100_int[3], NN_accuracy_200_int[3], NN_accuracy_500_int[3], NN_accuracy_1000_int[3]))
# Melt the data frame to long format
lc_interaction_04_long <- melt(lc_interaction_04, id.vars = "SampleSize", variable.name = "Model", value.name = "Accuracy")
names(lc_interaction_04_long) = c("SampleSize", "Model", "Accuracy") # name the variables

lc_interaction_04_long$SD <- c(LR_accuracy_100_int[4], LR_accuracy_200_int[4], LR_accuracy_500_int[4], LR_accuracy_1000_int[4], 
                                 SVM_lin_accuracy_100_int[4], SVM_lin_accuracy_200_int[4], SVM_lin_accuracy_500_int[4], SVM_lin_accuracy_1000_int[4],
                                 SVM_pol_accuracy_100_int[4], SVM_pol_accuracy_200_int[4], SVM_pol_accuracy_500_int[4], SVM_pol_accuracy_1000_int[4],
                                 DT_accuracy_100_int[4], DT_accuracy_200_int[4], DT_accuracy_500_int[4], DT_accuracy_1000_int[4],
                                 RF_accuracy_100_int[4], RF_accuracy_200_int[4], RF_accuracy_500_int[4], RF_accuracy_1000_int[4],
                                 NN_accuracy_100_int[4], NN_accuracy_200_int[4], NN_accuracy_500_int[4], NN_accuracy_1000_int[4])


# Add lower and upper bounds for error bars
lc_interaction_04_long <- lc_interaction_04_long %>%
  mutate(Accuracy_lower = Accuracy - SD, 
         Accuracy_upper = Accuracy + SD)





##------------4-way interaction data sets --------------------- ##

# interaction weak effect size
lc_interaction4_02 <- data.frame(SampleSize = sample_sizes,
                                LogisticRegression = c(LR_accuracy_100_int4[1], LR_accuracy_200_int4[1], LR_accuracy_500_int4[1], LR_accuracy_1000_int4[1]),
                                LinearSVM = c(SVM_lin_accuracy_100_int4[1], SVM_lin_accuracy_200_int4[1], SVM_lin_accuracy_500_int4[1], SVM_lin_accuracy_1000_int4[1]),
                                PolySVM = c(SVM_pol_accuracy_100_int4[1], SVM_pol_accuracy_200_int4[1], SVM_pol_accuracy_500_int4[1], SVM_pol_accuracy_1000_int4[1]),
                                DecisionTree = c(DT_accuracy_100_int4[1], DT_accuracy_200_int4[1], DT_accuracy_500_int4[1], DT_accuracy_1000_int4[1]),
                                RandomForest = c(RF_accuracy_100_int4[1], RF_accuracy_200_int4[1], RF_accuracy_500_int4[1], RF_accuracy_1000_int4[1]),
                                NeuralNetwork = c(NN_accuracy_100_int4[1], NN_accuracy_200_int4[1], NN_accuracy_500_int4[1], NN_accuracy_1000_int4[1]))
# Melt the data frame to long format
lc_interaction4_02_long <- melt(lc_interaction4_02, id.vars = "SampleSize", variable.name = "Model", value.name = "Accuracy")
names(lc_interaction4_02_long) = c("SampleSize", "Model", "Accuracy") # name the variables

lc_interaction4_02_long$SD <- c(LR_accuracy_100_int4[2], LR_accuracy_200_int4[2], LR_accuracy_500_int4[2], LR_accuracy_1000_int4[2], 
                               SVM_lin_accuracy_100_int4[2], SVM_lin_accuracy_200_int4[2], SVM_lin_accuracy_500_int4[2], SVM_lin_accuracy_1000_int4[2],
                               SVM_pol_accuracy_100_int4[2], SVM_pol_accuracy_200_int4[2], SVM_pol_accuracy_500_int4[2], SVM_pol_accuracy_1000_int4[2],
                               DT_accuracy_100_int4[2], DT_accuracy_200_int4[2], DT_accuracy_500_int4[2], DT_accuracy_1000_int4[2],
                               RF_accuracy_100_int4[2], RF_accuracy_200_int4[2], RF_accuracy_500_int4[2], RF_accuracy_1000_int4[2],
                               NN_accuracy_100_int4[2], NN_accuracy_200_int4[2], NN_accuracy_500_int4[2], NN_accuracy_1000_int4[2])


# Add lower and upper bounds for error bars
lc_interaction4_02_long <- lc_interaction4_02_long %>%
  mutate(Accuracy_lower = Accuracy - SD, 
         Accuracy_upper = Accuracy + SD)



# Interaction strong effect size
lc_interaction4_04 <- data.frame(SampleSize = sample_sizes,
                                LogisticRegression = c(LR_accuracy_100_int4[3], LR_accuracy_200_int4[3], LR_accuracy_500_int4[3], LR_accuracy_1000_int4[3]),
                                LinearSVM = c(SVM_lin_accuracy_100_int4[3], SVM_lin_accuracy_200_int4[3], SVM_lin_accuracy_500_int4[3], SVM_lin_accuracy_1000_int4[3]),
                                PolySVM = c(SVM_pol_accuracy_100_int4[3], SVM_pol_accuracy_200_int4[3], SVM_pol_accuracy_500_int4[3], SVM_pol_accuracy_1000_int4[3]),
                                DecisionTree = c(DT_accuracy_100_int4[3], DT_accuracy_200_int4[3], DT_accuracy_500_int4[3], DT_accuracy_1000_int4[3]),
                                RandomForest = c(RF_accuracy_100_int4[3], RF_accuracy_200_int4[3], RF_accuracy_500_int4[3], RF_accuracy_1000_int4[3]),
                                NeuralNetwork = c(NN_accuracy_100_int4[3], NN_accuracy_200_int4[3], NN_accuracy_500_int4[3], NN_accuracy_1000_int4[3]))
# Melt the data frame to long format
lc_interaction4_04_long <- melt(lc_interaction4_04, id.vars = "SampleSize", variable.name = "Model", value.name = "Accuracy")
names(lc_interaction4_04_long) = c("SampleSize", "Model", "Accuracy") # name the variables


lc_interaction4_04_long$SD <- c(LR_accuracy_100_int4[4], LR_accuracy_200_int4[4], LR_accuracy_500_int4[4], LR_accuracy_1000_int4[4], 
                                SVM_lin_accuracy_100_int4[4], SVM_lin_accuracy_200_int4[4], SVM_lin_accuracy_500_int4[4], SVM_lin_accuracy_1000_int4[4],
                                SVM_pol_accuracy_100_int4[4], SVM_pol_accuracy_200_int4[4], SVM_pol_accuracy_500_int4[4], SVM_pol_accuracy_1000_int4[4],
                                DT_accuracy_100_int4[4], DT_accuracy_200_int4[4], DT_accuracy_500_int4[4], DT_accuracy_1000_int4[4],
                                RF_accuracy_100_int4[4], RF_accuracy_200_int4[4], RF_accuracy_500_int4[4], RF_accuracy_1000_int4[4],
                                NN_accuracy_100_int4[4], NN_accuracy_200_int4[4], NN_accuracy_500_int4[4], NN_accuracy_1000_int4[4])


# Add lower and upper bounds for error bars
lc_interaction4_04_long <- lc_interaction4_04_long %>%
  mutate(Accuracy_lower = Accuracy - SD, 
         Accuracy_upper = Accuracy + SD)

### 5.3 Plot learning Curves

# 5.3.1 ------------ Linear data------------------------- # 

ggplot(lc_correlation_02_long, aes(x = SampleSize, y = Accuracy, color = Model)) +
  geom_line(linewidth = 0.7) +
  geom_point() +
  geom_errorbar(aes(ymin = Accuracy_lower, ymax = Accuracy_upper), width = 0.05) +
  labs(x = "Sample Size", y = "Accuracy", title = "Linear Data at Correlation .2") +
  ylim(0.4, 1) +
  scale_x_log10(breaks = c(100, 200, 500, 1000)) + 
  theme_minimal()


ggplot(lc_correlation_04_long, aes(x = SampleSize, y = Accuracy, color = Model)) +
  geom_line(linewidth = 0.7) +
  geom_point() +
  geom_errorbar(aes(ymin = Accuracy_lower, ymax = Accuracy_upper), width = 0.05) +
  labs(x = "Sample Size", y = "Accuracy", title = "Linear Data at Correlation .4") +
  ylim(0.4, 1) +
  scale_x_log10(breaks = c(100, 200, 500, 1000)) + 
  theme_minimal()

# 5.3.2 ------------------ Nonlinear  data---------------------------------- # 

ggplot(lc_nonlinear_weak_long, aes(x = SampleSize, y = Accuracy, color = Model)) +
  geom_line(linewidth = 0.7) +
  geom_point() +
  geom_errorbar(aes(ymin = Accuracy_lower, ymax = Accuracy_upper), width = 0.05) +
  labs(x = "Sample Size", y = "Accuracy", title = "Nonlinear weak effect sizee") +
  ylim(0.3, 1) +
  scale_x_log10(breaks = c(100, 200, 500, 1000)) + 
  theme_minimal()

ggplot(lc_nonlinear_strong_long, aes(x = SampleSize, y = Accuracy, color = Model)) +
  geom_line(linewidth = 0.7) +
  geom_point() +
  geom_errorbar(aes(ymin = Accuracy_lower, ymax = Accuracy_upper), width = 0.05) +
  labs(x = "Sample Size", y = "Accuracy", title = "Nonlinear strong effect size") +
  ylim(0.3, 1) +
  scale_x_log10(breaks = c(100, 200, 500, 1000)) + 
  theme_minimal()

# 5.3.3 ----------------------- 2-way Interactions -------------------------------- # 

ggplot(lc_interaction_02_long, aes(x = SampleSize, y = Accuracy, color = Model)) +
  geom_line(linewidth = 0.7) +
  geom_point() +
  geom_errorbar(aes(ymin = Accuracy_lower, ymax = Accuracy_upper), width = 0.05) +
  labs(x = "Sample Size", y = "Accuracy", title = "2-Variable Interaction Weak") +
  ylim(0.3, 1) +
  scale_x_log10(breaks = c(100, 200, 500, 1000)) + 
  theme_minimal()

ggplot(lc_interaction_04_long, aes(x = SampleSize, y = Accuracy, color = Model)) +
  geom_line(linewidth = 0.7) +
  geom_point() +
  geom_errorbar(aes(ymin = Accuracy_lower, ymax = Accuracy_upper), width = 0.05) +
  labs(x = "Sample Size", y = "Accuracy", title = "2-Variable Interaction Strong") +
  ylim(0.3, 1) +
  scale_x_log10(breaks = c(100, 200, 500, 1000)) + 
  theme_minimal()



# 5.3.4 ----------------------- 4-way Interactions -------------------------------- # 

ggplot(lc_interaction4_02_long, aes(x = SampleSize, y = Accuracy, color = Model)) +
  geom_line(linewidth = 0.7) +
  geom_point() +
  geom_errorbar(aes(ymin = Accuracy_lower, ymax = Accuracy_upper), width = 0.05) +
  labs(x = "Sample Size", y = "Accuracy", title = "4-Variable Interaction Small Effect") +
  ylim(0.3, 1) +
  scale_x_log10(breaks = c(100, 200, 500, 1000)) + 
  theme_minimal()

ggplot(lc_interaction4_04_long, aes(x = SampleSize, y = Accuracy, color = Model)) +
  geom_line(linewidth = 0.7) +
  geom_point() +
  geom_errorbar(aes(ymin = Accuracy_lower, ymax = Accuracy_upper), width = 0.05) +
  labs(x = "Sample Size", y = "Accuracy", title = "4-Variable Interaction Large Effect") +
  ylim(0.4, 1) +
  scale_x_log10(breaks = c(100, 200, 500, 1000)) + 
  theme_minimal()


# Create one unifying graph


p1 <- ggplot(lc_correlation_02_long, aes(x = SampleSize, y = Accuracy, color = Model)) +
  geom_line(linewidth = 0.7) +
  geom_point() +
  geom_errorbar(aes(ymin = Accuracy_lower, ymax = Accuracy_upper), width = 0.05) +
  labs(x = "Sample Size", y = "Accuracy", title = "a) Linear Data Small Effect") +
  ylim(0.4, 1) +
  scale_x_log10(breaks = c(100, 200, 500, 1000)) + 
  theme_minimal()

p2 <- ggplot(lc_correlation_04_long, aes(x = SampleSize, y = Accuracy, color = Model)) +
  geom_line(linewidth = 0.7) +
  geom_point() +
  geom_errorbar(aes(ymin = Accuracy_lower, ymax = Accuracy_upper), width = 0.05) +
  labs(x = "Sample Size", y = "Accuracy", title = "b) Linear Data Large Effect") +
  ylim(0.4, 1) +
  scale_x_log10(breaks = c(100, 200, 500, 1000)) + 
  theme_minimal() + 
  theme(legend.position = "none")

# 5.3.1 ------------------ Nonlinear  data---------------------------------- # 

p3 <- ggplot(lc_nonlinear_weak_long, aes(x = SampleSize, y = Accuracy, color = Model)) +
  geom_line(linewidth = 0.7) +
  geom_point() +
  geom_errorbar(aes(ymin = Accuracy_lower, ymax = Accuracy_upper), width = 0.05) +
  labs(x = "Sample Size", y = "Accuracy", title = "c) Nonlinear Data Small Effect") +
  ylim(0.3, 1) +
  scale_x_log10(breaks = c(100, 200, 500, 1000)) + 
  theme_minimal()

p4 <- ggplot(lc_nonlinear_strong_long, aes(x = SampleSize, y = Accuracy, color = Model)) +
  geom_line(linewidth = 0.7) +
  geom_point() +
  geom_errorbar(aes(ymin = Accuracy_lower, ymax = Accuracy_upper), width = 0.05) +
  labs(x = "Sample Size", y = "Accuracy", title = "d) Nonlinear Data Large Effect") +
  ylim(0.3, 1) +
  scale_x_log10(breaks = c(100, 200, 500, 1000)) + 
  theme_minimal() + 
  theme(legend.position = "none")

# 5.3.1 ----------------------- Interactions -------------------------------- # 

p5 <- ggplot(lc_interaction_02_long, aes(x = SampleSize, y = Accuracy, color = Model)) +
  geom_line(linewidth = 0.7) +
  geom_point() +
  geom_errorbar(aes(ymin = Accuracy_lower, ymax = Accuracy_upper), width = 0.05) +
  labs(x = "Sample Size", y = "Accuracy", title = "e) 2-Variable Interaction Small Effect") +
  ylim(0.4, 1) +
  scale_x_log10(breaks = c(100, 200, 500, 1000)) + 
  theme_minimal()

p6 <- ggplot(lc_interaction_04_long, aes(x = SampleSize, y = Accuracy, color = Model)) +
  geom_line(linewidth = 0.7) +
  geom_point() +
  geom_errorbar(aes(ymin = Accuracy_lower, ymax = Accuracy_upper), width = 0.05) +
  labs(x = "Sample Size", y = "Accuracy", title = "f) 2-Variable Interaction Large Effect") +
  ylim(0.4, 1) +
  scale_x_log10(breaks = c(100, 200, 500, 1000)) + 
  theme_minimal() + 
  theme(legend.position = "none")


summary_lc <- (p1 + p2) / (p3 + p4) / (p5 + p6)

q1 <- ggplot(lc_interaction4_02_long, aes(x = SampleSize, y = Accuracy, color = Model)) +
  geom_line(linewidth = 0.7) +
  geom_point() +
  geom_errorbar(aes(ymin = Accuracy_lower, ymax = Accuracy_upper), width = 0.05) +
  labs(x = "Sample Size", y = "Accuracy", title = "a) 4-Variable Interaction Small Effect") +
  ylim(0.3, .8) +
  scale_x_log10(breaks = c(100, 200, 500, 1000)) + 
  theme_minimal()

q2 <- ggplot(lc_interaction4_04_long, aes(x = SampleSize, y = Accuracy, color = Model)) +
  geom_line(linewidth = 0.7) +
  geom_point() +
  geom_errorbar(aes(ymin = Accuracy_lower, ymax = Accuracy_upper), width = 0.05) +
  labs(x = "Sample Size", y = "Accuracy", title = "b) 4-Variable Interaction Large Effect") +
  ylim(0.3, .8) +
  scale_x_log10(breaks = c(100, 200, 500, 1000)) + 
  theme_minimal() +
theme(legend.position = "none")

(q1+q2)

library(data.table)
library(dplyr)
library(dummies)
library(xgboost)

setwd('/projetos/machine-learning-challenge-2/nubank')

# Feature Engineering -------------------------------------------------------------------------

create_feature <- function (data) {
  data$project_id <- data$project_id

  if (class(as.factor(data$score_1)) != "factor")
    data$score_1 <- as.factor(data$score_1)
  data <- dummy.data.frame(data, names=c("score_1"), sep="OHE_")

  if (class(as.factor(data$score_2)) != "factor")
    data$score_2 <- as.factor(data$score_2)
  data <- dummy.data.frame(data, names=c("score_2"), sep="OHE_")
  
  data$log_score_5 <- log1p(data$score_5)
  
  data$has_facebook_profile <- as.numeric(data$facebook_profile == 'True')
  data$is_female <- as.numeric(data$gender == 'f')
  
  data$last_paymentDate <- as.Date(data$last_payment)
  data$end_last_loanDate <- as.Date(data$end_last_loan)
  
  #signos
  if (class(as.factor(data$sign)) != "factor")
    data$sign <- as.factor(data$sign)
  data <- dummy.data.frame(data, names=c("sign"), sep="OHE_")

  #channel
  data$channel_NotEmpty <- as.numeric(data$channel != '')
  
  #real_state
  if (class(as.factor(data$real_state)) != "factor")
    data$real_state <- as.factor(data$real_state)
  data <- dummy.data.frame(data, names=c("real_state"), sep="OHE_")

  data$end_last_loanFac <- as.factor(year(data$end_last_loanDate)*100 + month(data$end_last_loanDate))
  data <- dummy.data.frame(data, names=c("end_last_loanFac"), sep="OHE_")
  
  data$last_paymentFac <- as.factor(year(data$last_paymentDate)*100 + month(data$last_paymentDate))
  data <- dummy.data.frame(data, names=c("last_paymentFac"), sep="OHE_")

  data
}
train <- read.csv('puzzle_train_dataset.csv', stringsAsFactors = FALSE)

train <- create_feature(train)
train$default <- as.numeric(train$default == "True")

ntrain <- names(train)
score_1_cols <- ntrain[grep('^score_1OHE_', ntrain)]
real_state_cols <- ntrain[grep('^real_stateOHE_', ntrain)]
sign_cols <- ntrain[grep('^signOHE_', ntrain)]

test <- read.csv('puzzle_test_dataset.csv', stringsAsFactors = FALSE)
test <- create_feature(test)

ntest <- names(test)
last_paymentFac_trainCols <- ntrain[grep('^last_paymentFacOHE_', ntrain)]
last_paymentFac_testCols <- ntest[grep('^last_paymentFacOHE_', ntest)]
last_paymentFac_cols <- intersect(last_paymentFac_trainCols, last_paymentFac_testCols)

end_last_loanFac_trainCols <- ntrain[grep('^end_last_loanFacOHE_', ntrain)]
end_last_loanFac_testCols <- ntest[grep('^end_last_loanFacOHE_', ntest)]
end_last_loanFac_cols <- intersect(end_last_loanFac_trainCols, end_last_loanFac_testCols)

# Model setup ---------------------------------------------------------------------------------

cols_to_use <- c('risk_rate'
                 , 'amount_borrowed'
                 , 'borrowed_in_months'
                 , 'credit_limit'
                 , 'income'
                 , 'ok_since'
                 , 'n_bankruptcies'
                 , 'n_defaulted_loans'
                 , 'log_score_5'
                 , 'has_facebook_profile'
                 , 'is_female'
                 , 'channel_NotEmpty'
                 , score_1_cols
                 , real_state_cols
                 , sign_cols
                 , last_paymentFac_cols
                 , end_last_loanFac_cols
)

params <- list(
  eta = 0.025,
  max_depth = 6,
  min_child_weight = 5,
  gamma = 0,
  subsample = 0.8,
  colsample_bytree = 0.8,
  scale_pos_weight = 1,
  seed = 1,
  objective = "binary:logistic"
)
dtrain = xgb.DMatrix(data = as.matrix(train[,cols_to_use]), label = as.matrix(train$default))
dtest <- xgb.DMatrix(data = as.matrix(test[cols_to_use]))

# Train ---------------------------------------------------------------------------------------

big_cv <- xgb.cv(params = params
                 ,data = dtrain
                 ,nrounds = 5000
                 ,nfold = 5L
                 ,metrics = list("logloss") #list("error", "auc")
                 ,stratified = T
                 ,print_every_n = 100
                 ,early_stopping_rounds = 40)

iter <- big_cv$best_iteration

big_train <- xgb.train(params = params
                       ,data = dtrain
                       ,nrounds = iter)
#[1078]	train-error:0.061993+0.000438	train-auc:0.963888+0.000463	test-error:0.073895+0.001521	test-auc:0.944723+0.002793
#[1147]	train-error:0.061521+0.000572	train-auc:0.965054+0.000580	test-error:0.073957+0.002932	test-auc:0.944662+0.003005
#[1209]	train-logloss:0.158898+0.001239	test-logloss:0.192858+0.004951

# Eval ----------------------------------------------------------------------------------------
preds <- predict(big_train, dtest)
mean(preds)

imp <- xgb.importance(model = big_train, feature_names = colnames(dtrain))
xgb.plot.importance(imp)

#xgb.plot.tree(model = big_train, feature_names = colnames(dtrain), n_first_tree = 2)

# Write ---------------------------------------------------------------------------------------
sub <- data.table(ids = test$ids, predictions = preds)
fwrite(sub, "xgb_marcos_choma.csv")

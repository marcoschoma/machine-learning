library(dplyr)
library(dummies)
library(xgboost)
library(DiagrammeR)

setwd('/projetos/nubank')

# Feature Engineering -------------------------------------------------------------------------

create_feature <- function (data) {
  #score_1 <- OHE
  if (class(as.factor(data$score_1)) != "factor")
    data$score_1 <- as.factor(data$score_1)
  data <- dummy.data.frame(data, names=c("score_1"), sep="OHE_")
  #score_2 <- OHE
  if (class(as.factor(data$score_2)) != "factor")
    data$score_2 <- as.factor(data$score_2)
  data <- dummy.data.frame(data, names=c("score_2"), sep="OHE_")
  
  #score_5 <- aplicar log
  data$log_score_5 <- log1p(data$score_5) #não tem correlação com risk_rate
  
  #risk_rate, score_3, 4, e 6 -> distribuicao normal
  
  #variaveis correlacionadas: risk_rate, amount_borrowed e score_3
  data$has_facebook_profile <- as.numeric(data$facebook_profile == 'True')
  data$is_female <- as.numeric(data$gender == 'f')
  
  data$last_paymentDate <- as.Date(data$last_payment)
  data$end_last_loanDate <- as.Date(data$end_last_loan)
  
  #signos
  #dizem q os bons são: touro, cancer, virgem, libra, capricornio
  data <- dummy.data.frame(data, names=c("sign"), sep="_")

  
  #channel
  data$channel_NotEmpty <- as.numeric(data$channel != '')
  
  data
}
train <- read.csv('puzzle_train_dataset.csv', stringsAsFactors = FALSE)
train <- create_feature(train)
train$default <- as.numeric(train$default == "True")

test <- read.csv('puzzle_test_dataset.csv', stringsAsFactors = FALSE)
test <- create_feature(test)

#### devs
levels(as.factor(test$channel))

names(train)
str(train)
mean(as.numeric(is.na(train$credit_limit)))

train[is.na(train$sign),]

View(train[1:10,])
View(filter(train, default == 'True')[1:10,])

groupByState <- group_by(train, fac = as.numeric(as.factor(state))) %>% summarise(t = n())
hist(data = groupByState, x = groupByState$fac, freq = groupByState$t)
View(groupByState)

sample_train <- sample_frac(train, 0.2)
sample_train
mean(train$income, na.rm = TRUE)
mean(train$credit_limit, na.rm = TRUE)
mean(train$amount_borrowed, na.rm = TRUE)

mean(train$amount_borrowed/train$borrowed_in_months, na.rm = TRUE)

test <- read.csv('puzzle_test_dataset.csv', stringsAsFactors = FALSE)

levels(as.factor(train$sign))

train %>% group_by(default, signo_bom) %>% summarize(n)
mean(train$default)
mean(filter(train, default == 1)$signo_bom)
mean(train$signo_bom)


#reason
# r0 <- "mLVIVxoGY7TUDJ1FyFoSIZi1SFcaBmO01AydRchaEiGYtUhXGgZjtNQMnUXIWhIh"
# r1 <- str_replace_all(string = train$reason,r0, "")
# r2 <- str_sub(r1, 1, 22)
# levels(as.factor(r2))
# r3 <- str_sub(r1, 22, 43)
# r4 <- str_sub(r1, 44, 44)
# r5 <- str_sub(r1, 45)
# levels(as.factor(r5))
##Fs+LBbioEVdUtVclKd053
#levels(as.factor(test$score_2))


#converter last_payment, end_last_loan para data

#analisar NA's na variavel default
train_default <- train[train$default == "",] #4626
nrow(train_default)

#quantidades de default
train_default_true <- train[train$default == "True",]
train_default_false <- train[train$default == "False",]

str(train_default_true) #9510
nrow(train_default_false) #50456

boxplot(train$default)

# Model setup ---------------------------------------------------------------------------------

cols_to_use <- c('risk_rate'
                 , 'amount_borrowed'
                 , 'borrowed_in_months'
                 , 'credit_limit'
                 , 'income'
                 
                 , 'n_bankruptcies'
                 , 'n_defaulted_loans'
                 , 'n_accounts'
                 , 'n_issues'
                 
                 , 'log_score_5'
                 , 'has_facebook_profile'
                 , 'is_female'
                 , 'channel_NotEmpty'
                 , 'sign_aqua' ,'sign_arie' ,'sign_cance' ,'sign_capr' ,'sign_gemi' ,'sign_leo' ,'sign_libr' ,'sign_pisce' ,'sign_scor' ,'sign_taur' ,'sign_virg'
                 #,'sign_sagi'
                 
                 
                 , 'score_1OHE_'
                 , 'score_1OHE_1Rk8w4Ucd5yR3KcqZzLdow=='
                 , 'score_1OHE_4DLlLW62jReXaqbPaHp1vQ=='
                 , 'score_1OHE_8k8UDR4Yx0qasAjkGrUZLw=='
                 , 'score_1OHE_DGCQep2AE5QRkNCshIAlFQ=='
                 , 'score_1OHE_e4NYDor1NOw6XKGE60AWFw=='
                 , 'score_1OHE_fyrlulOiZ+5hoFqLa6UbDQ=='
                 , 'score_1OHE_smzX0nxh5QlePvtVf6EAeg=='
                 # , 'score_2OHE_'
                 # , 'score_2OHE_/tdlnWjXoZ3OjdtBXzdOJQ=='
                 # , 'score_2OHE_+2hzpeP1RWr8PEvL1WTUdw=='
                 # , 'score_2OHE_+CxEO4w7jv3QPI/BQbyqAA=='
                 # , 'score_2OHE_5/uMrqKj3OL/Xk5OrGx9fg=='
                 # , 'score_2OHE_55UK234RR1d7HIWJjmq9tw=='
                 # , 'score_2OHE_6J1ZMTzN5GKHXnhM4J1JbA=='
                 # , 'score_2OHE_7h+tk4z7O9brtBSe1rNjxA=='
                 # , 'score_2OHE_7h8PTkrlTWUPP3yuyP4rUg=='
                 # , 'score_2OHE_A+QuW1n/ABeiVVe/9CRZ9Q=='
                 # , 'score_2OHE_bopP0NxW3+r8tn9xIHTaOw=='
                 # , 'score_2OHE_cdpgyOyZS04uXerMNu7uCw=='
                 # , 'score_2OHE_d/7Hedyz7ovK9Pn1CYN4+A=='
                 # , 'score_2OHE_dCm9hFKfdRm7ej3jW+gyxw=='
                 # , 'score_2OHE_dWJRASUFMejk3AHZ1p1Gkg=='
                 # , 'score_2OHE_emS9xH8CLoRNie2uSmaDAQ=='
                 # , 'score_2OHE_Fv28Bz0YRTVAT5kl1bAV6g=='
                 # , 'score_2OHE_IOVu8au3ISbo6+zmfnYwMg=='
                 # , 'score_2OHE_ky19q4V1ZqgL3jnHX0wKDw=='
                 # , 'score_2OHE_LCak332j+TYFqHC3NDwiqg=='
                 # , 'score_2OHE_mX2VRRG38RPiHX+MfjefRw=='
                 # , 'score_2OHE_NLvAOzzmJba/0zolQnWF5Q=='
                 # , 'score_2OHE_O4i7FxcROACMVTCgI0WXuA=='
                 # , 'score_2OHE_OlDYtdljgSSYM/M1L2CRaQ=='
                 # , 'score_2OHE_osCzpM4hJrxugqWWuZmMWw=='
                 # , 'score_2OHE_pAzpxkhjPsjWldgSX21+zg=='
                 # , 'score_2OHE_rJZgTmANW3PjOCQLCcp4iQ=='
                 # , 'score_2OHE_RO7MTL+j4PH2gNzbhNTq/A=='
                 # , 'score_2OHE_SaamrHMo23l/3TwXOWgVzw=='
                 # , 'score_2OHE_tHpS8e9F8d9zg3iOQM9tsA=='
                 # , 'score_2OHE_tQUTfUyeuGkhRotd+6WjVg=='
                 # , 'score_2OHE_vJyc9xom9v7hwFMPTIpmKw=='
                 # , 'score_2OHE_w1miZqhB5+RSamEQJa0rqg=='
                 # , 'score_2OHE_wjdj2vxjWoDsEIk0l09ynw=='
                 # , 'score_2OHE_wkeCdGeu5sEv4/fjwR0aDg=='
                 # , 'score_2OHE_YLGMUI9hObSh6wD/xfanGg=='
                 
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
                 ,metrics = list("error","auc")#'error'
                 ,stratified = T
                 ,print_every_n = 100
                 ,early_stopping_rounds = 40)

iter <- big_cv$best_iteration

big_train <- xgb.train(params = params
                       ,data = dtrain
                       ,nrounds = iter)
#16:25 [383]	train-error:0.130465+0.000183	train-auc:0.807786+0.001239	test-error:0.140931+0.000578	test-auc:0.752506+0.004674
#sem as principais features (income, credit_limit, amout_borrowed e borrowed_in_months)
#   [253]	train-error:0.136476+0.000573	train-auc:0.769013+0.000838	test-error:0.141984+0.002143	test-auc:0.741568+0.003149
#com todas as features até então:
#[306]	train-error:0.131986+0.000488	train-auc:0.799066+0.001369	test-error:0.140327+0.002804	test-auc:0.752013+0.007232
#sem score_2:
#[268]	train-error:0.132698+0.000653	train-auc:0.793055+0.000894	test-error:0.140126+0.002578	test-auc:0.749747+0.003049
#sem signo de sagitário:
#

# Eval ----------------------------------------------------------------------------------------
xgb.plot.tree(model = big_train)

preds <- predict(big_train, dtest)
big_pred <- ifelse(preds > 0.5,1,0)
mean(ifelse(preds > 0.5,1,0))

imp <- xgb.importance(model = big_train, feature_names = colnames(dtrain))
imp
xgb.plot.importance(imp)

# Write ---------------------------------------------------------------------------------------
sub <- data.table(project_id = test$project_id, final_status = big_pred)
fwrite(sub, "xgb_with_feats.csv")

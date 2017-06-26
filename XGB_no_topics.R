library(data.table)
library(lubridate)
library(stringr)
library(RTextTools)
library(tidytext)
library(plyr)
library(dplyr)
library(gbm)
library(topicmodels)
library(ggplot2)
library(xgboost)
library(countrycode)
library(dummies)
set.seed(1)
setwd('/projetos/machine-learning-challenge-2/')
train <- fread("train.csv")
test <- fread("test.csv")

unix_feats <- c('deadline','state_changed_at','created_at','launched_at')
train[,c(unix_feats) := lapply(.SD, function(x) structure(x, class=c('POSIXct'))), .SDcols = unix_feats]
test[,c(unix_feats) := lapply(.SD, function(x) structure(x, class=c('POSIXct'))), .SDcols = unix_feats]

len_feats <- c('name_len','desc_len','keywords_len')
count_feats <- c('name_count','desc_count','keywords_count')
cols <- c('name','desc','keywords')

train[,c(len_feats) := lapply(.SD, function(x) str_count(x)), .SDcols = cols]
train[,c(count_feats) := lapply(.SD, function(x) str_count(x,"\\w+")), .SDcols = cols]
test[,c(len_feats) := lapply(.SD, function(x) str_count(x)), .SDcols = cols]
test[,c(count_feats) := lapply(.SD, function(x) str_count(x,"\\w+")), .SDcols = cols]

currencies = c('AUD', 'CAD', 'DKK', 'EUR', 'GBP', 'NOK', 'NZD', 'SEK', 'USD')
currency_mean = c(0.9451, 0.9546, 0.1507, 1.3291, 1.5883, 0.1666, 0.7787, 0.1451, 1)
my_currency <- data.frame(currencies, currency_mean, stringsAsFactors = FALSE)

createFeatures <- function(data) {
  data$prep_time <- as.numeric(difftime(data$launched_at, data$created_at, units = c("days")))
  data$duration <- as.numeric(difftime(data$deadline, data$launched_at, units = c("days")))
  data$extratime <- as.numeric(difftime(data$deadline, data$state_changed_at, units = c("days")))
  
  data$probablyCanceled <- as.numeric(difftime(data$deadline, data$state_changed_at, units = c("days")) > 0)
  data$bin_disable_communication <- as.numeric(data$disable_communication == "true")
  
  currencyJoinedData <- data %>% left_join(my_currency, c('currency' = 'currencies'))
  data$dollarValue <- as.double(currencyJoinedData$currency_mean) * currencyJoinedData$goal
  
  data$sazonalityStart <- week(data$created_at)
  data$sazonalityEnd <- week(data$deadline)
  data$isWeekend <- as.numeric(weekdays.POSIXt(data$deadline, abbreviate = TRUE) == "dom" || weekdays.POSIXt(data$deadline, abbreviate = TRUE) == "sab")
  
  data$launchHour <- hour(data$launched_at)

  data$isNorthernAmerica <- countrycode(data$country, 'iso2c', 'region') == "Northern America"
  #as.factor(countrycode(data$country, 'iso2c', 'region'))
  #dummy.data.frame(df, names=c("region"), sep="_")
  
  #data$is2009 <- as.numeric(year(data$created_at) == 2009)
  data$is2010 <- as.numeric(year(data$created_at) == 2010)
  data$is2011 <- as.numeric(year(data$created_at) == 2011)
  data$is2012 <- as.numeric(year(data$created_at) == 2012)
  data$is2013 <- as.numeric(year(data$created_at) == 2013)
  data$is2014 <- as.numeric(year(data$created_at) >= 2014)
  data$isAfter2015 <- as.numeric(year(data$created_at) >= 2015)
  #data$is2016 <- as.numeric(year(data$deadline) == 2016)
  #data$is2017 <- as.numeric(year(data$deadline) == 2017)
  
  data
}
train <- createFeatures(train)
test <- createFeatures(test)

positiveWords <- read.csv('positive-words.txt', comment.char = ';')
negativeWords <- read.csv('negative-words.txt', comment.char = ';')

pWord <- data.frame(term=positiveWords, score=1, stringsAsFactors = FALSE)
nWord <- data.frame(term=negativeWords, score=-1, stringsAsFactors = FALSE)
names(pWord) <- c('word', 'score')
names(nWord) <- c('word', 'score')
afinn_set <- rbind(pWord, nWord)

getSentimentScore <- function(data) {
  words <- data %>% 
    mutate(text = limparTexto(paste(desc, name, keywords))) %>%
    unnest_tokens(word, desc) %>%
    anti_join(., stop_words) %>%
    left_join(., afinn_set) %>%
    group_by(project_id) %>%
    summarise(sentiment_score = sum(score, na.rm = TRUE))
  
  left_join(data, words)
}

limparTexto <- function(data) {
  data <- gsub("\"", "", data)#984
  data <- gsub("[^[:alnum:]]", " ", data)#984
  data <- gsub("  ", " ", data)#984
  data
}

train <- getSentimentScore(train)
test <- getSentimentScore(test)

#### XGBOOST
cols_to_use <- c('sentiment_score'
                 ,'prep_time'
                 ,'duration'
                 ,'extratime'
                 ,'dollarValue'
                 ,'name_len','desc_len','keywords_len' #0.67068
                 ,'desc_count'
                 ,'bin_disable_communication'
                 ,'sazonalityStart'
                 ,'sazonalityEnd'
                 ,'launchHour'
                 ,'probablyCanceled'
                 ,'isNorthernAmerica'
                 ,'isWeekend'
                 ,'is2010'
                 ,'is2011'
                 ,'is2012'
                 ,'is2013'
                 ,'is2014'
                 ,'isAfter2015'
                 #,'is2009'
                 #,'is2016'
                 #,'is2017'
                 #,'as.factor.topic.1'
                 #,'as.factor.topic.2'
                 #,'as.factor.topic.3'
                 #,'as.factor.topic.4'
                 #,'as.factor.topic.5'
                 #,'as.factor.topic.6'
                 #,'as.factor.topic.7'
                )

dtrain = xgb.DMatrix(data = as.matrix(train[,cols_to_use]), label = as.matrix(train$final_status))

my_xgb <- xgboost(data = dtrain,
                  #label = train$final_status,
                  nround = 100,#5000,
                  stratified = TRUE,
                  eta = 0.025,
                  max_depth = 6,
                  min_child_weight = 5,
                  gamma = 0,
                  subsample = 0.8,
                  colsample_bytree = 0.8,
                  scale_pos_weight = 1,
                  seed = 1,
                  eval_metric="error",
                  objective = "binary:logistic",
                  early_stopping_rounds = 25,
                  print_every_n = 100,
                  lambda = 0.01
)#[2000]	train-auc:0.952219 - subsample = 0.8
#[3000]	train-auc:0.981061 - subsample = 0.8
#[3000]	train-error:0.075271  - subsample = 0.6
#[2476]	train-error:0.229837 - max_depth = 6, subsample = 0.6, eta = 0.02
#[3000]	train-error:0.219497 - max_depth = 6, subsample = 0.8, eta = 0.02
#[3995]	train-error:0.198420 - max_depth = 6, subsample = 0.8, eta = 0.02
#[5000]	train-error:0.008952 - max_depth = 10, subsample = 0.8, eta = 0.02
#[5000]	train-error:0.008952 - max_depth = 10, subsample = 0.8, eta = 0.02
#[5000]	train-error:0.075132 - max_depth = 8, subsample = 0.8, eta = 0.02
#[5000]	train-error:0.083484 
##mudando para: min_child_weight
#[5000]	train-error:0.172854 - max_depth = 6, subsample = 0.8, eta = 0.025
#[5000]	train-error:0.170431 - lambda = 0.01  max_depth = 6, subsample = 0.8, eta = 0.025

preds <- predict(my_xgb, data.matrix(test[,cols_to_use]))

vrange <- 1:500/1000 + 0.1
result = lapply(vrange, function (x) { ifelse(x < preds,1,0) })
medias <- lapply(result, mean)
plot(vrange, medias)
mean(ifelse(vrange/1000 + 0.1 < preds,1,0))

#0.65968%
#nrow(test) * 0.65968
#nrow(test) - 41866
#sort(preds, decreasing = TRUE)[21599] ->>0.3432834

xgb_subst <- data.table(project_id = test$project_id, final_status = ifelse(preds > 0.3432834,1,0)) #0.69312
mean(xgb_subst$final_status)

summary(my_xgb)
importance_matrix <- xgb.importance(feature_names = cols_to_use, model = my_xgb)
print(importance_matrix)
xgb.ggplot.importance(importance_matrix = importance_matrix)

fwrite(xgb_subst, "xgb.csv") #0.70698


plot_sample <- cbind(test, preds) #sample_frac(filter(train, extratime >= 0), 0.1)
ggplot(plot_sample, aes(x = preds, y = log10(dollarValue))) + geom_point(aes(colour=ifelse(preds > 0.4,1,0)))
ggplot(plot_sample, aes(x = preds, y = log10(duration))) + geom_point(aes(colour=ifelse(preds > 0.4,1,0)))
ggplot(filter(plot_sample, dollarValue < 1e+05), aes(x = dollarValue, y = duration)) + geom_point(aes(colour=ifelse(preds > 0.4,1,0)))

boxplot(duration ~ final_status, data = train)

year(train$created_at)
min(train$created_at)
max(train$created_at)
min(test$created_at)
max(test$created_at)

max(train$deadline)
max(test$deadline)
plot(train$created_at)

##com CV
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
#0.8 sample: [422]	train-error:0.276526+0.000616	test-error:0.294435+0.001734
#0.7 sample: [253]	train-error:0.284968+0.000245	test-error:0.295730+0.001837
#[765]	train-error:0.306625+0.001666	test-error:0.347178+0.002624
#[572]	train-error:0.269541+0.000689	train-auc:0.776459+0.001051	test-error:0.294843+0.003360	test-auc:0.732995+0.004272
#0.71798 - [613]	train-error:0.268998+0.000797	train-auc:0.777442+0.000571	test-error:0.294935+0.001695	test-auc:0.732785+0.002123 

dtrain = xgb.DMatrix(data = as.matrix(train[,cols_to_use]), label = as.matrix(train$final_status))
dtest <- xgb.DMatrix(data = as.matrix(test[cols_to_use]))

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

imp <- xgb.importance(model = big_train, feature_names = colnames(dtrain))
xgb.plot.importance(imp)

sort(preds, decreasing = TRUE)[21599]

preds <- predict(big_train, dtest)
big_pred <- ifelse(preds > 0.3432834,1,0)
mean(ifelse(preds > 0.3432834,1,0))
sub <- data.table(project_id = test$project_id, final_status = big_pred)
fwrite(sub, "xgb_with_feats.csv") #0.68225

#0.3 = 0.69920 (0.3350193)
#0.4 = 0.70383 (0.2167494)
#0.5 = 0.69513 (0.123706)
# 
# cols_to_use %in% names(test)
# names(test)
# test[,cols_to_use, with=FALSE]


##### features em dev
rolou = filter(train, final_status == 1 )
hist(x = week(rolou$created_at))

sazonality <- week(rolou$created_at)

nrolou = filter(train, final_status == 0 )
mfrow = c(2, 1)
hist(x = week(rolou$created_at))
hist(x = week(nrolou$created_at))


##outra abordagem
cv.ctrl <- trainControl(method = "repeatedcv", repeats = 1,number = 3, 
                        #summaryFunction = twoClassSummary,
                        classProbs = TRUE,
                        allowParallel=T)

xgb.grid <- expand.grid(
  nrounds = 1000,
  eta = c(0.01, 0.001, 0.0001),
  max_depth = c(2, 4, 6, 8, 10),
  gamma = 1,
  colsample_bytree = c(0.5,0.7,0.9),
  subsample = c(0.5,0.7,0.9),
  min_child_weight = c(1, 6, 2)
)

set.seed(45)
xgb_tune <-train(train$final_status~.,
                 data=train[,..cols_to_use],
                 method="xgbTree",
                 trControl=cv.ctrl,
                 tuneGrid=xgb.grid,
                 verbose=T,
                 metric="auc",
                 nthread =3
)

############# caret + CV + hyper tunning
#install.packages('caret')
library(caret)
#install.packages('readr')
#library(readr)

# xgboost fitting with arbitrary parameters
xgb_params_1 = list(
  objective = "binary:logistic",                                               # binary classification
  eta = 0.01,                                                                  # learning rate
  max.depth = 3,                                                               # max tree depth
  eval_metric = "auc"                                                          # evaluation/loss metric
)

# fit the model with the arbitrary parameters specified above
xgb_1 = xgboost(data = dtrain,
                params = xgb_params_1,
                nrounds = 100,                                                 # max number of trees to build
                verbose = TRUE,                                         
                print_every_n = 1,
                early_stopping_rounds = 10                                          # stop if no improvement within 10 trees
)

# cross-validate xgboost to get the accurate measure of error
xgb_cv_1 = xgb.cv(params = xgb_params_1,
                  data = dtrain,
                  nrounds = 100, 
                  nfold = 5,                                                   # number of folds in K-fold
                  prediction = TRUE,                                           # return the prediction using the final model 
                  showsd = TRUE,                                               # standard deviation of loss across folds
                  stratified = TRUE,                                           # sample is unbalanced; use stratified sampling
                  verbose = TRUE,
                  print_every_n = 1, 
                  early_stopping_rounds = 10,
                  metrics = "auc"
)

# plot the AUC for the training and testing samples
xgb_cv_1 %>%
#  select(-contains("std")) %>%
  mutate(IterationNum = 1:n()) %>%
  gather(TestOrTrain, AUC, -IterationNum) %>%
  ggplot(aes(x = IterationNum, y = AUC, group = TestOrTrain, color = TestOrTrain)) + 
  geom_line() + 
  theme_bw()

# set up the cross-validated hyper-parameter search
xgb_grid_1 = expand.grid(
  nrounds = 1000,
  eta = c(0.01, 0.001, 0.0001),
  max_depth = c(2, 4, 6, 8, 10),
  gamma = 1,
  colsample_bytree = c(0.5,0.7,0.9),
  subsample = c(0.5,0.7,0.9),
  min_child_weight = c(1, 6, 2)
)

# pack the training control parameters
xgb_trcontrol_1 = trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE,
  returnData = FALSE,
  returnResamp = "all",                                                        # save losses across all models
  classProbs = TRUE,                                                           # set to TRUE for AUC to be computed
  summaryFunction = twoClassSim,
  allowParallel = TRUE
)

# train the model for each parameter combination in the grid, 
#   using CV to evaluate
xgb_train_1 = train(
  x = as.matrix(train[,cols_to_use, with=FALSE]),
  y = as.matrix(train$final_status),
  trControl = xgb_trcontrol_1,
  tuneGrid = xgb_grid_1,
  method = "xgbTree"
)
xgb_grid_1
# scatter plot of the AUC against max_depth and eta
ggplot(xgb_train_1$results, aes(x = as.factor(eta), y = max_depth, size = ROC, color = ROC)) + 
  geom_point() + 
  theme_bw() + 
  scale_size_continuous(guide = "none")


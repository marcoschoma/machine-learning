library(data.table)
library(lubridate)
library(stringr)
library(RTextTools)
library(tidytext)
library(dplyr)
library(gbm)
library(topicmodels)
library(xgboost)
library(Matrix)
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



# x0 <- filter(test, project_id == 'kkst1382298322')
# abs(as.numeric(difftime(x0$deadline, x0$state_changed_at, units = c("secs"))))
# rolou = filter(train, final_status == 1 )
# rolou$aoMesmoTempo <- with(rolou, as.numeric(difftime(deadline, state_changed_at, units = c("secs"))) ==0)
# mean(rolou$aoMesmoTempo)
# 
# mean(with(rolou, abs(as.numeric(difftime(deadline, state_changed_at, units = c("secs"))))))
# 
# 
# n_rolou = filter(train, final_status == 0 )
# n_rolou$aoMesmoTempo <- with(n_rolou, as.numeric(difftime(deadline, state_changed_at, units = c("secs"))) ==0)
# 
# mean(n_rolou$aoMesmoTempo)
# 
# with(n_rolou, abs(as.numeric(difftime(deadline, state_changed_at, units = c("secs")))))
# mean(with(n_rolou, abs(as.numeric(difftime(deadline, state_changed_at, units = c("secs"))))))
# 
# paste(rolou$project_id, '/',rolou$keywords, sep = '')

#dates features
createFeatures <- function(data) {
  data$prep_time <- as.numeric(difftime(data$created_at, data$launched_at, units = c("days")))
  data$lifespan <- as.numeric(difftime(data$deadline, data$created_at, units = c("days")))
  data$duration <- as.numeric(difftime(data$deadline, data$launched_at, units = c("days")))
  data$extratime <- as.numeric(difftime(data$deadline, data$state_changed_at, units = c("days")))
  
  data$probablyCanceled <- as.numeric(difftime(data$deadline, data$state_changed_at, units = c("days")))
  
  data$disable_communication <- as.integer(as.factor(data$disable_communication))-1
  data <- data %>% left_join(my_currency, c('currency' = 'currencies')) %>%
    mutate('dollarValue' = as.double(currency_mean) * goal)
  
  
  
  #by_month <- train %>% group_by(deadlineStr, final_status) %>% summarise(count = n())
  #data$deadlineStr <- year(data$deadline) * 100 + month(data$deadline)
  #data$created_atStr <- year(data$created_at) * 100 + month(data$created_at)
  #data$launched_atStr <- year(data$launched_at) * 100 + month(data$launched_at)
  #data$state_changed_atStr <- year(data$state_changed_at) * 100 + month(data$state_changed_at)
  
  data$incomeRate <- data$dollarValue / data$duration
  data
}
train <- createFeatures(train)
test <- createFeatures(test)
# filter(test, project_id == 'kkst102036154')
#createFeatures(my_sample)
# my_sample <- sample_frac(train, 0.01)
# my_sample <- my_sample %>% inner_join(my_currency, c('currency' = 'currencies')) %>%
#   mutate('dollarValue' = as.double(currency_mean) * goal)

#my_sample %>% mutate(text = paste(desc, name, keywords)) %>% unnest_tokens(word, text)

#sentiment feature

afinn_set <- get_sentiments('afinn')
getSentimentScore <- function(data) {
  words <- data %>% 
    mutate(text = paste(desc, name, keywords)) %>%
    unnest_tokens(word, desc) %>%
    anti_join(., stop_words) %>%
    left_join(., afinn_set) %>%
    group_by(project_id) %>%
    summarise(sentiment_score = sum(score, na.rm = TRUE))
  
  left_join(data, words)
}

train <- getSentimentScore(train)
test <- getSentimentScore(test)

# cols to use in modeling
cols_to_use <- c('final_status'
                 ,'sentiment_score'
                 #,'disable_communication'
                 ,'prep_time'
                 ,'lifespan'
                 ,'duration'
                 ,'extratime'
                 ,'dollarValue'
                 ,'incomeRate' #0.67276!!
                 ,'name_len','desc_len','keywords_len' #0.67068
                 ,'name_count','desc_count','keywords_count'
                 #,'deadlineStr'
                 #,'state_changed_atStr'
                 #,'launched_atStr'
                 #,'created_atStr'
)

# GBM
clf_model <- gbm(final_status ~ .
                 ,data = train[,cols_to_use]
                 ,n.trees = 583
                 ,interaction.depth = 5
                 ,shrinkage = 0.03
                 ,train.fraction = 0.6
                 ,verbose = T
                 , distribution = "adaboost" #adaboost=0.66775 #bernoulli = 0.66595
                 , cv.folds = 3
)
summary(clf_model)

# make predictions
clf_pred <- predict(clf_model, newdata = test, n.trees = 581, type = 'response')
#subst <- data.table(project_id = test$project_id, final_status = ifelse(clf_pred >= 0.3521892,1,0)) #0.69312
subst <- data.table(project_id = test$project_id, final_status = ifelse(clf_pred > 0.4,1,0)) #0.69312
mean(subst$final_status)
#0.38 = 0.69961
#.395 = 0.70128
#0.4 = 0.70200
#0.45 = 0.70002
#0.5 = 0.69495
fwrite(subst, "gbm.csv")
View(test)

######################## XGBoost
cols_test = c('sentiment_score'
              ,'prep_time'
              ,'lifespan'
              ,'duration'
              ,'extratime'
              ,'dollarValue'
              ,'incomeRate'
              ,'name_len','desc_len','keywords_len'
              ,'name_count','desc_count','keywords_count'
              ,'deadlineStr'
              ,'state_changed_atStr'
              ,'launched_atStr'
              ,'created_atStr'
              )

dtrain = xgb.DMatrix(data = as.matrix(train[,cols_test]), label = as.matrix(train$final_status))

my_xgb <- xgboost(data = dtrain,
              #label = train$final_status,
              eta = 0.01,
              max_depth = 15,
              nround = 500,
              subsample = 0.6,
              colsample_bytree = 0.6,
              seed = 1,
              #eval_metric = "merror",
              objective = "binary:logistic",
              early_stopping_rounds = 15
              #num_class = 12,
              #nthread = 3
            )
summary(my_xgb)
preds <- predict(my_xgb, data.matrix(test[cols_test]))

importance_matrix <- xgb.importance(model = my_xgb)
print(importance_matrix)
xgb.plot.importance(importance_matrix = importance_matrix)

xgb_subst <- data.table(project_id = test$project_id, final_status = ifelse(preds > 0.4,1,0)) #0.69312
fwrite(xgb_subst, "xgb.csv")

names(test)
test[,cols_to_use]

my_sample <- sample_n(train, 10000)
dist_incomeRate = dist(my_sample$lifespan)
hc <- hclust(dist_incomeRate)
plot(hc,  labels = FALSE, hang = -1, main = "Original Tree")

memb <- cutree(hc, k = 10)
opar <- par(mfrow = c(1, 1))
par(opar)


# load data and libraries
library(data.table)
library(lubridate)
library(stringr)
library(RTextTools)
library(tidytext)
library(dplyr)
library(gbm)
set.seed(1)

train <- fread("datafiles/train.csv")
test <- fread("datafiles/test.csv")

# data dimension

sprintf("There are %s rows and %s columns in train data ",nrow(train),ncol(train))
sprintf("There are %s rows and %s columns in test data ",nrow(test),ncol(test))

# convert unix time format 

unix_feats <- c('deadline','state_changed_at','created_at','launched_at')
train[,c(unix_feats) := lapply(.SD, function(x) structure(x, class=c('POSIXct'))), .SDcols = unix_feats]
test[,c(unix_feats) := lapply(.SD, function(x) structure(x, class=c('POSIXct'))), .SDcols = unix_feats]

# create simple features

len_feats <- c('name_len','desc_len','keywords_len')
count_feats <- c('name_count','desc_count','keywords_count')
cols <- c('name','desc','keywords')

train[,c(len_feats) := lapply(.SD, function(x) str_count(x)), .SDcols = cols]
train[,c(count_feats) := lapply(.SD, function(x) str_count(x,"\\w+")), .SDcols = cols]

test[,c(len_feats) := lapply(.SD, function(x) str_count(x)), .SDcols = cols]
test[,c(count_feats) := lapply(.SD, function(x) str_count(x,"\\w+")), .SDcols = cols]

# encode features

train[,disable_communication := as.integer(as.factor(disable_communication))-1]
train[,country := as.integer(as.factor(country))-1]

test[,disable_communication := as.integer(as.factor(disable_communication))-1]
test[,country := as.integer(as.factor(country))-1]

#my feats

#prep_time
train$prep_time = as.numeric(train$launched_at - train$created_at)
test$prep_time = as.numeric(test$launched_at - test$created_at)

train$duration = as.numeric(train$deadline - train$launched_at)
test$duration = as.numeric(test$deadline - test$launched_at)

train$income_rate = train$goal / train$duration
test$income_rate = test$goal / test$duration

#sentiment
affin_set <- filter(sentiments, lexicon == 'AFINN')

desc_words <- train[, c('project_id', 'desc')] %>%
  unnest_tokens(word, desc) %>%
  anti_join(., stop_words) %>%
  left_join(., affin_set) %>%
  group_by(project_id, word) %>%
  summarise(desc_word_count = n(), sentiment_score = sum(score, na.rm = TRUE))

sentiment_score <- desc_words %>%
  group_by(project_id) %>%
  summarise(score = sum(sentiment_score))

train <- left_join(train,sentiment_score)

desc_words <- test[, c('project_id', 'desc')] %>%
  unnest_tokens(word, desc) %>%
  anti_join(., stop_words) %>%
  left_join(., affin_set) %>%
  group_by(project_id, word) %>%
  summarise(desc_word_count = n(), sentiment_score = sum(score, na.rm = TRUE))

sentiment_score <- desc_words %>%
  group_by(project_id) %>%
  summarise(score = sum(sentiment_score))

test <- left_join(test, sentiment_score)


# cols to use in modeling
cols_to_use <- c('final_status'
                 ,'score'
                 ,'prep_time'
                 ,'duration'
                 ,'income_rate'
                 #,'name_len'
                 #,'desc_len'
                 #,'keywords_len'
                 #,'name_count'
                 #,'desc_count'
                 #,'keywords_count'
                 )

# GBM

X_train <- copy(train)
X_train$final_status <- as.factor(X_train$final_status)

clf_model <- gbm(final_status ~ .
                 ,data = train[,cols_to_use]#,with=F
                 ,n.trees = 500
                 ,interaction.depth = 5
                 ,shrinkage = 0.3
                 ,train.fraction = 0.6
                 ,verbose = T)

#500        1.1931          1.0865     0.3000   -0.0001 sem feature de LEN
# 500        1.1761          1.0689     0.3000   -0.0001 com features de LEN

# check variable importance
summary(clf_model, n.trees = 125)

# make predictions
clf_pred <- predict(clf_model, newdata = test, n.trees = 232,type = 'response')
clf_pred <- ifelse(clf_pred > 0.5,1,0)

# write file
subst <- data.table(project_id = test$project_id, final_status = clf_pred)
fwrite(subst, "gbm_starter.csv") #0.65754









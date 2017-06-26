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

mean(train$backers_count)
mean(filter(train, final_status == 1)$backers_count)

rolou <- train %>% filter(final_status == 1)
rolou %>% group_by(country) %>% summarise(mean(backers_count))

albuns <- rolou %>% filter(grepl(' album ',desc)) %>% group_by(country) %>% summarise(mean(backers_count))
films <- rolou %>% filter(grepl(' film ',desc)) %>% group_by(country) %>% summarise(mean(backers_count))
game <- rolou %>% filter(grepl(' game ',desc)) %>% group_by(country) %>% summarise(mean(backers_count))
art <- rolou %>% filter(grepl(' art ',desc)) %>% group_by(country) %>% summarise(mean(backers_count))
product <- rolou %>% filter(grepl(' product ',desc)) %>% group_by(country) %>% summarise(mean(backers_count))
design <- rolou %>% filter(grepl(' design ',desc)) %>% group_by(country) %>% summarise(mean(backers_count))



get_topics <- function(data) {
  print('trabalhando palavras')
  
  data$desc <- limparTexto(paste(data$desc, data$name, data$keywords))
  
  words <- data %>%
    unnest_tokens(word, desc) %>%
    anti_join(., stop_words) %>%
    mutate(word = str_extract(word, "[a-z']+")) %>%
    filter(!is.na(word)) %>%
    count(project_id, word, sort = TRUE) %>%
    ungroup()
  #words
  
  print('criando dtm') 
  project_dtm <- words %>%
    cast_dtm(project_id, word, n)
  
  print('calculando LDA - 10 classes') 
  projects_lda <- LDA(project_dtm, k = 10, control = list(seed = 1234))
  str(projects_lda)
  
  print('criando tópicos - beta')
  project_topics <- tidy(projects_lda, matrix = "beta")#
  project_topics
}


topicos <- get_topics(rolou)
topicos %>% group_by(topic) %>% summarize(max(beta))
topicos %>%
  group_by(document) %>%
  top_n(1, gamma)

View(topicos)






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
  data$is2014 <- as.numeric(year(data$created_at) >= 2014) #mudar para >= piorou...
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
                )

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
fwrite(sub, "xgb_with_feats.csv")

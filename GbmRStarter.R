# load data and libraries

library(data.table)
library(lubridate)
library(stringr)
library(RTextTools)
library(tidytext)
library(dplyr)
library(gbm)
library(topicmodels)
set.seed(1)

train <- fread("train.csv")
test <- fread("test.csv")

# data dimension

sprintf("There are %s rows and %s columns in train data ",nrow(train),ncol(train))
sprintf("There are %s rows and %s columns in test data ",nrow(test),ncol(test))

# convert unix time format 

unix_feats <- c('deadline','state_changed_at','created_at','launched_at')
train[,c(unix_feats) := lapply(.SD, function(x) structure(x, class=c('POSIXct'))), .SDcols = unix_feats]
test[,c(unix_feats) := lapply(.SD, function(x) structure(x, class=c('POSIXct'))), .SDcols = unix_feats]

# create simple features

#len_feats <- c('name_len','desc_len','keywords_len')
#count_feats <- c('name_count','desc_count','keywords_count')
#cols <- c('name','desc','keywords')

#train[,c(len_feats) := lapply(.SD, function(x) str_count(x)), .SDcols = cols]
#train[,c(count_feats) := lapply(.SD, function(x) str_count(x,"\\w+")), .SDcols = cols]

#test[,c(len_feats) := lapply(.SD, function(x) str_count(x)), .SDcols = cols]
#test[,c(count_feats) := lapply(.SD, function(x) str_count(x,"\\w+")), .SDcols = cols]

#dates features
createFeatures <- function(data) {
  data$prep_time <- as.numeric(data$created_at) - as.numeric(data$launched_at)
  data$duration <- as.numeric(data$deadline) - as.numeric(data$created_at)
  data$extratime <- as.numeric(data$deadline) - as.numeric(data$state_changed_at)
  
  data$disable_communication <- as.integer(as.factor(data$disable_communication))-1
  #data$country <- as.integer(as.factor(data$country))-1
  data$currency <- as.integer(as.factor(data$currency))-1
  
  data$incomeRate <- data$goal / data$duration
  
  #data$final_status <- as.factor(X_train$final_status)
  data
}
train <- createFeatures(train)
test <- createFeatures(test)

#sentiment feature
#sample <- train[1:1000,]

afinn_set <- filter(sentiments, lexicon == 'AFINN')
getSentimentScore <- function(data) {
  words <- data %>% unnest_tokens(word, desc) %>%
    anti_join(., stop_words) %>%
    left_join(., afinn_set) %>%
    group_by(project_id) %>%
    summarise(sentiment_score = sum(score, na.rm = TRUE))
  
  left_join(data, words)
}

train <- getSentimentScore(train)
test <- getSentimentScore(test)

classify_project <- function(data) {
  print('trabalhando palavras')
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
  
  print('calculando LDA - 7 classes') 
  projects_lda <- LDA(project_dtm, k = 7, control = list(seed = 1234))
  #projects_lda
  
  print('criando tópicos - gamma')
  project_topics <- tidy(projects_lda, matrix = "gamma")#beta
  #View(project_topics)
  
  print('gerando classificações')
  project_classifications <- project_topics %>%
    group_by(document) %>%
    top_n(1, gamma) %>%
    ungroup()
  project_classifications
}

train_classifications <- classify_project(train)
train_classifications$project_id <- train_classifications$document
train <- train %>%
  inner_join(train_classifications)

# cols to use in modeling
cols_to_use <- c('final_status'
                 #,'topic'
                 ,'sentiment_score'
                 ,'currency'
                 #,'disable_communication'
                 #,'country'
                 ,'prep_time'
                 ,'duration'
                 ,'extratime'
                 ,'incomeRate'
                 )

# GBM
clf_model <- gbm(final_status ~ .
                 ,data = train[,cols_to_use]
                 ,n.trees = 700
                 ,interaction.depth = 5
                 ,shrinkage = 0.03
                 ,train.fraction = 0.6
                 ,verbose = T
                 , distribution = "adaboost" #bernoulli
                 , cv.folds = 5
                )

plot(clf_model$valid.error)

# check variable importance
summary(clf_model, n.trees = 500)

# make predictions
clf_pred <- predict(clf_model, newdata = test, n.trees = 500, type = 'response')

min(sort(clf_pred, decreasing = TRUE)[1:22212])

clf_pred <- ifelse(clf_pred >= 0.3521892,1,0)
mean(clf_pred)

clf_pred

#inv_pred <- predict(clf_model, newdata = filter(test, project_id == 'kkst2114901164'), type = 'response')
#inv_pred <- predict(clf_model, newdata = filter(test, grepl("(Canceled)", name, fixed = TRUE)), type = 'response')


#auc <- roc( ifelse(testDF[,outcomeName]=="yes",1,0), predictions[[2]])
#print(auc$auc)

# write file
subst <- data.table(project_id = test$project_id, final_status = clf_pred)
fwrite(subst, "gbm_0.6487857.csv") #0.65754

#library(randomForest) #--muito muito lento
#fit <- randomForest(final_status ~ .
#                    , data=train[,cols_to_use]
#                    , ntree=1
#                    , na.action = na.exclude)
#summary(fit)


#subst <- data.table(project_id = test$project_id, final_status = 0)
#fwrite(subst, "gbm_z.csv") #0.65754
#subindo um arquivo com tudo zero, encontrei 0.65968% de acerto.
#ou seja, 65,968% é zero;


##quero ver qual o sentimento dos projetos de cada pais
# t_sample <- sample(train, 0.1)
# words <- t_sample %>%
#   unnest_tokens(words, desc) %>%
#   anti_join(., stop_words) %>%
#   left_join(., get_sentiments("nrc"))
# t_sample <- sample_n(train, 1000)
# #t_sample <- filter(t_sample, final_status == 1)
# 
# ggplot(t_sample, aes(project_id, sentiment_score, fill = final_status)) +
#   geom_col(show.legend = FALSE) +
#   facet_wrap(~country, ncol = 2, scales = "free_x")
# 
# nrow(filter(t_sample, deadline > state_changed_at))
# nrow(filter(t_sample, deadline == state_changed_at))
# nrow(filter(t_sample, deadline < state_changed_at))
# 
# View(filter(train, deadline == state_changed_at))
# View(filter(train, deadline < state_changed_at))
# 
# t_sample <- train#sample_n(train, 1000)#


# 
# 
# filter(train_classifications, document == 'kkst1649897086')
# 
# top_terms <- project_topics %>%
#   group_by(topic) %>%
#   top_n(10, beta) %>%
#   ungroup() %>%
#   arrange(topic, -beta)
# View(top_terms)
# 
# View(train_classifications)
# filter(train_classifications, document == 'kkst1661697795')
# 
# 
# ap_lda <- LDA(train, k = 2, control = list(seed = 1234))

#################################
data <- sample_frac(test, 0.2)
print('trabalhando palavras')
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

print('calculando LDA - 7 classes')
projects_lda <- LDA(project_dtm, k = 7, control = list(seed = 1234))
#View(projects_lda)
projects_lda

print('criando tópicos - gamma')
project_topics <- tidy(projects_lda, matrix = "gamma")#beta
#View(project_topics)

print('gerando classificações')
project_classifications <- project_topics %>%
  group_by(document) %>%
  top_n(1, gamma) %>%
  ungroup()
project_classifications

x <- data.frame(project_classifications)
x$project_id <- x$document
data %>% left_join(x)
inner_join(data, x, by = c("project_id", "document"))


project_classifications$document
# 
# 
# 
# 
# 
# 
View(train %>% filter(final_status == 1) %>% group_by(country, topic) %>% summarise(total = n()))

View(test)
# p <- ggplot(train$incomeRate, aes(variable, Name))
#   + geom_tile(aes(fill = rescale), colour = "white")
#   + scale_fill_gradient(low = "white", high = "steelblue")
# 
rate <- train %>% order(incomeRate)
plot(rate)

order_by(10:1, cumsum(1:10))

plot(arrange(train, incomeRate))

successful <- filter(train, final_status == 1)
failed <- filter(train, final_status == 0)
summary(train$incomeRate)
boxplot(log10(train$incomeRate), log10(test$incomeRate))
hist(log10(train$incomeRate))
hist(log10(test$incomeRate))

boxplot(log10(successful$incomeRate), log10(failed$incomeRate))
names(train)

hist(log10(train$incomeRate))

arranged <- arrange(filter(train, backers_count > 1), desc(backers_count))
View(top_n(arranged, 100, backers_count))
filter(train, backers_count > 0)

View(test)

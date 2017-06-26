#install.packages('forcats')
#??mfrow
library(ggplot2)
library(forcats)

par(mfrow = c(1,1))
ggplot2::ggplot(train, aes(x = forcats::fct_infreq(score_1), fill = default))+ geom_histogram(stat = "count")
ggplot(train, aes(x = forcats::fct_infreq(state), fill = default))+ geom_histogram(stat = "count")
names(train)

## analise de correlação
install.packages('Hmisc')
library(Hmisc)
res<-rcorr(train[c("risk_rate", "amount_borrowed", "score_3", "score_4", "score_5", "score_6")])
signif(res$r, 2)

install.packages('PerformanceAnalytics')
library(PerformanceAnalytics)
chart.Correlation(train[c("risk_rate", "amount_borrowed", "score_3", "score_4", "score_5", "score_6")], histogram=TRUE, pch=19)
chart.Correlation(train[c("risk_rate", "score_3", "score_4", "score_6")], histogram=TRUE, pch=19, method = "pearson")
chart.Correlation(cbind(train$risk_rate, log1p(train$amount_borrowed), log1p(train$score_5)), histogram=TRUE, pch=19, method = "spearman")

chart.Correlation(cbind(as.numeric(train$default == "False"), train$risk_rate, log1p(train$amount_borrowed), train$score_5), histogram=TRUE, pch=19, method = "spearman")

chart.Correlation(cbind(as.numeric(train$default == "False"), train$signo_bom), histogram=TRUE, pch=19, method = "spearman")

chisq.test(train$default, train$n_bankruptcies)
aov1 <- aov(train$n_bankruptcies ~ train$default)

x <- by(train$default, train$n_issues, function(x) as.numeric(x == 1))
#str(x)
y <- lapply(x, mean)
y

summary(aov1)

chart.Correlation(cbind(
  train$default
  #, train$real_state
  , train$ok_since
  , train$n_bankruptcies
  , train$n_defaulted_loans
  , train$n_accounts
  , train$n_issues
), histogram=TRUE, pch=19, method = "spearman")

chart.Correlation(cbind(
train$default
, train$sign_aqua
, train$sign_arie
, train$sign_cance
, train$sign_capr
, train$sign_gemi
, train$sign_leo
, train$sign_libr
, train$sign_pisce
, train$sign_sagi
, train$sign_scor
, train$sign_taur
, train$sign_virg
), histogram=TRUE, pch=19, method = "spearman")

train %>% group_by(default, signo_bom) %>% summarize(n)


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




#### devs
levels(as.factor(train$real_state))

tl = levels(as.factor(test$state))
xl = levels(as.factor(train$state))

all(xl != tl)

xl
all.equal(tl, xl)

identical(xl, tl)
which(xl != tl)

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

mean(train$signOHE_sagi)
lapply(train[,sign_cols], mean)


comprometidos <- (train$amount_borrowed / train$borrowed_in_months) #> (train$income / 12)*0.3
head(train$amount_borrowed / train$borrowed_in_months)
mean(comprometidos, na.rm = TRUE)
str(train)
hist(year(train$last_paymentDate))
ggplot(train, aes(x = forcats::fct_infreq(as.factor(year(last_paymentDate))), fill = (default == 1)))+ geom_histogram(stat = "count")

ggplot(train, aes(x = forcats::fct_infreq(as.factor(month(last_paymentDate))), fill = (default == 1)))+ geom_histogram(stat = "count")

ggplot(train, aes(x = forcats::fct_infreq(as.factor(year(last_paymentDate)*100 + month(last_paymentDate))), fill = (default == 1)))+ geom_histogram(stat = "count")
ggplot(train, aes(x = forcats::fct_infreq(as.factor(year(last_paymentDate)*100 + month(end_last_loanDate))), fill = (default == 1)))+ geom_histogram(stat = "count")

ggplot(train, aes(x = forcats::fct_infreq(as.factor(year(last_paymentDate)*100 + month(end_last_loanDate))), fill = (default == 1)))+ geom_histogram(stat = "count")

ggplot(train, aes(x = forcats::fct_infreq(as.factor(year(end_last_loanDate)*100 + month(end_last_loanDate))), fill = (default == 1)))+ geom_histogram(stat = "count")
boxplot(year(train$last_paymentDate)*100 + month(train$end_last_loanDate))

levels(as.factor(year(train$last_paymentDate)*100 + month(train$end_last_loanDate)))


mean(train$default)



ntest <- names(test)
ntrain[which(ntrain != ntest)]
ntest[which(ntest != ntrain)]

all.equal(ntrain, ntest)

test[,intersect(ntrain, ntest)]
mean()
mean(test$last_paymentFacOHE_201111)

intersect(as.factor(train$state), as.factor(test$state))

ms = lapply(lapply(train[,end_last_loanFac_cols], as.numeric), mean) 
mm = as.matrix(ms)

plot(mm[mm > 0.2])

train[,end_last_loanFac_cols]

train$job_name
train %>% group_by 
mean(as.numeric(train$job_name == ''))
mean(as.numeric(calots$job_name == ''))


calots <- filter(train, default==1)

mean(as.numeric(train$zip == ''))
mean(as.numeric(calots$channel_NotEmpty))
mean(as.numeric(train$channel_NotEmpty))

install.packages('caret')
library(caret)
confusionMatrix(preds)



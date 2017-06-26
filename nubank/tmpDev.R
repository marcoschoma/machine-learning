#install.packages('forcats')
#??mfrow
par(mfrow = c(1,2))
ggplot2::ggplot(train, aes(x = forcats::fct_infreq(score_1), fill = default))+ geom_histogram(stat = "count")
ggplot2::ggplot(train, aes(x = forcats::fct_infreq(score_2), fill = default))+ geom_histogram(stat = "count")


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
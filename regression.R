library(data.table)
library(stringr)
library(dplyr)
library(ggplot2)
library(xgboost)
set.seed(1)
setwd('/projetos/machine-learning-challenge-2/')
train <- fread("train.csv")
test <- fread("test.csv")

my_sample <- sample_frac(train, 0.1)

library(quantmod)
?quantmod
install.packages('quantmod')
from <- c("CAD", "JPY", "USD")
to <- c("USD", "USD", "EUR")
getQuote(paste0(from, to, "=X"))

?xgboost

#reg:logistic
getSymbols("AAPL",src="yahoo") 
barChart(AAPL)

tail(AAPL, 10)$AAPL.Close

require(TTR) 
getSymbols("PETR4", src = "google")
chartSeries(PETR4)
addMACD()
addBBands()

str(PETR4)

row.names(PETR4)

PETR4[as.POSIXct.Date(today())]
PETR4['2017-06-01']:'2017-06-20']
#fonte: http://tidytextmining.com/tidytext.html#contrasting-tidy-text-with-other-data-structures
library(data.table)
library(tidytext)
library(dplyr)
library(ggplot2)

train <- fread("train.csv")
train_sample <- sample_frac(train, size = 0.1)
dim(train_sample)

names(train_sample)

words <- train_sample %>%
  unnest_tokens(word, desc) %>%
  anti_join(stop_words) %>%
  mutate(word = str_extract(word, "[a-z']+"))

#palavras mais usadas:

words %>%
  filter(is.na(word) == FALSE) %>%
  count(word, sort = TRUE) %>%
  filter(n > 400) %>%
  mutate(word = reorder(word, n)) %>%
  ggplot(aes(word, n)) +
  geom_col() +
  xlab(NULL) +
  coord_flip()

words
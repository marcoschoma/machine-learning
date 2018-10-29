from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
sc = SparkContext('local')
spark = SparkSession(sc)

from pyspark.ml.clustering import LDA
from pyspark.ml.feature import Tokenizer
from pyspark.ml.feature import CountVectorizer
from pyspark.sql.types import StructField, StringType, StructType
from pyspark.ml.feature import StopWordsRemover

labels = ["title","body","FROM_UNIXTIME"]

fields = [StructField(field_name, StringType(), True) for field_name in labels]
schema = StructType(fields)

# Loads data.
data_df = spark.read.csv("/home/marcos/code/data/noticias_small.csv", schema=schema)
#.map(lambda row: row.split("\r\n"))
print(data_df)

tokenizer = Tokenizer(inputCol="body", outputCol="words")
wordsDataFrame = tokenizer.transform(data_df)

stopWords = StopWordsRemover.loadDefaultStopWords("portuguese")
remover = StopWordsRemover(inputCol="words", outputCol="words_filtered", stopWords = stopWords)
wordsFiltered = remover.transform(wordsDataFrame)

cv_tmp = CountVectorizer(inputCol="words_filtered", outputCol="tmp_vectors")
cv_tmp_model = cv_tmp.fit(wordsFiltered)
df_vect = cv_tmp_model.transform(wordsFiltered)

def parseVectors(line):
    return [int(line[2]), line[1]]

sparsevector = df_vect.select("FROM_UNIXTIME", "tmp_vectors")

lda = LDA(k=10, maxIter=5, featuresCol="tmp_vectors")
ldaModel = lda.fit(sparsevector)
topics = ldaModel.topicsMatrix()
# model = LDA.train(sparsevector, k=5, seed=1)

# Describe topics.
topics = ldaModel.describeTopics(3)
# print("The topics described by their top-weighted terms:")
# topics.show(truncate=False)

# ldaModel.describeTopics(3)
topics\
    .select("termIndices", "termWeights")\
    .rdd\
    .map(lambda t: print(t[0],t[1]))
# for x, topic in enumerate(topics):
#     print('topic nr: ' + str(x))
#     words = topic["termIndices"]
#     weights = topic["termWeights"]
#     print(cv_tmp_model.vocabulary[int(words.getInt())] + ' ' + str(weights[0]))
#     print(cv_tmp_model.vocabulary[words[1,]] + ' ' + str(weights[1]))
#     print(cv_tmp_model.vocabulary[words[2,]] + ' ' + str(weights[2]))

# print(topics)

# for x, topic in enumerate(topics):
#     print('topic nr: ' + str(x))
#     words = topic[0]
#     weights = topic[1]
#     if len(words) > 0:
#         for n in range(len(words)):
#             print(df_vect.vocabulary[words[n]] + ' ' + str(weights[n]))

# data = spark.createDataFrame([
#     (1, "Hi I heard about Spark".split(" "), ),
#     (2, "I wish Java could use case classes".split(" "), ),
#     (3, "Logistic regression models are neat".split(" "), )
# ], ["id", "words"])

# countVectorizer = CountVectorizer(inputCol="words", outputCol="features", vocabSize=3, minDF=2.0)
# cvModel = countVectorizer.fit(data)
# result = cvModel.transform(data)
# corpus = result.select("id", "features")

# # Trains a LDA model.
# lda = LDA(k=2, maxIter=2)
# ldaModel = lda.fit(corpus)
# topics = ldaModel.topicsMatrix()

# print(topics)
# # vocabArray = ldaModel.vocabulary

# ll = ldaModel.logLikelihood(corpus)
# lp = ldaModel.logPerplexity(corpus)
# print("The lower bound on the log likelihood of the entire corpus: " + str(ll))
# print("The upper bound on perplexity: " + str(lp))

# # Describe topics.
# topics = ldaModel.describeTopics(3)
# print("The topics described by their top-weighted terms:")
# topics.show(truncate=False)

# # Shows the result
# transformed = ldaModel.transform(corpus)
# transformed.show(truncate=False)
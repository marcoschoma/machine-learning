from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
sc = SparkContext('local')
spark = SparkSession(sc)
from pyspark.ml.linalg import Vectors
from pyspark.ml.classification import LogisticRegression
import pyspark

training = spark.createDataFrame([
    (1.0, Vectors.dense([0.0, 1.1, 0.1])),
    (0.0,   Vectors.dense([2.0, 1.0, -1.0])),
    (0.0, Vectors.dense([2.0, 1.3, 1.0])),
    (1.0, Vectors.dense([0.0, 1.1, -0.5])),
],["label", "features"])

lr = LogisticRegression(maxIter=10, regParam=0.01)

print("LogisticRegression parameters:\n" + lr.explainParams() + '\n')

model1 = lr.fit(training)

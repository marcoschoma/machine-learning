from pyspark import SparkContext, SparkConf
from pyspark.sql.session import SparkSession

sparkContext = SparkContext('local')
spark = SparkSession(sparkContext)

# data = [1,2,3,4,5] # list(range(1,5))
# distData = sparkContext.parallelize(data)

# x = distData.reduce(lambda a, b: a + b)

from pyspark.ml.linalg import Vectors
from pyspark.ml.stat import Correlation

data = [(Vectors.sparse(4, [(0, 1.0), (3, -2.0)]),),
        (Vectors.dense([4.0, 5.0, 0.0, 3.0]),),
        (Vectors.dense([6.0, 7.0, 0.0, 8.0]),),
        (Vectors.sparse(4, [(0, 9.0), (3, 1.0)]),)]
df = spark.createDataFrame(data, ["features"])

# print dataframe
#  df.show()

# correlation!
# r1 = Correlation.corr(df, "features").head()
# print("Pearson correlation matrix: ", str(r1[0]), "\n")
# r2 = Correlation.corr(df, "features", "spearman").head()
# print("Spearman correlation matrix: ", str(r2[0]), "\n")

# save dataframe efficiently
df.write.parquet('dataframe.parquet')
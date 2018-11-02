# from pyspark.context import SparkContext
# from pyspark.sql.session import SparkSession
# sparkContext = SparkContext('local')
# spark = SparkSession(sparkContext)

# catalogDataframe = spark.read.json('/home/marcos/code/machine-learning/data/catalogo_sample_20181031.csv', multiLine=True)

# catalogDataframe['Name'] = catalogDataframe['Name'].tolower()
# catalogDataframe.show()

from unicodedata import normalize
from string import punctuation

def remover_acentos(txt):
    return normalize('NFKD', txt).encode('ASCII', 'ignore').decode('ASCII')

string = "“ç~^{}[]!@#$%*()_+1234567890-= áàãóòõâôéèẽêíìĩîûũúùç”/·€ŧßđðæß“«»©“«”þø→n”ĸn<br />-Capítulo"

def fixString(text):
    return remover_acentos(text.lower()).translate(None, punctuation)

print(fixString(string))
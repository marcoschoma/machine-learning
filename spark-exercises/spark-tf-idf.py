from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
sc = SparkContext('local')
spark = SparkSession(sc)

from pyspark.ml.feature import Tokenizer, IDF, HashingTF

corpus = spark.createDataFrame([
    (1, "CAFETEIRA EXPRESSO ARNO DOLCE GUSTO MINI ME AUTOMÁTICA - VERMELHA", "A Cafeteira Expresso Arno Dolce Gusto Mini Me vai deixar seu dia a dia muito mais saboroso e prático. Eficiente, prepara diversas bebidas como: Espresso, Cappucino, Chococcino, Mocha, Nestea e muito mais. Toda qualidade da Dolce Gusto agora em uma máquina automática mais moderna, basta selecionar o nível indicado na cápsula, escolher quente ou frio e pronto, sua bebida está preparada na medida certa. Seu a exclusivo sistema de cápsulas e pressão de 15 bar proporciona uma bebida cremosa e aromática na temperatura perfeita para consumo. Possui reservatório de água com capacidade para 800 ml de fácil abastecimento e limpeza. Além disso, conta com sistema Thermoblock que garante a temperatura ideal da bebida, desde a primeira até a última xícara preparada."),
    (2, "PENEIRA AÇO INOX BRINOX TOP PRATIC 2202/331", "Peneira Aço Inox Brinox"),
    (3, "NERF DE ÁGUA SUPER SOAKER H2OPS SQUALL SURGE HASBRO", "Nerf de Água Super Soaker H2OPS Squall Surge"),
    (4, "LITTLEST PET SHOP HASBRO AMIGOS FASHION - PHILIPE BOUDREAUX E ZOE TRENT", ""),
    (5, "BALCÃO COM TAMPO MULTIMÓVEIS 2 PORTAS 2541", "Balcão com Tampo Multimóveis 2 Portas"),
    (6, "KIT BAR 4 PEÇAS BON GOURMET 25589", "Kit Bar 4 Peças Bon Gourmet")
], ["id", "name", "description"])

# corpus.show()
tokenizer = Tokenizer(inputCol="name", outputCol="words")
wordsData = tokenizer.transform(corpus)

# wordsData.show()

hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
featurizedData = hashingTF.transform(wordsData)

featurizedData.show()

idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

# rescaledData.select("id", "features").show()
rescaledData.show()
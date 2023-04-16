from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.sql import SparkSession
import sys
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder.appName("app").getOrCreate()

# Load the trained model from disk
model = RandomForestClassificationModel.load("trained-model")

# Load the user input file
# data = spark.read.format("csv").load(sys.argv[1])
data = spark.read.options(delimiter=';').csv(sys.argv[1], header=True, inferSchema=True)


data_assembler = VectorAssembler(inputCols=data.columns[:-1], outputCol='features')
data = data_assembler.transform(data)
data = data.select(['features', '""""quality"""""'])

# Use the loaded model to make predictions on the user input file
predictions = model.transform(data)

# Output the predictions
predictions.show()

evaluator = MulticlassClassificationEvaluator(labelCol='""""quality"""""', predictionCol='prediction', metricName='f1')
f1_score = evaluator.evaluate(predictions)
print('F1-score:', f1_score)
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark = SparkSession.builder.appName("build_model").getOrCreate()

# Load the data into a DataFrame
df = spark.read.options(delimiter=';').csv('TrainingDataset.csv', header=True, inferSchema=True)

# Prepare the data for modeling
assembler = VectorAssembler(inputCols=df.columns[:-1], outputCol='features')
df = assembler.transform(df)
df = df.select(['features', '""""quality"""""'])

# Define the classification algorithm
rf = RandomForestClassifier(featuresCol='features', labelCol='""""quality"""""', numTrees=50)

# Fit the model to the training data
model = rf.fit(df)

# Load validation dataset into DataFrame
df_validation = spark.read.options(delimiter=';').csv('ValidationDataset.csv', header=True, inferSchema=True)

# Prepare the validation data for modeling
validation_assembler = VectorAssembler(inputCols=df_validation.columns[:-1], outputCol='features')
df_validation = validation_assembler.transform(df_validation)
df_validation = df_validation.select(['features', '""""quality"""""'])

# Use the model to make predictions on the test data
predictions = model.transform(df_validation)

# Evaluate the performance of the model
evaluator = MulticlassClassificationEvaluator(labelCol='""""quality"""""', predictionCol='prediction', metricName='f1')
f1_score = evaluator.evaluate(predictions)
print('F1-score:', f1_score)

# Save model into file
model.save('hdfs:///user/hadoop/trained-model') 

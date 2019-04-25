import numpy as np  
import pandas as pd  
import re  
import nltk  
import matplotlib.pyplot as plt  
import csv
import spark
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from Src.helper import cleanData
from pyspark.sql import SparkSession

sparkSession = SparkSession.builder.appName("aitomatos").config("spark.kryoserializer.buffer.max", "1g").config("spark.executor.memory","1g").getOrCreate()
print(sparkSession.sparkContext)
df_load = sparkSession.read.csv('hdfs:/tmp/dell_data/train.csv',header=True, mode="DROPMALFORMED")
df_load.show();

def PandaDF_from_Spark(numberOflines):
  print("Transforming Spark dataTable to Panda data table")
  resultsPerRating = numberOflines//5
  py_df1 = df_load.where("Rating == 1").limit(resultsPerRating)
  py_df2 = df_load.where("Rating == 2").limit(resultsPerRating)
  py_df3 = df_load.where("Rating == 3").limit(resultsPerRating)
  py_df4 = df_load.where("Rating == 4").limit(resultsPerRating)
  py_df5 = df_load.where("Rating == 5").limit(resultsPerRating)
  panda_dataframe = py_df1.union(py_df2).union(py_df3).union(py_df4).union(py_df5).toPandas().sample(frac=1)
  print(panda_dataframe)
  return panda_dataframe

#from helper import stemSentences
#from helper import cleanData
#%matplotlib inline

processed_reviews = []
data_source_url = "train.csv"

def executePreparation(numberoflines, testDataPercentage):
  
  if numberoflines > 0:
    csv_data = PandaDF_from_Spark(numberoflines)
    #csv_data = pd.read_csv(csvFile,nrows = numberoflines, skiprows = offset)
  else:
    csv_data = PandaDF_from_Spark(300000)
    #csv_data = pd.read_csv(csvFile)  
    
  print(csv_data)
  
  review_texts = csv_data.iloc[:,0]
  ratings = csv_data.iloc[:,1]

  print(review_texts)
  print(ratings)

  
  processed_reviews = cleanData(review_texts)
  
  vectorizer = CountVectorizer(max_features=2500, min_df=7, max_df =0.8, stop_words = stopwords.words('english'))
  processed_review_vector = vectorizer.fit_transform(processed_reviews).toarray()
  print(processed_review_vector)

  print("Spliting training and test data")
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(processed_review_vector, ratings, test_size=testDataPercentage, random_state=0)  

  return processed_review_vector, X_train, X_test, y_train, y_test, csv_data, review_texts, ratings , vectorizer

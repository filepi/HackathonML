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

#from helper import stemSentences
#from helper import cleanData
#%matplotlib inline

processed_reviews = []
data_source_url = "train.csv"

def executePreparation(numberoflines, testDataPercentage):
  
  if numberoflines > 0:
    csv_data = pd.read_csv(data_source_url,nrows = numberoflines)  
  else:
    csv_data = pd.read_csv(data_source_url)  
    
  print(csv_data);
  
  review_texts = csv_data.iloc[:,0]
  ratings = csv_data.iloc[:,1]

  #print(review_texts)
  #print(ratings)

  processed_reviews = cleanData(review_texts)
  
  vectorizer = CountVectorizer(max_features=2500, min_df=7, max_df =0.8, stop_words = stopwords.words('english'))
  processed_review_vector = vectorizer.fit_transform(processed_reviews).toarray()
  print(processed_review_vector)

  print("Spliting training and test data")
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(processed_review_vector, ratings, test_size=testDataPercentage, random_state=0)  

  return processed_review_vector, X_train, X_test, y_train, y_test, csv_data, review_texts, ratings , vectorizer
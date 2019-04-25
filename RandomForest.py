import numpy as np  
import pandas as pd  
import re  
import nltk  
import matplotlib.pyplot as plt  
import csv
import Src.DataPreparation
%matplotlib inline


processed_review_vector, X_train, X_test, y_train, y_test, csv_data, review_texts, ratings, vectorizer = executePreparation(1000, 0.3)

print("Training the model with Random Forest")

from sklearn.ensemble import RandomForestClassifier

text_classifier = RandomForestClassifier(n_estimators=850, random_state=0)
text_classifier.fit(X_train, y_train)

predictions_randomforest = text_classifier.predict(X_test)

print("Random Forest Predictions")
print(predictions_randomforest)

###############################

print("Evaluating model accuracy")

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print(confusion_matrix(y_test, predictions_randomforest))
print(classification_report(y_test, predictions_randomforest))
print(accuracy_score(y_test, predictions_randomforest))

print(predictions_randomforest)
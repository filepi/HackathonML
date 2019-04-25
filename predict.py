# Read the fitted model from the file model.pkl
# and define a function that uses the model to
# predict petal width from petal length

import pickle

#model = pickle.load(open('model.pkl', 'rb'))

#def predict(args):
#  iris_x = float(args.get('petal_length'))
#  result = model.predict(iris_x)
#  return result[0][0]


#model = pickle.load(open('model.pkl', 'rb'))

import csv


with open('train.csv', 'r') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        print(row)

csvFile.close();


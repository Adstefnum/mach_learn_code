import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

age = int(input('Enter any age pls:'))

data = pd.read_csv('age_classification_train.csv')

x= np.array(data[['Age']])
x.reshape(1,-1)

y= data['class']

Classifier = LogisticRegression(solver = 'lbfgs', multi_class = 'ovr', random_state = 0)
Classifier.fit(x,y)
print(Classifier.predict([[age]]))

#make better
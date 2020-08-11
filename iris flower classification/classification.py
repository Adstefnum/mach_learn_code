import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import numpy as np


df = pd.read_csv('iris.data.csv')

#Developing data for model
x = df.drop("variety",axis=1)
y = df['variety']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state = 42)

scale = StandardScaler()
x_train = scale.fit_transform(x_train)
x_test = scale.transform(x_test)


#
score = {}
algo_list = [LogisticRegression(solver = 'lbfgs', multi_class = 'ovr', random_state = 0),
				KNeighborsClassifier(),XGBClassifier(),RandomForestClassifier(n_estimators=100),
				DecisionTreeClassifier(),svm.SVC()

				]
'''
#all are 100 except this
clf_1 = LogisticRegression(solver = 'lbfgs', multi_class = 'ovr', random_state = 0)
clf_1.fit(x_train,y_train)
pred1=clf_1.predict(x_test)
#feeding data to check
check = clf_1.predict([[5.1,3.5,1.4,0.2]])
print('logreg',check)
s1=accuracy_score(y_test,pred1)
score['logisticreg'] = s1*100

clf_1 = KNeighborsClassifier()
clf_1.fit(x_train,y_train)
pred1=clf_1.predict(x_test)
#feeding data to check
check = clf_1.predict([[5.1,3.5,1.4,0.2]])
print('knn',check)
s1=accuracy_score(y_test,pred1)
score['knn'] = s1*100

clf_1 = XGBClassifier()
clf_1.fit(x_train,y_train)
pred1=clf_1.predict(x_test)
#feeding data to check
fv = [5.1,3.5,1.4,0.2]
fv = numpy.array(fv).reshape((1,-1))
check = clf_1.predict(fv)
print('xgb',check)
s1=accuracy_score(y_test,pred1)
score['xgb'] = s1*100

clf_1 = RandomForestClassifier(n_estimators=100)
clf_1.fit(x_train,y_train)
pred1=clf_1.predict(x_test)
#feeding data to check
check = clf_1.predict([[5.1,3.5,1.4,0.2]])
print('randfor',check)
s1=accuracy_score(y_test,pred1)
score['randfor'] = s1*100

clf_1 = DecisionTreeClassifier()
clf_1.fit(x_train,y_train)
pred1=clf_1.predict(x_test)
#feeding data to check
check = clf_1.predict([[5.1,3.5,1.4,0.2]])
print('dectree',check)
s1=accuracy_score(y_test,pred1)
score['dectree'] = s1*100

#gives wrong value
clf_1 = svm.SVC()
clf_1.fit(x_train,y_train)
pred1=clf_1.predict(x_test)
#feeding data to check
check = clf_1.predict([[6.1,2.8,4.0,1.3]])
print('svm',check)
s1=accuracy_score(y_test,pred1)
score['svm'] = s1*100
print(score)'''

for alg in algo_list:
	clf_1 = alg
	clf_1.fit(x_train,y_train)
	pred1=clf_1.predict(x_test)
	s1=accuracy_score(y_test,pred1)
	score[str(str(alg).split('(')[0])] = s1*100
print(score)

#implement neural networks
#start pickling
#it seems that even though accuracy can be high it sill gives wrong answers when you feed it 
#your own data

 
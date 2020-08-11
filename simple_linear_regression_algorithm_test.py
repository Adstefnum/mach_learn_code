import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random
from myalgs.simple_linear_regression_algorithm import SimpleLinearRegression
from sklearn.linear_model import LinearRegression
import pickle

style.use('ggplot')

def create_dataset(points,variance,step=2,correlation=False):
	val = 1 
	ys = []
	for i in range(points):
		y = val+ random.randrange(-variance,variance)
		ys.append(y)

		if correlation and correlation == 'pos':
			val += step
		elif correlation and correlation == 'neg':
			val -=step

	xs = [ i for i in range(len(ys))]

	return np.array(xs,dtype=np.float64) , np.array(ys,dtype=np.float64)

#my algorithm
s = SimpleLinearRegression()
xs,ys = create_dataset(40,100,2,correlation='pos')
m,c = s.get_slope_and_intercept(xs,ys)
regression_line = [(m*x)+c for x in xs]
r_squared = s.r_squared(ys,regression_line)
print("accuracy: {}".format(r_squared))

'''#pickling-useful only with models like those from sklearn
#unless you can find a way to make all the functions from the s class get calculated 
#inside the class and not likt this(it's possible) then it makes sense to pickle, if not
#all you're pickling is the class and not the model
with open('linreg.pickle','wb') as f:
	pickle.dump(s,f)

load_model = open('linreg.pickle','rb')
s = pickle.load(load_model)'''

'''#sklearn's algorithm- get it to work
sklearn_reg = LinearRegression()
sklearn_reg.fit([xs],[ys])
print('sklearn\'s accuracy:',sklearn_reg.score([xs],[ys]))'''

predict_x = 15
predict_y = (m*predict_x)+c

plt.scatter(xs,ys,color='#003F72',label='data')
plt.plot(xs,regression_line,label = 'regression line')
plt.scatter(predict_x,predict_y,color='g')
plt.legend(loc="upper left")
plt.show()
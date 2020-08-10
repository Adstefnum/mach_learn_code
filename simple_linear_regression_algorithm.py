from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

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



def get_slope_and_intercept(xs,ys):

	m = ((mean(xs)*mean(ys) - mean(xs*ys)) / (mean(xs)**2 - mean(xs**2)))

	c = mean(ys) - m*mean(xs)

	return m,c

def squared_error(ys_orig,ys_line):
	return sum((ys_line-ys_orig)**2)

def r_squared(ys_orig,ys_line):
	y_mean_line = [mean(ys_orig) for y in ys_orig]
	squared_regr_error = squared_error(ys_orig,ys_line)
	squared_mean_error = squared_error(ys_orig,y_mean_line)
	return 1 - (squared_regr_error/squared_mean_error)

xs,ys = create_dataset(40,100,2,correlation='pos')
m,c = get_slope_and_intercept(xs,ys)
regression_line = [(m*x)+c for x in xs]
r_squared = r_squared(ys,regression_line)
print("accuracy: {}".format(r_squared))

predict_x = 15
predict_y = (m*predict_x)+c

plt.scatter(xs,ys,color='#003F72',label='data')
plt.plot(xs,regression_line,label = 'regression line')
plt.scatter(predict_x,predict_y,color='g')
plt.legend(loc="upper left")
plt.show()
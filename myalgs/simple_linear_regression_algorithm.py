from statistics import mean

class SimpleLinearRegression:

	def get_slope_and_intercept(self,xs,ys):

		m = ((mean(xs)*mean(ys) - mean(xs*ys)) / (mean(xs)**2 - mean(xs**2)))

		c = mean(ys) - m*mean(xs)
		return m,c

	def squared_error(self,ys_orig,ys_line):
		return sum((ys_line-ys_orig)**2)

	def r_squared(self,ys_orig,ys_line):
		y_mean_line = [mean(ys_orig) for y in ys_orig]
		squared_regr_error = self.squared_error(ys_orig,ys_line)
		squared_mean_error = self.squared_error(ys_orig,y_mean_line)
		return 1 - (squared_regr_error/squared_mean_error)
		

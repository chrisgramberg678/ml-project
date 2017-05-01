import grad
import numpy as np
from scipy.stats import bernoulli

def main():
	# lls_test()
	# log_reg_test()
	stochastic_test()

# simple test 
def lls_test():
	print "linear least squares test:"
	x = np.random.rand(2,10)
	w = np.random.rand(2)
	# force w to be a column vector
	w.shape = (2,1)
	y = w.transpose().dot(x).flatten()
	m = grad.PyLLSModel()
	gd = grad.PyGradient_Descent(x,y)
	ans = gd.fit([0,0],.001,.000000001, m)
	print "w was:"
	print w.flatten()
	print "we got:"
	print ans

def log_reg_test():
	print "binary logistic regression test:"
	# 10 data points with 2 parts each
	x = 10 * np.random.randn(4,10000)
	print x
	# 2 coefficients 
	w = np.random.rand(4)
	# force it to be a column vector
	w.shape = (4,1)
	# for the bernoulli distribution which will give us y
	logit = np.exp(w.transpose().dot(x))/(1+np.exp(w.transpose().dot(x)))
	y = bernoulli.rvs(logit)
	# initialize the model
	m = grad.PyBLRModel()
	gd = grad.PyGradient_Descent(x,y)
	ans = gd.fit([0,0,0,0],.00001,.000000001, m)
	print "w was:"
	print w.flatten()
	print "we got:"
	print ans

def stochastic_test():
	print "stochastic test:"
	x = np.random.rand(2,10)
	w = np.random.rand(2)
	# force w to be a column vector
	w.shape = (2,1)
	y = w.transpose().dot(x).flatten()
	m = grad.PyLLSModel()
	gd = grad.PyGradient_Descent(x,y)
	# we can check this by moving the main loop of fit outside the gradient calculation
	precision = np.zeros(2) + .000000001
	curr, prev = np.array([0,0]), np.array([2,2])
	diff = np.absolute(prev - curr)
	flags = np.greater(diff, precision)
	while flags.any():
		# hold onto the old value
		prev = curr
		# get the new one
		curr = gd.stochastic_fit(prev,.001, m)
		# get the difference
		diff = np.absolute(prev - curr)
		print diff
		# update flags
		flags = np.greater(diff, precision)
	print "w was:"
	print w.flatten()
	print "we got:"
	print curr
	w = np.random.rand(2)

if __name__ == '__main__':
	main()
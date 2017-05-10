import grad
import numpy as np
from scipy.stats import bernoulli

def main():
	for i in range(10):
		print "Seed: {0}".format(i)
		np.random.seed(i)
		try:
			log_reg_test()
		except RuntimeError as e:
			print "Runtime Error: {0}".format(e)

# simple test 
def lls_test():
	print "linear least squares test:"
	x = np.random.rand(2,10)
	w = np.random.rand(2)
	# force w to be a column vector
	w.shape = (2,1)
	y = w.transpose().dot(x).flatten()
	m = grad.PyLLSModel()
	gd = grad.PyBatch_Gradient_Descent(x,y,m)
	ans = gd.fit([0,0],.001,.000000001)
	print "w was:"
	print w.flatten()
	print "we got:"
	print ans

def log_reg_test():
	print "binary logistic regression test:"
	x = 10 * np.random.randn(4,100)
	# 2 coefficients 
	w = np.random.rand(4)
	# force it to be a column vector
	w.shape = (4,1)
	# for the bernoulli distribution which will give us y
	logit = np.exp(w.transpose().dot(x))/(1+np.exp(w.transpose().dot(x)))
	y = bernoulli.rvs(logit)
	# initialize the model
	m = grad.PyBLRModel()
	gd = grad.PyBatch_Gradient_Descent(x,y,m)
	print "w was:"
	print w.flatten()
	ans = gd.fit([0,0,0,0],.00001,.000000001)
	print "we got:"
	print ans

def stochastic_test():
	print "stochastic test:"
	x = np.random.rand(2,100)
	w = np.random.rand(2)
	# force w to be a column vector
	w.shape = (2,1)
	y = w.transpose().dot(x).flatten()
	m = grad.PyLLSModel()
	gd = grad.PyGradient_Descent(x,y)
	# we can check this by moving the main loop of fit outside the gradient calculation
	precision = np.zeros(2) + .0000000001
	curr, prev = np.array([0,0]), np.array([2,2])
	diff = np.absolute(prev - curr)
	flags = np.greater(diff, precision)
	# while any of the old values are not within the precision of the new values
	losses = np.zeros(0)
	for i in range(1000000):
		# hold onto the old value
		prev = curr
		# get the new one
		# we want to hand in a single data point or a group of points
		curr = gd.stochastic_fit(prev,.001, m)
		# get the difference
		# diff = np.absolute(prev - curr)
		# update flags
		# flags = np.greater(diff, precision)
		# losses = gd
	print "w was:"
	print w.flatten()
	print "we got:"
	print curr
	w = np.random.rand(2)

if __name__ == '__main__':
	main()
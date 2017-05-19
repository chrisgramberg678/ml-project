import grad
import numpy as np
from scipy.stats import bernoulli
import matplotlib.pyplot as plt

def main():
	for i in range(10):
		print "Seed: {0}".format(i)
		np.random.seed(i)
		try:
			lls_test()
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
	ans = gd.fit([0,0],.001,"step_precision",.000000001)
	print "w was:"
	print w.flatten()
	print "we got:"
	print ans
	info = gd.get_loss()
	plt.plot(info)
	plt.ylabel("Loss")
	plt.xlabel("Iteration")
	plt.axis([1000,len(info),0,.2])
	plt.show()

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
	x = np.random.rand(2,1000)
	w = np.random.rand(2)
	# force w to be a column vector
	w.shape = (2,1)
	y = w.transpose().dot(x).flatten()
	m = grad.PyLLSModel()
	gd = grad.PyStochastic_Gradient_Descent(m)
	# we can check this by moving the main loop of fit outside the gradient calculation
	curr, prev = np.array([0,0]), np.array([2,2])
	# while any of the old values are not within the precision of the new values
	losses = np.zeros(0)
	for i in range(10000):
		# hold onto the old value
		prev = curr
		# we want to hand in a single data point or a group of points
		# curr = gd.fit(prev,.001,x,y) 
		# this should allow us to send j points at once
		j = 20
		n = i % (1000 - j)
		m = n + j
		curr = gd.fit(prev,.1,x[:,n:m],y[n:m])
	print "w was:"
	print w.flatten()
	print "we got:"
	print curr
	plt.plot(gd.get_loss())
	plt.ylabel("Loss")
	plt.xlabel("Iteration")
	plt.axis([6000, 10000, 0, .1])
	plt.show()

if __name__ == '__main__':
	main()
import unittest
import numpy as np
import grad
import test_util 

class TestBatchKernelRegression(unittest.TestCase):
	"""Tests for batch gradient descent with kernels."""

	# hyper-parameters for the sover and model
	# N should be between 1 and 1800 for the training data in this directory
	N = 300
	sigma = .2
	lam = 1
	a = 2
	c = 0
	d = .2
	init = np.zeros(N)
	step_szie = .001
	convergence_type = "loss_precision"
	precision = .000001
	# kernel = grad.PyPolynomialKernel(a,c,d)
	kernel = grad.PyGaussianKernel(sigma)
	
	def setUp(self):
		self.model = grad.PyKernelBLRModel(self.kernel, self.lam)

	def tearDown(self):
		self.model = None

	def f(self, w, kernel, X, x):
		return w.dot(kernel.gram_matrix(X,x))

	def guess(self, w, Xtrain, Xtest):
		return np.array([self.f(w, self.kernel, Xtrain[:,:self.N], Xtest[:,i]) for i in range(Xtest.shape[1])])

	def count_misses(slef, ytest, yguess):
		missed = 0
		for i in range(yguess.size):
			if (yguess[i] > 0 and ytest[i] == 0) or (yguess[i] < 0 and ytest[i] == 1):
				missed += 1
		return missed

	def framework(self, num):
		Xtrain, ytrain, Xtest, ytest = test_util.read_data(num)
		solver = grad.PyBatch_Gradient_Descent(Xtrain[:,:self.N],ytrain[:self.N],self.model)
		w = solver.fit(self.init, self.step_szie, self.convergence_type, self.precision)
		yguess = self.guess(w, Xtrain, Xtest)
		misses = self.count_misses(ytest, yguess)
		# we should miss less than 10%
		allowed_misses = .1 * self.N
		error = str.format("Expected fewer than {} misses. There were {} misses.", allowed_misses, misses)
		self.assertTrue(misses < allowed_misses, error)

	def test_classify_0(self):
		self.framework("0")		

	def test_classify_1(self):
		self.framework("1")

	def test_classify_2(self):
		self.framework("2")				

suite = unittest.TestLoader().loadTestsFromTestCase(TestBatchKernelRegression)
unittest.TextTestRunner(verbosity=2).run(suite)

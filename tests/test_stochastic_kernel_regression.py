import unittest
import numpy as np
import grad
import test_util

class TestStochasticKernelRegression(unittest.TestCase):
	"""Tests for stochastic gradient descent with kernels."""

	# hyper-parameters for the solver and model
	batch_size = 32
	# use a multiple of batch_size so that we can do some mini-batching
	# N < 1800 b/c that is the full size of our training set
	N = batch_size * 20
	sigma = .2
	lam = 1
	init = np.zeros(N)
	kernel = grad.PyGaussianKernel(sigma)
	loss_precision = 0.00001
	step_size = 0.001

	def setUp(self):
		self.model = grad.PyStochasticKLRModel(self.kernel, self.lam)

	def tearDown(self):
		self.model = None

	def framework(self, num, batch_size):
		Xtrain, ytrain, Xtest, ytest = test_util.read_data(num)
		solver = grad.PyStochastic_Gradient_Descent(self.model)
		# set the loss to something high and run until it changes by less than the loss_precision
		prev_loss = 10000
		next_loss = 1
		loops = 0
		while abs(prev_loss - next_loss) > self.loss_precision:
			print(loops,prev_loss, next_loss)
			loops += 1
			# shuffle the data before training by batch size
			np.random.seed(prev_loss)
			np.random.shuffle(Xtrain)
			np.random.seed(prev_loss)
			np.random.shuffle(ytrain)
			weights = np.array([1])
			for i in range(0, self.N + 1, batch_size):
				weights = solver.fit(weights, self.step_size, Xtrain[:,i:i+batch_size], ytrain[i:i+batch_size])
			# update the loss 
			prev_loss = next_loss
			next_loss = solver.get_loss()[-1]
		print(loops,prev_loss, next_loss)
		
		# still need to define estimating class labels for models

		# we should miss less than 10%
		# allowed_misses = .1 * self.N
		# error = str.format("Expected fewer than {} misses. There were {} misses.", allowed_misses, misses)
		# self.assertTrue(misses < allowed_misses, error)

	def test_classify_0_no_batch(self):
		self.framework("0", 1)

suite = unittest.TestLoader().loadTestsFromTestCase(TestStochasticKernelRegression)
unittest.TextTestRunner(verbosity=2).run(suite)
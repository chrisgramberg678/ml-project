import unittest
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import bernoulli
import ml_project as ml
import test_util

class TestStochasticGradientDescent(unittest.TestCase):

	def setUp(self):
		self.loss_precision = 0.000001
		self.step_size = 0.01
		self.N = 1800

	def train(self, batch_size, model, data):
		Xtrain, ytrain = data
		solver = ml.SGD(model)
		# set the loss to something high and run until it changes by less than the loss_precision
		prev_loss = np.random.rand() * 100
		next_loss = 1
		loops = 0
		# weights = np.zeros((self.weights,1))
		weights = np.random.rand(self.weights,1)
		for e in range(self.epochs):
			# shuffle the data before training by batch size, using the loss as a seed to keep data lined up
			seed = e + int(abs(prev_loss*1000000)%2**32)
			np.random.seed(seed)
			np.random.shuffle(Xtrain)
			np.random.seed(seed)
			np.random.shuffle(ytrain)
			for i in range(0, self.N, batch_size):
				weights = solver.fit(weights, self.step_size, Xtrain[i:i+batch_size], ytrain[i:i+batch_size])
			# update the loss 
			prev_loss = next_loss
			next_loss = solver.get_loss_values()[-1]
			# print("epoch: {}. loss: {}".format(e, np.mean(solver.get_loss_values())))
		return solver.get_loss_values()

	# @unittest.skip("skip")
	def test_lls(self):
		self.weights = 2
		self.N = 480
		self.epochs = 1000
		m = ml.lls_model()
		xs = 10 * np.random.rand(512,self.weights)
		ws = np.random.rand(self.weights,1)
		ys = xs.dot(ws)
		data = (xs[:self.N],ys[:self.N])
		self.train(32,m,data)
		predictions = m.predict(xs[self.N:])
		self.assertTrue(np.allclose(predictions, ys[self.N:], atol=.2))

	# @unittest.skip("skip")
	def test_blr(self):
		self.weights = 2
		self.epochs = 2000
		for i in range(3):
			m = ml.blr_model()
			train_data, train_labels, validation_data, validation_labels = test_util.read_data(str(i))
			losses = self.train(16, m, (train_data[:self.N], train_labels[:self.N]))
			predictions = m.predict(validation_data).flatten()
			correct = predictions == validation_labels
			missed = 0
			for c in correct:
				if not c:
					missed+=1
			error = "Data set: {}. Missed: {}/{}".format(i,missed,validation_labels[:int(self.N)].size)
			self.assertTrue(missed < .15*validation_labels.size, error)

	# @unittest.skip("skip")
	def test_stochastic_kblr(self):
		self.weights = 1
		self.epochs = 100
		self.N = 200
		self.step_size = 0.1
		for i in range(1):
			k = ml.gaussian_kernel(.1)
			m = ml.sklr_model(k, 0)
			train_data, train_labels, validation_data, validation_labels = test_util.read_data(str(i))
			losses = self.train(1, m, (train_data[:self.N], train_labels[:self.N]))
			predictions = m.predict(validation_data).flatten()
			correct = predictions == validation_labels
			missed = 0
			for c in correct:
				if not c:
					missed+=1
			error = "Data set: {}. Missed: {}/{}".format(i,missed,validation_labels[:int(self.N)].size)
			self.assertTrue(missed < .1*validation_labels.size, error)


suite = unittest.TestLoader().loadTestsFromTestCase(TestStochasticGradientDescent)
unittest.TextTestRunner(verbosity=2).run(suite)
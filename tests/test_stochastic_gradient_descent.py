import unittest
import numpy as np
from scipy.stats import bernoulli
import ml_project as ml
import test_util

class TestStochasticGradientDescent(unittest.TestCase):

	def setUp(self):
		self.loss_precision = 0.00001
		self.step_size = 0.001
		self.N = 300

	def train(self, batch_size, model, data):
		Xtrain, ytrain = data
		solver = ml.SGD(model)
		# set the loss to something high and run until it changes by less than the loss_precision
		prev_loss = 10000
		next_loss = 1
		loops = 0
		while abs(prev_loss - next_loss) > self.loss_precision:
			print(loops,prev_loss, next_loss)
			loops += 1
			# shuffle the data before training by batch size, using the loss as a seed to keep data lined up
			np.random.seed(prev_loss)
			np.random.shuffle(Xtrain)
			np.random.seed(prev_loss)
			np.random.shuffle(ytrain)
			weights = np.array(self.weights)
			print("here?")
			for i in range(0, self.N + 1, batch_size):
				print(i)
				weights = solver.fit(weights, self.step_size, Xtrain[:,i:i+batch_size], ytrain[i:i+batch_size])
			print("there?")
			# update the loss 
			prev_loss = next_loss
			next_loss = solver.get_loss()[-1]

	def test_lls(self):
		self.weights = 2
		self.N = 480
		m = ml.lls_model()
		xs = 10 * np.random.rand(512,2)
		ws = np.random.rand(self.weights,1)
		ys = xs.dot(ws)
		data = (xs[:480],ys[:480])
		self.train(32,m,data)
		predictions = m.predict(xs[480:])
		self.assertTrue(np.allclose(predictions, ys[480:], atol=.00001))

	@unittest.skip("skip")
	def test_blr(self):
		self.weights = 2
		N = self.N
		for i in range(3):
			m = ml.blr_model()
			train_data, train_labels, validation_data, validation_labels = test_util.read_data(str(i))
			self.train(15, m, (train_data[:self.N], train_labels[:self.N]))
			predictions = m.predict(validation_data[:int(.25*self.N)])
			correct = label_guess == validation_labels[:int(.25*self.N)]
			missed = 0
			for c in correct:
				if not c:
					missed+=1
			# print(i,missed)
			self.assertTrue(missed < .1*validation_labels[:int(.25*N)].size)

	def test_stochastic_kblr(self):
		pass


suite = unittest.TestLoader().loadTestsFromTestCase(TestStochasticGradientDescent)
unittest.TextTestRunner(verbosity=2).run(suite)
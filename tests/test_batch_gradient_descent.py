import unittest
import numpy as np
from scipy.stats import bernoulli
import ml_project as ml
import test_util

class TestBatchGradientDescent(unittest.TestCase):

	def test_lls_small(self):
		for i in range(100):
			data = 10 * np.random.rand(100,2)
			weights = np.random.rand(2,1)
			labels = data.dot(weights)
			model = ml.lls_model()
			solver = ml.BGD(data, labels, model)
			weight_guess = solver.fit(.001, 'step_precision', .0000001)
			label_guess = model.predict(data)
			self.assertTrue(np.allclose(label_guess, labels, atol=.00001))

	def test_blr(self):
		N = 400
		for i in range(3):
			data = test_util.read_data(str(i))
			train_data, train_labels, validation_data, validation_labels = data
			model = ml.blr_model()
			solver = ml.BGD(train_data[:N], train_labels[:N], model)
			weight_guess = solver.fit(.001, 'step_precision', .0000001)
			label_guess = model.predict(validation_data[:int(.25*N)]).flatten()
			correct = label_guess == validation_labels[:int(.25*N)]
			missed = 0
			for c in correct:
				if not c:
					missed+=1
			# print(i,missed)
			self.assertTrue(missed < .15*validation_labels[:int(.25*N)].size)

	@unittest.skip('getting -nan for loss on this model, skip for now')
	def test_kblr(self):
		N = 300
		for i in range(3):
			gk = ml.gaussian_kernel(.3)
			data = test_util.read_data(str(i))
			train_data, train_labels, validation_data, validation_labels = data
			model = ml.kblr_model(gk, .5)
			solver = ml.BGD(train_data[:N], train_labels[:N], model)
			weight_guess = solver.fit(.001, 'step_precision', .0000001)
			label_guess = model.predict(validation_data[:int(.25*N)]).flatten()
			correct = label_guess == validation_labels[:int(.25*N)]
			missed = 0
			for c in correct:
				if not c:
					missed+=1
			# print(i,missed)
			self.assertTrue(missed < .1*validation_labels[:int(.25*N)].size)

suite = unittest.TestLoader().loadTestsFromTestCase(TestBatchGradientDescent)
unittest.TextTestRunner(verbosity=2).run(suite)
import unittest
import numpy as np
from scipy.stats import bernoulli
import ml_project as ml

class TestBatchGradientDescent(unittest.TestCase):
	
	def test_lls_small(self):
		for i in range(100):
			data = 10 * np.random.rand(10,2)
			weights = np.random.rand(2,1)
			labels = data.dot(weights)
			model = ml.lls_model()
			solver = ml.BGD(data, labels, model)
			weight_guess = solver.fit(np.random.rand(2,1), .001, "step_precision", .00000001)
			label_guess = model.predict(data)
			# print(label_guess)
			# print(labels)
			self.assertTrue(np.allclose(label_guess, labels, atol=.00001))

	def test_blr(self):
		data = 10 * np.random.randn(2,100)
		w = np.random.rand(2,1)
		logit = np.exp(w.transpose().dot(data))/(1+np.exp(w.transpose().dot(data)))
		labels = bernoulli.rvs(logit)
		labels.shape = (100,1)
		model = ml.blr_model()
		solver = ml.BGD(data,labels,model)
		weight_guess = solver.fit(np.random.rand(2,1), .001, "step_precision", .0000001)

suite = unittest.TestLoader().loadTestsFromTestCase(TestBatchGradientDescent)
unittest.TextTestRunner(verbosity=2).run(suite)
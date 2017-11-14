# unit tests for kernels
import unittest
import ml_project as ml
import numpy as np
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.metrics.pairwise import rbf_kernel

class TestKernels(unittest.TestCase):
	"""Tests for kernels"""

	# constants for filters and kernels
	#polynomial
	pa = 1.5
	pc = .2
	pd = 3
	#rbf/gaussian
	gs = .5
	g = 1/(2*gs*gs)
	# my kernels
	lk = ml.linear_kernel(0)
	pk = ml.polynomial_kernel(pa,pc,pd)
	gk = ml.gaussian_kernel(gs)

	def compare_to_sklearn(self, x, y):
		"""Helper for comparing the three kernels to avoid copy-pasting """
		sk_lk = linear_kernel(x, y)
		my_lk = self.lk.gram_matrix(x, y)
		self.assertTrue(np.allclose(my_lk, sk_lk))

		sk_pk = polynomial_kernel(x, y, self.pd, self.pa, self.pc)
		my_pk = self.pk.gram_matrix(x, y)
		self.assertTrue(np.allclose(my_pk, sk_pk))

		sk_gk = rbf_kernel(x, y, self.g)
		my_gk = self.gk.gram_matrix(x, y)
		self.assertTrue(np.allclose(my_gk, sk_gk))

	# we're expecting exceptions here so there's no need to compare anything
	def check_exceptions(self, x, y):
		message = "to compute a Gram Matrix both input matrices must have the same number of rows."
		with self.assertRaises(ValueError):
			self.lk.gram_matrix(x, y)
		with self.assertRaises(ValueError):
			self.pk.gram_matrix(x, y)
		with self.assertRaises(ValueError):
			self.gk.gram_matrix(x, y)

	def test_2vectors_same_samples(self):
		x = np.random.rand(3,1)
		y = np.random.rand(3,1)
		self.compare_to_sklearn(x, y)

	def test_2vectors_diff_samples(self):
		x = np.random.rand(3,1)
		y = np.random.rand(4,1)
		self.compare_to_sklearn(x,y)

	def test_2vectors_diff_features(self):
		x = np.random.rand(3,1)
		y = np.random.rand(3,2)
		self.check_exceptions(x, y)
	
	def test_1vector_1matrix_good(self):
		x = np.random.rand(1,5)
		y = np.random.rand(5,5)
		self.compare_to_sklearn(x, y)

	def test_1vector_1matrix_bad(self):
		x = np.random.rand(5,1)
		y = np.random.rand(5,5)
		self.check_exceptions(x, y)

	def test_2same_samples_matrices_good(self):
		x = np.random.rand(5,5)
		y = np.random.rand(5,5)
		self.compare_to_sklearn(x, y)

	def test_2matrices_diff_features_bad(self):
		x = np.random.rand(2,2)
		y = np.random.rand(2,3)
		self.check_exceptions(x, y)

	def test_2diff_samples_matrices_good(self):
		x = np.random.rand(3,3)
		y = np.random.rand(2,3)
		self.compare_to_sklearn(x, y)

	def test_2diff_samples_and_features_matrices_bad(self):
		x = np.random.rand(3,3)
		y = np.random.rand(2,4)
		self.check_exceptions(x, y)

	def test_large_input(self):
		x = np.random.rand(2000,2000)
		y = np.random.rand(2000,2000)
		self.compare_to_sklearn(x, y)

	def test_call_kernel_gram_matrix(self):
		k = ml.kernel()
		a = np.zeros(1)
		with self.assertRaises(Exception) as e:
			k.gram_matrix(a,a)

suite = unittest.TestLoader().loadTestsFromTestCase(TestKernels)
unittest.TextTestRunner(verbosity=2).run(suite)
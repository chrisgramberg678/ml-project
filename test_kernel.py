# unit tests for kernels
import unittest
import grad
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
	lk = grad.PyLinearKernel(0)
	pk = grad.PyPolynomialKernel(pa,pc,pd)
	gk = grad.PyGaussianKernel(gs)

	def compare_to_sklearn(self, x, y):
		"""Helper for comparing the three kernels to avoid copy-pasting """
		sk_lk = linear_kernel(x.transpose(), y.transpose())
		my_lk = self.lk.gram_matrix(x, y)
		self.assertTrue(np.allclose(my_lk, sk_lk))

		sk_pk = polynomial_kernel(x.transpose(), y.transpose(), self.pd, self.pa, self.pc)
		my_pk = self.pk.gram_matrix(x, y)
		self.assertTrue(np.allclose(my_pk, sk_pk))

		sk_gk = rbf_kernel(x.transpose(), y.transpose(), self.g)
		my_gk = self.gk.gram_matrix(x, y)
		self.assertTrue(np.allclose(my_gk, sk_gk))

	# we're expecting exceptions here so there's no need to compare anything
	def check_exceptions(self, x, y):
		message = "to compute a Gram Matrix both input matrices must have the same number of rows."
		failed = "we should have gotten an exception here"
		try:
			my_lk = self.lk.gram_matrix(x, y)
			self.fail(failed)
		except Exception as e:
			self.assertTrue(e.message, message)

		try:
			my_pk = self.pk.gram_matrix(x, y)
			self.fail(failed)
		except Exception as e:
			self.assertTrue(e.message, message)

		try:
			my_gk = self.gk.gram_matrix(x, y)
			self.fail(failed)
		except Exception as e:
			self.assertTrue(e.message, message)

	def test_2vectors_good(self):
		# test data 
		x = np.array([1,2,3,4])
		y = np.array([5,6,7,8])
		x.shape = y.shape = (4,1)
		self.compare_to_sklearn(x, y)

	def test_2vectors_bad(self):
		# note the mismatch in shapes here
		x = np.array([1,2,3])
		y = np.array([1,2])
		x.shape = (3,1)
		y.shape = (2,1)
		self.check_exceptions(x, y)

	def test_1vector_1matrix_good(self):
		x = np.array([1,2])
		x.shape = (2,1)
		y = np.array([[1,2],[3,4]])
		self.compare_to_sklearn(x, y)

	def test_1vector_1matrix_bad(self):
		x = np.array([1,2,3])
		x.shape = (3,1)
		y = np.array([[1,2],[3,4]])
		self.check_exceptions(x, y)

	def test_2same_samples_matrices_good(self):
		x = np.array([[1,2,3],[4,5,6]])
		y = np.array([[7,8,9],[1,2,3]])
		self.compare_to_sklearn(x, y)

	def test_2same_samples_matrices_bad(self):
		x = np.array([[1,2],[3,4]])
		y = np.array([[1,2,3],[4,5,6]])
		self.check_exceptions(x, y)

	def test_2diff_samples_matrices_good(self):
		x = np.array([[1,2],[3,4]])
		y = np.array([[9,8,7],[6,5,4]])
		self.compare_to_sklearn(x, y)

	def test_2diff_samples_matrices_bad(self):
		x = np.array([[1,2],[3,4],[5,6]])
		y = np.array([[1,2,3],[4,5,6]])
		self.check_exceptions(x, y)

	@unittest.skip("this one takes about 50 seconds on my laptop")
	def test_stress(self):
		"""Let's see what happens when we use big input"""
		x = np.random.rand(2000,2000)
		y = np.random.rand(2000,2000)
		self.compare_to_sklearn(x, y)

suite = unittest.TestLoader().loadTestsFromTestCase(TestKernels)
unittest.TextTestRunner(verbosity=2).run(suite)
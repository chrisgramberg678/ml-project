from eigency.core cimport *
from libcpp.vector cimport vector
from decl cimport kernel as _kernel
from decl cimport linear_kernel as _linear_kernel
from decl cimport polynomial_kernel as _polynomial_kernel
from decl cimport gaussian_kernel as _gaussian_kernel
from decl cimport model as _model
from decl cimport linear_least_squares_model as _lls_model
from decl cimport binary_logistic_regression_model as _blr_model
from decl cimport kernel_binary_logistic_regression_model as _kblr_model
from decl cimport stochastic_kernel_logistic_regression_model as _sklr_model
from decl cimport optomization_solver_base as _solver_base
from decl cimport batch_gradient_descent as _BGD
from decl cimport stochastic_gradient_descent as SGD
import numpy as np

def col_major(n):
	"""convert the array to column-major for Eigen"""
	# transpose matricies with either one sample or one feature to compensate for switching from row major to column major
	if n.shape[0] == 1 or n.shape[1] == 1:
		n = n.transpose()
	return np.reshape(n,n.shape,order='F')

cdef class kernel:
	"""Abstract Class that serves as a base for kernels and provides an implementation of gram_matrix()"""
	cdef _kernel *thisptr;

	def __cinit__(self):
		thisptr = NULL

	def __dealloc__(self):
		pass

	def gram_matrix(self, np.ndarray X, np.ndarray Y):
		if self.thisptr is NULL:
			raise Exception("Cannot call gram_matrix() on kernel base class!")
		else:
			return ndarray_copy(self.thisptr.gram_matrix(Map[MatrixXd](col_major(X)), Map[MatrixXd](col_major(Y))))

cdef class linear_kernel(kernel):
	"""linear kernel, impl based on:
	http://crsouza.com/2010/03/17/kernel-functions-for-machine-learning-applications/#linear
	"""
	
	def __cinit__(self, double c):
		self.thisptr = new _linear_kernel(c)

	def __dealloc__(self):
		del self.thisptr

cdef class polynomial_kernel(kernel):
	"""polynomial kernel, impl based on:
	http://crsouza.com/2010/03/17/kernel-functions-for-machine-learning-applications/#polynomial
	"""

	def __cinit__(self, double a, double c, double d):
		self.thisptr = new _polynomial_kernel(a, c, d)

	def __dealloc__(self):
		del self.thisptr

cdef class gaussian_kernel(kernel):
	"""gaussian kernel, impl based on:
	http://crsouza.com/2010/03/17/kernel-functions-for-machine-learning-applications/#gaussian
	"""
	
	def __cinit__(self, double s):
		self.thisptr = new _gaussian_kernel(s)

	def __dealloc__(self):
		del self.thisptr

cdef class model:
	"""Abstract Base class for models"""
	cdef _model *thisptr

	def __cinit__(self):
		thisptr = NULL

	def __dealloc__(self):
		pass

	def predict(self, np.ndarray weights, np.ndarray X):
		if self.thisptr is NULL:
			raise Exception("Cannot call predict() on model base class!")
		else:
			return ndarray_copy(self.thisptr.predict(Map[VectorXd](col_major(weights)), Map[MatrixXd](col_major(X))))

cdef class lls_model(model):

	def __cinit__(self):
		self.thisptr = new _lls_model()

	def __dealloc__(self):
		del self.thisptr

cdef class blr_model(model):

	def __cinit__(self):
		self.thisptr = new _blr_model()

	def __dealloc__(self):
		del self.thisptr

cdef class kblr_model(model):

	def __cinit__(self, kernel k, double l):
		self.thisptr = new _kblr_model(k.thisptr, l)

	def __dealloc__(self):
		del self.thisptr

cdef class sklr_model(model):

	def __cinit__(self, kernel k, double l):
		self.thisptr = new _sklr_model(k.thisptr, l)

	def __dealloc__(self):
		del self.thisptr

cdef class solver:
	"""Abstract base class for solvers"""
	cdef _solver_base *thisptr

	def __cinit__(self):
		self.thisptr = NULL

	def __dealloc__(self):
		pass

	def get_loss_values(self):
		if self.thisptr is NULL:
			raise Exception("Cannot call get_loss_values on solver base class!")
		else:
			return np.array(list(self.thisptr.get_loss_values()))

cdef class BGD(solver):
	cdef _BGD *bgdptr

	def __cinit__(self, np.ndarray x, np.ndarray y, model m not None):
		print(m)
		self.thisptr = bgdptr = new _BGD(Map[MatrixXd](col_major(x)), Map[VectorXd](col_major(y)), m.thisptr)

	def __dealloc__(self):
		del self.thisptr

	def fit(self, np.ndarray init, double step_size, str conv_type, double conv_val):
		return ndarray_copy(self.bgdptr.fit(Map[VectorXd](col_major(init)), step_size, conv_type, conv_val))
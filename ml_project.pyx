from eigency.core cimport *
from decl cimport kernel as _kernel
from decl cimport linear_kernel as _linear_kernel
from decl cimport polynomial_kernel as _polynomial_kernel
from decl cimport gaussian_kernel as _gaussian_kernel
from decl cimport model as _model
from decl cimport linear_least_squares_model as _lls_model
from decl cimport binary_logistic_regression_model as _blr_model
from decl cimport kernel_binary_logistic_regression_model as _kblr_model
from decl cimport stochastic_kernel_logistic_regression_model as _sklr_model
import numpy as np

def col_major(n):
	"""convert the array to column-major for Eigen"""
	# transpose matricies with either one sample or one feature to compensate for switching from row major to column major
	if n.shape[0] == 1 or n.shape[1] == 1:
		n = n.transpose()
	return np.reshape(n,n.shape,order='F')

cdef class kernel:
	"""Abstract Class that serves as a base for kernels and provides an implementation of gram_matrix()"""
	cdef _kernel* thisptr;

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
	cdef _linear_kernel* linearptr

	def __cinit__(self, double c):
		self.thisptr = self.linearptr = new _linear_kernel(c)
		if self.thisptr is NULL:
			raise MemoryError()

	def __dealloc__(self):
		del self.linearptr

cdef class polynomial_kernel(kernel):
	"""polynomial kernel, impl based on:
	http://crsouza.com/2010/03/17/kernel-functions-for-machine-learning-applications/#polynomial
	"""
	cdef _polynomial_kernel* polyptr

	def __cinit__(self, double a, double c, double d):
		self.thisptr = self.polyptr = new _polynomial_kernel(a, c, d)
		if self.thisptr is NULL:
			raise MemoryError()

	def __dealloc__(self):
		del self.polyptr

cdef class gaussian_kernel(kernel):
	"""gaussian kernel, impl based on:
	http://crsouza.com/2010/03/17/kernel-functions-for-machine-learning-applications/#gaussian
	"""
	cdef _gaussian_kernel* gaussptr

	def __cinit__(self, double s):
		self.thisptr = self.gaussptr = new _gaussian_kernel(s)
		if self.thisptr is NULL:
			raise MemoryError()

	def __dealloc__(self):
		del self._gaussian_kernel

cdef class model:
	"""Abstract Base class for models"""
	cdef _model* thisptr

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
	cdef _lls_model* llsptr

	def __cinit__(self):
		self.thisptr = self.llsptr = new _lls_model()

	def __dealloc__(self):
		del self.llsptr

cdef class blr_model(model):
	cdef _blr_model* blrptr

	def __cinit__(self):
		self.thisptr = self.blrptr = new _blr_model()

	def __dealloc__(self):
		del self.blrptr

cdef class kblr_model(model):
	cdef _kblr_model* kblrptr

	def __cinit__(self, kernel k, double l):
		self.thisptr = self.kblrptr = new _kblr_model(k.thisptr, l)

	def __dealloc__(self):
		del self.kblrptr

cdef class sklr_model(model):
	cdef _sklr_model* sklrptr

	def __cinit__(self, kernel k, double l):
		self.thisptr = self.sklrptr = new _sklr_model(k.thisptr, l)

	def __dealloc__(self):
		del self.sklrptr
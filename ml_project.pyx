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
from decl cimport stochastic_gradient_descent as _SGD
import numpy as np

def col_major(n):
	"""convert the array to column-major for Eigen"""
	# if the shape of n is (len(n),) then it is treated as having len(n) samples with 1 feature so we'll adjust the shape
	if len(n.shape) == 1:
		n.shape = len(n),1
	return np.array(n.transpose(),order='F')

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
			_x = col_major(X)
			_y = col_major(Y)
			return ndarray_copy(self.thisptr.gram_matrix(Map[MatrixXd](_x), Map[MatrixXd](_y)))

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
	cdef _model* thisptr

	def __cinit__(self):
		thisptr = NULL

	def __dealloc__(self):
		pass

	def predict(self, np.ndarray X):
		if self.thisptr is NULL:
			raise Exception("Cannot call predict() on model base class!")
		else:
			_x = col_major(X)
			return ndarray_copy(self.thisptr.predict(Map[MatrixXd](_x)))

cdef class lls_model(model):
	'''Linear Least Squares Model'''
	def __cinit__(self):
		self.thisptr = new _lls_model()

	def __dealloc__(self):
		del self.thisptr

cdef class blr_model(model):
	'''Binary Logistic Regression Model'''
	def __cinit__(self):
		self.thisptr = new _blr_model()

	def __dealloc__(self):
		del self.thisptr

cdef class kblr_model(model):
	'''Kernel Binary Logistic Regression Model'''
	def __cinit__(self, kernel k, double l):
		self.thisptr = new _kblr_model(k.thisptr, l)

	def __dealloc__(self):
		del self.thisptr

cdef class sklr_model(model):
	'''Stochastic Kernel Binary Logistic Regression Model'''
	def __cinit__(self, kernel k, double l):
		self.thisptr = new _sklr_model(k.thisptr, l)

	def __dealloc__(self):
		del self.thisptr

cdef class solver:
	"""Abstract base class for solvers"""
	cdef _solver_base* thisptr

	def __cinit__(self):
		thisptr = NULL

	def __dealloc__(self):
		pass

	def get_loss_values(self):
		if self.thisptr is NULL:
			raise Exception("Cannot call get_loss_values on solver base class!")
		else:
			return np.array(list(self.thisptr.get_loss_values()))

cdef class BGD(solver):
	'''Batch Gradient Descent Solver'''
	cdef _BGD* bgdptr

	def __cinit__(self, np.ndarray x, np.ndarray y, model m not None):
		cdef _model* mod = m.thisptr
		_x = col_major(x)
		_y = col_major(y)
		self.thisptr = self.bgdptr = new _BGD(Map[MatrixXd](_x), Map[VectorXd](_y), mod)

	def __dealloc__(self):
		del self.thisptr
		
	def fit(self, np.ndarray init, double step_size, str conv_type, double conv_val):
		_init = col_major(init)
		return ndarray_copy(self.bgdptr.fit(Map[VectorXd](_init), step_size, conv_type, conv_val))

cdef class SGD(solver):
	'''Stochastic Gradient Descent Solver'''
	cdef _SGD* sgdptr

	def __cinit__(self, model m not None):
		cdef _model* mod = m.thisptr
		self.thisptr = self.sgdptr = new _SGD(mod)

	def __dealloc__(self):
		del self.thisptr

	def fit(self, np.ndarray init, double step_size, np.ndarray data, np.ndarray labels):
		print("sup")
		_init = col_major(init)
		_data = col_major(data)
		_labels = col_major(labels)
		print("hmm")
		return ndarray_copy(self.sgdptr.fit(Map[VectorXd](_init), step_size, Map[MatrixXd](_data), Map[VectorXd](_labels)))
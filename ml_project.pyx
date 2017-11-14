from eigency.core cimport *
from decl cimport kernel as _kernel
from decl cimport linear_kernel as _linear_kernel
from decl cimport polynomial_kernel as _polynomial_kernel
from decl cimport gaussian_kernel as _gaussian_kernel
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
			raise Exception("Cannot call gram_matrix on kernel base class!")
		else:
			return ndarray_copy(self.thisptr.gram_matrix(Map[MatrixXd](col_major(X)), Map[MatrixXd](col_major(Y))))

cdef class linear_kernel(kernel):
	"""linear kernel, impl based on:
	http://crsouza.com/2010/03/17/kernel-functions-for-machine-learning-applications/#linear
	"""
	cdef _linear_kernel *linearptr

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
	cdef _polynomial_kernel *polyptr

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
	cdef _gaussian_kernel *gaussptr

	def __cinit__(self, double s):
		self.thisptr = self.gaussptr = new _gaussian_kernel(s)
		if self.thisptr is NULL:
			raise MemoryError()

	def __dealloc__(self):
		del self._gaussian_kernel

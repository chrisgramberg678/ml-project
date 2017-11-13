from eigency.core cimport *

cdef extern from "kernel.h":
	# kernel classes 
	cdef cppclass kernel "kernel":
		kernel() except+
		MatrixXd gram_matrix(Map[MatrixXd], Map[MatrixXd]) except+
	cdef cppclass linear_kernel(kernel):
		linear_kernel(double) except+
	cdef cppclass polynomial_kernel(kernel):
		polynomial_kernel(double, double, double) except+
	cdef cppclass gaussian_kernel(kernel):
		gaussian_kernel(double) except+
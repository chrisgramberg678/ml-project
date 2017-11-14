from eigency.core cimport *

cdef extern from "model.h":
	
	cdef cppclass kernel:
		kernel() except+
		MatrixXd gram_matrix(Map[MatrixXd], Map[MatrixXd]) except+
	cdef cppclass linear_kernel(kernel):
		linear_kernel(double) except+
	cdef cppclass polynomial_kernel(kernel):
		polynomial_kernel(double, double, double) except+
	cdef cppclass gaussian_kernel(kernel):
		gaussian_kernel(double) except+

	cdef cppclass model:
		model() except+
		model(bool) except+
		VectorXd predict(Map[VectorXd], Map[MatrixXd]) except+
	cdef cppclass linear_least_squares_model(model):
		linear_least_squares_model() except+
	cdef cppclass binary_logistic_regression_model(model):
		binary_logistic_regression_model() except+
	cdef cppclass kernel_binary_logistic_regression_model(model):
		kernel_binary_logistic_regression_model(kernel*, double) except+
	cdef cppclass stochastic_kernel_logistic_regression_model(model):
		stochastic_kernel_logistic_regression_model(kernel*, double) except+
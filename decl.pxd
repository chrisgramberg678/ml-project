from eigency.core cimport *
from libcpp.vector cimport vector
from libcpp.string cimport string

cdef extern from "gradient_descent.h":
	
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
		VectorXd predict(Map[MatrixXd]) except+
	cdef cppclass linear_least_squares_model(model):
		linear_least_squares_model() except+
	cdef cppclass binary_logistic_regression_model(model):
		binary_logistic_regression_model() except+
	cdef cppclass kernel_binary_logistic_regression_model(model):
		kernel_binary_logistic_regression_model(kernel*, double) except+
	cdef cppclass stochastic_kernel_logistic_regression_model(model):
		stochastic_kernel_logistic_regression_model(kernel*, double) except+

	cdef cppclass optomization_solver_base:
		optomization_solver_base() except+
		vector[double] get_loss_values() except+
	cdef cppclass batch_gradient_descent(optomization_solver_base):
		batch_gradient_descent() except+
		batch_gradient_descent(Map[MatrixXd], Map[VectorXd], model*) except+
		VectorXd fit(Map[VectorXd], double, string, double) except+
	cdef cppclass stochastic_gradient_descent(optomization_solver_base):
		stochastic_gradient_descent() except+
		stochastic_gradient_descent(model*) except+
		VectorXd fit(Map[VectorXd], double, Map[MatrixXd], Map[MatrixXd]) except+
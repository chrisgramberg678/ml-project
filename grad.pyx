# distutils: language = c++
# distutils: sources = gradient_descent.cpp model.cpp kernel.cpp utilities.cpp
from libcpp.vector cimport vector
from libcpp.string cimport string
import numpy as np

cdef extern from "kernel.h" :
	cdef cppclass kernel:
		kernel() except +
		vector[ vector[double] ] py_gram_matrix(vector[ vector[double] ], vector[ vector[double] ]) 
	cdef cppclass linear_kernel(kernel):
		linear_kernel(double) except +
	cdef cppclass polynomial_kernel(kernel):
		polynomial_kernel(double, double, double) except +
	cdef cppclass gaussian_kernel(kernel):
		gaussian_kernel(double) except +

cdef extern from "gradient_descent.h" :
	# model classes
	cdef cppclass model:
		model() except +
	cdef cppclass linear_least_squares_model(model):
		linear_least_squares_model() except +
	cdef cppclass binary_logistic_regression_model(model):
		binary_logistic_regression_model() except +
	# optomization solver classes
	cdef cppclass optomization_solver_base:
		optomization_solver_base() except +
		vector[double] get_loss()
	cdef cppclass batch_gradient_descent(optomization_solver_base):
		batch_gradient_descent() except + 
		batch_gradient_descent(vector[ vector[double] ], vector[double], model*) except +
		vector[double] py_fit(vector[double], double, string, double) except +
	cdef cppclass stochastic_gradient_descent(optomization_solver_base):
		stochastic_gradient_descent() except +
		stochastic_gradient_descent(model*) except +
		vector[double] py_fit(vector[double], double, vector[ vector[double] ], vector[double]) except +

# helper for converting numpy matrices to vector[ vector[double] ]
# this is mostly just a check to see if we need to force row vectors to be column vectors
def np_to_stl(m):
	cdef vector[ vector[double] ] _m
	if m.ndim == 2:
		_m = list(list(m))
	elif m.ndim ==1:
		a = m.shape[0]
		m.shape = (a,1)
		_m = list(list(m))
	return _m

# classes for kernels

cdef class PyKernel:
	cdef kernel* kernelptr
	def __cinit__(self):
		# you're not allowed to make instances of this class so we're not going to do anything if you try
		pass
	def gram_matrix(self,X,Y):
		return self.kernelptr.py_gram_matrix(np_to_stl(X),np_to_stl(Y))

cdef class PyLinearKernel(PyKernel):
	cdef linear_kernel* lkptr
	def __cinit__(self, c):
		if type(self) is PyLinearKernel:
			self.lkptr = self.kernelptr = new linear_kernel(c)

# the python classes that wrap around the c++ classes for models
cdef class PyModel:
	cdef model* modelptr
	def __cinit__ (self):
		# you're not allowed to make instances of this class so we're not going to do anything if you try
		pass
	
cdef class PyLLSModel(PyModel):
	cdef linear_least_squares_model* LLSptr
	def __cinit__ (self):
		if type(self) is PyLLSModel:
			self.LLSptr = self.modelptr = new linear_least_squares_model()

cdef class PyBLRModel(PyModel):
	cdef binary_logistic_regression_model* BLRptr
	def __cinit__ (self):
		if type(self) is PyBLRModel:
			self.BLRptr = self.modelptr = new binary_logistic_regression_model()

# the python class that wraps around the C++ classes for optomization solvers
cdef class PyOptomization_Solver_Base:
	cdef optomization_solver_base* solverptr
	def __cinit__(self):
		# you're not allowed to make instances of this class so we're not going to do anything if you try
		pass
	def get_loss(self):
		# simply call the function on the pointer
		return np.array(list(self.solverptr.get_loss()))

# classes for our solvers
cdef class PyBatch_Gradient_Descent(PyOptomization_Solver_Base):
	# the C++ object that does gradient_descent
	cdef batch_gradient_descent* batchptr	
	def __cinit__(self, x, y, PyModel M):
		# convert the lists x and y into vectors that we can use
		cdef vector[ vector [double] ] _x = list(list(x))
		cdef vector[double] _y = list(y)
		cdef model* m = M.modelptr
		if type(self) is PyBatch_Gradient_Descent:
			self.batchptr = self.solverptr = new batch_gradient_descent(_x, _y, m)
	def fit(self, init, double gamma, str conversion_type = "none", double conv = 1000000):
		# convert the parameters
		cdef vector[double] _init = init
		cdef string s = conversion_type
		# then call the function and convert the resulting vector[double] to a numpy array
		cdef vector[double] result = self.batchptr.py_fit(_init, gamma, s, conv)
		return np.array(list(result))

cdef class PyStochastic_Gradient_Descent(PyOptomization_Solver_Base):
	# the c++ object we're wrapping
	cdef stochastic_gradient_descent* stochasticptr
	def __cinit__(self, PyModel M):
		cdef model* m = M.modelptr
		if type(self) is PyStochastic_Gradient_Descent:
			self.stochasticptr = self.solverptr = new stochastic_gradient_descent(m)
	def fit(self, prev, double gamma, x, y):
		cdef vector[double] _prev = list(prev)
		# we need to check whether this x has 1 data point or many
		cdef vector[double] _y
		if y.ndim == 1:
			_y = list(y)
		elif y.ndim == 0:
			_y.push_back(y)
		cdef vector[double] res = self.stochasticptr.py_fit(_prev, gamma, np_to_stl(x), _y)
		return np.array(list(res))
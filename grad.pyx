# distutils: language = c++
# distutils: sources = gradient_descent.cpp model.cpp
from libcpp.vector cimport vector
import numpy as np

cdef extern from "gradient_descent.h" :
	# model base classes
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
		batch_gradient_descent(vector[ vector[double] ], vector[double]) except +
		vector[double] py_fit(vector[double], double, double, model*) except +

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

cdef class PyBatch_Gradient_Descent(PyOptomization_Solver_Base):
	# the C++ object that does gradient_descent
	cdef batch_gradient_descent* batchptr	
	def __cinit__(self, x, y):
		# convert the lists x and y into vectors that we can use
		cdef vector[ vector [double] ] _x = list(list(x))
		cdef vector[double] _y = list(y)
		if type(self) is PyBatch_Gradient_Descent:
			self.batchptr = self.solverptr = new batch_gradient_descent(_x, _y)
	def fit(self, init, double gamma, double precision, PyModel M):
		# convert the parameters
		cdef vector[double] _init = init
		cdef model* m = M.modelptr
		# then call the function and convert the resulting
		# vector[double] to a numpy array
		cdef vector[double] result = self.batchptr.py_fit(_init, gamma, precision, m)
		return np.array(list(result))
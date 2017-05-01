# distutils: language = c++
# distutils: sources = gradient_descent.cpp model.cpp
from libcpp.vector cimport vector
import numpy as np

cdef extern from "gradient_descent.h" :
	cdef cppclass model:
		model() except +
	cdef cppclass linear_least_squares_model(model):
		linear_least_squares_model() except +

	cdef cppclass binary_logistic_regression_model(model):
		binary_logistic_regression_model() except +

	cdef cppclass gradient_descent:
		gradient_descent() except + 
		gradient_descent(vector[ vector[double] ], vector[double]) except +
		vector[double] py_fit(vector[double], double, double, model*) except +
		vector[double] py_stochastic_fit(vector[double], double, model*) except +
	

#import the model class so we can wrap it up its children (linear least squares and logistic regression)
# cdef extern from "model.h" :

# the python classes that wrap around the c++ classes for models
cdef class PyModel:
	cdef model* modelptr
	def __cinit__ (self):
		if type(self) is PyModel:
			self.modelptr = new model()
	
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

# the python class that wraps around the C++ class "gradient_descent"
cdef class PyGradient_Descent:
	# the C++ object that does gradient_descent
	cdef gradient_descent c_gd	
	def __cinit__(self, x, y):
		# convert the lists x and y into vectors that we can use
		cdef vector[ vector [double] ] _x = list(list(x))
		cdef vector[double] _y = list(y)
		self.c_gd = gradient_descent(_x, _y)
	def fit(self, init, double gamma, double precision, PyModel M):
		# convert the parameters
		cdef vector[double] _init = init
		cdef model* m = M.modelptr
		# then call the function and convert the resulting
		# vector[double] to a numpy array
		cdef vector[double] result = self.c_gd.py_fit(_init, gamma, precision, m)
		return np.array(list(result))
	def stochastic_fit(self, prev, double gamma, PyModel M):
		# convert the parameters
		cdef vector[double] _prev = prev
		cdef model* m = M.modelptr
		cdef vector[double] result = self.c_gd.py_stochastic_fit(_prev, gamma, m)
		return np.array(list(result))
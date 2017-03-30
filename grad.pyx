# distutils: language = c++
# distutils: sources = gradient_descent.cpp
from libcpp.vector cimport vector
import numpy as np
cdef extern from "gradient_descent.h" :
	cdef cppclass gradient_descent:
		gradient_descent() except + 
		gradient_descent(vector[ vector[double] ], vector[double]) except +
		vector[double] py_fit(vector[double], double, double) except +

cdef class PyGradient_Descent:
	cdef gradient_descent c_gd	# the C++ object that does gradient_descent
	def __cinit__(self, x, y):
		# convert the lists x and y into vectors that we can use
		cdef vector[ vector [double] ] _x = list(list(x))
		cdef vector[double] _y = list(y)
		self.c_gd = gradient_descent(_x, _y)
	def fit(self, init, double gamma, double precision):
		# convert the initial values to a vector of doubles
		cdef vector[double] _init = init
		# then call the function and convert the resulting
		# vector[double] to a numpy array
		cdef vector[double] _res = self.c_gd.py_fit(_init, gamma, precision)
		return np.array(list(_res))
# simple test 
def test():
	xs = np.array([[7, 5, 3, 5], [6, 2, 9, 1]])
	ys = np.array([50.32, 37.97, 12.2, 39.69])
	gd = PyGradient_Descent(xs,ys)
	print gd.fit([0,0],.001,.000000001)
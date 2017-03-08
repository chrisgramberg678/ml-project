# distutils: language = c++
# distutils: sources = gradient_descent.cpp
from libcpp.vector cimport vector
import numpy as np
cdef extern from "gradient_descent.h" :
	cdef cppclass gradient_descent:
		gradient_descent() except + 
		gradient_descent(vector[int], vector[int]) except +
		vector[double] py_fit(vector[int], double, double, int) except +

cdef class PyGradient_Descent:
	cdef gradient_descent c_gd	# the C++ object that does gradient_descent
	def __cinit__(self, x, y):
		# convert the lists x and y into vectors that we can use
		cdef vector[int] _x = list(x)
		cdef vector[int] _y = list(y)
		self.c_gd = gradient_descent(_x, _y)
	def fit(self, ab, double gamma, double precision):
		# convert the vector[double] to a numpy array
		cdef vector[int] _ab = ab
		cdef vector[double] _res = self.c_gd.py_fit(_ab, gamma, precision, 0)
		return np.array(list(_res))
# simple test 
def test():
	xs = np.array([1,2,3,4])
	ys = np.array([6,5,7,10])
	gd = PyGradient_Descent(xs,ys)
	print gd.fit([0,0],.001,.000000001)
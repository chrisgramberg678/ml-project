# super simple sanity check for kernels
import grad
import numpy as np
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.metrics.pairwise import rbf_kernel

# constants for filters

#linear
lc = 0 

#polynomial
pa = 1
pc = 0
pd = 1

#rbf/gaussian
gs = .5
g = 1/(2*gs*gs)

# my kernels
lk = grad.PyLinearKernel(lc)
pk = grad.PyPolynomialKernel(pa,pc,pd)
gk = grad.PyGaussianKernel(gs)

# test data 
x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])


print "sklearn linear kernel gram matrix"
print linear_kernel(x,y)
print "my linear kernel gram matrix"
print lk.gram_matrix(x,y)
print "sklearn linear kernel gram matrix with transpose of the data"
print linear_kernel(x.transpose(),y.transpose())

# print "polynomial"
# print polynomial_kernel(x,y,pa,pc,pd)
# print pk.gram_matrix(x,y)

# print "rbf/gaussian"
# print rbf_kernel(x,y,g)
# print gk.gram_matrix(x,y,gs)
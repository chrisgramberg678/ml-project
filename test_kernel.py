# super simple sanity check for kernels
import grad
import numpy as np

lk = grad.PyLinearKernel(0)
x = np.array([[1,2],[3,4]])
y = np.array([5,6])

print lk.gram_matrix(x,y)
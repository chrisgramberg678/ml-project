# ml-project

To test the C++ code on it's own compile using:

`g++ -std=c++11 gradient_descent.cpp test_gradient_descent.cpp -o gradient_descent`

and run using

'./gradient_descent'

The python module doesn't work right now b/c it needs to updated to work with the new way gradient descent is implemented.

To use this as a Python module compile it using the command: 

`python gradient_descent_setup.py build_ext --inplace`

This will create a grad.so file which can be imported into Python using grad.so.
There is a test method `grad.test()` and new gradient_descent objects can be created by calling in this way:

`gd = PyGradient_Descent([x-vals],[y-vals])`

And calling 

`gd.fit([start-x,start-y],step-size,precision)`

will return the coeffecients which fit the line through the points.

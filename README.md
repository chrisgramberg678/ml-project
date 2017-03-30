# ml-project

To test the C++ code on it's own compile using:

`g++ -std=c++11 gradient_descent.cpp test_gradient_descent.cpp -o gradient_descent`

and run using

`./gradient_descent`

To use this as a Python module compile it using the command: 

`python gradient_descent_setup.py build_ext --inplace`

This will create a grad.so file which can be imported into Python using `import grad`.
There is a test method `grad.test()` which demonstrates a simple example of using the PyGradient_Descent class. New gradient_descent objects can be created in this way:

`gd = PyGradient_Descent([X],[y])`

Here, `X` is defined as a dxn Matrix where n is the number of data points and d is the dimensionality of each point. `y` is defined as a 1xn vector of labels for the n data points.

And calling 

`gd.fit([inital_values],step_size,precision)`

will return a 1xn vector of the coeffecients which fit the given X and y.
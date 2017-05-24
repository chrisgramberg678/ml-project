# ml-project

This library implements batch gradient descent and stochastic gradient descent for both a Linear Least Squares Model and a Binary Logistic Regression Model. The library is written in C++ and Cython is used to compile it to a Python module.

It depends on Eigen (eigen.tuxfamily.org) and numpy (numpy.org).

To use this as a Python module compile it using the command: 

`python gradient_descent_setup.py build_ext --inplace`

This will create grad.so which can be imported into Python using `import grad` like a normal python module.

To use this library you'll need to intialize a model as: 

`m = grad.PyLLSModel()` or `m = grad.PyBLRModel()`

and then use that model to initialize a optomization solver object:

`gd = PyBatch Gradient_Descent(X,y, m)` or `gd = grad.PyStochastic_Gradient_Descent(m)`

Here, `X` is defined as a dxn Matrix where n is the number of data points and d is the dimensionality of each point. `y` is defined as a 1xn vector of labels for the n data points.

Calling the fit() function on a solver will give the w which fits the model. 

`gd.fit(inital_values,step_size,convergence_type,convergence_value)`


will return a 1xn vector of the coeffecients which fit the given X and y.

There are three valid convergence types which are passed in as 

strings:
`step_precision` - stops when the difference between two consecutive steps is less than `convergence_value`
`loss_precision` - stops when the difference between two consecutive loss values at a step is less than `convergence_value`
`iterations` - stops after `convergence_value` iterations

These two parameters are optional and their default values are "iterations" and 1000000

Regardless of the convergence type you pass in the fit function will stop after 1,000,000,000 iterations or if the loss value (which is calculated at each step) gets to the value `numeric_limits<double>::infinity()`.

The `fit()` function for stochastic gradient descent works similarly:

`gd.fit(prev,step_size,X,y)`

however instead of taking an ititial value and computing the fit based on some precision value, it takes a single step. Here X can either be a single data point in the form of a column vector with a single label y, or a matrix of datapoints with a vector of labels y.

All solvers have a function called `get_loss()` which gives a list of all loss values. The batch gradient descent solver stores a loss value for each iteration of its main loop and the stochastic solver stores a loss value for each call to `fit()`.

The file test_grad.py contains some simple tests which demonstrate how to use the library.

It's sometimes useful to compile the C++ library only in order to run tests to ensure it is working before writing the interface to python for new features. To test the C++ code on it's own compile using:

`g++ -std=c++11 model.cpp gradient_descent.cpp test_gradient_descent.cpp -o gradient_descent`

and run using

`./gradient_descent`
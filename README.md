# ml-project

This library implements batch gradient descent and stochastic gradient descent for both a Linear Least Squares Model, a Binary Logistic Regression Model, and a Binary Logistic Regression Model with Kernels. The library is written in C++ and Cython is used to compile it to a Python module.

## Installation

This library has several dependencies, all of which can be installed via pip

### Pip and Python dependencies

Install numpy, scipy, cython, and eigency using `pip install numpy scipy cython eigency` (this may require sudo)

We also use matplotlib to create various figures so if you want to use those you'll need matplotlib as well.

### Installing Eigen

If you'd like to compile the C++ portions of this library on their own you'll need to include the path to the eigen header files. This can be done by installing eigen as described below or by using the version of Eigen provided by eigency. To find the path to eigen from eigency open a python terminal and `import eigency` and call `eigency.get_includes()`. 

To install eigen manually we'll need cmake: `sudo apt install cmake`

Then, go download the latest version of [Eigen](https://eigen.tuxfamily.org).

Extract it into your downloads folder and then install using the following commands:

```
mkdir build_dir
cd build_dir
cmake ~/Downloads/$name_of_extracted_folder
sudo make install
```

## Use

To use this as a Python module compile it using the command: 

`make module`

If you don't want to use make you can also just run `python stepup.py build_ext`

This will create ml_project.so which can be imported into Python using `import ml_project as ml`.

To use this library you'll need to intialize a model as: 

`m = ml.lls_model()` or `m = ml.blr_model()`

and then use that model to initialize a optomization solver object:

`batch_gradient_descent = ml.BGD(data, labels, m)` or `stochastic_gradient_descent = ml.SGD(m)`

Here, `data` is a NxM numpy array with N samples and M features and labels are size N.

Calling the fit() function on a solver will give the w which fits the model. 

__BGD.fit()__

`batch_gradient_descent.fit(inital_values, step_size, convergence_type, convergence_value)`.

There are three valid convergence types which are passed in as strings:

`step_precision` - stops when the difference between two consecutive gradient steps is less than `convergence_value`
`loss_precision` - stops when the difference between two consecutive loss values between gradient steps is less than `convergence_value`
`iterations` - stops after `convergence_value` iterations

These two parameters are optional and their default values are "iterations" and 1000000

Regardless of the convergence type you pass in the fit function will stop after 1,000,000,000 iterations or if the loss value (which is calculated at each step) gets to the value `numeric_limits<double>::infinity()`.

__SGD.fit()__

`stochastic_gradient_descent.fit(prev, step_size, data, labels)`

However instead of taking an ititial value and computing the fit based on some precision value, it takes a single step. Here X can either be a single data point in the form of a column vector with a single label y, or a batch of data as a matrix with a vector of labels y.

All solvers have a function called `get_loss()` which gives a list of all loss values. The batch gradient descent solver stores a loss value for each iteration of its main loop and the stochastic solver stores a loss value for each call to `fit()`.

## Tests

Python tests have been moved into the tests directory. They can be run using `make test` There aren't many yet, but I'm planning to add more.

It's often useful to compile the C++ library as a sanity check before writing the interface to python for new features. To test the C++ code on it's own compile using:

`g++ -std=c++11 -I /usr/local/include/eigen3 model.cpp gradient_descent.cpp test_gradient_descent.cpp -o gradient_descent`

and run using

`./gradient_descent`
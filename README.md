# ml-project

This library implements batch gradient descent and stochastic gradient descent for both a Linear Least Squares Model and a Binary Logistic Regression Model. The library is written in C++ and Cython is used to compile it to a Python module.

## Installation

This library has several dependencies. To start make sure you have the universe repository enabled:

`sudo apt-add-repository universe`

Now update:

`sudo apt-get update`

If you're installing to a usb or cd (or a vm emulating a usb or cd) you may encouter an error which looks something like: `Possible error ** (appstreamcli:9372): CRITICAL **: Error while moving old database out of the way.`

This can be solved using the command: `sudo chmod -R a+rX,u+w /var/cache/app-info/xapian/default`

and an explantion can be found [here](https://askubuntu.com/questions/761592/unable-to-apt-get-dist-upgrade-on-a-persistent-ubuntu-16-04-usb)

Next you'll need git `sudo apt install git`

so you can clone this repo: `git clone https://github.com/chrisgramberg678/ml-project.git` 

### Pip and Python dependencies

Install pip using `sudo apt-get install python-pip` 

and make sure it's updated to the latest version using: `pip install --upgrade pip`

Then install numpy, scipy, and cython using `pip install numpy scipy cython` (this may require sudo)

We'll also need matplotlib for the included test file: `sudo apt install python-matplotlib`

### Installing Eigen

First we'll need cmake: `sudo apt install cmake`

Then, go download the latest version of [Eigen](eigen.tuxfamily.org).

Extract it into your downloads folder and then install using the following commands:

```
mkdir build_dir
cd build_dir
cmake ~/Downloads/$name_of_extracted_folder
sudo make install
```

## Use

To use this as a Python module compile it using the command: 

`python gradient_descent_setup.py build_ext --inplace -I /usr/local/include/eigen3`

This will create grad.so which can be imported into Python using `import grad` like a normal python module.

To see an example run `python test_grad.py`. You can change the function call on line 11 of this file to see the different tests I've set up.

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

It's often useful to compile the C++ library as a sanity check before writing the interface to python for new features. To test the C++ code on it's own compile using:

`g++ -std=c++11 -I /usr/local/include/eigen3 model.cpp gradient_descent.cpp test_gradient_descent.cpp -o gradient_descent`

and run using

`./gradient_descent`
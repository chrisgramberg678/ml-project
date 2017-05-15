#include <iostream>
#include <vector>
#include <limits>
#include <string>
#include "model.h"
using namespace std;
using Eigen::ArrayXd;

/*
  This class serves as a base for solvers that do optomization based on datam, in the form
  of Eigen Matrices, and a Model object
 */
class optomization_solver_base{
	public:
		vector<double> get_loss();
	protected:
		// protected so that no one can create objects of this type
		optomization_solver_base();
		// all solvers need a model
		model* m;

		// all solvers will track the loss value at each step so that we can plot it
		vector<double> loss_values;

		// helpers for interfacting to Cython
		vector<double> eigen_to_stl(VectorXd v);
		VectorXd stl_to_eigen(vector<double> v);
		MatrixXd stl_to_eigen(vector< vector<double> > v);
};


/* 
  this class is initialized with data and its fit function uses batch gradient descent.
  meaning that all of the data is taken into consideration at each gradient step
 */
class batch_gradient_descent : public optomization_solver_base{
	private:
		// the data we want to fit
		// data
		MatrixXd _X;
		//labels
		VectorXd _y;

		// helper for the main loop in fit
		bool done(VectorXd w, double precision);

	public:
		// need a default constructor to appease Cython
		batch_gradient_descent();
		
		/* construction using arbitrary size data X and y
		   dimensionality of X should be dxN where
		   d is the dimensionality of the problem and N is the number of data points
		   dimensionality of y is 1xN
		 */
		batch_gradient_descent(MatrixXd X, VectorXd y, model* M);
		
		//construction using STL vectors so that we can interface with Cython
		batch_gradient_descent(vector< vector<double> > X, vector<double> y, model* M);

		// does the actual work of fitting a model to the data
		VectorXd fit(VectorXd init, double gamma, double precision);

		// a slightly different version of fit so that I don't have to wrap the VexctorXd class for Cython
		// This works because Cython comes with automatic conversion from STL vectors to Python lists
		vector<double> py_fit(vector<double> init, double gamma, double precision);
};


/*
  This class is initalized with just the model.
  Its fit function takes some labeled data and calculates a single gradient step 
  based on the model. It doesn't hold onto any of the data it is passed
 */
class stochastic_gradient_descent : public optomization_solver_base{



	public:
		// need a default constructor to appease Cython
		stochastic_gradient_descent();

		// initialize with just the model
		stochastic_gradient_descent(model* M);
		
		// does a single step using only set of data
		VectorXd fit(VectorXd prev, double gamma, MatrixXd X, VectorXd y);
		// wrapper for Cython to call stochastic_fit
		vector<double> py_fit(vector<double> prev, double gamma, vector< vector<double> > X, vector<double> y);
};
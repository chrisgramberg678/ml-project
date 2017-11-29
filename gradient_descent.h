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
		vector<double> get_loss_values();
	protected:
		// protected so that no one can create objects of this type
		optomization_solver_base();
		
		// all solvers need a model
		model* _model;

		// all solvers will track the loss value at each step so that we can plot it
		vector<double> _loss_values;
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
		bool done(string type, double conv, int iteration, VectorXd step_diff, double loss_diff);

	public:
		batch_gradient_descent();
		/** 
		 * construction using arbitrary size data X and y
		 * dimensions of X are dxN where
		 * d is the number of features and N is the number of data points
		 * dimensions of y are 1xN
		 */
		batch_gradient_descent(Map<MatrixXd> X, Map<VectorXd> y, model* M);
		
		// does the actual work of fitting a model to the data
		VectorXd fit(Map<VectorXd> init, double gamma, string convergence_type = "none", double conv = 1000000);
};


/*
  This class is initalized with just the model.
  Its fit function takes some labeled data and calculates a single gradient step 
  based on the model. It doesn't hold onto any of the data it is passed
 */
class stochastic_gradient_descent : public optomization_solver_base{
	public:
		stochastic_gradient_descent();

		// initialize with just the model
		stochastic_gradient_descent(model* M);
		
		// does a single step using only set of data
		// needs to query the gradient and update the model, but the model should not update itself
		VectorXd fit(Map<VectorXd> prev, double gamma, Map<MatrixXd> X, Map<VectorXd> y);
		
};
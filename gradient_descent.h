#include <iostream>
#include <vector>
#include <limits>
#include <string>
#include "model.h"
using namespace std;
using Eigen::ArrayXd;

class gradient_descent{
	private:
		// the data we want to fit
		// data
		MatrixXd _X;
		//labels
		VectorXd _y;

		// keeps track of where we are in _X when using stochastic fit
		int stochastic_step;

		// helper for the main loop in fit
		bool done(VectorXd w, double precision);

		// helpers for interfacting to Cython
		vector<double> eigen_to_stl(VectorXd v);
		VectorXd stl_to_eigen(vector<double> v);
		MatrixXd stl_to_eigen(vector< vector<double> > v);

	public:
		// need a default constructor to appease Cython
		gradient_descent();
		// construction using arbitrary size data X and y
		// dimensionality of X should be dxN where
		// d is the dimensionality of the problem and N is the number of data points
		// dimensionality of y is 1xN
		gradient_descent(MatrixXd X, VectorXd y);
		//construction using STL vectors so that we can interface with Cython
		gradient_descent(vector< vector<double> > X, vector<double> y);

		// does the actual work of fitting a model to the data
		VectorXd fit(VectorXd init, double gamma, double precision, model* M, bool verbose = false);

		// does a single step using only one of the data points
		VectorXd stochastic_fit(VectorXd prev, double gamma, model* M, bool verbose = false);

		// a slightly different version of fit so that I don't have 
		// to wrap the VexctorXd class for Cython
		vector<double> py_fit(vector<double> init, double gamma, double precision, model* M);

		// wrapper for Cython to call stochastic_fit
		vector<double> py_stochastic_fit(vector<double> prev, double gamma, model* M);


		// for stochastic we'll have python do the outer loop
		// be able to ask for W and also ask for an update to W
};
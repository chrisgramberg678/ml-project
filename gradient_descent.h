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
		VectorXd fit(VectorXd init, double gamma, double precision, Model* M, bool verbose = false);

		// a slightly different version of fit so that I don't have 
		// to wrap the entire Eigen library for Cython
		vector<double> py_fit(vector<double> init, double gamma, double precision, Model* M);
};
#include <iostream>
#include <Eigen/Dense>
#include <vector>
#include <limits>
using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::ArrayXd;

class gradient_descent{
	private:
		// the data we want to fit
		// data
		MatrixXd _X;
		//labels
		VectorXd _y;

		// computes the gradient vector by doing the summation over the training data 
		VectorXd gradient(VectorXd w);

		// L(w) = 1/N sum (w^T*x_i - y_i)^2
		// we know that our fit is working when the value of the loss function gets smaller
		double Loss(VectorXd w);

		// helper for the main loop in fit
		bool done(VectorXd w, double precision);

	public:
		// need a default constructor to appease Cython
		gradient_descent();
		// construction using arbitrary size data X and y
		// dimensionality of X should be dxN where
		// d is the dimensionality of the problem and N is the number of data points
		// dimensionality of y is 1xN
		gradient_descent(MatrixXd X, VectorXd y);

		// does the actual work of fitting a model to the data
		VectorXd fit(VectorXd init, double gamma, double precision, bool verbose = false);

		// a slightly different version of fit so that I don't have 
		// to wrap the entire Eigen library for Cython
		vector<double> py_fit(vector<int> ab, double gamma, double precision, bool verbose = false);
};
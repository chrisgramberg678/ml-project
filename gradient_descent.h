#include <iostream>
#include <Eigen/Dense>
#include <math.h> // used for fabs() -- absolute value of float
#include <vector>
using namespace std;
using Eigen::Vector2d;

class gradient_descent{
	private:
		// the data we're going to fit a line to
		vector<int> x,y;

		// dL/da = 1/N sum(-2x(y-xa-b))
		// where x and y come from the training data and N is the size of x and y
		double dLda(double a, double b);

		// dL/db = 1/N sum(-2(y-ax-b))
		// where x and y come from the training data and N is the size of x and y
		double dLdb(double a, double b);

		// computes a new vector by doing the summation over the training data 
		Vector2d dL(Vector2d ab);

		// L(a,b) = 1/N sum (y-ax-b)^2
		// we know this is working when the value of the loss function gets smaller
		double Loss(double a, double b);
	public:
		// need a default constructor to appease Cython
		gradient_descent();
		// construction uses to vectors to represent data
		gradient_descent(vector<int> n, vector<int> m);

		// does the actual work of fitting a function to the data
		Vector2d fit(Vector2d ab, double gamma, double precision, bool verbose = false);

		// a slightly different version of fit so that I don't have 
		// to wrap the entire Eigen library for Cython
		vector<double> py_fit(vector<int> ab, double gamma, double precision, bool verbose = false);
};
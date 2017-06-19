/* 
 * the kernel class is an abstract class that provides the function for calculating the kernel map/Gram matrix of two matrices
 * the actual kernel calculation is left virtual so that the child classes can be created for various kernel types 
 * ie: linear, polynomial, gaussian
 */

#include "utilities.h"

class kernel{
	// the base class for kernels only provides the nullary constructor
	// child classes are expected to provide constructors for initializing their parameters
	// as well as the implementation for k
	private:
		virtual double k(VectorXd x_i, VectorXd y_j) = 0;
	public:
		MatrixXd gram_matrix(MatrixXd X, MatrixXd Y);
		// for Cython
		vector< vector<double> > py_gram_matrix(vector< vector<double> > X, vector< vector<double> > Y); 
};

class linear_kernel : public kernel{
	private:
		double _c;
		double k(VectorXd x_i, VectorXd y_j);
	public:
		linear_kernel(double c);
};

class polynomial_kernel : public kernel{
	private:
		double _a, _c, _d;
		double k(VectorXd x_i, VectorXd y_j);

	public:
		polynomial_kernel(double a, double c, double d);
};

class gaussian_kernel : public kernel{
	private:
		// sigma
		double _s;
		double k(VectorXd x_i, VectorXd y_j);
	public:
		gaussian_kernel(double s);
};
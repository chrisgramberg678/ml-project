/* 
 * the kernel class is an abstract class that provides the function for calculating the kernel map/Gram matrix of two matrices
 * the actual kernel calculation is left virtual so that the child classes can be created for various kernel types 
 * ie: linear, polynomial, gaussian
 */

#include <Eigen/Dense>
#include <iostream>
#include "math.h"
using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

class kernel{
	// the base class for kernels only provides the nullary constructor
	// child classes are expected to provide constructors for initializing their parameters
	// as well as the implementation for k
	public:
		virtual double k(VectorXd x_i, VectorXd y_j) = 0;
		MatrixXd gram_matrix(MatrixXd X, MatrixXd Y);
};

class linear_kernel : public kernel{
	private:
		double _c;
	public:
		linear_kernel(double c);
		double k(VectorXd x_i, VectorXd y_j);
};

class polynomial_kernel : public kernel{
	private:
		double _a, _c, _d;

	public:
		polynomial_kernel(double a, double c, double d);
		double k(VectorXd x_i, VectorXd y_j);
};

class gaussian_kernel : public kernel{
	private:
		// sigma
		double _s;
	public:
		gaussian_kernel(double s);
		double k(VectorXd x_i, VectorXd y_j);
};
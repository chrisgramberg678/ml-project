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
	// the base class for kernel's only provides the nullary constructor
	// child classes are expected to provide constructors for initializing their parameters
	public:
		virtual double k(VectorXd x_i, VectorXd y_j);
		MatrixXd gram_matrix(MatrixXd X, MatrixXd Y);
};

class linear_kernel{
	private:
		double _c;
	public:
		// http://crsouza.com/2010/03/17/kernel-functions-for-machine-learning-applications/#linear
		linear_kernel(double c);
		double k(VectorXd x_i, VectorXd y_j);
};

class polynomial_kernel{
	private:
		double _a, _c, _d;

	public:
		//http://crsouza.com/2010/03/17/kernel-functions-for-machine-learning-applications/#polynomial
		polynomial_kernel(double a, double c, double d);
		double k(VectorXd x_i, VectorXd y_j);
};

class gaussian_kernel{
	private:
		// sigma
		double _s;
	public:
		// http://crsouza.com/2010/03/17/kernel-functions-for-machine-learning-applications/#gaussian
		polynomial_kernel(double s);
		double k(VectorXd x_i, VectorXd y_j);
};